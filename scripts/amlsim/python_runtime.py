import csv
import os

import networkx as nx

from amlsim.entities import Account, SARAccount, Branch, AccountGroup, Alert
from amlsim.java_random import JavaRandom
from amlsim.model_parameters import ModelParameters
from amlsim.models import (
    AbstractTransactionModel,
    AMLTypology,
    CashInModel,
    CashOutModel,
    EmptyModel,
    FanInTransactionModel,
    FanOutTransactionModel,
    ForwardTransactionModel,
    MutualTransactionModel,
    PeriodicalTransactionModel,
    SingleTransactionModel,
)
from amlsim.transaction_repository import TransactionRepository


def _parse_bool(value):
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes"}


class PythonAMLSim:
    def __init__(self, sim_properties):
        self.sim_properties = sim_properties
        self.random = JavaRandom(sim_properties.get_seed())
        self.min_transaction_amount = sim_properties.get_min_transaction_amount()
        self.max_transaction_amount = sim_properties.get_max_transaction_amount()
        Account.all_tx_types = []
        self.accounts = {}
        self.account_order = []
        self.branches = []
        self.account_groups = {}
        self.alerts = {}
        self.diameter_edges = set()
        self.diameter_writer = None

        tx_size = 100000
        self.transaction_repository = TransactionRepository(tx_size)
        transaction_limit = sim_properties.get_transaction_limit()
        if transaction_limit > 0:
            self.transaction_repository.setLimit(transaction_limit)

        self._configure_cash_models()
        self._init_branches()

    def execute(self):
        output_dir = self.sim_properties.get_output_dir()
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(self.sim_properties.work_dir, exist_ok=True)

        tx_log_file = self.sim_properties.get_output_tx_log_file()
        self.transaction_repository.initLogWriter(tx_log_file)
        if self.sim_properties.is_compute_diameter():
            self.diameter_writer = open(self.sim_properties.get_diameter_log_file(), "w", newline="")

        has_inputs = self._load_simulation_inputs()
        if has_inputs:
            self._run_steps()

        self.transaction_repository.flushLog()
        self.transaction_repository.closeLogWriter()
        self.transaction_repository.writeCounterLog(self.sim_properties.get_steps(), self.sim_properties.get_counter_log_file())

        if self.diameter_writer is not None:
            self.diameter_writer.close()
            self.diameter_writer = None

    def _load_simulation_inputs(self):
        input_files = [
            self.sim_properties.get_input_acct_file(),
            self.sim_properties.get_input_tx_file(),
            self.sim_properties.get_normal_models_file(),
        ]
        if not all(os.path.exists(path) for path in input_files):
            return False

        self._load_accounts()
        self._load_transactions()
        self._load_normal_models()
        self._load_alert_members()
        return True

    def _configure_cash_models(self):
        CashInModel.set_param(
            self.sim_properties.get_cash_tx_interval(True, False),
            self.sim_properties.get_cash_tx_interval(True, True),
            self.sim_properties.get_cash_tx_min_amount(True, False),
            self.sim_properties.get_cash_tx_max_amount(True, False),
            self.sim_properties.get_cash_tx_min_amount(True, True),
            self.sim_properties.get_cash_tx_max_amount(True, True),
        )
        CashOutModel.set_param(
            self.sim_properties.get_cash_tx_interval(False, False),
            self.sim_properties.get_cash_tx_interval(False, True),
            self.sim_properties.get_cash_tx_min_amount(False, False),
            self.sim_properties.get_cash_tx_max_amount(False, False),
            self.sim_properties.get_cash_tx_min_amount(False, True),
            self.sim_properties.get_cash_tx_max_amount(False, True),
        )

    def _init_branches(self):
        num_branches = self.sim_properties.get_num_branches()
        if num_branches <= 0:
            raise ValueError("numBranches must be positive")
        self.branches = [Branch(i, self.random) for i in range(num_branches)]

    def _load_accounts(self):
        with open(self.sim_properties.get_input_acct_file(), "r", newline="") as rf:
            reader = csv.DictReader(rf)
            for index, row in enumerate(reader):
                account_id = row["ACCOUNT_ID"]
                init_balance = float(row.get("INIT_BALANCE", 0.0))
                bank_id = row.get("BANK_ID", "")
                is_sar = _parse_bool(row.get("IS_SAR", False))

                account = SARAccount(account_id, init_balance, bank_id, self.random) if is_sar \
                    else Account(account_id, init_balance, bank_id, self.random, is_sar=False)
                account.set_branch(self.branches[index % len(self.branches)])
                account.cash_in_model = CashInModel(account, self.random)
                account.cash_out_model = CashOutModel(account, self.random)
                self.accounts[account_id] = account
                self.account_order.append(account)

    def _load_transactions(self):
        with open(self.sim_properties.get_input_tx_file(), "r", newline="") as rf:
            reader = csv.DictReader(rf)
            for row in reader:
                src_id = row.get("src")
                dst_id = row.get("dst")
                tx_type = row.get("ttype", "TRANSFER")
                src = self.accounts.get(src_id)
                dst = self.accounts.get(dst_id)
                if src is None or dst is None:
                    continue
                if ModelParameters.should_add_edge(src, dst):
                    src.add_bene_account(dst)
                src.add_tx_type(dst, tx_type)

    def _load_normal_models(self):
        total_steps = self.sim_properties.get_steps()
        interval = self.sim_properties.get_normal_transaction_interval()

        with open(self.sim_properties.get_normal_models_file(), "r", newline="") as rf:
            reader = csv.DictReader(rf)
            for row in reader:
                group_id = int(row.get("modelID", -1))
                model_type = str(row.get("type", "")).strip().lower()
                account_id = row.get("accountID")
                is_main = _parse_bool(row.get("isMain", False))

                account = self.accounts.get(account_id)
                if account is None:
                    continue

                group = self.account_groups.get(group_id)
                if group is None:
                    group = AccountGroup(group_id)
                    self.account_groups[group_id] = group
                    model = self._create_normal_model(model_type)
                    if model is not None:
                        if isinstance(model, SingleTransactionModel):
                            model.set_parameters(interval, -1, -1, total_steps)
                        else:
                            model.set_parameters(interval, -1, -1)
                        group.set_model(model)

                group.add_member(account)
                account.add_account_group(group)
                if is_main:
                    group.set_main_account(account)

    def _create_normal_model(self, model_type):
        if model_type == AbstractTransactionModel.SINGLE:
            return SingleTransactionModel(self.random)
        if model_type == AbstractTransactionModel.FAN_IN:
            return FanInTransactionModel(self.random)
        if model_type == AbstractTransactionModel.FAN_OUT:
            return FanOutTransactionModel(self.random)
        if model_type == AbstractTransactionModel.FORWARD:
            return ForwardTransactionModel(self.random)
        if model_type == AbstractTransactionModel.MUTUAL:
            return MutualTransactionModel(self.random)
        if model_type == AbstractTransactionModel.PERIODICAL:
            return PeriodicalTransactionModel(self.random)
        return EmptyModel(self.random)

    def _load_alert_members(self):
        path = self.sim_properties.get_input_alert_member_file()
        if not os.path.exists(path):
            return

        schedule_models = {}
        margin_ratio = self.sim_properties.get_margin_ratio()
        total_steps = self.sim_properties.get_steps()

        with open(path, "r", newline="") as rf:
            reader = csv.DictReader(rf)
            for row in reader:
                alert_id = int(row.get("alertID", -1))
                account_id = row.get("accountID")
                is_main = _parse_bool(row.get("isMain", False))
                is_sar = _parse_bool(row.get("isSAR", False))
                model_id = int(row.get("modelID", 0))
                min_amount = float(row.get("minAmount", 0.0))
                max_amount = float(row.get("maxAmount", 0.0))
                start_step = int(row.get("startStep", 0))
                end_step = int(row.get("endStep", 0))
                schedule_id = int(row.get("scheduleID", AMLTypology.UNORDERED))

                if alert_id in self.alerts:
                    alert = self.alerts[alert_id]
                    model = alert.model
                    model.update_min_amount(min_amount)
                    model.update_max_amount(max_amount)
                    model.update_start_step(start_step)
                    model.update_end_step(end_step)
                else:
                    model = AMLTypology.create_typology(
                        model_id, min_amount, max_amount, start_step, end_step,
                        self.random, margin_ratio, total_steps
                    )
                    alert = Alert(alert_id, model)
                    self.alerts[alert_id] = alert

                account = self.accounts.get(account_id)
                if account is None:
                    continue
                alert.add_member(account)
                if is_main:
                    alert.set_main_account(account)
                account.set_sar(is_sar)
                schedule_models[alert_id] = schedule_id

        for alert_id, schedule_id in schedule_models.items():
            self.alerts[alert_id].model.set_parameters(schedule_id)

    def _run_steps(self):
        total_steps = self.sim_properties.get_steps()
        alerts = list(self.alerts.values())
        groups = list(self.account_groups.values())
        accounts = list(self.account_order)
        for step in range(total_steps):
            for alert in alerts:
                if alert.main_account is not None:
                    alert.register_transactions(step, self)
            for group in groups:
                if group.get_main_account() is not None:
                    group.register_transactions(step, self)
            for account in accounts:
                account.cash_in_model.make_cash(step, self)
                account.cash_out_model.make_cash(step, self)
            if self.diameter_writer is not None and step % 10 == 0 and step > 0:
                diameter, avg = self._compute_diameter_snapshot()
                self.diameter_writer.write(f"{step},{diameter},{avg}\n")
                self.diameter_writer.flush()

    def _compute_diameter_snapshot(self):
        if not self.diameter_edges:
            return 0, 0.0

        graph = nx.DiGraph()
        graph.add_edges_from(self.diameter_edges)
        if graph.number_of_nodes() == 0:
            return 0, 0.0

        max_distance = 0
        total_distance = 0
        count = 0
        for source, lengths in nx.all_pairs_shortest_path_length(graph):
            for dest, distance in lengths.items():
                if source == dest:
                    continue
                max_distance = max(max_distance, distance)
                total_distance += distance
                count += 1
        if count == 0:
            return 0, 0.0
        return max_distance, float(total_distance) / float(count)

    def handle_transaction(self, step, desc, amount, orig, bene, is_sar=False, alert_id=-1):
        orig_before = float(orig.get_balance())
        orig.withdraw(amount)
        orig_after = float(orig.get_balance())

        bene_before = float(bene.get_balance())
        bene.deposit(amount)
        bene_after = float(bene.get_balance())

        self.transaction_repository.addTransaction(
            step, desc, amount, orig.get_id(), bene.get_id(),
            orig_before, orig_after, bene_before, bene_after, is_sar, alert_id
        )
        if orig.get_id() != "-" and bene.get_id() != "-":
            self.diameter_edges.add((orig.get_id(), bene.get_id()))
