import json
import os


class SimProperties:
    """Python equivalent of Java SimProperties."""

    def __init__(self, json_name, simulation_name=None):
        with open(json_name, "r") as rf:
            json_object = json.load(rf)

        self.general_prop = json_object.get("general", {})
        self.sim_prop = json_object.get("simulator", {})
        self.input_prop = json_object.get("temporal", {})
        self.output_prop = json_object.get("output", {})
        self.default_prop = json_object.get("default", {})

        self.cash_in_prop = self.default_prop.get("cash_in", {})
        self.cash_out_prop = self.default_prop.get("cash_out", {})
        self.normal_tx_interval = int(self.sim_prop.get("transaction_interval", 1))
        self.min_tx_amount = float(self.default_prop.get("min_amount", 0.0))
        self.max_tx_amount = float(self.default_prop.get("max_amount", 0.0))
        self.margin_ratio = float(self.default_prop.get("margin_ratio", 0.0))

        env_seed = os.getenv("RANDOM_SEED")
        self.seed = int(env_seed) if env_seed is not None else int(self.general_prop.get("random_seed", 0))

        if simulation_name is not None:
            self.sim_name = simulation_name
        else:
            env_sim_name = os.getenv("SIMULATION_NAME")
            self.sim_name = env_sim_name if env_sim_name is not None else self.general_prop.get("simulation_name", "sample")

        base_input_dir = self.input_prop.get("directory", "tmp")
        self.work_dir = os.path.join(base_input_dir, self.sim_name)

    def get_sim_name(self):
        return self.sim_name

    def get_seed(self):
        return self.seed

    def get_steps(self):
        return int(self.general_prop.get("total_steps", 0))

    def is_compute_diameter(self):
        return bool(self.sim_prop.get("compute_diameter", False))

    def get_transaction_limit(self):
        return int(self.sim_prop.get("transaction_limit", 0))

    def get_normal_transaction_interval(self):
        return self.normal_tx_interval

    def get_min_transaction_amount(self):
        return self.min_tx_amount

    def get_max_transaction_amount(self):
        return self.max_tx_amount

    def get_margin_ratio(self):
        return self.margin_ratio

    def get_num_branches(self):
        return int(self.sim_prop.get("numBranches", 1))

    def get_input_acct_file(self):
        return os.path.join(self.work_dir, self.input_prop.get("accounts", "accounts.csv"))

    def get_input_tx_file(self):
        return os.path.join(self.work_dir, self.input_prop.get("transactions", "transactions.csv"))

    def get_input_alert_member_file(self):
        return os.path.join(self.work_dir, self.input_prop.get("alert_members", "alert_members.csv"))

    def get_normal_models_file(self):
        return os.path.join(self.work_dir, self.input_prop.get("normal_models", "normal_models.csv"))

    def get_output_dir(self):
        return os.path.join(self.output_prop.get("directory", "outputs"), self.sim_name)

    def get_output_tx_log_file(self):
        return os.path.join(self.get_output_dir(), self.output_prop.get("transaction_log", "tx_log.csv"))

    def get_counter_log_file(self):
        return os.path.join(self.get_output_dir(), self.output_prop.get("counter_log", "tx_count.csv"))

    def get_diameter_log_file(self):
        return os.path.join(self.work_dir, self.output_prop.get("diameter_log", "diameter.csv"))

    def get_cash_tx_interval(self, is_cash_in, is_sar):
        key = "fraud_interval" if is_sar else "normal_interval"
        prop = self.cash_in_prop if is_cash_in else self.cash_out_prop
        return int(prop.get(key, 0))

    def get_cash_tx_min_amount(self, is_cash_in, is_sar):
        key = "fraud_min_amount" if is_sar else "normal_min_amount"
        prop = self.cash_in_prop if is_cash_in else self.cash_out_prop
        return float(prop.get(key, 0.0))

    def get_cash_tx_max_amount(self, is_cash_in, is_sar):
        key = "fraud_max_amount" if is_sar else "normal_max_amount"
        prop = self.cash_in_prop if is_cash_in else self.cash_out_prop
        return float(prop.get(key, 0.0))
