from typing import List


class TargetedTransactionAmount:
    def __init__(self, target, sim_properties, random_source):
        self.sim_properties = sim_properties
        self.random = random_source
        self.target = float(target)

    @staticmethod
    def sample(target, min_transaction_amount, max_transaction_amount, random_source):
        target = float(target)
        min_transaction_amount = float(min_transaction_amount)
        max_transaction_amount = float(max_transaction_amount)

        max_amount = target if target < max_transaction_amount else max_transaction_amount
        min_amount = target if target < min_transaction_amount else min_transaction_amount

        if max_amount - min_amount <= 0:
            return target
        if target - min_amount <= 100:
            return target
        return min_amount + random_source.next_double() * (max_amount - min_amount)

    def double_value(self):
        return self.sample(
            self.target,
            self.sim_properties.get_min_transaction_amount(),
            self.sim_properties.get_max_transaction_amount(),
            self.random,
        )


class AbstractTransactionModel:
    SINGLE = "single"
    FAN_OUT = "fan_out"
    FAN_IN = "fan_in"
    MUTUAL = "mutual"
    FORWARD = "forward"
    PERIODICAL = "periodical"

    def __init__(self, random_source):
        self.random = random_source
        self.interval = 1
        self.start_step = -1
        self.end_step = -1

    @staticmethod
    def generate_start_step(random_source, range_size):
        if range_size <= 0:
            return 0
        return random_source.next_int(int(range_size))

    def set_parameters(self, interval, start, end):
        self.interval = int(interval)
        self.start_step = int(start)
        self.end_step = int(end)

    def get_number_of_transactions(self, total_steps):
        if self.interval <= 0:
            return 0
        return int(total_steps) // int(self.interval)

    def send_transactions(self, step, account, runtime):
        raise NotImplementedError

    def make_transaction(self, step, amount, orig, dest, runtime, is_sar=False, alert_id=-1):
        if amount <= 0:
            return
        tx_type = orig.get_tx_type(dest)
        runtime.handle_transaction(step, tx_type, amount, orig, dest, is_sar=is_sar, alert_id=alert_id)

    def make_cash_transaction(self, step, amount, orig, dest, tx_type, runtime):
        runtime.handle_transaction(step, tx_type, amount, orig, dest, is_sar=False, alert_id=-1)

    def targeted_amount(self, target, runtime):
        return TargetedTransactionAmount.sample(
            target,
            runtime.min_transaction_amount,
            runtime.max_transaction_amount,
            self.random,
        )


class EmptyModel(AbstractTransactionModel):
    def __init__(self, random_source):
        super().__init__(random_source)

    def send_transactions(self, step, account, runtime):
        return


class FanInTransactionModel(AbstractTransactionModel):
    def __init__(self, random_source):
        super().__init__(random_source)
        self.index = 0

    def set_parameters(self, interval, start, end):
        super().set_parameters(interval, start, end)
        if self.start_step < 0:
            self.start_step = self.generate_start_step(self.random, interval)

    def send_transactions(self, step, account, runtime):
        bene_list = account.get_bene_list()
        num_origs = len(bene_list)
        if num_origs == 0 or (step - self.start_step) % self.interval != 0:
            return
        if self.index >= num_origs:
            self.index = 0

        amount = self.targeted_amount(account.get_balance(), runtime)
        bene = bene_list[self.index]
        self.make_transaction(step, amount, account, bene, runtime)
        self.index += 1


class FanOutTransactionModel(AbstractTransactionModel):
    def __init__(self, random_source):
        super().__init__(random_source)
        self.index = 0

    def set_parameters(self, interval, start, end):
        super().set_parameters(interval, start, end)
        if self.start_step < 0:
            self.start_step = self.generate_start_step(self.random, interval)

    def send_transactions(self, step, account, runtime):
        bene_list = account.get_bene_list()
        num_bene = len(bene_list)
        if num_bene == 0 or (step - self.start_step) % self.interval != 0:
            return
        if self.index >= num_bene:
            self.index = 0

        amount = self.targeted_amount(account.get_balance(), runtime)
        bene = bene_list[self.index]
        if amount > 0:
            self.make_transaction(step, amount, account, bene, runtime)
        self.index += 1


class ForwardTransactionModel(AbstractTransactionModel):
    def __init__(self, random_source):
        super().__init__(random_source)
        self.index = 0

    def set_parameters(self, interval, start, end):
        super().set_parameters(interval, start, end)
        if self.start_step < 0:
            self.start_step = self.generate_start_step(self.random, interval)

    def send_transactions(self, step, account, runtime):
        dests = account.get_bene_list()
        num_dests = len(dests)
        if num_dests == 0:
            return
        if (step - self.start_step) % self.interval != 0:
            return
        if self.index >= num_dests:
            self.index = 0

        amount = self.targeted_amount(account.get_balance(), runtime)
        dest = dests[self.index]
        self.make_transaction(step, amount, account, dest, runtime)
        self.index += 1


class MutualTransactionModel(AbstractTransactionModel):
    def __init__(self, random_source):
        super().__init__(random_source)

    def set_parameters(self, interval, start, end):
        super().set_parameters(interval, start, end)
        if self.start_step < 0:
            self.start_step = self.generate_start_step(self.random, interval)

    def send_transactions(self, step, account, runtime):
        if (step - self.start_step) % self.interval != 0:
            return

        counterpart = account.get_prev_orig()
        if counterpart is None:
            origs = account.get_orig_list()
            if not origs:
                return
            counterpart = origs[0]

        amount = self.targeted_amount(account.get_balance(), runtime)
        if counterpart not in account.get_bene_list():
            account.add_bene_account(counterpart)
        self.make_transaction(step, amount, account, counterpart, runtime)


class PeriodicalTransactionModel(AbstractTransactionModel):
    def __init__(self, random_source):
        super().__init__(random_source)
        self.index = 0

    def set_parameters(self, interval, start, end):
        super().set_parameters(interval, start, end)
        if self.start_step < 0:
            self.start_step = self.generate_start_step(self.random, interval)

    def send_transactions(self, step, account, runtime):
        bene_list = account.get_bene_list()
        if not bene_list:
            return
        if (step - self.start_step) % self.interval != 0:
            return

        num_dests = len(bene_list)
        if self.index >= num_dests:
            self.index = 0

        total_count = self.get_number_of_transactions(runtime.sim_properties.get_steps())
        each_count = 1 if num_dests < total_count or total_count <= 0 else num_dests // total_count
        base_amount = account.get_balance() / max(each_count, 1)

        for _ in range(each_count):
            dest = bene_list[self.index]
            self.make_transaction(step, self.targeted_amount(base_amount, runtime), account, dest, runtime)
            self.index += 1
            if self.index >= num_dests:
                break
        self.index = 0


class SingleTransactionModel(AbstractTransactionModel):
    def __init__(self, random_source):
        super().__init__(random_source)
        self.tx_step = -1

    def set_parameters(self, interval, start, end, total_steps):
        super().set_parameters(interval, start, end)
        if self.start_step < 0:
            self.start_step = 0
        if self.end_step < 0:
            self.end_step = int(total_steps)

        range_len = int(self.end_step - self.start_step + 1)
        if range_len <= 0:
            self.tx_step = self.start_step
        else:
            self.tx_step = self.start_step + self.random.next_int(range_len)

    def send_transactions(self, step, account, runtime):
        bene_list = account.get_bene_list()
        num_bene = len(bene_list)
        if step != self.tx_step or num_bene == 0:
            return

        amount = self.targeted_amount(account.get_balance(), runtime)
        index = self.random.next_int(num_bene)
        dest = bene_list[index]
        self.make_transaction(step, amount, account, dest, runtime)


class CashInModel(AbstractTransactionModel):
    NORMAL_INTERVAL = 1
    SUSPICIOUS_INTERVAL = 1
    NORMAL_MIN = 10.0
    NORMAL_MAX = 100.0
    SUSPICIOUS_MIN = 10.0
    SUSPICIOUS_MAX = 100.0

    @classmethod
    def set_param(cls, norm_int, case_int, norm_min, norm_max, case_min, case_max):
        cls.NORMAL_INTERVAL = int(norm_int)
        cls.SUSPICIOUS_INTERVAL = int(case_int)
        cls.NORMAL_MIN = float(norm_min)
        cls.NORMAL_MAX = float(norm_max)
        cls.SUSPICIOUS_MIN = float(case_min)
        cls.SUSPICIOUS_MAX = float(case_max)

    def __init__(self, account, random_source):
        super().__init__(random_source)
        self.account = account

    def is_next_step(self, step):
        interval = self.SUSPICIOUS_INTERVAL if self.account.is_sar_account() else self.NORMAL_INTERVAL
        return interval > 0 and step % interval == 0

    def compute_amount(self):
        if self.account.is_sar_account():
            return self.SUSPICIOUS_MIN + self.random.next_float() * (self.SUSPICIOUS_MAX - self.SUSPICIOUS_MIN)
        return self.NORMAL_MIN + self.random.next_float() * (self.NORMAL_MAX - self.NORMAL_MIN)

    def make_cash(self, step, runtime):
        if self.is_next_step(step):
            branch = self.account.get_branch()
            amount = self.compute_amount()
            self.make_cash_transaction(step, amount, self.account, branch, "CASH-IN", runtime)


class CashOutModel(AbstractTransactionModel):
    NORMAL_INTERVAL = 1
    SUSPICIOUS_INTERVAL = 1
    NORMAL_MIN = 10.0
    NORMAL_MAX = 100.0
    SUSPICIOUS_MIN = 10.0
    SUSPICIOUS_MAX = 100.0

    @classmethod
    def set_param(cls, norm_int, case_int, norm_min, norm_max, case_min, case_max):
        cls.NORMAL_INTERVAL = int(norm_int)
        cls.SUSPICIOUS_INTERVAL = int(case_int)
        cls.NORMAL_MIN = float(norm_min)
        cls.NORMAL_MAX = float(norm_max)
        cls.SUSPICIOUS_MIN = float(case_min)
        cls.SUSPICIOUS_MAX = float(case_max)

    def __init__(self, account, random_source):
        super().__init__(random_source)
        self.account = account

    def is_next_step(self, step):
        interval = self.SUSPICIOUS_INTERVAL if self.account.is_sar_account() else self.NORMAL_INTERVAL
        return interval > 0 and step % interval == 0

    def compute_amount(self):
        if self.account.is_sar_account():
            return self.SUSPICIOUS_MIN + self.random.next_float() * (self.SUSPICIOUS_MAX - self.SUSPICIOUS_MIN)
        return self.NORMAL_MIN + self.random.next_float() * (self.NORMAL_MAX - self.NORMAL_MIN)

    def make_cash(self, step, runtime):
        if self.is_next_step(step):
            branch = self.account.get_branch()
            amount = self.compute_amount()
            self.make_cash_transaction(step, amount, branch, self.account, "CASH-OUT", runtime)


class AMLTypology(AbstractTransactionModel):
    AML_FAN_OUT = 1
    AML_FAN_IN = 2
    CYCLE = 3
    BIPARTITE = 4
    STACK = 5
    RANDOM = 6
    SCATTER_GATHER = 7
    GATHER_SCATTER = 8

    FIXED_INTERVAL = 0
    RANDOM_INTERVAL = 1
    UNORDERED = 2
    SIMULTANEOUS = 3

    def __init__(self, min_amount, max_amount, start_step, end_step, random_source, margin_ratio, total_steps):
        super().__init__(random_source)
        self.alert = None
        self.min_amount = float(min_amount)
        self.max_amount = float(max_amount)
        self.start_step = int(start_step)
        self.end_step = int(end_step)
        self.margin_ratio = float(margin_ratio)
        self.total_steps = int(total_steps)

    @staticmethod
    def create_typology(model_id, min_amount, max_amount, start_step, end_step, random_source, margin_ratio, total_steps):
        if model_id == AMLTypology.AML_FAN_OUT:
            return FanOutTypology(min_amount, max_amount, start_step, end_step, random_source, margin_ratio, total_steps)
        if model_id == AMLTypology.AML_FAN_IN:
            return FanInTypology(min_amount, max_amount, start_step, end_step, random_source, margin_ratio, total_steps)
        if model_id == AMLTypology.CYCLE:
            return CycleTypology(min_amount, max_amount, start_step, end_step, random_source, margin_ratio, total_steps)
        if model_id == AMLTypology.BIPARTITE:
            return BipartiteTypology(min_amount, max_amount, start_step, end_step, random_source, margin_ratio, total_steps)
        if model_id == AMLTypology.STACK:
            return StackTypology(min_amount, max_amount, start_step, end_step, random_source, margin_ratio, total_steps)
        if model_id == AMLTypology.RANDOM:
            return RandomTypology(min_amount, max_amount, start_step, end_step, random_source, margin_ratio, total_steps)
        if model_id == AMLTypology.SCATTER_GATHER:
            return ScatterGatherTypology(min_amount, max_amount, start_step, end_step, random_source, margin_ratio, total_steps)
        if model_id == AMLTypology.GATHER_SCATTER:
            return GatherScatterTypology(min_amount, max_amount, start_step, end_step, random_source, margin_ratio, total_steps)
        raise ValueError(f"Unknown typology model ID: {model_id}")

    def set_alert(self, alert):
        self.alert = alert

    def is_valid_step(self, step):
        return self.start_step <= step <= self.end_step

    def get_step_range(self):
        return int(self.end_step - self.start_step + 1)

    def update_min_amount(self, min_amount):
        self.min_amount = min(self.min_amount, float(min_amount))

    def update_max_amount(self, max_amount):
        self.max_amount = max(self.max_amount, float(max_amount))

    def update_start_step(self, start_step):
        self.start_step = min(self.start_step, int(start_step))

    def update_end_step(self, end_step):
        self.end_step = max(self.end_step, int(end_step))

    def get_random_amount(self):
        return self.random.next_double() * (self.max_amount - self.min_amount) + self.min_amount

    def get_random_step(self):
        range_size = self.get_step_range()
        if range_size <= 0:
            return self.start_step
        return self.random.next_long(range_size) + self.start_step

    def get_random_step_range(self, start, end):
        start = int(start)
        end = int(end)
        if start < self.start_step or self.end_step < end:
            raise ValueError("start/end must be inside typology bounds")
        if end < start:
            raise ValueError("start/end are unordered")
        range_size = int(end - start + 1)
        if range_size <= 0:
            return start
        return self.random.next_long(range_size) + start

    def make_transaction_noop(self, step):
        return


class FanInTypology(AMLTypology):
    def __init__(self, min_amount, max_amount, start_step, end_step, random_source, margin_ratio, total_steps):
        super().__init__(min_amount, max_amount, start_step, end_step, random_source, margin_ratio, total_steps)
        self.bene = None
        self.orig_list: List = []
        self.steps = []

    def set_parameters(self, scheduling_id):
        members = self.alert.members
        main = self.alert.main_account
        self.bene = main if main is not None else members[0]
        self.orig_list = [orig for orig in members if orig is not self.bene]

        num_origs = len(self.orig_list)
        if num_origs == 0:
            self.steps = []
            return
        total_step = int(self.end_step - self.start_step + 1)
        default_interval = max(total_step // num_origs, 1)
        self.start_step = self.generate_start_step(self.random, default_interval)

        self.steps = [0] * num_origs
        if scheduling_id == self.SIMULTANEOUS:
            step = self.get_random_step()
            self.steps = [step] * num_origs
        elif scheduling_id == self.FIXED_INTERVAL:
            range_size = int(self.end_step - self.start_step + 1)
            if num_origs < range_size:
                self.interval = range_size // num_origs
                for i in range(num_origs):
                    self.steps[i] = self.start_step + self.interval * i
            else:
                batch = max(num_origs // max(range_size, 1), 1)
                for i in range(num_origs):
                    self.steps[i] = self.start_step + i // batch
        else:
            for i in range(num_origs):
                self.steps[i] = self.get_random_step()

    def send_transactions(self, step, account, runtime):
        alert_id = self.alert.alert_id
        is_sar = self.alert.is_sar()
        for i, orig in enumerate(self.orig_list):
            if self.steps[i] == step:
                amount = self.targeted_amount(orig.get_balance(), runtime)
                self.make_transaction(step, amount, orig, self.bene, runtime, is_sar=is_sar, alert_id=alert_id)


class FanOutTypology(AMLTypology):
    def __init__(self, min_amount, max_amount, start_step, end_step, random_source, margin_ratio, total_steps):
        super().__init__(min_amount, max_amount, start_step, end_step, random_source, margin_ratio, total_steps)
        self.orig = None
        self.bene_list = []
        self.steps = []

    def set_parameters(self, schedule_id):
        members = self.alert.members
        main = self.alert.main_account
        self.orig = main if main is not None else members[0]
        self.bene_list = [bene for bene in members if bene is not self.orig]

        num_benes = len(self.bene_list)
        if num_benes == 0:
            self.steps = []
            return
        total_step = int(self.end_step - self.start_step + 1)
        default_interval = max(total_step // num_benes, 1)
        self.start_step = self.generate_start_step(self.random, default_interval)

        self.steps = [0] * num_benes
        if schedule_id == self.SIMULTANEOUS:
            step = self.get_random_step()
            self.steps = [step] * num_benes
        elif schedule_id == self.FIXED_INTERVAL:
            range_size = int(self.end_step - self.start_step + 1)
            if num_benes < range_size:
                self.interval = range_size // num_benes
                for i in range(num_benes):
                    self.steps[i] = self.start_step + self.interval * i
            else:
                batch = max(num_benes // max(range_size, 1), 1)
                for i in range(num_benes):
                    self.steps[i] = self.start_step + i // batch
        else:
            for i in range(num_benes):
                self.steps[i] = self.get_random_step()

    def send_transactions(self, step, account, runtime):
        if self.orig is None or self.orig.get_id() != account.get_id():
            return
        alert_id = self.alert.alert_id
        is_sar = self.alert.is_sar()
        num_bene = len(self.bene_list)
        amount = self.targeted_amount(self.orig.get_balance() / num_bene if num_bene else 0.0, runtime)
        for i, bene in enumerate(self.bene_list):
            if self.steps[i] == step:
                self.make_transaction(step, amount, self.orig, bene, runtime, is_sar=is_sar, alert_id=alert_id)


class CycleTypology(AMLTypology):
    def __init__(self, min_amount, max_amount, start_step, end_step, random_source, margin_ratio, total_steps):
        super().__init__(min_amount, max_amount, start_step, end_step, random_source, margin_ratio, total_steps)
        self.steps = []

    def set_parameters(self, model_id):
        members = self.alert.members
        length = len(members)
        if length == 0:
            self.steps = []
            return
        self.steps = [0] * length

        all_step = int(self.total_steps)
        period = int(self.end_step - self.start_step)
        self.start_step = self.generate_start_step(self.random, all_step - period)
        self.end_step = min(self.start_step + period, all_step)

        if model_id == self.FIXED_INTERVAL:
            period = int(self.end_step - self.start_step)
            if period <= 0:
                self.steps = [self.start_step] * length
                return
            if length < period:
                self.interval = period // length
                for i in range(length - 1):
                    self.steps[i] = self.start_step + self.interval * i
                self.steps[length - 1] = self.end_step
            else:
                self.interval = 1
                batch = max(length // period, 1)
                for i in range(length - 1):
                    self.steps[i] = self.start_step + i // batch
                self.steps[length - 1] = self.end_step
        else:
            self.interval = 1
            self.steps[0] = self.start_step
            if length > 1:
                self.steps[1] = self.end_step
            for i in range(2, length):
                self.steps[i] = self.get_random_step()
            if model_id == self.RANDOM_INTERVAL:
                self.steps.sort()

    def send_transactions(self, step, account, runtime):
        length = len(self.alert.members)
        alert_id = self.alert.alert_id
        is_sar = self.alert.is_sar()
        amount = float("inf")
        for i in range(length):
            if self.steps[i] == step:
                j = (i + 1) % length
                src = self.alert.members[i]
                dst = self.alert.members[j]
                if src.get_balance() < amount:
                    amount = src.get_balance()
                tx_amount = self.targeted_amount(amount, runtime)
                self.make_transaction(step, tx_amount, src, dst, runtime, is_sar=is_sar, alert_id=alert_id)
                margin = amount * self.margin_ratio
                amount = amount - margin


class BipartiteTypology(AMLTypology):
    def set_parameters(self, model_id):
        return

    def send_transactions(self, step, account, runtime):
        members = self.alert.members
        last_orig_index = len(members) // 2
        for i in range(last_orig_index):
            orig = members[i]
            if orig.get_id() != account.get_id():
                continue
            num_bene = len(members) - last_orig_index
            base_amount = orig.get_balance() / num_bene if num_bene else 0.0
            for j in range(last_orig_index, len(members)):
                bene = members[j]
                self.make_transaction(step, self.targeted_amount(base_amount, runtime), orig, bene, runtime)


class StackTypology(AMLTypology):
    def set_parameters(self, model_id):
        return

    def send_transactions(self, step, account, runtime):
        members = self.alert.members
        total_members = len(members)
        orig_members = total_members // 3
        mid_members = orig_members
        for i in range(orig_members):
            orig = members[i]
            if orig.get_id() != account.get_id():
                continue
            num_bene = (orig_members + mid_members) - orig_members
            base_amount = orig.get_balance() / num_bene if num_bene else 0.0
            for j in range(orig_members, orig_members + mid_members):
                bene = members[j]
                self.make_transaction(step, self.targeted_amount(base_amount, runtime), orig, bene, runtime)

        for i in range(orig_members, orig_members + mid_members):
            orig = members[i]
            if orig.get_id() != account.get_id():
                continue
            num_bene = total_members - (orig_members + mid_members)
            base_amount = orig.get_balance() / num_bene if num_bene else 0.0
            for j in range(orig_members + mid_members, total_members):
                bene = members[j]
                self.make_transaction(step, self.targeted_amount(base_amount, runtime), orig, bene, runtime)


class RandomTypology(AMLTypology):
    def __init__(self, min_amount, max_amount, start_step, end_step, random_source, margin_ratio, total_steps):
        super().__init__(min_amount, max_amount, start_step, end_step, random_source, margin_ratio, total_steps)
        self.steps = set()
        self.next_orig = None

    def set_parameters(self, model_id):
        self.steps = set()
        for _ in range(len(self.alert.members)):
            self.steps.add(self.get_random_step())
        self.next_orig = self.alert.main_account

    def is_valid_step(self, step):
        return super().is_valid_step(step) and step in self.steps

    def send_transactions(self, step, account, runtime):
        if not self.is_valid_step(step) or self.next_orig is None:
            return
        bene_list = self.next_orig.get_bene_list()
        if not bene_list:
            return
        idx = self.random.next_int(len(bene_list))
        bene = bene_list[idx]
        amount = self.targeted_amount(self.next_orig.get_balance(), runtime)
        self.make_transaction(step, amount, self.next_orig, bene, runtime, is_sar=self.alert.is_sar(), alert_id=self.alert.alert_id)
        self.next_orig = bene


class ScatterGatherTypology(AMLTypology):
    def __init__(self, min_amount, max_amount, start_step, end_step, random_source, margin_ratio, total_steps):
        super().__init__(min_amount, max_amount, start_step, end_step, random_source, margin_ratio, total_steps)
        self.orig = None
        self.bene = None
        self.intermediate = []
        self.scatter_steps = []
        self.gather_steps = []
        self.scatter_amount = 0.0
        self.gather_amount = 0.0

    def set_parameters(self, model_id):
        self.scatter_amount = self.max_amount
        margin = self.scatter_amount * self.margin_ratio
        self.gather_amount = max(self.scatter_amount - margin, self.min_amount)

        self.intermediate = []
        self.orig = self.alert.main_account
        self.bene = None
        for acct in self.alert.members:
            if acct is self.orig:
                continue
            if self.bene is None:
                self.bene = acct
            else:
                self.intermediate.append(acct)

        size = len(self.alert.members) - 2
        if size <= 0:
            self.scatter_steps = []
            self.gather_steps = []
            return
        self.scatter_steps = [0] * size
        self.gather_steps = [0] * size
        middle_step = (self.end_step + self.start_step) // 2
        self.scatter_steps[0] = self.start_step
        self.gather_steps[0] = self.end_step
        for i in range(1, size):
            self.scatter_steps[i] = self.get_random_step_range(self.start_step, middle_step)
            self.gather_steps[i] = self.get_random_step_range(middle_step + 1, self.end_step)

    def send_transactions(self, step, account, runtime):
        alert_id = self.alert.alert_id
        is_sar = self.alert.is_sar()
        num_mid = len(self.alert.members) - 2
        for i in range(num_mid):
            if self.scatter_steps[i] == step:
                bene = self.intermediate[i]
                target = min(self.orig.get_balance(), self.scatter_amount)
                amount = self.targeted_amount(target, runtime)
                self.make_transaction(step, amount, self.orig, bene, runtime, is_sar=is_sar, alert_id=alert_id)
            elif self.gather_steps[i] == step:
                orig = self.intermediate[i]
                target = min(orig.get_balance(), self.scatter_amount)
                amount = self.targeted_amount(target, runtime)
                self.make_transaction(step, amount, orig, self.bene, runtime, is_sar=is_sar, alert_id=alert_id)


class GatherScatterTypology(AMLTypology):
    def __init__(self, min_amount, max_amount, start_step, end_step, random_source, margin_ratio, total_steps):
        super().__init__(min_amount, max_amount, start_step, end_step, random_source, margin_ratio, total_steps)
        self.orig_accts = []
        self.bene_accts = []
        self.gather_steps = []
        self.scatter_steps = []
        self.middle_step = 0
        self.total_received_amount = 0.0
        self.scatter_amount = 0.0

    def set_parameters(self, model_id):
        self.middle_step = (self.start_step + self.end_step) // 2
        self.orig_accts = []
        self.bene_accts = []
        self.total_received_amount = 0.0
        self.scatter_amount = 0.0

        num_sub = len(self.alert.members) - 1
        num_orig = num_sub // 2
        num_bene = num_sub - num_orig

        self.gather_steps = [0] * num_orig
        self.scatter_steps = [0] * num_bene

        main = self.alert.main_account
        sub_members = [acct for acct in self.alert.members if acct is not main]
        for i, acct in enumerate(sub_members):
            if i < num_orig:
                self.orig_accts.append(acct)
            else:
                self.bene_accts.append(acct)

        if num_orig > 0:
            self.gather_steps[0] = self.start_step
        for i in range(1, num_orig):
            self.gather_steps[i] = self.get_random_step_range(self.start_step, self.middle_step)
        if num_bene > 0:
            self.scatter_steps[0] = self.end_step
        for i in range(1, num_bene):
            self.scatter_steps[i] = self.get_random_step_range(self.middle_step + 1, self.end_step)

    def send_transactions(self, step, account, runtime):
        alert_id = self.alert.alert_id
        is_sar = self.alert.is_sar()
        num_gather = len(self.gather_steps)
        num_scatter = len(self.scatter_steps)
        main = self.alert.main_account

        if step <= self.middle_step:
            for i in range(num_gather):
                if self.gather_steps[i] == step:
                    orig = self.orig_accts[i]
                    amount = self.targeted_amount(orig.get_balance(), runtime)
                    self.make_transaction(step, amount, orig, main, runtime, is_sar=is_sar, alert_id=alert_id)
                    self.total_received_amount += amount
        else:
            for i in range(num_scatter):
                if self.scatter_steps[i] == step:
                    bene = self.bene_accts[i]
                    target = min(main.get_balance(), self.scatter_amount)
                    amount = self.targeted_amount(target, runtime)
                    self.make_transaction(step, amount, main, bene, runtime, is_sar=is_sar, alert_id=alert_id)

        if step == self.middle_step and num_scatter > 0:
            margin = self.total_received_amount * self.margin_ratio
            self.scatter_amount = (self.total_received_amount - margin) / num_scatter
