class Account:
    all_tx_types = []

    def __init__(self, account_id, init_balance, bank_id, random_source, is_sar=False):
        self.id = str(account_id)
        self.balance = float(init_balance)
        self.bank_id = str(bank_id)
        self.random = random_source
        self.is_sar = bool(is_sar)

        self.bene_accounts = []
        self.bene_account_ids = set()
        self.orig_accounts = []
        self.orig_account_ids = set()
        self.num_sar_bene = 0
        self.prev_orig = None
        self.tx_types = {}
        self._tx_type_values_cache = None
        self.alerts = []
        self.account_groups = []
        self.branch = None

        self.start_step = 0
        self.end_step = 0

    def get_id(self):
        return self.id

    def get_balance(self):
        return self.balance

    def set_balance(self, balance):
        self.balance = float(balance)

    def withdraw(self, amount):
        amount = float(amount)
        if self.balance < amount:
            self.balance = 0.0
        else:
            self.balance -= amount

    def deposit(self, amount):
        self.balance += float(amount)

    def add_bene_account(self, bene):
        if bene.id in self.bene_account_ids:
            return
        self.bene_accounts.append(bene)
        self.bene_account_ids.add(bene.id)
        bene.orig_accounts.append(self)
        bene.orig_account_ids.add(self.id)
        if bene.is_sar:
            self.num_sar_bene += 1

    def add_tx_type(self, bene, tx_type):
        tx_type = str(tx_type)
        self.tx_types[bene.id] = tx_type
        self._tx_type_values_cache = None
        Account.all_tx_types.append(tx_type)

    def get_tx_type(self, bene):
        dest_id = bene.id
        if dest_id in self.tx_types:
            return self.tx_types[dest_id]
        if self.tx_types:
            if self._tx_type_values_cache is None:
                self._tx_type_values_cache = list(self.tx_types.values())
            target = self.random.next_int(len(self._tx_type_values_cache))
            return self._tx_type_values_cache[target]
        if Account.all_tx_types:
            return Account.all_tx_types[self.random.next_int(len(Account.all_tx_types))]
        return "TRANSFER"

    def get_bene_list(self):
        return self.bene_accounts

    def get_orig_list(self):
        return self.orig_accounts

    def get_prev_orig(self):
        return self.prev_orig

    def get_num_sar_bene(self):
        return self.num_sar_bene

    def get_prop_sar_bene(self):
        if self.num_sar_bene == 0 or not self.bene_accounts:
            return 0.0
        return float(self.num_sar_bene) / float(len(self.bene_accounts))

    def add_account_group(self, account_group):
        self.account_groups.append(account_group)

    def add_alert(self, alert):
        self.alerts.append(alert)

    def set_sar(self, flag):
        self.is_sar = bool(flag)

    def is_sar_account(self):
        return self.is_sar

    def set_branch(self, branch):
        self.branch = branch

    def get_branch(self):
        return self.branch


class SARAccount(Account):
    def __init__(self, account_id, init_balance, bank_id, random_source):
        super().__init__(account_id, init_balance, bank_id, random_source, is_sar=True)


class Branch(Account):
    def __init__(self, branch_id, random_source):
        super().__init__("-", 0.0, "branch", random_source, is_sar=False)
        self.branch_id = int(branch_id)
        self.limit_amount = 100.0

    def get_limit_amount(self):
        return self.limit_amount

    def get_name(self):
        return f"B{self.branch_id}"


class AccountGroup:
    def __init__(self, account_group_id):
        self.account_group_id = int(account_group_id)
        self.members = []
        self.main_account = None
        self.model = None

    def set_model(self, model):
        self.model = model

    def register_transactions(self, step, runtime):
        if self.model is None or self.main_account is None:
            return
        self.model.send_transactions(step, self.main_account, runtime)

    def add_member(self, account):
        self.members.append(account)

    def set_main_account(self, account):
        self.main_account = account

    def get_main_account(self):
        return self.main_account


class Alert:
    def __init__(self, alert_id, model):
        self.alert_id = int(alert_id)
        self.members = []
        self.main_account = None
        self.model = model
        self.model.set_alert(self)

    def add_member(self, account):
        self.members.append(account)
        account.add_alert(self)

    def set_main_account(self, account):
        self.main_account = account

    def is_sar(self):
        return self.main_account is not None and self.main_account.is_sar_account()

    def register_transactions(self, step, runtime):
        if self.model.is_valid_step(step):
            main = self.main_account if self.main_account is not None else (self.members[0] if self.members else None)
            if main is not None:
                self.model.send_transactions(step, main, runtime)
