from collections import defaultdict


class TransactionRepository:
    """Python port of the Java transaction repository."""

    HEADER = "step,type,amount,nameOrig,oldbalanceOrig,newbalanceOrig,nameDest,oldbalanceDest,newbalanceDest,isSAR,alertID\n"

    def __init__(self, size):
        self.size = int(size)
        self.index = 0
        self.count = 0
        self.limit = (1 << 31) - 1

        self.steps = [0] * self.size
        self.descriptions = [""] * self.size
        self.amounts = [0.0] * self.size
        self.orig_ids = [""] * self.size
        self.dest_ids = [""] * self.size

        self.orig_before = [0.0] * self.size
        self.orig_after = [0.0] * self.size
        self.dest_before = [0.0] * self.size
        self.dest_after = [0.0] * self.size
        self.is_sar = [False] * self.size
        self.alert_ids = [0] * self.size

        self.tx_counter = defaultdict(int)
        self.sar_tx_counter = defaultdict(int)
        self.log_writer = None

    def setLimit(self, limit):
        self.limit = int(limit)

    def initLogWriter(self, log_file_name):
        self.closeLogWriter()
        self.log_writer = open(log_file_name, "w", newline="")
        self.log_writer.write(self.HEADER)
        self.log_writer.flush()

    def closeLogWriter(self):
        if self.log_writer is not None:
            self.log_writer.close()
            self.log_writer = None

    @staticmethod
    def _double_precision(value):
        # Keep Java semantics: cast to int truncates toward zero.
        return int(value * 100) / 100.0

    def addTransaction(self, step, desc, amt, orig_id, dest_id, orig_before, orig_after,
                       dest_before, dest_after, is_sar, alert_id):
        if self.count >= self.limit:
            if self.count == self.limit:
                print("Warning: the number of output transactions has reached the limit:", self.limit)
                self.flushLog()
                self.count += 1
            return

        self.steps[self.index] = int(step)
        self.descriptions[self.index] = str(desc)
        self.amounts[self.index] = float(amt)
        self.orig_ids[self.index] = str(orig_id)
        self.dest_ids[self.index] = str(dest_id)
        self.orig_before[self.index] = float(orig_before)
        self.orig_after[self.index] = float(orig_after)
        self.dest_before[self.index] = float(dest_before)
        self.dest_after[self.index] = float(dest_after)
        self.is_sar[self.index] = bool(is_sar)
        self.alert_ids[self.index] = int(alert_id)

        if is_sar:
            self.sar_tx_counter[step] += 1
        elif "CASH-" not in desc:
            self.tx_counter[step] += 1

        self.count += 1
        self.index += 1
        if self.index >= self.size:
            self.flushLog()

    def writeCounterLog(self, steps, log_file):
        with open(log_file, "w", newline="") as writer:
            writer.write("step,normal,SAR\n")
            tx_counter = self.tx_counter
            sar_tx_counter = self.sar_tx_counter
            writer.writelines(
                f"{step},{tx_counter[step]},{sar_tx_counter[step]}\n"
                for step in range(int(steps))
            )
            writer.flush()

    def flushLog(self):
        if self.index == 0:
            return
        if self.log_writer is None:
            raise RuntimeError("Transaction log writer is not initialized")

        size = self.index
        steps = self.steps
        descriptions = self.descriptions
        amounts = self.amounts
        orig_ids = self.orig_ids
        dest_ids = self.dest_ids
        orig_before = self.orig_before
        orig_after = self.orig_after
        dest_before = self.dest_before
        dest_after = self.dest_after
        is_sar = self.is_sar
        alert_ids = self.alert_ids
        double_precision = self._double_precision

        self.log_writer.writelines(
            f"{steps[i]},{descriptions[i]},{double_precision(amounts[i])},"
            f"{orig_ids[i]},{double_precision(orig_before[i])},"
            f"{double_precision(orig_after[i])},{dest_ids[i]},"
            f"{double_precision(dest_before[i])},{double_precision(dest_after[i])},"
            f"{'1' if is_sar[i] else '0'},{alert_ids[i]}\n"
            for i in range(size)
        )
        self.log_writer.flush()
        self.index = 0
