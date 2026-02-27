import tempfile
import unittest
from pathlib import Path

from amlsim.transaction_repository import TransactionRepository


class TransactionRepositoryTests(unittest.TestCase):

    def test_write_log_and_counters(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            log_file = tmp_path / "tx_log.csv"
            counter_file = tmp_path / "tx_count.csv"

            repository = TransactionRepository(size=2)
            repository.initLogWriter(str(log_file))

            repository.addTransaction(0, "TRANSFER", 12.3499, "1", "2", 100.129, 87.779, 10.991, 23.344, False, -1)
            repository.addTransaction(0, "CASH-IN", 5.555, "3", "4", 20.0, 14.445, 0.0, 5.555, False, -1)
            repository.addTransaction(1, "TRANSFER", 9.999, "2", "1", 87.779, 77.78, 23.344, 33.343, True, 99)
            repository.flushLog()
            repository.closeLogWriter()

            repository.writeCounterLog(3, str(counter_file))

            self.assertTrue(log_file.exists())
            self.assertTrue(counter_file.exists())

            log_lines = log_file.read_text().strip().splitlines()
            self.assertEqual(log_lines[0], TransactionRepository.HEADER.strip())
            self.assertIn("0,TRANSFER,12.34,1,100.12,87.77,2,10.99,23.34,0,-1", log_lines)
            self.assertIn("0,CASH-IN,5.55,3,20.0,14.44,4,0.0,5.55,0,-1", log_lines)
            self.assertIn("1,TRANSFER,9.99,2,87.77,77.78,1,23.34,33.34,1,99", log_lines)

            counter_lines = counter_file.read_text().strip().splitlines()
            self.assertEqual(counter_lines[0], "step,normal,SAR")
            self.assertEqual(counter_lines[1], "0,1,0")
            self.assertEqual(counter_lines[2], "1,0,1")
            self.assertEqual(counter_lines[3], "2,0,0")

    def test_limit_stops_writes_after_warning_boundary(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            log_file = Path(tmp_dir) / "tx_log.csv"
            repository = TransactionRepository(size=10)
            repository.setLimit(1)
            repository.initLogWriter(str(log_file))

            repository.addTransaction(0, "TRANSFER", 1.0, "1", "2", 10.0, 9.0, 0.0, 1.0, False, -1)
            repository.addTransaction(0, "TRANSFER", 2.0, "1", "2", 9.0, 7.0, 1.0, 3.0, False, -1)
            repository.addTransaction(0, "TRANSFER", 3.0, "1", "2", 7.0, 4.0, 3.0, 6.0, False, -1)
            repository.closeLogWriter()

            lines = log_file.read_text().strip().splitlines()
            self.assertEqual(len(lines), 2)


if __name__ == ' main ':
    unittest.main()
