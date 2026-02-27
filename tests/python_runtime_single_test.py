import json
import tempfile
import unittest
from pathlib import Path

from run_py_AMLSim import run_python_simulator


class PythonRuntimeSingleModelTests(unittest.TestCase):

    def test_single_model_produces_transaction_and_counter(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            conf_path = tmp_path / "conf.json"
            sim_name = "single_case"
            conf = {
                "general": {"random_seed": 0, "simulation_name": sim_name, "total_steps": 10},
                "default": {
                    "min_amount": 100.0,
                    "max_amount": 1000.0,
                    "margin_ratio": 0.1,
                    "cash_in": {},
                    "cash_out": {}
                },
                "simulator": {
                    "compute_diameter": False,
                    "transaction_limit": 0,
                    "transaction_interval": 7,
                    "numBranches": 2
                },
                "temporal": {
                    "directory": str(tmp_path / "tmp"),
                    "accounts": "accounts.csv",
                    "transactions": "transactions.csv",
                    "alert_members": "alert_members.csv",
                    "normal_models": "normal_models.csv"
                },
                "output": {
                    "directory": str(tmp_path / "outputs"),
                    "transaction_log": "tx_log.csv",
                    "counter_log": "tx_count.csv",
                    "diameter_log": "diameter.csv"
                }
            }
            conf_path.write_text(json.dumps(conf))

            input_dir = tmp_path / "tmp" / sim_name
            input_dir.mkdir(parents=True, exist_ok=True)
            (input_dir / "accounts.csv").write_text(
                "ACCOUNT_ID,CUSTOMER_ID,INIT_BALANCE,COUNTRY,ACCOUNT_TYPE,IS_SAR,BANK_ID\n"
                "0,C_0,1000.0,US,I,false,bank\n"
                "1,C_1,1000.0,US,I,false,bank\n"
            )
            (input_dir / "transactions.csv").write_text(
                "id,src,dst,ttype\n"
                "0,0,1,TRANSFER\n"
            )
            (input_dir / "normal_models.csv").write_text(
                "modelID,type,accountID,isMain,isSAR,scheduleID\n"
                "1,single,0,True,False,2\n"
                "1,single,1,False,False,2\n"
            )

            run_python_simulator(str(conf_path))

            out_dir = tmp_path / "outputs" / sim_name
            tx_log_lines = (out_dir / "tx_log.csv").read_text().strip().splitlines()
            counter_lines = (out_dir / "tx_count.csv").read_text().strip().splitlines()

            self.assertGreaterEqual(len(tx_log_lines), 2)  # header + at least 1 tx
            self.assertTrue(tx_log_lines[1].startswith("0,TRANSFER,"))
            self.assertEqual(counter_lines[0], "step,normal,SAR")
            self.assertEqual(counter_lines[1], "0,1,0")


if __name__ == ' main ':
    unittest.main()
