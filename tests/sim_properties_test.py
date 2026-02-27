import json
import os
import tempfile
import unittest
from pathlib import Path

from amlsim.sim_properties import SimProperties


class SimPropertiesTests(unittest.TestCase):

    def test_loads_paths_and_defaults(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            conf_path = tmp_path / "conf.json"
            conf = {
                "general": {"random_seed": 7, "simulation_name": "base", "total_steps": 10},
                "default": {
                    "min_amount": 100,
                    "max_amount": 1000,
                    "margin_ratio": 0.1,
                    "cash_in": {"normal_interval": 5, "fraud_interval": 2, "normal_min_amount": 1, "normal_max_amount": 2,
                                "fraud_min_amount": 3, "fraud_max_amount": 4},
                    "cash_out": {"normal_interval": 6, "fraud_interval": 3, "normal_min_amount": 5, "normal_max_amount": 6,
                                 "fraud_min_amount": 7, "fraud_max_amount": 8}
                },
                "simulator": {"compute_diameter": False, "transaction_limit": 0, "transaction_interval": 9, "numBranches": 5},
                "temporal": {"directory": "tmp", "accounts": "accounts.csv", "transactions": "transactions.csv",
                             "alert_members": "alert_members.csv", "normal_models": "normal_models.csv"},
                "output": {"directory": "outputs", "transaction_log": "tx_log.csv", "counter_log": "tx_count.csv",
                           "diameter_log": "diameter.csv"}
            }
            conf_path.write_text(json.dumps(conf))

            props = SimProperties(str(conf_path))
            self.assertEqual(props.get_seed(), 7)
            self.assertEqual(props.get_sim_name(), "base")
            self.assertEqual(props.get_steps(), 10)
            self.assertEqual(props.get_input_acct_file(), os.path.join("tmp", "base", "accounts.csv"))
            self.assertEqual(props.get_output_tx_log_file(), os.path.join("outputs", "base", "tx_log.csv"))
            self.assertEqual(props.get_cash_tx_interval(True, False), 5)
            self.assertEqual(props.get_cash_tx_max_amount(False, True), 8.0)

    def test_env_seed_and_sim_name_override(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            conf_path = tmp_path / "conf.json"
            conf = {
                "general": {"random_seed": 1, "simulation_name": "conf_name", "total_steps": 1},
                "default": {"min_amount": 1, "max_amount": 2, "cash_in": {}, "cash_out": {}},
                "simulator": {},
                "temporal": {"directory": "tmp"},
                "output": {"directory": "outputs"}
            }
            conf_path.write_text(json.dumps(conf))

            old_seed = os.environ.get("RANDOM_SEED")
            old_name = os.environ.get("SIMULATION_NAME")
            os.environ["RANDOM_SEED"] = "99"
            os.environ["SIMULATION_NAME"] = "env_name"
            try:
                props = SimProperties(str(conf_path))
            finally:
                if old_seed is None:
                    os.environ.pop("RANDOM_SEED", None)
                else:
                    os.environ["RANDOM_SEED"] = old_seed
                if old_name is None:
                    os.environ.pop("SIMULATION_NAME", None)
                else:
                    os.environ["SIMULATION_NAME"] = old_name

            self.assertEqual(props.get_seed(), 99)
            self.assertEqual(props.get_sim_name(), "env_name")


if __name__ == ' main ':
    unittest.main()
