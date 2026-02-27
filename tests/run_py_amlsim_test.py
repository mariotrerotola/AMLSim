import json
import tempfile
import unittest
from pathlib import Path

from run_py_AMLSim import run_python_simulator


class RunPyAMLSimTests(unittest.TestCase):

    def test_skeleton_runtime_writes_contract_files(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            conf_path = tmp_path / "conf.json"
            conf = {
                "general": {
                    "simulation_name": "sample",
                    "total_steps": 5
                },
                "simulator": {
                    "compute_diameter": True,
                    "transaction_limit": 10
                },
                "temporal": {
                    "directory": str(tmp_path / "tmp")
                },
                "output": {
                    "directory": str(tmp_path / "outputs"),
                    "transaction_log": "tx_log.csv",
                    "counter_log": "tx_count.csv",
                    "diameter_log": "diameter.csv"
                }
            }
            conf_path.write_text(json.dumps(conf))

            run_python_simulator(str(conf_path))

            tx_log = tmp_path / "outputs" / "sample" / "tx_log.csv"
            counter_log = tmp_path / "outputs" / "sample" / "tx_count.csv"
            diameter_log = tmp_path / "tmp" / "sample" / "diameter.csv"

            self.assertTrue(tx_log.exists())
            self.assertTrue(counter_log.exists())
            self.assertTrue(diameter_log.exists())

            tx_lines = tx_log.read_text().strip().splitlines()
            self.assertEqual(len(tx_lines), 1)
            self.assertTrue(tx_lines[0].startswith("step,type,amount,nameOrig"))

            counter_lines = counter_log.read_text().strip().splitlines()
            self.assertEqual(counter_lines[0], "step,normal,SAR")
            self.assertEqual(len(counter_lines), 6)  # header + 5 steps


if __name__ == ' main ':
    unittest.main()
