#!/usr/bin/env python3
import os
import sys

from amlsim.python_runtime import PythonAMLSim
from amlsim.sim_properties import SimProperties


def _resolve_sim_name(cli_sim_name):
    if cli_sim_name:
        return cli_sim_name
    env_name = os.getenv("SIMULATION_NAME")
    if env_name:
        return env_name
    return None


def run_python_simulator(conf_path, cli_sim_name=None):
    sim_name = _resolve_sim_name(cli_sim_name)
    sim_properties = SimProperties(conf_path, sim_name)
    runtime = PythonAMLSim(sim_properties)
    runtime.execute()
    print(f"[py-amlsim] Runtime completed for simulation '{sim_properties.get_sim_name()}'.")


def main():
    if len(sys.argv) < 2:
        print(f"Usage: python3 {sys.argv[0]} [ConfJSON] [OptionalSimulationName]")
        raise SystemExit(1)

    conf_path = sys.argv[1]
    sim_name = sys.argv[2] if len(sys.argv) >= 3 else None
    run_python_simulator(conf_path, sim_name)


if __name__ == "__main__":
    main()
