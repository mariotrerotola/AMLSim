#!/usr/bin/env bash

if [[ $# -lt 1 ]]; then
    echo "Usage: sh $0 [ConfJSON] [OptionalSimulationName]"
    exit 1
fi

CONF_JSON=$1
SIM_NAME=${2:-}

export PYTHONPATH="scripts:scripts/amlsim:${PYTHONPATH}"

if [[ -n "${SIM_NAME}" ]]; then
    python3 scripts/run_py_AMLSim.py "${CONF_JSON}" "${SIM_NAME}"
else
    python3 scripts/run_py_AMLSim.py "${CONF_JSON}"
fi
