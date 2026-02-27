#!/usr/bin/env bash

if [[ $# -ne 1 ]]; then
    echo "Usage: sh $0 [ConfJSON]"
    exit 1
fi

CONF_JSON=$1

echo 'Python runtime selected.'
sh scripts/run_py_AMLSim.sh "${CONF_JSON}"

# Cleanup temporal outputs of AMLSim
rm -f outputs/_*.csv outputs/_*.txt outputs/summary.csv
