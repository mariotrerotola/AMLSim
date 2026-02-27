#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

CONF_FILE="${1:-conf.json}"
SIM_NAME="${2:-bench_$(date +%Y%m%d_%H%M%S)}"
BENCH_DIR="${3:-outputs/benchmarks}"

mkdir -p "$BENCH_DIR"
REPORT_FILE="$BENCH_DIR/${SIM_NAME}.md"

if [[ -d /opt/homebrew/opt/openjdk@17/libexec/openjdk.jdk/Contents/Home ]]; then
  export JAVA_HOME=/opt/homebrew/opt/openjdk@17/libexec/openjdk.jdk/Contents/Home
  export PATH="$JAVA_HOME/bin:$PATH"
fi

export PYTHONPATH="scripts:scripts/amlsim"

CONF_FILE_RESOLVED="$BENCH_DIR/${SIM_NAME}_conf.json"
python3 - "$CONF_FILE" "$CONF_FILE_RESOLVED" "$SIM_NAME" <<'PY'
import json
import sys

src, dst, sim_name = sys.argv[1], sys.argv[2], sys.argv[3]
with open(src, "r") as rf:
    conf = json.load(rf)
conf.setdefault("general", {})["simulation_name"] = sim_name
with open(dst, "w") as wf:
    json.dump(conf, wf, indent=2)
PY

STEPS=()

run_timed_step() {
  local label="$1"
  shift
  local log_file="$BENCH_DIR/${SIM_NAME}_${label// /_}.log"
  local start_ts=$SECONDS
  "$@" >"$log_file" 2>&1
  local duration=$((SECONDS - start_ts))
  STEPS+=("$label|$duration|$log_file")
}

run_timed_step "generate_graph" python3 scripts/transaction_graph_generator.py "$CONF_FILE_RESOLVED"
run_timed_step "compile_java" mvn -q -DskipTests compile
run_timed_step "run_simulator" mvn -q -Dexec.mainClass=amlsim.AMLSim -Dexec.args="$CONF_FILE_RESOLVED" exec:java
run_timed_step "convert_logs" python3 scripts/convert_logs.py "$CONF_FILE_RESOLVED"

TMP_DIR="tmp/$SIM_NAME"
OUT_DIR="outputs/$SIM_NAME"

TMP_ACCTS=$(wc -l < "$TMP_DIR/accounts.csv" | tr -d '[:space:]')
TMP_TXS=$(wc -l < "$TMP_DIR/transactions.csv" | tr -d '[:space:]')
TMP_ALERTS=$(wc -l < "$TMP_DIR/alert_members.csv" | tr -d '[:space:]')

OUT_ACCTS=$(wc -l < "$OUT_DIR/accounts.csv" | tr -d '[:space:]')
OUT_TXS=$(wc -l < "$OUT_DIR/transactions.csv" | tr -d '[:space:]')
OUT_CASH_TXS=$(wc -l < "$OUT_DIR/cash_tx.csv" | tr -d '[:space:]')
OUT_ALERT_TXS=$(wc -l < "$OUT_DIR/alert_transactions.csv" | tr -d '[:space:]')
OUT_SAR_ACCTS=$(wc -l < "$OUT_DIR/sar_accounts.csv" | tr -d '[:space:]')

{
  echo "# Smoke Benchmark: $SIM_NAME"
  echo
  echo "- Config: $CONF_FILE"
  echo "- Effective config: $CONF_FILE_RESOLVED"
  echo "- Timestamp: $(date '+%Y-%m-%d %H:%M:%S %z')"
  echo
  echo "## Step timings"
  echo
  echo "| Step | Duration (s) | Log |"
  echo "| --- | ---: | --- |"
  for row in "${STEPS[@]}"; do
    IFS='|' read -r label duration log_file <<< "$row"
    echo "| $label | $duration | $log_file |"
  done
  echo
  echo "## Output sanity"
  echo
  echo "| File | Rows |"
  echo "| --- | ---: |"
  echo "| tmp/accounts.csv | $TMP_ACCTS |"
  echo "| tmp/transactions.csv | $TMP_TXS |"
  echo "| tmp/alert_members.csv | $TMP_ALERTS |"
  echo "| outputs/accounts.csv | $OUT_ACCTS |"
  echo "| outputs/transactions.csv | $OUT_TXS |"
  echo "| outputs/cash_tx.csv | $OUT_CASH_TXS |"
  echo "| outputs/alert_transactions.csv | $OUT_ALERT_TXS |"
  echo "| outputs/sar_accounts.csv | $OUT_SAR_ACCTS |"
} > "$REPORT_FILE"

echo "Benchmark report written to $REPORT_FILE"
