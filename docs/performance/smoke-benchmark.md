# Smoke Benchmark

This repository now includes a reproducible smoke benchmark script:

```bash
./scripts/benchmark_smoke.sh conf.json
```

The script runs:
1. Transaction graph generation
2. Python AMLSim simulation
3. Log conversion

It writes a full report and logs under `outputs/benchmarks/`.

## Latest run

- Date: 2026-02-27 15:28:52 +0100
- Config: `conf.json` (1K parameter set)
- Effective simulation name: `bench_20260227_152848`

### Step timings

| Step | Duration (s) |
| --- | ---: |
| generate_graph | 1 |
| run_simulator | 2 |
| convert_logs | 1 |

### Output sanity

| File | Rows |
| --- | ---: |
| tmp/accounts.csv | 1447 |
| tmp/transactions.csv | 7978 |
| tmp/alert_members.csv | 74 |
| outputs/accounts.csv | 1447 |
| outputs/transactions.csv | 132316 |
| outputs/cash_tx.csv | 111520 |
| outputs/alert_transactions.csv | 66 |
| outputs/sar_accounts.csv | 72 |
