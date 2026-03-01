# AMLSim

AMLSim is a Python-based anti-money-laundering simulator for generating synthetic financial data used in research, model development, and pipeline validation.

## Highlights

- Python-only runtime and tooling
- Simple API: generate a full AML dataset in 3 lines of code
- Reproducible simulation runs with configurable seeds
- Built-in normal behavior models and AML typology injection
- End-to-end outputs as pandas DataFrames or CSV files

## Requirements

- macOS or Linux
- Python 3.10+

```bash
pip3 install -r requirements.txt
```

## Quick Start (Python API)

```python
from amlsim import AMLSim

sim = AMLSim(num_accounts=1000, num_steps=720, seed=42)
sim.run()

transactions = sim.to_dataframe()
sar_accounts = sim.get_sar_accounts()
alerts = sim.get_alerts()
```

Run from the `scripts/` directory or add it to your `PYTHONPATH`:

```bash
cd scripts
python3 -c "
from amlsim import AMLSim
sim = AMLSim(num_accounts=1000, num_steps=365, seed=42)
sim.run()
print(sim.to_dataframe().head())
"
```

### Parameters

All parameters have default values. Pass only what you need to customize.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `num_accounts` | int | 1000 | Number of bank accounts to generate |
| `num_steps` | int | 720 | Number of simulated days |
| `base_date` | str | "2017-01-01" | Start date of the simulation (YYYY-MM-DD) |
| `seed` | int | 0 | Random seed for reproducibility |
| `simulation_name` | str | "sample" | Name of the simulation run (used for output folder) |
| `min_amount` | float | 100.0 | Minimum transaction amount for normal transactions |
| `max_amount` | float | 1000.0 | Maximum transaction amount for normal transactions |
| `min_balance` | float | 50000.0 | Minimum initial account balance |
| `max_balance` | float | 100000.0 | Maximum initial account balance |
| `num_fraud_patterns` | int | 10 | Number of suspicious (SAR) patterns to inject |
| `fraud_types` | list | ["fan_in", "fan_out", "cycle"] | Types of fraud patterns to generate (see below) |
| `fraud_min_amount` | float | 100.0 | Minimum transaction amount in fraud patterns |
| `fraud_max_amount` | float | 200.0 | Maximum transaction amount in fraud patterns |
| `fraud_min_accounts` | int | 5 | Minimum accounts involved in a fraud pattern |
| `fraud_max_accounts` | int | 10 | Maximum accounts involved in a fraud pattern |
| `output_dir` | str | "outputs" | Directory where results are saved |
| `margin_ratio` | float | 0.1 | Fraction of amount retained by intermediary accounts |
| `degree_threshold` | int | 10 | Minimum degree for hub account selection |
| `transaction_interval` | int | 7 | Base interval (days) between normal transactions |
| `num_branches` | int | 1000 | Number of bank branches in the simulation |

### Fraud types

Available values for the `fraud_types` parameter:

| Type | Description |
|---|---|
| `fan_in` | Multiple accounts send money to a single account |
| `fan_out` | A single account sends money to multiple accounts |
| `cycle` | Money circulates through a ring of accounts |
| `bipartite` | Many-to-many transactions between two groups |
| `stack` | Layered bipartite (originator -> intermediary -> beneficiary) |
| `random` | Random transaction chain among members |
| `scatter_gather` | Fan-out followed by fan-in through intermediaries |
| `gather_scatter` | Fan-in followed by fan-out through a central account |

### Output methods

| Method | Returns | Description |
|---|---|---|
| `sim.to_dataframe()` | `pd.DataFrame` | All generated transactions |
| `sim.get_sar_accounts()` | `pd.DataFrame` | Accounts flagged as suspicious (SAR) |
| `sim.get_alerts()` | `pd.DataFrame` | Alert transactions linked to fraud patterns |

### Advanced: using `SimulationConfig` directly

```python
from amlsim import AMLSim, SimulationConfig

config = SimulationConfig(
    num_accounts=5000,
    num_steps=365,
    fraud_types=["fan_in", "cycle", "scatter_gather"],
    num_fraud_patterns=50,
)
sim = AMLSim(**config.__dict__)
sim.run()
```

## Alternative: CLI Pipeline

For users who prefer the command-line workflow:

```bash
# 1. Generate graph
python3 scripts/transaction_graph_generator.py conf.json

# 2. Run simulation
sh scripts/run_AMLSim.sh conf.json

# 3. Convert logs
python3 scripts/convert_logs.py conf.json

# Or all at once:
sh scripts/run_batch.sh conf.json
```

CLI configuration is managed through `conf.json`. See `paramFiles/` for presets.

## Testing

```bash
PYTHONPATH=scripts pytest -q
```

## Smoke Benchmark

```bash
./scripts/benchmark_smoke.sh conf.json
```

Reference: `docs/performance/smoke-benchmark.md`

## Project Layout

```
scripts/amlsim/          Core library (API, models, runtime)
scripts/                  CLI entry points and utilities
paramFiles/               Parameter presets by dataset scale
tests/                    Test suite
docs/                     Documentation
outputs/                  Generated results
```

## Citation

If you use AMLSim in publications, please cite:

```bibtex
@misc{AMLSim,
  author = {Toyotaro Suzumura and Hiroki Kanezashi},
  title = {{Anti-Money Laundering Datasets}: {InPlusLab} Anti-Money Laundering DataDatasets},
  howpublished = {\url{http://github.com/IBM/AMLSim/}},
  year = 2021
}
```

- EvolveGCN: Evolving Graph Convolutional Networks for Dynamic Graphs
  https://arxiv.org/abs/1902.10191
- Scalable Graph Learning for Anti-Money Laundering: A First Look
  https://arxiv.org/abs/1812.00076

## License

This project is distributed under the Apache 2.0 License. See `LICENSE`.
