# AMLSim

AMLSim is a Python-based anti-money-laundering simulator for generating synthetic financial data used in research, model development, and pipeline validation.

The project includes:
- Transaction graph generation
- Time-stepped simulation runtime
- Log conversion to analytics-ready CSV outputs

## Highlights

- Python-only runtime and tooling
- Reproducible simulation runs with configurable seeds
- Built-in normal behavior models and AML typology injection
- End-to-end outputs for graph analytics and ML workflows
- Automated tests and smoke benchmark support

## Requirements

- macOS or Linux
- Python 3.10+
- Python dependencies from `requirements.txt`

Install dependencies:

```bash
pip3 install -r requirements.txt
```

## Quick Start

1. Generate temporal graph inputs:

```bash
python3 scripts/transaction_graph_generator.py conf.json
```

2. Run the simulator:

```bash
sh scripts/run_AMLSim.sh conf.json
```

3. Convert simulation logs to final datasets:

```bash
python3 scripts/convert_logs.py conf.json
```

Optional single-command pipeline:

```bash
sh scripts/run_batch.sh conf.json
```

Optional utilities:

```bash
python3 scripts/visualize/plot_distributions.py conf.json
python3 scripts/validation/validate_alerts.py conf.json
sh scripts/clean_logs.sh
```

## Configuration

Main runtime configuration file: `conf.json`.

Primary sections:
- `general`: simulation metadata (`simulation_name`, `total_steps`, `base_date`, `random_seed`)
- `default`: fallback financial and behavior parameters
- `input`: source parameter files (`accounts`, `degree`, `alert_patterns`, `normal_models`, `schema`)
- `temporal`: intermediate files produced before simulation
- `simulator`: runtime options (`transaction_interval`, `transaction_limit`, `compute_diameter`, `numBranches`)
- `output`: final exported file names and destination directory
- `graph_generator`: graph generation controls

Environment overrides:
- `RANDOM_SEED`: overrides `general.random_seed`
- `SIMULATION_NAME`: overrides `general.simulation_name`

For larger scenarios, start from presets under `paramFiles/` and adapt `conf.json`.

## Testing

Run all Python tests:

```bash
PYTHONPATH=scripts:scripts/amlsim pytest -q
```

## Smoke Benchmark

Run a reproducible end-to-end smoke benchmark:

```bash
./scripts/benchmark_smoke.sh conf.json
```

Artifacts:
- Step logs and timing report under `outputs/benchmarks/`
- Effective benchmark config with a unique `simulation_name`

Reference: `docs/performance/smoke-benchmark.md`

## Performance Notes

Recent optimizations include:
- Buffered transaction and counter log writes in runtime repository
- Cache for transaction-type fallback lookup on accounts
- Reduced per-transaction object allocations in simulation models
- Faster dedupe key handling and date-column processing in log conversion
- Optimized active-edge marking and alert amount aggregation in graph generation

## Project Layout

- `scripts/`: runtime, generators, conversion, validation, utilities
- `scripts/amlsim/`: core simulation models and runtime components
- `paramFiles/`: parameter presets by dataset scale and scenario
- `tests/`: Python test suite
- `docs/`: documentation (migration and performance notes)
- `outputs/`, `tmp/`: generated artifacts

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
