# AMLSim

AMLSim is a multi-agent anti-money-laundering simulator that generates synthetic financial transaction data for research, experimentation, and model validation.

The project combines:
- Python preprocessing for graph and parameter generation
- Python simulation for time-stepped transaction execution
- Python post-processing for export and analysis

## Why AMLSim

- Produces reproducible synthetic AML datasets from configurable scenarios
- Supports alert pattern injection and SAR-oriented outputs
- Generates CSV artifacts suitable for graph analytics and ML pipelines
- Includes automated tests and a smoke benchmark workflow

## Requirements

- macOS or Linux
- Python 3.10+
- Python dependencies from `requirements.txt`

Install Python dependencies:

```bash
pip3 install -r requirements.txt
```

## Quick Start

1. Generate transaction graph inputs:

```bash
python3 scripts/transaction_graph_generator.py conf.json
```

2. Build AMLSim:

```bash
sh scripts/build_AMLSim.sh
```

3. Run simulation:

```bash
sh scripts/run_AMLSim.sh conf.json
```

4. Convert raw logs to final CSV outputs:

```bash
python3 scripts/convert_logs.py conf.json
```

Optional utilities:

```bash
python3 scripts/visualize/plot_distributions.py conf.json
python3 scripts/validation/validate_alerts.py conf.json
sh scripts/clean_logs.sh
```

## Configuration

The main runtime configuration is `conf.json`.

Key sections:
- `input`: parameter directory and schema source files
- `general`: simulation metadata (`simulation_name`, `total_steps`, `base_date`)
- `output`: generated file names and destination directory

For production-scale runs, copy one of the templates under `paramFiles/` and update `conf.json` accordingly.

## Testing

Run Python tests:

```bash
PYTHONPATH=scripts:scripts/amlsim pytest -q
```

## Smoke Benchmark

Run a full end-to-end smoke benchmark:

```bash
./scripts/benchmark_smoke.sh conf.json
```

Outputs:
- Step logs and timing report in `outputs/benchmarks/`
- Effective benchmark config stored with unique `simulation_name`

Details: `docs/performance/smoke-benchmark.md`

## Project Layout

- `scripts/`: Python data generation and processing scripts
- `paramFiles/`: parameter presets
- `tests/`: Python test suite
- `docs/`: project documentation
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
