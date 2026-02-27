# Python-Only Migration Plan (Java Decommission)

## Objective

Replace the Java simulation engine with a Python implementation while preserving:
- Functional behavior
- Output schema and file contracts
- Reproducibility guarantees (seed-driven runs)
- Operational reliability for existing pipelines

## Current Pipeline

1. `scripts/transaction_graph_generator.py` builds temporal inputs (`tmp/<simulation_name>/...`)
2. Python simulator (`scripts/run_py_AMLSim.py`) generates runtime logs (`tx_log.csv`, counters, diameter)
3. `scripts/convert_logs.py` converts runtime logs into final CSV outputs

Target state:
- Step 2 is replaced by a Python engine.
- Steps 1 and 3 remain compatible (no breaking change for users).

## Migration Strategy

Use an incremental parallel-run strategy:

1. Build Python engine behind a new entrypoint (do not remove Java yet).
2. Run Java and Python in parallel on identical seeds/configuration.
3. Compare outputs automatically with contract tests.
4. Switch default runtime to Python only after parity and performance gates pass.
5. Decommission legacy Java runtime and Maven build paths.

## Scope Inventory (Porting Backlog)

Core runtime:
- `amlsim/AMLSim.java`
- `amlsim/Account.java`, `SARAccount.java`, `Branch.java`
- `amlsim/Alert.java`, `AccountGroup.java`
- `amlsim/TransactionRepository.java`
- `amlsim/SimProperties.java`

Models:
- Normal: `single`, `fan_in`, `fan_out`, `forward`, `mutual`, `periodical`, `empty`
- Cash: `cash_in`, `cash_out`
- AML typologies: `fan_in`, `fan_out`, `cycle`, `bipartite`, `gather_scatter`, `scatter_gather`, `stack`, `random`

Statistics:
- `amlsim/stat/Diameter.java`

## Work Phases

### Phase 0 - Freeze and Baseline (2-3 days)

- Freeze current Java behavior with regression fixtures.
- Capture baseline outputs for representative configs (`1K`, `10K`, typology-focused).
- Add output-contract checker (schema + counts + key invariants).

Deliverables:
- Baseline artifact set under `outputs/baselines/`
- Validation script for diffing Java vs Python outputs

### Phase 1 - Python Simulation Kernel (1 week)

- Implement scheduler/event loop in Python.
- Port core entities (`Account`, `SARAccount`, `Branch`, `Alert`, `AccountGroup`).
- Port transaction logging repository and file writer.
- Port config loading semantics from `SimProperties`.
- Ensure deterministic random stream compatibility.

Deliverables:
- `scripts/pyamlsim/` kernel package
- CLI entrypoint equivalent to Java run step

### Phase 2 - Normal and Cash Models (1-1.5 weeks)

- Port normal transaction models.
- Port cash in/out models.
- Validate output parity for non-AML scenarios.

Deliverables:
- Model registry + factory
- Parity report for normal/cash-only runs

### Phase 3 - AML Typologies (1.5-2 weeks)

- Port all AML typologies.
- Validate alert member and alert transaction parity.

Deliverables:
- Full typology coverage
- Parity report for typology-heavy parameter sets

### Phase 4 - Statistics and Diameter (1 week)

- Reimplement diameter computation in Python.
- Choose backend:
  - Option A: NetworkX approximation for fast iteration
  - Option B: Graph-tool/igraph for higher performance
- Match CSV contract for `diameter.csv`.

Deliverables:
- Python stats module + benchmark comparison

### Phase 5 - Cutover and Java Removal (1 week)

- Make Python engine default in scripts/CI.
- Keep Java path behind temporary fallback flag.
- Java/Maven integration has been removed from operational scripts and CI.

Deliverables:
- Python default runtime
- Decommissioned Java path and updated docs

## Estimated Effort

- Single engineer: 7-9 weeks
- Two engineers in parallel: 4-6 weeks

Assumptions:
- No major schema redesign.
- Existing parameter files remain valid.
- Performance target is near current Java runtime for typical workloads.

## Key Risks and Mitigations

1. RNG mismatch leads to non-comparable outputs
- Mitigation: use Java-compatible RNG implementation in Python.

2. Runtime slowdown on large datasets
- Mitigation: profile early; optimize hotspots with vectorization/Cython/PyPy where needed.

3. Hidden behavior in Java scheduling/model interactions
- Mitigation: phase-by-phase parity tests and fixtures before cutover.

4. Diameter/statistics divergence
- Mitigation: define tolerance thresholds and document accepted approximation behavior.

## Acceptance Criteria

Functional:
- Python pipeline produces all expected output files with unchanged schema.
- Invariants (row counts, unique keys, alert coverage) match Java baseline on regression suite.

Determinism:
- Same seed/config -> stable Python outputs across repeated runs.

Performance:
- End-to-end runtime within agreed envelope (target: <= 1.5x Java baseline on 10K profile).

Operational:
- CI runs Python-only pipeline without Java/Maven dependency.

## Immediate Next 3 Tasks

1. Add Java-compatible RNG utility in Python and tests (determinism foundation).
2. Implement Python `TransactionRepository` equivalent and CSV contract tests.
3. Introduce `run_py_AMLSim.sh` entrypoint with partial kernel wiring.

## Execution Status (2026-02-27)

- Done: Java-compatible RNG utility in Python (`scripts/amlsim/java_random.py`) with reference tests.
- Done: Python `TransactionRepository` port (`scripts/amlsim/transaction_repository.py`) with contract tests.
- Done: Python runtime entrypoint (`scripts/run_py_AMLSim.py`, `scripts/run_py_AMLSim.sh`) wired to a new runtime core.
- Done: runtime core with account/transaction loading, scheduler loop, and output contracts.
- Done: normal models (`single`, `fan_in`, `fan_out`, `forward`, `mutual`, `periodical`, `empty`).
- Done: cash models (`cash_in`, `cash_out`) and Python diameter snapshot output.
- Done: AML typologies (`fan_in`, `fan_out`, `cycle`, `bipartite`, `stack`, `random`, `scatter_gather`, `gather_scatter`).
- Done: CI switched to Python-only pipeline.
- Next: tighten Java/Python parity checks with dataset-level diff reports and decommission legacy Java source tree.
