# Selected instrumentation

These five scripts expose the L40S experiment's workload, arm definitions, interleaving, capture, and analysis. They are unchanged copies of the research instruments, published to make the method inspectable. They are not a complete reproduction kit or a general GPU benchmarking framework.

## Files and responsibilities

| File | Responsibility |
|---|---|
| [lab-05-workload.py](lab-05-workload.py) | Synthetic-token causal-transformer training; CUDA and PyTorch DDP; emits a `LAB05SUMMARY` JSON record. |
| [lab-05-protocol.sh](lab-05-protocol.sh) | Interleaves ten application arms; captures logs, host state, and GPU telemetry. |
| [lab-05-analysis.py](lab-05-analysis.py) | Summarizes application runs, observed spreads, and within-run timing drift. |
| [lab-05-nccl-protocol.sh](lab-05-nccl-protocol.sh) | Interleaves ten collective arms using `all_reduce_perf`; captures logs, host state, and GPU telemetry. |
| [lab-05-nccl-analysis.py](lab-05-nccl-analysis.py) | Summarizes collective runs, excludes parsed runs with correctness errors, and compares latency and bandwidth. |

The V100 instruments and platform qualification harness are outside this selection. Their role is described in the [platform qualification methodology](../methodology/platform-qualification.md).

## Dependencies and working directory

The analysis scripts use Python 3's standard library only. They do not import PyTorch or require a GPU.

The launch scripts require Linux, Bash with associative arrays, NVIDIA drivers and `nvidia-smi`, procfs, and ordinary shell utilities. Their masks assume four GPUs with the placement established on the tested L40S instance; the same GPU numbers do not establish the same topology on another host.

The collective launcher requires `NCCL_TESTS_DIR` pointing to an existing nccl-tests checkout containing `build/all_reduce_perf`. It runs one process with `-g` selecting the GPU count. The application launcher requires a CUDA-enabled PyTorch environment with `python` and `torchrun` under `TORCH_VENV/bin/`; its default environment is `$HOME/gputopo-tools/torch-venv`. Neither launcher installs dependencies, qualifies a platform, or provisions resources.

Run from a working directory containing `scripts/`. The instruments use `results/lab-05/`, relative to that directory:

| Consumer | Input | Output |
|---|---|---|
| Application analysis | `runs/<arm>-run-*.log`; optional `lab-05-runs.csv` | `lab-05-analysis.txt` |
| Collective analysis | `nccl/<arm>-run-*.log`; optional `lab-05-nccl-runs.csv` | `lab-05-nccl-analysis.txt` |

All paths in this table sit beneath `results/lab-05/`. The launchers also write host snapshots and telemetry. They overwrite CSV files and named run logs on rerun; use a fresh working directory for a new capture.

After supplying compatible logs in that layout, analysis is invoked with:

```bash
python3 scripts/lab-05-nccl-analysis.py
python3 scripts/lab-05-analysis.py
```

The public [evidence package](../evidence/README.md) preserves a different root: `evidence/results/`. The scripts do not automatically read it. Its four representative collective logs support only partial analysis, with below-floor warnings and missing-arm comparisons. Raw application runs and the full collective run archive are not published, so the recorded full analyses cannot be regenerated from this repository.

## Execution and interpretation assumptions

- Default-policy arms inherit the calling environment. Start with `NCCL_P2P_LEVEL` unset and inspect other NCCL overrides before using a launcher; its “default” label does not clear inherited settings.
- Both launchers request at least ten measured rounds and interleave arms. A successful launcher exit does not guarantee ten successful runs per arm: inspect counts, failures, correctness checks, and analysis warnings. See the [run-count decision](../decisions/EDR-001-run-count-floor.md).
- Arms are unbound. Temperature and host state are captured, but there is no thermal gate.
- The application defaults to 40 optimizer steps, measuring from step 10; the accumulation arm uses 16 steps, measuring from step 4. Accumulation changes optimizer-update frequency as well as synchronization. The [workload decision](../decisions/EDR-002-workload-selection.md) governs interpretation of the original script's stronger “exposed-communication bound” wording.
- The workload's per-run “median” selects the upper middle observation for an even sample count. The analysis scripts use `statistics.median` across runs. This historical behavior is preserved.
- “Within spread” is a descriptive comparison of medians with observed min–max ranges, not a statistical equivalence test. Arm transport labels describe observations on the tested instance; verify transport anew on another system.
- Original comments and next-step messages reference private guides and companion scripts. `lab-05-transports.sh` and `lab-05-telemetry-summary.py` are not bundled. Those references are archival context, not additional required public files. Captures can include host identifiers and local paths; review new outputs before publishing them.

## Provenance and license

Copied without code changes from `Robert-N-Myhre/gpu-communication-boundaries`. Source Git blob identifiers make the published versions identifiable:

| File | Source blob |
|---|---|
| `lab-05-workload.py` | `6ec96604b695b745348386bfe4c5ebeecd081784` |
| `lab-05-analysis.py` | `3d95fbee2eb0048b5221c6b90a15a98cab26d204` |
| `lab-05-nccl-analysis.py` | `fb69ef492ca5c58107ad6d2cb19797c2805cce3f` |
| `lab-05-protocol.sh` | `c342cbdad8f954865b085551379ead4ff7b84af2` |
| `lab-05-nccl-protocol.sh` | `9a5fca7d69f799bfee4d3c7398286ec53b95e98a` |

The original [MIT license and copyright notice](../LICENSE) are retained. Documentation clarifications above explain the instruments' limits without changing historical measurements or behavior.
