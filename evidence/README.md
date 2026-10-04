# Selected Evidence

This directory supports the principal findings in the [GPU Communication Boundaries case study](../gpu-communication-boundaries-case-study.md). It contains selected analysis outputs, topology and capability captures, transport captures, and four representative L40S benchmark logs from the original investigation.

The original `results/` paths are preserved beneath this directory. For example, a source reference to `results/lab-05/lab-05-analysis.txt` resolves here as [results/lab-05/lab-05-analysis.txt](results/lab-05/lab-05-analysis.txt).

## Findings and supporting files

| Finding | V100 evidence | L40S evidence |
|---|---|---|
| Matched transports show no measurable collective penalty from crossing the socket boundary | [Collective analysis](results/lab-03/lab-03-nccl-analysis.txt): AU versus U, both unbound and host-staged | [Collective analysis](results/lab-05/lab-05-nccl-analysis.txt): A versus B with P2P; AL versus BL with host staging |
| Communication mechanism changes pair bandwidth | [Collective analysis](results/lab-03/lab-03-nccl-analysis.txt): A and C arm tables; [transport captures](results/lab-03/lab-03-nccl-transports.txt) | [Collective analysis](results/lab-05/lab-05-nccl-analysis.txt): A versus AL and B versus BL; representative pair logs below |
| Hardware capability does not establish NCCL's selected transport | [Topology and P2P capability](results/lab-03/lab-03-preflight.txt), [per-run transports](results/lab-03/lab-03-nccl-transports.txt), and [ring captures](results/lab-03/lab-03-nccl-graph.txt) | [Topology and P2P capability](results/lab-05/lab-05-preflight.txt), [ring captures](results/lab-05/lab-05-nccl-graph.txt), and representative pair logs below |
| The tested mixed-transport rings underperform the all-host-staged rings | [Collective analysis](results/lab-03/lab-03-nccl-analysis.txt): Q versus P; [ring captures](results/lab-03/lab-03-nccl-graph.txt) | [Collective analysis](results/lab-05/lab-05-nccl-analysis.txt): Q versus QH; [ring captures](results/lab-05/lab-05-nccl-graph.txt) |
| Collective gains and losses do not translate directly into application gains and losses | [Application analysis](results/lab-04/lab-04-analysis.txt), compared with the collective analysis; [application transport captures](results/lab-04/lab-04-transports.txt) | [Application analysis](results/lab-05/lab-05-analysis.txt), compared with the collective analysis; [application transport captures](results/lab-05/lab-05-transports.txt) |

## Representative L40S pair logs

| Placement | Default policy: observed P2P | LOC policy: observed host staging |
|---|---|---|
| Same NUMA, GPUs 0 and 1 | [A-run-01.log](results/lab-05/nccl/A-run-01.log) | [AL-run-01.log](results/lab-05/nccl/AL-run-01.log) |
| Cross NUMA, GPUs 0 and 2 | [B-run-01.log](results/lab-05/nccl/B-run-01.log) | [BL-run-01.log](results/lab-05/nccl/BL-run-01.log) |

These logs illustrate the selected paths and individual measurements. They are not the complete repeated-run dataset and cannot independently reproduce its statistics.

## How to interpret the evidence

- Analysis files are derived outputs. They report arm definitions, run counts, medians, spreads, and comparisons from the measured runs.
- The protocol's “within spread” criterion compares medians with observed min–max ranges. It is a descriptive rule, not a statistical equivalence test.
- NCCL transport strings establish the selected path; they do not explain why the library selected it.
- Benchmark and application captures are separate because the PyTorch installation bundles its own NCCL version.
- Bus bandwidth is the nccl-tests normalized collective metric, not a direct measurement of a physical link's bandwidth. Its normalization changes with rank count.
- Compare placements and policies within each platform. The V100 and L40S systems are not a controlled hardware comparison.

## Scope and provenance

The captures and analysis outputs originate from LAB-03, LAB-04, and LAB-05 of an independent investigation on one four-V100 workstation and one four-L40S cloud instance. Timestamps, commands, versions, and source references remain in the selected files.

This is a selected evidence package, not a complete reproduction kit: the full run archive and analysis scripts are not included. The case study's latency investigation, thermal incident history, and provisioning timeline are contextual observations outside the selected package's main coverage. Cross-node communication was not tested.
