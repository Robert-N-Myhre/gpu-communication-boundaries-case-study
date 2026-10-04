# GPU Communication Boundaries

An evidence-driven architecture case study of GPU placement, NCCL transport selection, and collective versus application performance on a four-V100 workstation and a four-L40S cloud instance.

**Architectural question:** When GPU communication crosses a CPU socket boundary, what changes performance: physical placement, communication mechanism, or both—and which measurements are sufficient to explain the observed behavior?

The investigation began with an expectation that crossing the socket boundary would impose a performance penalty. Controlled comparisons found no measurable collective penalty when transport was held constant. Changing the communication mechanism mattered more, and the application's response often differed from the benchmark's.

[Read the case study](gpu-communication-boundaries-case-study.md) · [Inspect the selected evidence](evidence/README.md) · [How the platforms were qualified](methodology/platform-qualification.md) · [Inspect the instrumentation](scripts/README.md)

## Key findings

| Observation on the tested systems | Architectural implication |
|---|---|
| Same-NUMA and cross-NUMA pairs showed no measurable collective penalty under matched transports: host staging on V100; P2P and host staging on L40S. | A topology label alone does not establish a performance cost. |
| Forcing P2P improved V100 same-socket pair bandwidth by 15%; disabling P2P reduced L40S pair bandwidth by approximately 18% at both placements. | Separate placement from mechanism before attributing a difference to the boundary. |
| NCCL selected host staging where P2P was available. On L40S, default selection changed between two-GPU and four-GPU jobs. | Hardware capability and runtime selection require separate verification. |
| Mixed-transport rings lost approximately 10% of collective bandwidth on V100 and 37% on L40S relative to their all-host-staged defaults. | Individual link capabilities do not predict the behavior of the complete collective. |
| An approximately 18% L40S pair-bandwidth reduction produced a training-throughput change within the observed spread at the same-NUMA placement. Some benchmark improvements accompanied application regressions. | Measure workload impact before choosing an infrastructure optimization. |

## How the conclusions were tested

The study combined topology and P2P capability captures, NCCL's observed transport strings, repeated all-reduce measurements, and a 124-million-parameter PyTorch DDP training workload using synthetic tokens in fp32.

Placement and transport policies were varied within each platform. Runs were interleaved, results retained medians and observed spreads, and application transports were captured separately because PyTorch bundled a different NCCL version. Unexpected results changed the investigation; unresolved causes remain explicitly unknown.

The [workload-selection decision](decisions/EDR-002-workload-selection.md) explains the constraints, alternatives, and tradeoffs behind the application experiment. The [repetition-count decision](decisions/EDR-001-run-count-floor.md) records why later experiments moved from five to ten measured runs.

The [L40S transport-discovery field note](field-notes/l40s-transport-discovery.md) preserves the unexpected observations that forced two revisions of the experiment.

## Architecture takeaway

Topology constrains the available paths. Runtime selection determines which paths are used. Collective measurements expose communication behavior, while workload structure determines how much of that behavior reaches application performance.

Before paying for topology-aware placement or forcing a transport policy, verify the selected mechanism and measure whether the workload benefits.

## Scope and limits

These results describe one V100 workstation, one L40S cloud instance, and one training workload. The platforms differ in hardware, software, and operating conditions; their comparison concerns patterns, not a controlled V100-versus-L40S ranking. “No measurable penalty” refers to the tested comparisons and observed spreads, not universal equivalence.

Cross-node communication was deferred to a separate investigation. NVLink, RDMA, and network fabrics were not tested here.

This repository contains the case study, selected supporting evidence, and selected research instruments. The [evidence index](evidence/README.md) explains coverage and interpretation; the [instrumentation guide](scripts/README.md) documents dependencies and assumptions. The full experimental archive and platform provisioning toolkit are not included.

**Author:** Robert N. Myhre · **Study completed:** October 4, 2026
