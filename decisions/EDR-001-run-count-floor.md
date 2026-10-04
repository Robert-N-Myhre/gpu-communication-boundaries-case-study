# EDR-001: Raising the Minimum Repetition Count

**Record type:** Engineering Decision Record  
**Status:** Accepted  
**Original decision:** August 24, 2026  
**Public adaptation:** October 4, 2026  
**Scope:** Application experiments from LAB-04 onward and the LAB-05 collective experiments.

## Context

The investigation began with one warm-up and five measured runs per arm, interleaved across configurations. Early large-message bandwidth measurements were stable enough to support the recorded comparisons under the study's descriptive spread rule.

Small-message latency behaved differently. The original LAB-A1 investigation characterized two launch-scoped levels, approximately 3.75 and 8.3 microseconds, with the higher level appearing in roughly 12–18% of 1,600 single-GPU runs across three tested GPUs. The underlying cause remained unidentified.

A five-run sample could represent that mixture poorly. In LAB-03, the single-GPU control produced an unexpectedly high median. Its [published analysis](../evidence/results/lab-03/lab-03-nccl-analysis.txt) retains the five-run count, small-message variability, and per-run timing table.

The LAB-A1 archive itself is not included in this public evidence selection. Its characterization supplies the recorded context for the decision; the selected LAB-03 evidence illustrates the sampling concern.

## Decision

Raise the minimum to **ten measured runs per arm** for subsequent experiments, using one rule across metrics and platforms.

Preserve the remaining controls: warm-up, interleaving, medians, observed min–max ranges, coefficient of variation, transport captures, and per-run host state. Observed variability can justify additional runs.

The decision applied prospectively. Completed five-run experiments and their instruments were retained with their actual run counts; they were not relabeled as ten-run experiments.

## Alternatives considered

| Alternative | Reason not selected |
|---|---|
| Retain five runs for every metric | The measured launch-scoped latency structure showed a sampling concern that low bandwidth variation did not address. |
| Use five for stable bandwidth comparisons and ten or more for latency | Would reduce paid runtime, but introduce metric-specific protocol rules. The investigator chose a single floor for operational simplicity. |
| Require twenty or more runs universally | Would increase runtime without evidence that every comparison needed the larger count. The latency mixture also required inspecting per-run structure rather than relying on a larger sample's median alone. |

This was an engineering tradeoff between measurement coverage, protocol complexity, and cost. It was not a sample-size calculation establishing ten as an optimal statistical threshold.

## Consequences

- More observations per arm reduced dependence on a very small sample and increased the chance of observing the less frequent latency level.
- One floor made the protocol easier to apply consistently, including during paid cloud sessions.
- Doubling measured repetitions increased runtime and rental exposure.
- Per-run inspection remained necessary; a floor could not explain or remove the latency mechanism.
- Earlier claims retained their original evidential limits. The source decision assessed them as still supported, rather than treating the new floor as retrospective invalidation.

## Implementation and subsequent verification

The LAB-05 collective launcher defaults to ten measured runs and rejects a lower configured count before execution. Application protocols adopted the same minimum.

The public [L40S collective analysis](../evidence/results/lab-05/lab-05-nccl-analysis.txt) and [V100 application analysis](../evidence/results/lab-04/lab-04-analysis.txt) report ten measured runs per arm, demonstrating that the later experiments used the revised protocol.

## What ten runs do not establish

Ten observations do not guarantee both latency levels will appear, statistical independence, representative sampling, or adequate power for a particular effect size.

The study's comparison rule asks whether medians sit outside the other configuration's observed min–max range. That is a descriptive criterion, not a confidence interval, hypothesis test, or proof of equivalence when results are within spread.

Larger samples can improve estimates of a stable distribution. They cannot by themselves identify the mechanism behind multiple levels or correct uncontrolled changes between sessions. The original rationale is therefore preserved as an operational choice, with these statistical limits made explicit.

## Revisit conditions

Reconsider the rule if session changes dominate the comparisons, if a formal inferential question requires a justified sample size, if the latency mechanism becomes controllable, or if paid-runtime constraints warrant metric-specific repetition rules.

## Provenance

Adapted from `artifacts/EDR/EDR-001-protocol-run-floor-10.md` in the private investigation repository; enforcement checked against `scripts/lab-05-nccl-protocol.sh`. Course dependencies and unexecuted future scope were removed. Statistical interpretation is clarified here without changing the historical decision or measurements.
