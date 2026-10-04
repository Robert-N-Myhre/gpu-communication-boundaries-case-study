# Field Note: Transport Discovery Changed the Experiment Twice

**Experiment:** LAB-05, four-L40S cloud instance  
**Date:** October 4, 2026  
**Publication status:** Adapted from the original session record

The [case study](../gpu-communication-boundaries-case-study.md) presents the final comparisons. This note preserves how the experiment had to change before those comparisons could answer the intended question.

## The assumption under test

The initial plan expected default NCCL communication to use the cross-socket P2P path that the driver reported as available. Restricting P2P to PHB distance was expected to remove cross-socket P2P and produce a controlled mechanism comparison.

Transport capture was deliberately scheduled before interpreting benchmark results. The weakness was that the initial probe used four GPUs only, although the experiment also included two-GPU jobs.

## First discovery: the four-GPU default already used host staging

The four-GPU capture reported `via SHM/direct` on every ring link, including the same-NUMA pairs. NCCL 2.31.2 was not using the P2P capability shown by the driver.

The additional policy probes produced three distinct rings:

| Policy on the tested four-GPU communicator | Observed transport |
|---|---|
| Default | Host staging on every link |
| PHB | P2P within each NUMA pair; host staging on the socket crossings |
| SYS | P2P on every ring link, including the socket crossings |

The [ring capture](../evidence/results/lab-05/lab-05-nccl-graph.txt) exposes these paths.

**Experiment change:** add the SYS configuration and revise the pair and ring arms. The planned policy change could no longer be interpreted as simply removing a default cross-socket P2P path.

## Second discovery: the two-GPU default used P2P

After the first collective pass, the pair logs showed P2P at default for both same-NUMA and cross-NUMA placements. The four-GPU observation did not describe the two-GPU communicator.

Consequently, the explicit PHB same-NUMA pair and SYS cross-NUMA pair did not supply the expected opposite mechanism: they selected P2P just as their default counterparts did.

**Experiment change:** add LOC pair configurations to obtain host staging at both placements. Retain the same-transport duplicates as replicate controls rather than mislabel them as mechanism contrasts.

| Placement | Default: observed P2P | LOC: observed host staging |
|---|---|---|
| Same NUMA, GPUs 0 and 1 | [A-run-01.log](../evidence/results/lab-05/nccl/A-run-01.log) | [AL-run-01.log](../evidence/results/lab-05/nccl/AL-run-01.log) |
| Cross NUMA, GPUs 0 and 2 | [B-run-01.log](../evidence/results/lab-05/nccl/B-run-01.log) | [BL-run-01.log](../evidence/results/lab-05/nccl/BL-run-01.log) |

These public logs are representative final-pass captures. They do not independently establish consistency across every repetition.

### Reflection

This was the point where I stopped trusting the four-GPU result as a predictor of what NCCL would do in the pair tests. I had already adjusted the experiment once, after finding that the default four-GPU ring used host staging. Then the two-GPU jobs selected P2P by default, which meant communicator size was another variable I had not accounted for.

I did not know why NCCL was making those choices, and I still don't. I could not infer the transport from the topology or the configuration label, so from then on I checked which transport each comparison had actually used before trusting its performance result.

## The application needed its own check

The application bundled NCCL 2.27.5 rather than the benchmark's system NCCL 2.31.2. Its transports therefore required independent capture.

The original note records operator-reported P2P in the running two-GPU application's warm-up logs. That workload pass was stopped during its second measured round and was not committed as a complete dataset. The application arms were revised to include LOC at both pair placements, and both collective and workload protocols were rerun in full.

The public [application transport capture](../evidence/results/lab-05/lab-05-transports.txt) shows the four-GPU paths; the complete two-GPU application logs and superseded discovery pass are outside this public selection.

The ten-run floor was retained. Discovery required additional probes, one interrupted workload pass, and reruns within a session recorded as 80.9 metered minutes. That figure is the total session duration, not an isolated measurement of redesign cost.

## What the evidence establishes

**Observed:** the tested default pairs used P2P while the tested default four-GPU ring used host staging. The driver reported P2P capability across the topology.

**Design consequence:** a configuration label did not identify the mechanism contrast. The experiment needed the transport actually selected at each placement and communicator size.

**Unknown:** why NCCL selected these defaults. Communicator size was associated with the observed difference; the library's internal selection rule was not identified. This is not a general rule for all L40S systems or NCCL versions.

The original note offered an assistant-generated hypothesis about topology search, explicitly untested. It also recorded no contemporaneous operator explanation of the selection rule. This public adaptation does not promote that hypothesis into a finding or invent such an explanation.

## Method lesson

Probe the selected transport at every communicator size used by the experiment, and check the application runtime separately. If a planned intervention does not change the mechanism, revise the contrast before attributing its performance to that mechanism.

Capture enough host information to support later investigation: the instance's host CPU model was not captured, leaving an explanatory gap. Discovery remained a valid outcome, but its cause remained open.

## Provenance

Adapted from `artifacts/field-notes/LAB-05-field-note-01.md` in the private investigation repository. The original records operator-approved probes and redesigns in an AI-assisted investigation. This publication connects the final paths to [selected public evidence](../evidence/README.md), distinguishes session history from the published subset, and replaces unfinished interpretation fields with explicit limits.
