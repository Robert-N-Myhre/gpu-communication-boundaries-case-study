# Platform Qualification and Evidence Acquisition

**Purpose:** Explain how the infrastructure behind the [GPU Communication Boundaries case study](../gpu-communication-boundaries-case-study.md) was qualified, how its state was captured, and how paid experiment sessions were controlled.

A repeatable benchmark can consistently measure the wrong platform or a degraded node. Qualification therefore preceded interpretation: establish the machine's shape, inspect its health, and exercise communication before treating experiment results as evidence.

## Separate platform qualification from the experiment

The private `gpu-platform-baseline` repository owns the collector harness, accepted baseline registry, health criteria, and acquisition/release automation. Experiment repositories use that authority and retain run identifiers and acceptance verdicts rather than maintaining independent copies of the criteria.

This separates reusable platform knowledge from a particular research question. A changed driver, topology, or health criterion can be reviewed at its source, while the experiment records which platform and verdict it used.

The three acceptance gates answer different questions:

| Gate | Question | Evidence and response |
|---|---|---|
| Identity and drift | Is this the expected platform and software state? | Compare a fresh capture with an accepted baseline. Normalize known identity changes and transient state; review remaining differences. Software drift requires assessment, while topology changes can require redesigning placement. |
| Health | Is the node suitable for this experiment? | Inspect PCIe capabilities and width, P2P capability, ECC history, remap state, and applicable error logs and diagnostics. Hard failures reject the node; advisory findings require operator review. |
| Functional smoke | Does communication work correctly and perform plausibly? | Run a short all-reduce sweep. Incorrect results reject the run regardless of speed. Compare performance with a recorded baseline where one exists; a first measurement establishes a baseline rather than independently proving expected performance. |

The automation can skip functional smoke explicitly, but the procedure still requires it before experiment work. A successful acquisition command is therefore read together with the individual gate verdicts and outstanding review items.

## Capture evidence without hiding missing observations

The harness collects system, CPU, NUMA, PCIe, GPU, software, kernel, and applicable interconnect information into a directory identified by its run ID.

Its collection design preserves partial evidence:

- External collector commands run through a wrapper with bounded execution time.
- The manifest records commands, timing, exit status, output locations, and missing binaries.
- A collector failure is recorded while other collectors continue.
- The run manifest records expected platform attributes and the harness revision.
- Packaging produces file checksums and an archive checksum for integrity checks during transfer.

**Capture completion is not a health verdict.** A completed capture may contain failed commands or missing observations. Checksums establish file integrity, not measurement validity; qualification still requires reading the evidence and gate outcomes.

Raw baseline captures can contain machine and network identifiers. The baseline workflow keeps those collections private and publishes selected derivatives. The [public evidence index](../evidence/README.md) describes this case study's selection; public files retain some original provenance fields, so selection alone is not a claim of complete anonymization.

## Two incidents that changed the qualification criteria

These incidents are recorded in the baseline project's acceptance history. Their underlying acceptance archives are not included in this public evidence selection.

### Idle PCIe speed caused a false rejection

On September 8, 2026, an L40S node reported current PCIe Gen1 speed against Gen4 capability. The initial check treated that difference as a fault, although link width, ECC, diagnostics, and other inspected state were clean.

The acceptance record attributed the reading to idle link power state and revised the criteria to distinguish maximum capability, current width, and current speed. Current speed became advisory; functional measurement was needed to assess performance.

**Architectural lesson:** a snapshot of runtime state cannot be substituted for a hardware capability or an under-load performance measurement.

### A quick diagnostic missed recorded ECC history

On September 9, 2026, another L40S node reported six aggregate uncorrectable DRAM ECC errors while volatile counters were zero. The quick DCGM diagnostic passed, and no Xid was observed. The experiment's acceptance policy rejected the node based on its recorded history.

That observation did not establish an active failure during the session. It established that a passing diagnostic and clean current-session counters did not satisfy the chosen acceptance policy.

**Architectural lesson:** diagnostic verdicts, current-session counters, and lifetime history cover different failure evidence.

## Control the paid session as well as the measurements

The acquisition workflow checks capacity, quota, configuration, and budget before provisioning. It then captures and qualifies the node, leaving it available for the experiment.

Its failure path attempts a control-plane stop unless the operator deliberately requests retention. Release orders operations as **stop, destroy, verify cleanup, then record the ledger**. Stopping addresses ongoing compute spend before potentially slow teardown. Verification checks for remaining resources; an emergency stop path does not depend on Terraform state.

Cost estimates and provider billing are recorded separately. These controls limit exposure and make failures inspectable; they do not guarantee availability, termination, or exact billing.

## What qualification establishes—and what it leaves open

Qualification supports a defensible starting point. It does not establish the transport NCCL will select, explain that selection, or predict application behavior. Those require the experiment's own transport captures and workload measurements.

Nor does a short acceptance test prove sustained workload stability. The V100 training experiment subsequently exposed thermal behavior that the collective benchmarks had not. Qualification was supplemented by per-run telemetry and workload-specific controls, as described in the case study.

The transferable practice is to maintain explicit boundaries between **captured state, acceptance judgment, runtime behavior, and application impact**.

## Source and publication scope

Adapted from `gpu-platform-baseline/docs/node-acceptance.md`, the collector harness, and the acquisition/release scripts. This document describes their role in the investigation; it does not publish the baseline archive, provider provisioning implementation, or a runnable harness.

The public [V100 preflight](../evidence/results/lab-03/lab-03-preflight.txt) and [L40S preflight](../evidence/results/lab-05/lab-05-preflight.txt) expose selected experiment-time topology, capability, and version information. They are not substitutes for the complete acceptance records.
