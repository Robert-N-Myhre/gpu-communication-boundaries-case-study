# Case Study: What a Socket Boundary Costs GPU Traffic — and What the Training Step Keeps

**Date:** 2026-10-04 · **Derived from:** LAB-01, LAB-02, LAB-A1, LAB-03, LAB-04, LAB-05

## Summary

Multi-GPU servers put some GPU pairs on the same CPU socket and others across a socket
boundary, and folklore says the crossing is expensive. I measured it on two platforms — a
four-V100 PCIe workstation I own and a four-L40S cloud instance rented for 81 minutes —
with a fixed protocol, NCCL's own transport log in every run, and a real distributed
training step beside every benchmark. The boundary itself cost nothing I could measure on
either machine. What cost something was the *mechanism* — whether bytes went GPU-to-GPU or
bounced through host memory — and that choice was made for me by the communication library,
differently on each platform and, on the L40S, differently depending on how many GPUs were
in the job. Even then, an 18 % swing in collective bandwidth reached the training step as
nothing at all, and twice the transport that won the benchmark lost the training step. The
actionable finding is unglamorous: measure whether your workload exposes enough
communication to care before paying for topology optimization, and read the library's
transport log before trusting the topology diagram. The investigation concluded at this
boundary on 2026-10-04; the cross-node chapter it originally planned was deferred to a separate project,
and nothing here speaks to networks.

## The Problem

A four-GPU node is not four equal neighbors. On both machines in this study, GPU0 and GPU1
hang off one CPU socket and GPU2 and GPU3 off the other, so a `{0,1}` pair talks across a
PCIe host bridge and a `{0,2}` pair talks across the socket interconnect too. Topology tools
label the first link `PHB` and the second `SYS`. The driver separately reports whether a
direct GPU-to-GPU (P2P) copy is possible on each link — on the V100 box it is not across the
socket; on the L40S instance it is everywhere.

The question was: how much do placement (which pair) and mechanism (P2P or host-staged) each
cost, first in a raw collective and then in an application — and which measurements are
enough to explain the result rather than infer it from the diagram. The V100 could not
separate the two axes, because its cross-socket link offers only host staging. The L40S,
where the boundary keeps P2P, was the platform where they could come apart. Nothing here is
a V100-versus-L40S ranking; the platforms differ in silicon, PCIe generation, NCCL version,
host, and virtualization, and every cross-platform statement below is about the *shape* of a
result, not its magnitude.

## Method

**Instrument.** `nccl-tests` all-reduce from 8 bytes to 256 MB on one, two, and four GPUs,
built from the same pinned commit on both platforms. Every run carried `NCCL_DEBUG=INFO`, so
each run's log states the transport NCCL actually used on each link (`via SHM` or
`via P2P`). That line, not the topology matrix, is what every comparison cites.

**Protocol.** One warm-up plus five measured runs per arm on the V100 (raised to ten by an
engineering decision once run-to-run structure appeared), ten on the L40S; arms interleaved
round-robin so drift hits all of them alike; median with min–max and coefficient of
variation; a difference counts only when the medians sit outside each other's spread. Host
state and one-second GPU telemetry beside every run. Placement by `CUDA_VISIBLE_DEVICES`
with PCI bus-ID ordering, proven against the bus IDs before any number was read.

**Arms.** Same-NUMA pair `{0,1}`, cross-NUMA pair `{0,2}`, and the four-GPU ring, each at
NCCL's default and with the P2P policy forced the other way (`NCCL_P2P_LEVEL=PHB`, and on
the L40S `SYS` and `LOC`). On the V100, the arms also ran with and without CPU binding, which
turned out to matter.

**Application.** A 124 M-parameter causal transformer trained with PyTorch DDP in fp32 on
synthetic tokens, batch 8 × sequence 512 per GPU. Primary arms ran forty optimizer steps
with the first ten discarded. The gradient-accumulation arm ran sixteen optimizer steps
with the first four discarded, synchronizing every fourth micro-batch. It also reduced
optimizer-update frequency per token, so its throughput difference measures workload
sensitivity to the combined change, not isolated communication time. Placements and
transport policies followed the collective comparisons. The wheel bundles its own NCCL,
so the application's transports were captured separately and cited separately. See the
[workload-selection decision](decisions/EDR-002-workload-selection.md) for interpretation limits.

**Success and failure.** A result counted if every claim had a transport string and a spread
behind it. "Large collective differences, minimal workload impact" was declared a valid
outcome before the first run. Two expectations were written as hypotheses and both were
disproved; the labs were still complete.

## Findings

**1. Crossing the socket boundary produced no measurable collective penalty when transport was 
held constant: host staging on V100, and both P2P and host staging on L40S.** On the V100 with CPU
binding removed, the cross-socket pair matched the same-socket pair at 256 MB (6.70 vs
6.64 GB/s, the crossing marginally *ahead*). An earlier 2 % "boundary cost" was a binding
artefact, not the socket. On the L40S the cross-NUMA pair matched the same-NUMA pair within
spread at every size with P2P on both sides (21.05 vs 21.11 GB/s) and again with P2P off
(17.32 vs 17.28 GB/s).

**2. Communication mechanism affected pair bandwidth: forcing P2P improved V100 same-socket 
bandwidth by 15 %, while disabling P2P reduced L40S bandwidth by approximately 18 % at both 
placements.** V100: forcing P2P on the same-socket pair, +15 % (7.66 vs 6.65 GB/s). L40S:
disabling P2P cost 18 % on the same-NUMA pair (17.28 vs 21.11) and 18 % across the boundary
(17.32 vs 21.05). The cost is per byte: it vanished at 8 bytes and was 16–18 % of time at
64 KB.

**3. The library chose the mechanism, and its choice was neither the driver's capability nor
a property of the link.** On the V100, NCCL 2.26.2 staged every pair through host memory
even where the driver allowed P2P. On the L40S, NCCL 2.31.2 used P2P for every two-GPU job —
across the socket boundary included — and host memory on every link of the four-GPU ring,
pairs included. The application's bundled NCCL 2.27.5 made the identical choices. The
four-GPU default left 4.8 % on the table against an all-P2P ring (20.02 vs 20.99 GB/s).

**4. The tested mixed-transport rings were worse than the all-host-staged rings on both
platforms.** Pairs on P2P with the two socket crossings on host memory ran 10 % below the
all-host-memory default on the V100 (7.48 vs 8.27 GB/s) and 37 % below it on the L40S
(12.61 vs 20.02 GB/s). The L40S mixed ring also underperformed its all-P2P ring
(12.61 vs 20.99 GB/s). An all-P2P ring was unavailable on the V100, whose cross-socket
links did not support P2P.

**5. Almost none of this reached the training step, and when it did, the sign flipped.**
Per-GPU throughput, outside spread unless noted:

| Change forced | Collective bandwidth | Training throughput per GPU |
|---|---|---|
| V100: P2P on the pair | +15 % | −3.2 % |
| V100: mixed ring | −10 % | −3.8 % |
| L40S: host staging on the same-NUMA pair | −18 % | +0.1 % (within spread) |
| L40S: host staging across the boundary | −18 % | −1.7 % |
| L40S: mixed ring | −37 % | −4.9 % |
| L40S: all-P2P ring | +4.8 % | −1.1 % |

The cross-NUMA pair was 2.7 % *faster* than the same-NUMA pair in the L40S training step,
replicated by an independent control arm to within 0.1 %. The four-micro-batch accumulation
configuration increased per-GPU throughput by 12 % on the V100 and 17 % on the L40S.
Those are measured workload-level improvements; synchronization and optimizer updates both
became less frequent per token. They do not establish an 11–14 % upper bound on the
communication share of a step. The historical analysis outputs retain the original
“exposed-communication bound” label; this public interpretation accounts for the additional
change visible in the workload code.

**6. Two things nobody predicted.** On the V100, small-message latency sat at one of two
levels — about 3.75 or 8.3 µs — chosen per process launch, machine-wide, about one launch in
seven high, independent of placement, transport, idle time, and GPU clock; 1,600 single-GPU
runs characterized it without identifying the mechanism. On the L40S, ten draws showed nine
in a 0.14 µs band and one 1.7× higher: a weak null. And the V100 chassis, clean through three
benchmark labs, hit 84 °C under sustained training and ran 55–75 % slow until a thermal gate
was added; the L40S instance held a steady 49–52 °C at its 350 W power cap with no gate.

**7. The cloud session cost 80.9 metered minutes against a four-hour budget**, including
two redesigns of the arm tables after the transport captures contradicted the plan, and a
full rerun of both protocols.

## What This Does and Doesn't Prove

- **One workstation and one cloud instance.** Every number is about a specific machine
  in the recorded experiment sessions. The L40S instance is a KVM guest whose host CPU
  model was never captured, and two earlier instances of the same type failed acceptance (one
  with uncorrectable ECC history that no summary diagnostic reported). "The L40S" is not a
  thing this study measured.
- **One workload, one shape, fp32.** A 124 M-parameter model at batch 8 with DDP. Mixed
  precision, larger models, other parallelisms, and other frameworks were not run. The
  application findings describe this workload and configuration; they do not establish
  gradient traffic or performance under mixed precision.
- **Patterns across platforms, not controlled comparisons.** The two machines differ in
  everything but their four-GPU, two-socket shape. What travelled was the direction of
  each result; the magnitudes did not and were not expected to.
- **Mechanisms behind three results are unmeasured.** Why NCCL splits its default by
  communicator size, why a mixed-transport ring collapses, and why the cross-NUMA pair
  ran the training step faster are all open; hypotheses are recorded as hypotheses.
- **The transport-selection rule was not identified.** NCCL's documentation says only
  that, unset, it will "attempt to optimally select a value based on the architecture and
  environment"; the debug output prints the choice and not the reason.
- **The latency-level null on the L40S is weak**: ten draws, one high, which is also what
  the V100's per-launch probability would produce.
- **The V100 application numbers come from a cold-conditioned regime** (every run started
  at or below 50 °C); the L40S ran warm at its power cap. The comparisons within each
  platform hold; the operating points differ between them.

## Takeaway

On two very different four-GPU machines, crossing the socket boundary cost a two-GPU
collective nothing measurable and a training step nothing measurable or slightly less than
nothing. The cost people attribute to the boundary belonged to the mechanism — host staging
versus direct GPU writes — and the mechanism was selected by the communication library, not
by the hardware diagram, with a rule that depended on the size of the job. Even an 18 %
collective swing produced a training-throughput difference within the observed spread at
the same-NUMA placement. The workload uses DDP to overlap gradient synchronization with
backward computation, but these measurements do not isolate how much communication was
hidden. The transport that won the microbenchmark lost the step twice.

Three habits made that visible, and they transfer to any system: read the library's
transport log for every run rather than the topology tool's label; put a real application
step beside every collective number before drawing a conclusion; and write expectations down
as hypotheses, so that being wrong on paid time is a finding rather than a failure. The
decision this supports is plain: before paying for topology-aware placement or forcing a
transport, measure whether the workload benefits. In the tested application comparisons,
transport-policy changes produced per-GPU throughput differences of approximately 5 % or
less, even where collective bandwidth changed much more. The accumulation experiment
showed larger workload-level gains, but did not isolate communication time.
