# EDR-002: Selecting the Application Workload

**Record type:** Engineering Decision Record  
**Status:** Accepted  
**Original decision:** August 24, 2026  
**Public adaptation:** October 4, 2026  
**Scope:** The V100 and L40S application experiments in the [GPU Communication Boundaries case study](../gpu-communication-boundaries-case-study.md).

This record preserves a workload choice made before the application results existed. The subsequent validation is reported separately below. Cross-node experiments appeared in the original plan but were deferred; this record makes no claim about their execution.

## Problem and constraints

Collective benchmarks showed what placement and transport policies did to communication. The investigation also needed an application measurement to determine whether those differences mattered to useful work.

The workload had to:

- Execute on both V100 and L40S GPUs and fit a 16 GB V100.
- Support controlled one-, two-, and four-GPU placement.
- Produce step time and throughput metrics.
- Avoid dataset, storage, and dataloader effects obscuring the communication question.
- Generate gradient communication in the size range already characterized by the collective experiments.
- Allow communication pressure to change without selecting an unrelated model.

Compatibility was a constraint to validate empirically, not an assumption based solely on an installation succeeding.

## Decision

Use a small causal-transformer language model trained with PyTorch DistributedDataParallel (DDP) on synthetic tokens generated on the GPU.

| Parameter | Selected configuration |
|---|---|
| Model | Approximately 124 million parameters; 12 layers, width 768, 12 attention heads, vocabulary 50,257 |
| Embeddings | Tied input and output weights |
| Arithmetic | fp32 |
| Optimization | AdamW; dropout disabled |
| Primary workload per GPU | Batch 8, sequence length 512 |
| Distribution | DDP on one, two, or four selected GPUs |
| Communication-pressure variation | Gradient accumulation: synchronize every micro-batch or every fourth micro-batch |
| Primary measurements | Step time, aggregate tokens/s, and tokens/s/GPU |

The recorded environment selection was the PyTorch 2.10 CUDA 12.6 wheel line, subsequently captured as `2.10.0+cu126`. This is historical experiment provenance, not a recommendation for a current installation.

Approximately 124 million fp32 parameters imply roughly 0.5 GB of gradient data. DDP bucketed and overlapped its synchronization with backward computation. That made the workload suitable for confronting the collective results with an application's scheduling and compute behavior.

## Alternatives considered

These were design judgments at selection time. The alternatives were not benchmarked against each other.

| Alternative | Reason not selected |
|---|---|
| ResNet-50-class vision training | Expected lower communication pressure and a less direct way to vary it within the desired experiment. This was a suitability judgment, not a measured claim about vision workloads generally. |
| Tensor-parallel inference | Runtime compatibility risk on V100 would complicate staging the investigation locally before renting the L40S platform. |
| Several unrelated models | Would introduce model differences into a study intended to isolate placement and transport effects. |

## Tradeoffs and interpretation limits

**Control gained.** Synthetic on-device inputs remove an external dataset and dataloader from the measured step. Fixed model shape and per-GPU micro-batch make the placement and transport comparisons easier to interpret. fp32 avoids adding mixed-precision behavior to the experiment.

**Representativeness sacrificed.** This is an application-level training instrument, not a model-quality study or a representative production training benchmark. Loss progression is a sanity observation. The model does not represent tensor parallelism, pipeline parallelism, or large-model scaling, and fp32 results should not be generalized to mixed-precision workloads.

**Scaling must be explicit.** Each GPU processes the same primary micro-batch. Aggregate token throughput therefore grows with rank count by construction; per-GPU throughput and step time answer the infrastructure-efficiency question more directly.

**Accumulation is not a direct communication timer.** Synchronizing every fourth micro-batch also changes optimizer-update frequency per token and the amount of work per optimizer step. The model and micro-batch shape remain fixed, but the entire execution is not otherwise identical. Compare token rates, not raw optimizer-step times, and treat the result as a workload-level sensitivity estimate rather than an isolated measurement of communication time.

**Runtime must be observed independently.** PyTorch bundled its own NCCL, distinct from the system library used by nccl-tests. The application required separate version and transport captures; benchmark transport selection could not simply be assumed to carry over.

## Subsequent validation and outcome

The workload executed on both platforms. The published [V100 application analysis](../evidence/results/lab-04/lab-04-analysis.txt) and [L40S application analysis](../evidence/results/lab-05/lab-05-analysis.txt) report repeated-run step times, throughput, memory observations, and comparisons. The [L40S preflight](../evidence/results/lab-05/lab-05-preflight.txt) records the installed runtime and GPU execution check.

The application results did not reproduce the collective gains proportionally. At the L40S same-NUMA pair, an approximately 18% collective-bandwidth reduction accompanied an application-throughput change within the observed spread. Other transport changes reversed the direction of the benchmark benefit.

That was a valid outcome under the original experiment design: the workload was chosen to test whether communication differences reached the application, not to guarantee that they would.

## Revisit conditions

Reconsider this workload if the research question changes to production mixed precision, model quality, different parallelism, or scale-out behavior; if compatibility prevents comparable execution; or if a different workload is needed to investigate stronger exposed communication.

A small measured application effect alone is not a reason to discard the workload. It can be the answer to the question being investigated.

## Provenance

Adapted from `artifacts/EDR/EDR-002-ddp-transformer-workload.md` in the private investigation repository, with implementation checked against `scripts/lab-05-workload.py`. Course-specific dependencies and pending-work instructions were removed. The accumulation interpretation is clarified here; the recorded instrument and measurements have not been changed.
