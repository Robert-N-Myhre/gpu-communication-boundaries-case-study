#!/usr/bin/env python3
"""LAB-05: the course workload — a small causal-transformer language model trained on
synthetic tokens under PyTorch DDP. Deliberately minimal and self-contained: no dataset,
no dataloader, no AMP, fp32 throughout, dropout 0. Communication per optimizer step is
the DDP gradient all-reduce (~ model size in fp32), overlapped with backward in default
~25 MB buckets; --accum N syncs gradients only every Nth micro-batch (comm-pressure knob).

Learner-run instrument, launched by lab-05-{transports,protocol}.sh — single GPU via
plain python, multi-GPU via torchrun (env RANK/WORLD_SIZE/LOCAL_RANK). Rank 0 prints one
line per optimizer step and a final "LAB05SUMMARY {json}" line; the launching script owns
the evidence file. Exits 3 on a non-finite loss (a failed run, recorded, never retried here).

Copied from scripts/lab-04-workload.py on 2026-08-27 (identifiers aside, behavior identical at
copy time) so that any platform-specific change needed on the L40S stays out of the LAB-04
evidence chain. If this file diverges, the divergence is declared here and in the findings.
"""
import argparse, contextlib, json, os, sys, time

import torch
import torch.nn as nn
import torch.nn.functional as F


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=40, help="optimizer steps")
    p.add_argument("--measure-from", type=int, default=10, help="first steady-state step (0-based)")
    p.add_argument("--batch", type=int, default=8, help="micro-batch per GPU")
    p.add_argument("--seq", type=int, default=512)
    p.add_argument("--layers", type=int, default=12)
    p.add_argument("--width", type=int, default=768)
    p.add_argument("--heads", type=int, default=12)
    p.add_argument("--vocab", type=int, default=50257)
    p.add_argument("--accum", type=int, default=1, help="micro-batches per optimizer step (grad sync every Nth)")
    p.add_argument("--lr", type=float, default=3e-4)
    return p.parse_args()


class LM(nn.Module):
    def __init__(self, a):
        super().__init__()
        self.tok = nn.Embedding(a.vocab, a.width)
        self.pos = nn.Embedding(a.seq, a.width)
        layer = nn.TransformerEncoderLayer(d_model=a.width, nhead=a.heads, dim_feedforward=4 * a.width,
                                           dropout=0.0, activation="gelu", batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, num_layers=a.layers)
        self.head = nn.Linear(a.width, a.vocab, bias=False)
        self.head.weight = self.tok.weight  # tied as in GPT-2 — the ~124 M count and ~0.5 GB gradient sync assume it
        nn.init.normal_(self.tok.weight, std=0.02)  # GPT-2's embedding scale; default N(0,1) under a tied head
        nn.init.normal_(self.pos.weight, std=0.02)  # started loss at ~771 instead of ln(vocab) (smoke, 2026-08-27)

    def forward(self, x, mask):
        h = self.tok(x) + self.pos.weight[None, : x.shape[1], :]
        h = self.enc(h, mask=mask, is_causal=True)
        return self.head(h)


def main():
    a = parse_args()
    rank = int(os.environ.get("RANK", "0")); world = int(os.environ.get("WORLD_SIZE", "1"))
    local = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local); device = torch.device("cuda", local)
    torch.manual_seed(1234 + rank)
    dist = None
    if world > 1:
        import torch.distributed as dist
        dist.init_process_group("nccl")
    model = LM(a).to(device)
    nparams = sum(p.numel() for p in model.parameters())
    if world > 1:
        from torch.nn.parallel import DistributedDataParallel as DDP
        model = DDP(model, device_ids=[local])
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr)
    mask = nn.Transformer.generate_square_subsequent_mask(a.seq, device=device)
    gen = torch.Generator(device=device); gen.manual_seed(4321 + rank)

    if rank == 0:
        nccl = ".".join(map(str, torch.cuda.nccl.version()))
        print(f"# torch {torch.__version__} nccl {nccl} cuda {torch.version.cuda} device {torch.cuda.get_device_name(local)}", flush=True)
        print(f"# params {nparams} world {world} batch/gpu {a.batch} seq {a.seq} accum {a.accum} steps {a.steps} fp32 AdamW", flush=True)

    times, losses = [], []
    for s in range(a.steps):
        torch.cuda.synchronize(); t0 = time.perf_counter()
        for micro in range(a.accum):
            x = torch.randint(0, a.vocab, (a.batch, a.seq), device=device, generator=gen)
            sync = (world > 1 and micro < a.accum - 1)
            ctx = model.no_sync() if sync else contextlib.nullcontext()
            with ctx:
                logits = model(x, mask)
                loss = F.cross_entropy(logits[:, :-1].reshape(-1, a.vocab), x[:, 1:].reshape(-1))
                (loss / a.accum).backward()
        opt.step(); opt.zero_grad(set_to_none=True)
        torch.cuda.synchronize(); dt = time.perf_counter() - t0
        lv = loss.item()
        if not (lv == lv and abs(lv) != float("inf")):
            if rank == 0: print(f"FAIL: non-finite loss at step {s}", flush=True)
            sys.exit(3)
        times.append(dt); losses.append(lv)
        if rank == 0: print(f"step {s:03d}  {dt*1000:8.1f} ms  loss {lv:.4f}", flush=True)

    steady = times[a.measure_from:]
    med = sorted(steady)[len(steady) // 2]
    if rank == 0:
        toks = a.batch * a.seq * a.accum * world
        print("LAB05SUMMARY " + json.dumps({
            "world": world, "params": nparams, "batch_per_gpu": a.batch, "seq": a.seq, "accum": a.accum,
            "steps": a.steps, "measure_from": a.measure_from, "steady_steps": len(steady),
            "steady_median_step_ms": round(med * 1000, 2),
            "steady_min_step_ms": round(min(steady) * 1000, 2), "steady_max_step_ms": round(max(steady) * 1000, 2),
            "tokens_per_s": round(toks / med, 1), "tokens_per_s_per_gpu": round(toks / med / world, 1),
            "max_mem_mib": round(torch.cuda.max_memory_allocated(device) / 2**20, 1),
            "loss_first": round(losses[0], 4), "loss_last": round(losses[-1], 4),
            "step_ms": [round(t * 1000, 2) for t in times],
            "torch": torch.__version__, "nccl": ".".join(map(str, torch.cuda.nccl.version())),
        }), flush=True)
    if world > 1:
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    sys.exit(main())
