#!/usr/bin/env python3
"""LAB-05: per-arm workload statistics on the L40S and the six comparisons, each one knob.

Learner-run instrument. Reads results/lab-05/runs/<arm>-run-*.log (measured runs only; the
LAB05SUMMARY json line each carries) and the runs CSV for host state. Writes
results/lab-05/lab-05-analysis.txt and prints it.

Scaling definition (declared in the guide, matching LAB-03's): fixed per-rank workload —
every GPU processes the same batch x seq per micro-step at every world size; global tokens
grow with GPU count. tokens_per_s_per_gpu and step time answer the caller's question.
Comparisons use the course spread test (median outside the other side's min-max); the
EDR-001 floor of 10 measured runs applies. The comm-fraction lever: Q4 vs QL4 differ only
in gradient-sync frequency (every step vs every 4th micro-batch), so their per-token rates
bound how much of a step is exposed communication.
"""
import datetime, json, socket, statistics, sys
from pathlib import Path

ARMS = {"S1": "1 GPU {0} default", "A2": "2 GPU {0,1} same-NUMA default (P2P, measured)", "A2L": "2 GPU {0,1} same-NUMA LOC (host-staged)",
        "B2": "2 GPU {0,2} cross-NUMA default (P2P, measured)", "BS2": "2 GPU {0,2} cross-NUMA SYS (P2P — replicate of B2)",
        "B2L": "2 GPU {0,2} cross-NUMA LOC (host-staged)", "Q4": "4 GPU default (all SHM)",
        "QH4": "4 GPU PHB (mixed ring: pair P2P, SYS SHM)", "QS4": "4 GPU SYS (all-P2P ring)",
        "QL4": "4 GPU default, accum 4 (1/4 sync rate)"}
# Arm set revised 2026-10-04 on the meter: Step 2 measured SHM on every link at default (results/lab-05/lab-05-nccl-graph.txt).
COMPARISONS = [
    ("S1", "A2", "scaling 1 -> 2 GPUs, same NUMA, default"),
    ("A2", "Q4", "scaling 2 -> 4 GPUs, default"),
    ("A2", "B2", "same-NUMA vs cross-NUMA pair, default — the SYS boundary with P2P on both sides (measured transport)"),
    ("A2", "A2L", "pair mechanism at the app — P2P vs host-staged, same placement (LAB-04 C2 cell, second platform)"),
    ("B2", "B2L", "boundary mechanism at the app — P2P vs host-staged across SYS, same placement"),
    ("A2L", "B2L", "same-NUMA vs cross-NUMA pair, both host-staged — the V100 comparison like for like"),
    ("B2", "BS2", "replicate control — same transport (P2P), different knob label: the app-level noise floor"),
    ("Q4", "QH4", "all-SHM ring vs mixed ring (pair P2P) — LAB-04 P4 pattern on L40S"),
    ("Q4", "QS4", "all-SHM ring vs all-P2P ring at the app"),
    ("Q4", "QL4", "sync every step vs every 4th micro-batch — exposed-communication bound"),
]


def load_runs(dirp, arm):
    out = {}
    for f in sorted(Path(dirp).glob(f"{arm}-run-*.log")):
        for line in f.read_text(errors="replace").splitlines():
            if line.startswith("LAB05SUMMARY "):
                out[f.name] = json.loads(line[len("LAB05SUMMARY "):]); break
    return out


def st(v):
    med = statistics.median(v); mean = statistics.mean(v); sd = statistics.stdev(v) if len(v) > 1 else 0.0
    return med, min(v), max(v), (sd / mean * 100 if mean else 0.0)


def main():
    root = Path("results/lab-05")
    now = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    lines = ["# lab-05-analysis", f"captured_utc: {now}", f"hostname: {socket.gethostname()}",
             f"command: {' '.join(sys.argv)}", "source: results/lab-05/runs/<arm>-run-*.log LAB05SUMMARY lines (measured runs only)",
             "scaling_definition: fixed per-rank workload (guide §1); per-GPU rate and step time answer the caller's question",
             "spread test: a difference is a finding only where medians sit outside the other side's min-max", ""]
    data = {arm: load_runs(root / "runs", arm) for arm in ARMS}
    data = {k: v for k, v in data.items() if v}
    if not data:
        print("FAIL: no measured runs parsed under results/lab-05/runs/", file=sys.stderr); return 1

    # host state, right-anchored (mask is quoted but stay defensive)
    hs = {}
    csvp = root / "lab-05-runs.csv"
    TAIL = ["p2p_level", "accum", "phase", "run", "start_utc", "status", "duration_s", "pre_max_temp_c",
            "load1", "procs_running", "mhz_min", "mhz_max", "steady_median_step_ms", "tokens_per_s"]
    if csvp.exists():
        for line in csvp.read_text().splitlines():
            if not line or line.startswith("#") or line.startswith("arm,"): continue
            f = line.split(",")
            if len(f) < 15: continue
            r = dict(zip(TAIL, [x.strip('"') for x in f[-14:]])); r["arm"] = f[0]
            if r["phase"] == "run": hs[(r["arm"], r["run"])] = r

    for arm, runs in data.items():
        meds = [d["steady_median_step_ms"] for d in runs.values()]
        tps = [d["tokens_per_s"] for d in runs.values()]
        tpg = [d["tokens_per_s_per_gpu"] for d in runs.values()]
        mem = max(d["max_mem_mib"] for d in runs.values())
        m, lo, hi, cv = st(meds); tm, tlo, thi, tcv = st(tps); gm = statistics.median(tpg)
        floor_flag = "" if len(runs) >= 10 else f"  ** BELOW EDR-001 FLOOR: only {len(runs)} parsed measured runs (<10) **"
        drifts = []
        for d in runs.values():
            steady_ms = d["step_ms"][d["measure_from"]:]
            half = len(steady_ms) // 2
            if half >= 2 and statistics.median(steady_ms[:half]):
                drifts.append(statistics.median(steady_ms[half:]) / statistics.median(steady_ms[:half]))
        drift = f"  within-run drift (2nd/1st steady half): median {statistics.median(drifts):.3f}" if drifts else ""
        lines += [f"## arm {arm} — {ARMS[arm]} — {len(runs)} runs{floor_flag}",
                  f"  steady median step ms: median {m:.1f}  min-max {lo:.1f}-{hi:.1f}  cv% {cv:.2f}{drift}",
                  f"  tokens/s: median {tm:.0f}  min-max {tlo:.0f}-{thi:.0f}  cv% {tcv:.2f}   tokens/s/GPU: {gm:.0f}",
                  f"  max rank-0 mem MiB: {mem:.0f}   loss first->last (run 01): {list(runs.values())[0]['loss_first']} -> {list(runs.values())[0]['loss_last']}", ""]

    lines.append("## comparisons (ratio = second/first; step-time ratio < 1 = second faster; per-GPU-rate ratio > 1 = second faster)")
    for a, b, why in COMPARISONS:
        if a not in data or b not in data:
            lines += ["", f"### {a} vs {b} — {why}: SKIPPED (missing arm)"]; continue
        lines += ["", f"### {a} vs {b} — {why}", ""]
        for label, key, better_high in [("step_ms", "steady_median_step_ms", False), ("tok/s/GPU", "tokens_per_s_per_gpu", True)]:
            if (a, b) == ("Q4", "QL4") and key == "steady_median_step_ms":
                lines.append(f"  {label:>10}: not comparable — a QL4 optimizer step is 4 micro-batches by construction; tok/s/GPU carries this comparison")
                continue
            va = [d[key] for d in data[a].values()]; vb = [d[key] for d in data[b].values()]
            ma, la, ha, _ = st(va); mb, lb, hb, _ = st(vb)
            within = "within spread" if (lb <= ma <= hb or la <= mb <= ha) else ""
            lines.append(f"  {label:>10}: {ma:>9.1f} vs {mb:>9.1f}  ratio {mb/ma:>6.3f}  spreads {la:.1f}-{ha:.1f} | {lb:.1f}-{hb:.1f}  {within}")

    if "Q4" in data and "QL4" in data:
        q = statistics.median([d["tokens_per_s_per_gpu"] for d in data["Q4"].values()])
        ql = statistics.median([d["tokens_per_s_per_gpu"] for d in data["QL4"].values()])
        lines += ["", "## exposed-communication bound (derived from Q4 vs QL4 medians above)",
                  f"  per-GPU token rate at full sync {q:.0f} vs quarter sync {ql:.0f}: "
                  f"rate ratio {ql/q:.3f} — the gap bounds what full-rate gradient sync costs this workload on this topology.",
                  "  (Interpretation belongs to the findings; DDP overlaps sync with backward, so this is a bound, not a comm time.)"]

    lines += ["", "## per-run table with host state (latency-level watch: RSR-004 — one process, one draw)", "",
              f"{'arm':>5}{'run':>4}{'step_ms':>9}{'tok/s':>9}{'load1':>7}{'mhz_max':>8}{'pre_C':>6}"]
    for arm, runs in data.items():
        for name, d in runs.items():
            run = name.split("-run-")[1].split(".")[0]
            h = hs.get((arm, run), {})
            g = lambda k: h.get(k, "-")
            lines.append(f"{arm:>5}{run:>4}{d['steady_median_step_ms']:>9.1f}{d['tokens_per_s']:>9.0f}{g('load1'):>7}{g('mhz_max'):>8}{g('pre_max_temp_c'):>6}")
    lines += ["", "A difference is a finding only where medians sit outside each other's spread; attribute it to a",
              "mechanism only with lab-05-transports.txt behind it. Per-GPU rate and step time answer the caller's",
              "question; aggregate tokens/s rises with world size under this scaling definition by construction."]
    out = root / "lab-05-analysis.txt"; out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines)); print(f"evidence: {out}"); return 0


if __name__ == "__main__":
    sys.exit(main())
