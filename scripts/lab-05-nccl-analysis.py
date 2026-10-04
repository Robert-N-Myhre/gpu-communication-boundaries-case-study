#!/usr/bin/env python3
"""LAB-05: per-arm collective statistics on the L40S and the six comparisons, plus the per-run
small-message table (RSR-004's L40S question: do the run-scoped latency levels exist here?).

Learner-run instrument, off-meter: runs on any clone after the instance's evidence is pulled.
Reads results/lab-05/nccl/<arm>-run-*.log (measured runs only) and the runs CSV for host state.
Writes results/lab-05/lab-05-nccl-analysis.txt and prints it.

Row parsed (all_reduce_perf): size count type redop root time algbw busbw #wrong -> (time_us, algbw, busbw).
Same reading rules as LAB-03 (fixed per-rank buffer; busbw = algbw * 2(n-1)/n): latency range on time,
bandwidth range on busbw AND algbw. Cross-platform comparison against the V100 numbers belongs to
findings/LAB-05.md as patterns — this script is single-platform on purpose (config: cross-platform
comparisons are not controlled experiments).
"""
import datetime, socket, statistics, sys
from pathlib import Path

ARMS = {"S1": "{0} 1 GPU (no communication)", "A": "{0,1} same-NUMA default (P2P, measured)", "AP": "{0,1} same-NUMA PHB (P2P — replicate of A)",
        "AL": "{0,1} same-NUMA LOC (host-staged)",
        "B": "{0,2} cross-NUMA default (P2P, measured)", "BS": "{0,2} cross-NUMA SYS (P2P — replicate of B)",
        "BL": "{0,2} cross-NUMA LOC (host-staged)",
        "Q": "{0,1,2,3} default (all SHM)", "QH": "{0,1,2,3} PHB (mixed ring: pair P2P, SYS SHM)",
        "QS": "{0,1,2,3} SYS (all-P2P ring)"}
# Arm set revised 2026-10-04 on the meter: Step 2 measured SHM on every link at default (results/lab-05/lab-05-nccl-graph.txt).
COMPARISONS = [
    ("S1", "A", "what communication costs at all: 1 GPU vs 2 GPUs (time at every size)", "time"),
    ("A", "B", "same-NUMA vs cross-NUMA pair, default — the SYS boundary with P2P on both sides (measured transport)", "both"),
    ("A", "AL", "pair mechanism — P2P vs host-staged at PHB distance, same placement (RSR-003 cell, second platform)", "both"),
    ("B", "BL", "boundary mechanism — P2P vs host-staged across SYS, same placement, one knob", "both"),
    ("AL", "BL", "same-NUMA vs cross-NUMA pair, both host-staged — the V100 comparison like for like", "both"),
    ("A", "AP", "replicate control — same transport (P2P), different knob label: the noise floor", "both"),
    ("B", "BS", "replicate control — same transport (P2P), different knob label: the noise floor", "both"),
    ("A", "Q", "scaling 2 -> 4 GPUs at default transport", "both"),
    ("Q", "QH", "all-SHM ring vs mixed ring (pair P2P) — the V100 P4 pattern on L40S", "both"),
    ("Q", "QS", "all-SHM ring vs all-P2P ring — the cell the V100 could not build", "both"),
    ("QH", "QS", "mixed ring vs all-P2P ring — what P2P on the two SYS links adds", "both"),
]
LAT_MAX, BW_MIN, FLAG_PCT = 65536, 1048576, 2.0


def parse(path):
    rows, wrong = {}, 0
    for line in path.read_text(errors="replace").splitlines():
        t = line.split()
        if len(t) >= 12 and t[0].isdigit() and t[1].isdigit():
            try: rows[int(t[0])] = (float(t[5]), float(t[6]), float(t[7]))
            except ValueError: continue
            for w in t[8:9] + t[12:13]:  # out-of-place and in-place #wrong columns
                if w.isdigit() and int(w) != 0: wrong += 1
    return rows, wrong


def st(v):
    med = statistics.median(v); mean = statistics.mean(v); sd = statistics.stdev(v) if len(v) > 1 else 0.0
    return med, min(v), max(v), (sd / mean * 100 if mean else 0.0)


def load_runs(pattern_dir, glob):
    parsed = {l.name: parse(l) for l in sorted(Path(pattern_dir).glob(glob))}
    ok = {k: r for k, (r, w) in parsed.items() if r and w == 0}
    bad = {k: w for k, (r, w) in parsed.items() if r and w > 0}
    return ok, bad


def series(runs, size, idx):
    return [(r, d[size][idx]) for r, d in runs.items() if size in d]


def compare(lines, a, ra, b, rb, why, mode):
    lines += ["", f"### {a} vs {b} — {why}", "",
              f"{'size_B':>11}{'metric':>8}{a+'_med':>10}{b+'_med':>10}{'ratio':>8}{'spread_'+a:>14}{'spread_'+b:>14}"]
    sizes = sorted(set(next(iter(ra.values()))) & set(next(iter(rb.values()))))
    for s in sizes:
        if mode == "time": metrics = [(0, "time_us")]
        elif s <= LAT_MAX: metrics = [(0, "time_us")]
        elif s >= BW_MIN: metrics = [(2, "busbw"), (1, "algbw")]
        else: continue
        for idx, metric in metrics:
            va = [x for _, x in series(ra, s, idx)]; vb = [x for _, x in series(rb, s, idx)]
            if not va or not vb: continue
            ma, la, ha, _ = st(va); mb, lb, hb, _ = st(vb)
            within = "within spread" if (lb <= ma <= hb or la <= mb <= ha) else ""
            lines.append(f"{s:>11}{metric:>8}{ma:>10.2f}{mb:>10.2f}{(mb/ma if ma else 0):>8.3f}{f'{la:.2f}-{ha:.2f}':>14}{f'{lb:.2f}-{hb:.2f}':>14}  {within}")


def main():
    root = Path("results/lab-05")
    now = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    lines = ["# lab-05-nccl-analysis", f"captured_utc: {now}", f"hostname: {socket.gethostname()} (off-meter analysis host; capture host in the run logs)",
             f"command: {' '.join(sys.argv)}", "source: results/lab-05/nccl/<arm>-run-*.log (measured runs only)",
             "scaling_definition: fixed per-rank buffer (learner, 2026-08-23); busbw = algbw * 2(n-1)/n (nccl-tests doc/PERFORMANCE.md)",
             f"ranges: latency <= {LAT_MAX} B on time (us); bandwidth >= {BW_MIN} B on busbw AND algbw (GB/s)",
             f"flag_rule: > {FLAG_PCT} % from median, bandwidth range (busbw) only; latency range: CV reported, spread is the test",
             "cross-platform: V100 comparisons belong to findings/LAB-05.md as patterns, not to this script",
             "wrong_check: a run with any #wrong != 0 row is a failed run (guide Setup) — reported per arm and excluded from stats", ""]
    data, wrongd = {}, {}
    for arm in ARMS:
        data[arm], wrongd[arm] = load_runs(root / "nccl", f"{arm}-run-*.log")
    for arm, bad in wrongd.items():
        if bad and not data.get(arm):
            lines.append(f"arm {arm}: ALL {len(bad)} parsed runs have #wrong != 0 — failed runs (guide Setup), arm excluded entirely")
    data = {k: v for k, v in data.items() if v}
    if not data:
        print("FAIL: no measured runs parsed under results/lab-05/nccl/", file=sys.stderr); return 1

    for arm, runs in data.items():
        floor_flag = "" if len(runs) >= 10 else f"  ** BELOW EDR-001 FLOOR: only {len(runs)} parsed measured runs (<10) **"
        sizes = sorted(set().union(*[set(d) for d in runs.values()]))
        lines += [f"## arm {arm} — {ARMS[arm]} — {len(runs)} runs{floor_flag}", ""]
        if wrongd.get(arm):
            lines += [f"  ** #wrong != 0 in {len(wrongd[arm])} run(s) — failed runs (guide Setup), excluded from stats: {', '.join(sorted(wrongd[arm]))} **"]
        lines += [
                  f"{'size_B':>11}{'n':>3}{'time_med':>10}{'time_cv%':>9}{'algbw_med':>10}{'busbw_med':>10}{'bw_min':>8}{'bw_max':>8}{'bw_cv%':>8}  bw_flags"]
        for s in sizes:
            tv = [x for _, x in series(runs, s, 0)]; av = [x for _, x in series(runs, s, 1)]; bv = series(runs, s, 2)
            tm, _, _, tcv = st(tv); am = statistics.median(av); bm, blo, bhi, bcv = st([x for _, x in bv])
            flags = [f"{r}:{x:.2f}" for r, x in bv if s >= BW_MIN and bm and abs(x - bm) / bm * 100 > FLAG_PCT]
            lines.append(f"{s:>11}{len(tv):>3}{tm:>10.2f}{tcv:>9.2f}{am:>10.2f}{bm:>10.2f}{blo:>8.2f}{bhi:>8.2f}{bcv:>8.2f}  {' '.join(flags) or '-'}")
        lines.append("")

    lines.append("## comparisons (ratio = second / first; time ratio < 1 = second faster; bandwidth ratio > 1 = second faster)")
    for a, b, why, mode in COMPARISONS:
        if a not in data or b not in data:
            lines += ["", f"### {a} vs {b} — {why}: SKIPPED (missing arm)"]; continue
        compare(lines, a, data[a], b, data[b], why, mode)

    lines += ["", "## per-run 8 B time with host state (RSR-004: do the run-scoped levels exist on this platform?)", "",
              f"{'arm':>4}{'run':>4}{'time_8B':>9}{'time_64B':>9}{'time_16K':>9}{'time_64K':>9}{'load1':>7}{'running':>8}{'mhz_min':>8}{'mhz_max':>8}{'pre_C':>6}"]
    hs = {}
    csvp = root / "lab-05-nccl-runs.csv"
    if csvp.exists():
        # Right-anchored: the quoted mask may split on commas; the last 12 fields are fixed.
        TAIL = ["ngpus", "p2p_level", "phase", "run", "start_utc", "status",
                "duration_s", "pre_max_temp_c", "load1", "procs_running", "cpu_mhz_min", "cpu_mhz_max"]
        for line in csvp.read_text().splitlines():
            if not line or line.startswith("#") or line.startswith("arm,"): continue
            f = line.split(",")
            if len(f) < 13: continue
            r = dict(zip(TAIL, [x.strip('"') for x in f[-12:]]))
            r["arm"] = f[0]
            if r["phase"] == "run": hs[(r["arm"], r["run"])] = r
    for arm, runs in data.items():
        for name, d in runs.items():
            run = name.split("-run-")[1].split(".")[0]
            h = hs.get((arm, run), {})
            g = lambda k: h.get(k, "-")
            lines.append(f"{arm:>4}{run:>4}{d.get(8,(0,))[0]:>9.2f}{d.get(64,(0,))[0]:>9.2f}{d.get(16384,(0,))[0]:>9.2f}{d.get(65536,(0,))[0]:>9.2f}{g('load1'):>7}{g('procs_running'):>8}{g('cpu_mhz_min'):>8}{g('cpu_mhz_max'):>8}{g('pre_max_temp_c'):>6}")
    lines += ["", "A difference is a finding only where the medians sit outside each other's min-max spread;",
              "attribute it to a mechanism only with the transport strings from lab-05-nccl-graph.txt behind it.",
              "busbw rises with GPU count by construction (2(n-1)/n); only algbw and time say whether the caller got faster."]
    out = root / "lab-05-nccl-analysis.txt"; out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines)); print(f"evidence: {out}"); return 0


if __name__ == "__main__":
    sys.exit(main())
