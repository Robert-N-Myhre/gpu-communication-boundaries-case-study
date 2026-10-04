#!/usr/bin/env bash
# LAB-05: ten workload arms on the L40S — EDR-001 floor of 10 measured runs, interleaved,
# one warm-up round, NCCL_DEBUG=INFO in every run's log, host state before each run, telemetry
# throughout. NO THERMAL GATE (learner decision 2026-08-27: a paid datacenter instance is assumed
# adequately cooled; pre-run temps + telemetry + the analysis drift check adjudicate the assumption
# after the fact). ALL ARMS UNBOUND (guide Setup; V100 evidence, L40S untested).
# Revised on the meter 2026-10-04 (Step 2: default is SHM on every link; PHB grants pair P2P; SYS grants
# P2P across the boundary too). Second revision, same session: the first pass showed every 2-GPU DDP arm
# running P2P/CUMEM at default (A2/B2 warm-up logs, learner-reported on the meter), so A2L/B2L at
# NCCL_P2P_LEVEL=LOC ("Never use P2P", NCCL docs 2.32.3, retrieved 2026-10-04) supply the host-staged
# side; BS2 stays as a same-transport replicate control. QS4 uses SYS; QH4 keeps PHB (the mixed ring).
# Learner-run instrument. Evidence: results/lab-05/runs/<arm>-{warmup,run}-NN.log (raw, authoritative)
#                                   results/lab-05/whost/<arm>-{warmup,run}-NN.txt  (workload host state; nccl protocol owns host/)
#                                   results/lab-05/runs/telemetry.csv
#                                   results/lab-05/lab-05-runs.csv
set -euo pipefail
export CUDA_DEVICE_ORDER=PCI_BUS_ID
RUNS="${RUNS:-10}"; WARMUP="${WARMUP:-1}"
[ "${RUNS}" -ge 10 ] || { echo "FAIL: RUNS=${RUNS} below the course floor of 10 (EDR-001)" >&2; exit 1; }
VENV="${TORCH_VENV:-${HOME}/gputopo-tools/torch-venv}"; PY="${VENV}/bin/python"; TORCHRUN="${VENV}/bin/torchrun"
[ -x "${PY}" ] || { echo "FAIL: ${PY} missing" >&2; exit 1; }

# arm|GPU mask|ngpus|NCCL_P2P_LEVEL (empty=default)|accum|steps|measure_from — do not edit between rounds.
ARMS=(
  "S1|0|1||1|40|10"
  "A2|0,1|2||1|40|10"
  "A2L|0,1|2|LOC|1|40|10"
  "B2|0,2|2||1|40|10"
  "BS2|0,2|2|SYS|1|40|10"
  "B2L|0,2|2|LOC|1|40|10"
  "Q4|0,1,2,3|4||1|40|10"
  "QH4|0,1,2,3|4|PHB|1|40|10"
  "QS4|0,1,2,3|4|SYS|1|40|10"
  "QL4|0,1,2,3|4||4|16|4"
)

OUT_DIR="results/lab-05"; RUN_DIR="${OUT_DIR}/runs"; HOST_DIR="${OUT_DIR}/whost"; CSV="${OUT_DIR}/lab-05-runs.csv"; TELEM="${RUN_DIR}/telemetry.csv"
mkdir -p "${RUN_DIR}" "${HOST_DIR}"
{
  echo "# lab-05-runs"; echo "# captured_utc: $(date -u +%Y-%m-%dT%H:%M:%SZ)"; echo "# hostname: $(hostname)"
  echo "# command: $0 $* (workload: scripts/lab-05-workload.py; warmup rounds=${WARMUP}; measured rounds=${RUNS}; interleaved; all arms unbound)"
  echo "# arms: ${ARMS[*]}"
  echo "# host-state note: ps pcpu is lifetime-average CPU share, not instantaneous"
  echo "arm,\"mask\",ngpus,p2p_level,accum,phase,run,start_utc,status,duration_s,pre_max_temp_c,load1,procs_running,mhz_min,mhz_max,steady_median_step_ms,tokens_per_s"
} > "${CSV}"
{ echo "# lab-05 telemetry (1 sample/s, whole step)"; echo "# captured_utc: $(date -u +%Y-%m-%dT%H:%M:%SZ)"; echo "# hostname: $(hostname)"; } > "${TELEM}"
Q="timestamp,index,power.draw,clocks.sm,temperature.gpu,utilization.gpu,memory.used"
nvidia-smi --query-gpu="${Q}" --format=csv | awk 'NR==1' >> "${TELEM}"
( while true; do nvidia-smi --query-gpu="${Q}" --format=csv,noheader; sleep 1; done ) >> "${TELEM}" &
TELEM_PID=$!; trap 'kill "${TELEM_PID}" 2>/dev/null || true' EXIT

host_state() { # $1=file ; prints "load1,procs_running,mhz_min,mhz_max"
  local LA RUNNING MHZ
  LA=$(cut -d' ' -f1 /proc/loadavg); RUNNING=$(cut -d' ' -f4 /proc/loadavg | cut -d/ -f1)
  MHZ=$(awk -F': ' '/^cpu MHz/{print $2}' /proc/cpuinfo | sort -n | awk 'NR==1{mn=$1} {mx=$1} END{printf "%.0f,%.0f", mn, mx}')
  { echo "# host state before run ($(date -u +%Y-%m-%dT%H:%M:%SZ))"; echo "loadavg: $(cat /proc/loadavg)"
    echo "cpu_mhz_min,max: ${MHZ}"; echo "## top CPU consumers by lifetime-average pcpu, not instantaneous (ps -eo pcpu,pid,comm --sort=-pcpu | head -8)"
    ps -eo pcpu,pid,comm --sort=-pcpu | head -8; } > "$1"
  echo "${LA},${RUNNING},${MHZ}"
}

declare -A OKC FAILC
do_run() { # $1=arm-spec $2=phase $3=NN
  local ARM MASK NG LVL ACC STEPS MFROM LOG PRE START T0 STATUS=ok DUR HS SUM MED TPS LAUNCH
  IFS='|' read -r ARM MASK NG LVL ACC STEPS MFROM <<< "$1"
  LOG="${RUN_DIR}/${ARM}-$2-$3.log"
  HS=$(host_state "${HOST_DIR}/${ARM}-$2-$3.txt")
  PRE=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits | sort -n | tail -1)
  START=$(date -u +%Y-%m-%dT%H:%M:%SZ); T0=$(date +%s)
  if [ "${NG}" = 1 ]; then LAUNCH="${PY}"; else LAUNCH="${TORCHRUN} --standalone --nproc-per-node ${NG}"; fi
  { echo "# lab-05 arm ${ARM} $2 $3"; echo "# start_utc: ${START}"; echo "# hostname: $(hostname)"
    echo "# mask: CUDA_VISIBLE_DEVICES=${MASK}  ngpus: ${NG}  nccl_p2p_level: ${LVL:-<default>}  accum: ${ACC}  host: unbound"
    echo "# command: ${LAUNCH} scripts/lab-05-workload.py --steps ${STEPS} --measure-from ${MFROM} --accum ${ACC}"
    echo "# cuda_device_order: ${CUDA_DEVICE_ORDER}"; echo "# host_state: load1,procs_running,cpu_mhz_min,cpu_mhz_max = ${HS}"
    echo "# pre_run_max_gpu_temp_c: ${PRE} (no thermal gate on this platform — guide Setup)"; echo; } > "${LOG}"
  echo "--- arm ${ARM} $2 $3: CUDA_VISIBLE_DEVICES=${MASK} ngpus ${NG} NCCL_P2P_LEVEL=${LVL:-<default>} accum ${ACC} pre-temp ${PRE}C load1=${HS%%,*} ---"
  if [ -n "${LVL}" ]; then
    CUDA_VISIBLE_DEVICES="${MASK}" NCCL_DEBUG=INFO NCCL_P2P_LEVEL="${LVL}" ${LAUNCH} scripts/lab-05-workload.py --steps "${STEPS}" --measure-from "${MFROM}" --accum "${ACC}" 2>&1 | tee -a "${LOG}" | (grep -E '^step |^LAB05SUMMARY|^# torch| via ' || true) || STATUS=failed
  else
    CUDA_VISIBLE_DEVICES="${MASK}" NCCL_DEBUG=INFO ${LAUNCH} scripts/lab-05-workload.py --steps "${STEPS}" --measure-from "${MFROM}" --accum "${ACC}" 2>&1 | tee -a "${LOG}" | (grep -E '^step |^LAB05SUMMARY|^# torch| via ' || true) || STATUS=failed
  fi
  DUR=$(( $(date +%s) - T0 ))
  SUM=$(grep -m1 '^LAB05SUMMARY ' "${LOG}" || true)
  MED=$(grep -oE '"steady_median_step_ms": [0-9.]+' <<< "${SUM}" | grep -oE '[0-9.]+' || echo "-")
  TPS=$(grep -oE '"tokens_per_s": [0-9.]+' <<< "${SUM}" | grep -oE '[0-9.]+' || echo "-")
  [ "${MED}" != "-" ] || STATUS=failed
  echo "${ARM},\"${MASK}\",${NG},${LVL:-default},${ACC},$2,$3,${START},${STATUS},${DUR},${PRE},${HS},${MED},${TPS}" >> "${CSV}"
  echo "--- round $2 $3 arm ${ARM}: ${STATUS} (${DUR}s, median step ${MED} ms, ${TPS} tokens/s) ---"
  if [ "$2" = run ]; then
    if [ "${STATUS}" = ok ]; then OKC[${ARM}]=$(( ${OKC[${ARM}]:-0} + 1 )); else FAILC[${ARM}]=$(( ${FAILC[${ARM}]:-0} + 1 )); fi
  fi
  return 0
}

echo "=== lab-05 protocol: 10 arms interleaved (S1 A2 A2L B2 BS2 B2L Q4 QH4 QS4 QL4), ${WARMUP} warm-up + ${RUNS} measured rounds; EDR-001 floor 10; all arms unbound; no thermal gate (learner decision, guide Setup) ==="
r=1; while [ "${r}" -le "${WARMUP}" ]; do for spec in "${ARMS[@]}"; do do_run "${spec}" warmup "$(printf '%02d' "${r}")"; done; r=$((r+1)); done
r=1; while [ "${r}" -le "${RUNS}" ]; do for spec in "${ARMS[@]}"; do do_run "${spec}" run "$(printf '%02d' "${r}")"; done; r=$((r+1)); done

kill "${TELEM_PID}" 2>/dev/null || true; wait "${TELEM_PID}" 2>/dev/null || true; trap - EXIT
echo "=== lab-05 protocol summary ==="
echo "evidence: ${CSV} + ${RUN_DIR}/ + ${HOST_DIR}/ (raw logs authoritative) + ${TELEM}"
RC=0
for spec in "${ARMS[@]}"; do ARM="${spec%%|*}"; echo "arm ${ARM}: measured_ok=${OKC[${ARM}]:-0} measured_failed=${FAILC[${ARM}]:-0}"; [ "${OKC[${ARM}]:-0}" -gt 0 ] || RC=1; done
[ "${RC}" -eq 0 ] || { echo "FAIL: an arm has zero successful measured runs" >&2; exit 1; }
echo "Next: python3 scripts/lab-05-analysis.py, then python3 scripts/lab-05-telemetry-summary.py"
