#!/usr/bin/env bash
# LAB-05: eight all-reduce arms on the L40S (1, 2 and 4 GPUs; default, NCCL_P2P_LEVEL=PHB and =SYS).
# Revised on the meter 2026-10-04 after Step 2 measured SHM on every link at default: PHB grants pair
# P2P (as on the V100) and SYS grants P2P across the NUMA boundary too — so the mechanism pairs are
# A/AP (pair) and B/BS (boundary), and Q/QH/QS is the ring ladder. Second revision, same session,
# after the first 8-arm pass: every 2-GPU arm ran P2P at default (results/lab-05/nccl/{A,B}-run-*.log,
# committed d669bb4), so A/AP and B/BS were duplicate transports. AL and BL (NCCL_P2P_LEVEL=LOC:
# "Never use P2P", NCCL docs 2.32.3, retrieved 2026-10-04) supply the host-staged side; AP and BS stay
# as same-transport replicate controls (the noise floor). Ten arms. Run under the course
# protocol, INTERLEAVED, NCCL_DEBUG=INFO so each run's transport is in its own log, host-state
# snapshot beside every run, EDR-001 floor of 10. ALL ARMS UNBOUND (declared in the guide Setup).
# Learner-run instrument. Evidence: results/lab-05/nccl/<arm>-{warmup,run}-NN.log (raw, authoritative)
#                                   results/lab-05/host/<arm>-{warmup,run}-NN.txt
#                                   results/lab-05/nccl/telemetry.csv
#                                   results/lab-05/lab-05-nccl-runs.csv
set -euo pipefail
export CUDA_DEVICE_ORDER=PCI_BUS_ID
RUNS="${RUNS:-10}"; WARMUP="${WARMUP:-1}"
[ "${RUNS}" -ge 10 ] || { echo "FAIL: RUNS=${RUNS} below the course floor of 10 (EDR-001)" >&2; exit 1; }
NCCL_ARGS="${NCCL_ARGS:--b 8 -e 256M -f 2}"
BIN="${NCCL_TESTS_DIR:?FAIL: NCCL_TESTS_DIR unset — source ~/gputopo-tools/env.sh}/build/all_reduce_perf"
[ -x "${BIN}" ] || { echo "FAIL: ${BIN} missing" >&2; exit 1; }

# arm|GPU mask|ngpus|NCCL_P2P_LEVEL (empty = default) — do not edit between rounds.
ARMS=(
  "S1|0|1|"
  "A|0,1|2|"
  "AP|0,1|2|PHB"
  "AL|0,1|2|LOC"
  "B|0,2|2|"
  "BS|0,2|2|SYS"
  "BL|0,2|2|LOC"
  "Q|0,1,2,3|4|"
  "QH|0,1,2,3|4|PHB"
  "QS|0,1,2,3|4|SYS"
)

OUT_DIR="results/lab-05"; RUN_DIR="${OUT_DIR}/nccl"; HOST_DIR="${OUT_DIR}/host"; CSV="${OUT_DIR}/lab-05-nccl-runs.csv"; TELEM="${RUN_DIR}/telemetry.csv"
mkdir -p "${RUN_DIR}" "${HOST_DIR}"
{
  echo "# lab-05-nccl-runs"; echo "# captured_utc: $(date -u +%Y-%m-%dT%H:%M:%SZ)"; echo "# hostname: $(hostname)"
  echo "# command: $0 $* (tool: ${BIN} ${NCCL_ARGS} -g <ngpus>; warmup rounds=${WARMUP}; measured rounds=${RUNS}; interleaved; all arms unbound)"
  echo "# arms: ${ARMS[*]}"
  echo "# host-state note: ps pcpu is lifetime-average CPU share, not instantaneous"
  echo "arm,\"mask\",ngpus,p2p_level,phase,run,start_utc,status,duration_s,pre_max_temp_c,load1,procs_running,cpu_mhz_min,cpu_mhz_max"
} > "${CSV}"
{ echo "# lab-05 nccl telemetry (1 sample/s, whole step)"; echo "# captured_utc: $(date -u +%Y-%m-%dT%H:%M:%SZ)"; echo "# hostname: $(hostname)"; } > "${TELEM}"
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
  local ARM MASK NG LVL LOG PRE START T0 STATUS=ok DUR HS
  IFS='|' read -r ARM MASK NG LVL <<< "$1"
  LOG="${RUN_DIR}/${ARM}-$2-$3.log"
  HS=$(host_state "${HOST_DIR}/${ARM}-$2-$3.txt")
  PRE=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits | sort -n | tail -1)
  START=$(date -u +%Y-%m-%dT%H:%M:%SZ); T0=$(date +%s)
  { echo "# lab-05 nccl arm ${ARM} $2 $3"; echo "# start_utc: ${START}"; echo "# hostname: $(hostname)"
    echo "# mask: CUDA_VISIBLE_DEVICES=${MASK}  ngpus: ${NG}  nccl_p2p_level: ${LVL:-<default>}  host: unbound"
    echo "# command: ${BIN} ${NCCL_ARGS} -g ${NG}"; echo "# cuda_device_order: ${CUDA_DEVICE_ORDER}"
    echo "# host_state: load1,procs_running,cpu_mhz_min,cpu_mhz_max = ${HS}"; echo; } > "${LOG}"
  echo "--- arm ${ARM} $2 $3: CUDA_VISIBLE_DEVICES=${MASK} -g ${NG} NCCL_P2P_LEVEL=${LVL:-<default>} load1=${HS%%,*} ---"
  if [ -n "${LVL}" ]; then
    CUDA_VISIBLE_DEVICES="${MASK}" NCCL_DEBUG=INFO NCCL_P2P_LEVEL="${LVL}" "${BIN}" ${NCCL_ARGS} -g "${NG}" 2>&1 | tee -a "${LOG}" | (grep -E ' via |^#  +size|^ +[0-9]+ +[0-9]+ +float|Avg bus bandwidth' || true) || STATUS=failed
  else
    CUDA_VISIBLE_DEVICES="${MASK}" NCCL_DEBUG=INFO "${BIN}" ${NCCL_ARGS} -g "${NG}" 2>&1 | tee -a "${LOG}" | (grep -E ' via |^#  +size|^ +[0-9]+ +[0-9]+ +float|Avg bus bandwidth' || true) || STATUS=failed
  fi
  DUR=$(( $(date +%s) - T0 ))
  echo "${ARM},\"${MASK}\",${NG},${LVL:-default},$2,$3,${START},${STATUS},${DUR},${PRE},${HS}" >> "${CSV}"
  echo "--- round $2 $3 arm ${ARM}: ${STATUS} (${DUR}s, pre-temp ${PRE}C) ---"
  if [ "$2" = run ]; then
    if [ "${STATUS}" = ok ]; then OKC[${ARM}]=$(( ${OKC[${ARM}]:-0} + 1 )); else FAILC[${ARM}]=$(( ${FAILC[${ARM}]:-0} + 1 )); fi
  fi
  return 0
}

echo "=== lab-05 nccl protocol: 10 arms interleaved (S1 A AP AL B BS BL Q QH QS), ${WARMUP} warm-up + ${RUNS} measured rounds; EDR-001 floor 10; all arms unbound ==="
r=1; while [ "${r}" -le "${WARMUP}" ]; do for spec in "${ARMS[@]}"; do do_run "${spec}" warmup "$(printf '%02d' "${r}")"; done; r=$((r+1)); done
r=1; while [ "${r}" -le "${RUNS}" ]; do for spec in "${ARMS[@]}"; do do_run "${spec}" run "$(printf '%02d' "${r}")"; done; r=$((r+1)); done

kill "${TELEM_PID}" 2>/dev/null || true; wait "${TELEM_PID}" 2>/dev/null || true; trap - EXIT
echo "=== lab-05 nccl protocol summary ==="
echo "evidence: ${CSV} + ${RUN_DIR}/ + ${HOST_DIR}/ (raw logs authoritative) + ${TELEM}"
RC=0
for spec in "${ARMS[@]}"; do ARM="${spec%%|*}"; echo "arm ${ARM}: measured_ok=${OKC[${ARM}]:-0} measured_failed=${FAILC[${ARM}]:-0}"; [ "${OKC[${ARM}]:-0}" -gt 0 ] || RC=1; done
[ "${RC}" -eq 0 ] || { echo "FAIL: an arm has zero successful measured runs" >&2; exit 1; }
echo "Next (on the instance): bash scripts/lab-05-transports.sh, then bash scripts/lab-05-protocol.sh"
