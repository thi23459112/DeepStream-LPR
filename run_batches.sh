#!/usr/bin/env bash
# 批次執行：每 BATCH_SIZE 部影片一批，跑完一批自動接下一批。放在專案根目錄。
set -uo pipefail

# ============ Conda ============
CONDA_ENV="tracking"
CONDA_BASE="${CONDA_BASE:-$(conda info --base 2>/dev/null)}"
if [ -z "$CONDA_BASE" ] || [ ! -f "$CONDA_BASE/etc/profile.d/conda.sh" ]; then
  echo "[ERROR] 找不到 conda，請先 conda activate $CONDA_ENV 再執行，或設定 CONDA_BASE。"; exit 1
fi
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV" || { echo "[ERROR] 無法啟用 conda 環境：$CONDA_ENV"; exit 1; }
echo "[INFO] 已啟用 conda：$CONDA_ENV ($(which python))"
export PYTHONUNBUFFERED=1          # 讓 python 的 print 即時吐出（配 tee 才看得到）

# ============ 參數 ============
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
POOL_DIR="${PROJECT_DIR}/ds_yaml"
BATCH_SIZE=2
PYTHON_BIN="python"
WORK_ROOT="${PROJECT_DIR}/.batch_runs"

# 自動偵測本專案的設定產生器（traffic_count_txt.py / person_count_txt.py / LPR_txt.py）
GEN_SCRIPT="$(find "$PROJECT_DIR" -maxdepth 1 -name '*_txt.py' | head -1)"
MAIN_SCRIPT="${PROJECT_DIR}/main.py"
if [ -z "$GEN_SCRIPT" ]; then echo "[ERROR] 找不到 *_txt.py 設定產生器"; exit 1; fi
echo "[INFO] 產生器：$(basename "$GEN_SCRIPT")"

mkdir -p "$WORK_ROOT"
mapfile -t ALL_YAML < <(find "$POOL_DIR" -maxdepth 1 -name '*.yaml' | LC_ALL=C sort)
TOTAL=${#ALL_YAML[@]}
[ "$TOTAL" -eq 0 ] && { echo "[ERROR] $POOL_DIR 沒有 .yaml"; exit 1; }
TOTAL_BATCHES=$(( (TOTAL + BATCH_SIZE - 1) / BATCH_SIZE ))
echo "[INFO] 共 $TOTAL 部，每批 $BATCH_SIZE，總共 $TOTAL_BATCHES 批"

CHILD_PID=""
cleanup(){ echo; echo "[INFO] 中斷，停止批次..."; [ -n "$CHILD_PID" ] && { kill -INT "$CHILD_PID" 2>/dev/null; wait "$CHILD_PID" 2>/dev/null; }; exit 130; }
trap cleanup INT TERM

batch_no=0
for (( start=0; start<TOTAL; start+=BATCH_SIZE )); do
  batch_no=$((batch_no+1)); end=$(( start+BATCH_SIZE )); [ "$end" -gt "$TOTAL" ] && end=$TOTAL
  batch_dir="${WORK_ROOT}/batch_$(printf '%03d' "$batch_no")"
  rm -rf "$batch_dir"; mkdir -p "$batch_dir"
  echo "============================================================"
  echo "[INFO] 第 $batch_no/$TOTAL_BATCHES 批（第 $((start+1)) ~ $end 部）"
  for (( i=start; i<end; i++ )); do f="${ALL_YAML[$i]}"; ln -sf "$f" "${batch_dir}/$(basename "$f")"; echo "        - $(basename "$f")"; done
  log_file="${batch_dir}/run.log"

  echo "[INFO] 產生設定檔..."
  if ! DS_YAML_DIR="$batch_dir" "$PYTHON_BIN" "$GEN_SCRIPT" > >(tee "$log_file") 2>&1; then
    echo "[ERROR] 第 $batch_no 批產生設定檔失敗，略過。詳見 $log_file"; continue; fi

  echo "[INFO] 開始推論，畫面即時顯示，log → $log_file"
  # 用 process substitution 接 tee：main.py 仍是直接的背景子程序，
  # $! 抓到的是 main.py 本身（不是 tee），exit code 與 Ctrl+C 才能正確對到。
  DS_YAML_DIR="$batch_dir" "$PYTHON_BIN" "$MAIN_SCRIPT" </dev/null > >(tee -a "$log_file") 2>&1 &
  CHILD_PID=$!; wait "$CHILD_PID"; status=$?; CHILD_PID=""

  if [ "$status" -ne 0 ]; then echo "[WARNING] 第 $batch_no 批非 0 結束 (exit=$status)，詳見 $log_file"; else echo "[INFO] 第 $batch_no 批完成。"; fi
done
echo "============================================================"
echo "[INFO] 全部 $TOTAL 部、共 $batch_no 批 執行完畢。"
