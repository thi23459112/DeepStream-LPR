"""
SQLite 事件紀錄與軌跡狀態管理（車牌辨識 LPR 版）

主要功能：
1. 多 cam 獨立 SQLite DB：每路 cam 一個 .db 檔，避免寫入衝突
2. 軌跡狀態維護：每台車一份 track_history，記錄位置、方向、ROI 命中、車種投票、車牌投票
3. 消失時結算：物件連續 N 幀未出現 → 對每個達門檻的 ROI 各 emit 一筆 DB 紀錄
4. 方向過濾：只有 IN / OUT 才寫 DB；NA (抖動誤判) 整筆丟掉
5. 車牌投票機制：每幀 assemble_plate_probe 累積一票，結算時取最多票字串
6. 截圖寫檔：結算時把得票最高幀的 JPEG bytes（由 probes.py 用 Object Encoder 編好）寫檔，
   並把相對路徑寫進 DB 的 ClassImg / PlateImg 欄位
7. 批次 flush 機制：累積在記憶體 pending_records，定期批次寫入 DB
8. save_output_db=false 旗標：純跑統計、不開連線、零 DB IO（截圖獨立判斷）
9. local_id 循環機制：每路 cam 累積到 LOCAL_ID_MAX 後歸 1 重新計算
"""

import os
import re
import time
import sqlite3
import threading
from collections import Counter
from datetime import timedelta

from logic.config import SOURCE_CONFIGS, LOCAL_ID_MAX, BASE_DIR
from logic.color import CLASS_MAP


# ==========================================
# 1. 系統配置區 (System Configuration)
# ==========================================

# --- 全域狀態字典 (供 probes.py 直接 import 使用) ---
track_history    = {}    # (pad_index, obj_id) → 軌跡狀態 dict
pending_records  = {}    # pad_index → 待寫入 DB 的 tuple list
last_flush_times = {}    # pad_index → 上次 flush 的時間戳
fps_streams      = {}    # pad_index → {"current_fps", "timestamps"}
local_id_maps    = {}    # pad_index → {global_id: local_id}
next_local_ids   = {}    # pad_index → 下一個可分配的 local_id（達 LOCAL_ID_MAX 後歸 1）

# --- SQLite 連線管理 ---
_db_conns = {}                    # pad_index → sqlite3.Connection
_db_lock  = threading.Lock()      # 寫入批次的執行緒鎖

# --- DB Schema ---
# Plate 欄位是 LPR 獨有，從 plate_votes.most_common(1) 取值；沒辨識到時填 "N/A"
# ClassImg / PlateImg 存截圖的相對路徑（相對專案根目錄），供外部 db_to_excel.py 取用
_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS events (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    DeviceCode  TEXT    NOT NULL,
    CameraCode  TEXT    NOT NULL,
    TrackID     INTEGER NOT NULL,
    Plate       TEXT,
    Class       TEXT,
    ROI         TEXT    NOT NULL,
    Direction   TEXT    NOT NULL,
    HitCount    INTEGER NOT NULL,
    VideoTime   TEXT,
    CreateTime  TEXT    NOT NULL,
    ClassImg    TEXT,
    PlateImg    TEXT
);

CREATE INDEX IF NOT EXISTS idx_camera_time
    ON events (CameraCode, CreateTime);

CREATE INDEX IF NOT EXISTS idx_roi
    ON events (ROI);

CREATE INDEX IF NOT EXISTS idx_direction
    ON events (Direction);

CREATE INDEX IF NOT EXISTS idx_plate
    ON events (Plate);
"""


# ==========================================
# 2. DB 連線輔助 (Connection Helper)
# ==========================================

def _get_db_path(cfg, pad_index):
    """
    從 cfg["excel_path"] 推算 DB 路徑（向下相容鍵名）

    參數：
        cfg (dict): 該路 cam 的 YAML 設定
        pad_index (int): 哪一路 cam

    返回：
        str: .db 檔絕對路徑
    """
    excel_path = cfg.get("excel_path", f"output_db/cam_{pad_index}.db")
    base, _ = os.path.splitext(excel_path)
    return f"{base}.db"


def _open_db(pad_index, cfg):
    """
    為指定 cam 開啟 SQLite 連線並建立 schema

    使用 WAL 模式提升併發寫入效能，synchronous=NORMAL 兼顧速度與資料安全

    參數：
        pad_index (int): 哪一路 cam
        cfg (dict): 該路 cam 的 YAML 設定

    返回：
        sqlite3.Connection: 已建立 schema 的 DB 連線
    """
    db_path = _get_db_path(cfg, pad_index)
    db_dir = os.path.dirname(db_path)
    if db_dir:
        os.makedirs(db_dir, exist_ok=True)

    conn = sqlite3.connect(db_path, check_same_thread=False, isolation_level=None)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.executescript(_SCHEMA_SQL)
    print(f"[INFO] SQLite DB 開啟: {db_path}")
    return conn


def _format_video_time(vsec):
    """
    秒數轉成 HH:MM:SS 字串

    參數：
        vsec (float): 秒數

    返回：
        str: "HH:MM:SS" 格式；負值或 None 回傳 "00:00:00"
    """
    if vsec is None or vsec < 0:
        return "00:00:00"
    return time.strftime("%H:%M:%S", time.gmtime(int(vsec)))


# ==========================================
# 3. 截圖檔名與寫檔輔助 (Screenshot File Helper)
# ==========================================

def _sanitize_for_filename(text):
    """
    把字串清成適合當檔名的格式

    車牌字串可能含特殊字元（雖然少見），這裡把非英數、減號、底線以外的字元換成底線，
    避免寫檔失敗

    參數：
        text (str): 原始字串（如車牌號碼）

    返回：
        str: 清理後字串
    """
    if not text:
        return "NA"
    return re.sub(r"[^0-9A-Za-z\-_]", "_", str(text))


def _save_jpg_bytes(jpg_bytes, dir_path, filename):
    """
    把 JPEG bytes 寫入指定路徑，回傳相對於專案根目錄的路徑

    參數：
        jpg_bytes (bytes): JPEG 影像位元組（probes.py 用 Object Encoder 編好的）
        dir_path (str): 目標資料夾（絕對路徑）
        filename (str): 檔名（含 .jpg）

    返回：
        str | None: 成功回傳「相對專案根目錄」的路徑（如 screenshot/test3/class/car/37_xxx.jpg），
                    失敗回傳 None
    """
    if not jpg_bytes:
        return None
    try:
        os.makedirs(dir_path, exist_ok=True)
        full_path = os.path.join(dir_path, filename)
        with open(full_path, "wb") as f:
            f.write(jpg_bytes)
        # 轉相對路徑（相對專案根目錄），方便搬機器後由 db_to_excel.py 補 BASE_DIR
        rel_path = os.path.relpath(full_path, BASE_DIR)
        return rel_path
    except Exception as e:
        print(f"[WARNING] 截圖寫檔失敗 ({filename}): {e}")
        return None


# ==========================================
# 4. 啟動初始化 (Startup Initialization)
# ==========================================

def initialize_state_managers():
    """
    為每一路 cam 初始化狀態字典與 DB 連線

    處理流程：
    1. 為每路 cam 初始化所有狀態字典（pending / fps / local_id 等）
    2. 依 cfg["save_output_db"] 決定是否開 DB 連線
       - true（預設）：呼叫 _open_db 建立連線
       - false       ：跳過連線開啟，emit/flush 走 no-op 分支

    註：本函式應在 main.py 啟動時呼叫一次
    """
    for pad_index, cfg in SOURCE_CONFIGS.items():
        # 步驟 1: 狀態字典初始化
        pending_records[pad_index] = []
        last_flush_times[pad_index] = time.time()
        fps_streams[pad_index] = {"current_fps": 0.0}
        local_id_maps[pad_index] = {}
        next_local_ids[pad_index] = 1

        # 步驟 2: 依旗標決定是否開 DB
        if cfg.get("save_output_db", True):
            _db_conns[pad_index] = _open_db(pad_index, cfg)
        else:
            cam_name = cfg.get("source_id", f"cam_{pad_index}")
            print(f"[INFO] {cam_name} save_output_db=false，停用 DB 寫入（純跑統計）")


# ==========================================
# 5. ID 管理 (ID Mapping)
# ==========================================

def get_local_id(pad_index, global_id):
    """
    將追蹤器給的 global_id 映射成該路 cam 內的短 local_id

    循環機制：local_id 從 1 累加到 LOCAL_ID_MAX，下一個 global_id 拿到的會是 1
              （每路 cam 各自獨立循環，互不干擾）
              撞號的紀錄靠 DB CreateTime 區分，查詢時記得帶時間範圍

    參數：
        pad_index (int): 哪一路 cam
        global_id (int): 追蹤器給的物件 ID

    返回：
        int: 該路內遞增的短 ID（範圍 1 ~ LOCAL_ID_MAX，達上限後歸 1）
    """
    if global_id not in local_id_maps[pad_index]:
        local_id_maps[pad_index][global_id] = next_local_ids[pad_index]

        # 達上限歸 1，否則 +1
        if next_local_ids[pad_index] >= LOCAL_ID_MAX:
            cam_name = SOURCE_CONFIGS.get(pad_index, {}).get("source_id", f"cam_{pad_index}")
            print(f"[INFO] {cam_name} local_id 達上限 {LOCAL_ID_MAX}，下一個歸 1 重新計算")
            next_local_ids[pad_index] = 1
        else:
            next_local_ids[pad_index] += 1

    return local_id_maps[pad_index][global_id]


# ==========================================
# 6. 軌跡結算 (Trajectory Finalization)
# ==========================================

def _finalize_one(m_key, state, force=False):
    """
    結算單一車輛軌跡，把符合條件的 ROI 紀錄各 emit 一筆到 pending_records，
    並（若啟用截圖）把得票最高幀的 JPEG bytes 寫檔

    結算條件：
    1. 方向必須是 IN 或 OUT（NA 表示位移不足，視為抖動誤判 → 整筆丟掉）
    2. 對該軌跡的每個 ROI：命中數 >= min_roi_hits → emit 一筆紀錄
       （多 ROI 機制：一台車經過 N 個 ROI，會 emit N 筆紀錄）

    每筆紀錄欄位：
        DeviceCode / CameraCode / TrackID / Plate / Class / ROI / Direction /
        HitCount / VideoTime / CreateTime / ClassImg / PlateImg

    截圖（多 ROI 時各只存一張，與觸發幾個 ROI 無關）：
        車種圖 → screenshot/<根名>/class/<車種>/ID_yyyymmdd_hhmmss.jpg
        車牌圖 → screenshot/<根名>/LPR/ID_車牌_yyyymmdd_hhmmss.jpg
    車種圖來源優先序：
        1. best_class_jpg（與車牌同幀截的，車身最完整）
        2. fallback_class_jpg（整段都沒辨識到車牌時，用車種票數最高幀）
    同一台車的多筆 ROI 紀錄共用同一組 ClassImg / PlateImg 相對路徑

    參數：
        m_key (tuple): (pad_index, obj_id)
        state (dict): 該軌跡的狀態字典
        force (bool): 是否為強制結算（程式結束時用，影響 log 標籤）
    """
    pad_index, obj_id = m_key
    cfg = SOURCE_CONFIGS.get(pad_index, {})
    cam_name = cfg.get("source_id", f"cam_{pad_index}")
    min_hits = cfg.get("track_logic", {}).get("min_roi_hits", 2)

    # 步驟 1: 方向過濾（NA 整筆丟掉）
    if state.get("direction", "NA") == "NA":
        return

    # 步驟 2: 找出所有達門檻的 ROI（沒有就整筆丟掉）
    triggered_rois = {
        roi_name: hits
        for roi_name, hits in state.get("roi_hits", {}).items()
        if hits >= min_hits
    }
    if not triggered_rois:
        return

    # 步驟 3: 共用欄位計算
    local_id = get_local_id(pad_index, obj_id)
    device_code = cfg.get("device_code", "UNKNOWN")
    direction = state["direction"]

    # 車種投票：取票數最多的類別
    if state.get("class_votes"):
        best_class_id = state["class_votes"].most_common(1)[0][0]
        cls_name = CLASS_MAP.get(best_class_id, f"Class_{best_class_id}")
    else:
        cls_name = "Unknown"

    # 車牌投票：取票數最多的車牌字串（LPR 獨有）
    plate_votes = state.get("plate_votes", Counter())
    if plate_votes:
        best_plate = plate_votes.most_common(1)[0][0]
    else:
        best_plate = "N/A"

    # VideoTime：軌跡最後出現幀號 → 影片內秒數
    vsec = state["last_frame_num"] / cfg.get("stream_fps", 30.0)
    time_axis = _format_video_time(vsec)

    # CreateTime：檔案模式 = start_time + vsec；即時串流 = 系統當下時間
    start_dt = cfg.get("start_time_dt")
    if start_dt is not None:
        event_dt = start_dt + timedelta(seconds=vsec)
        create_time_str = event_dt.strftime("%Y-%m-%d %H:%M:%S")
    else:
        create_time_str = time.strftime("%Y-%m-%d %H:%M:%S")

    # 步驟 4: 截圖寫檔（多 ROI 時各只存一張，與 ROI 數量無關）
    # 檔名時間戳用 CreateTime（與 DB 一致），格式化成 yyyymmdd_hhmmss
    # class_img_rel / plate_img_rel 為相對路徑，會寫進 DB；沒截到圖則為 None
    class_img_rel = None
    plate_img_rel = None

    if cfg.get("save_screenshot", False):
        ts_for_name = create_time_str.replace("-", "").replace(":", "").replace(" ", "_")

        # 車種截圖：優先用車牌同幀的，沒有才用 fallback（車種票數最高幀）
        class_jpg = state.get("best_class_jpg") or state.get("fallback_class_jpg")
        if class_jpg:
            class_sub_dir = os.path.join(cfg.get("screenshot_dir_class", ""), cls_name)
            fname = f"{local_id}_{ts_for_name}.jpg"
            class_img_rel = _save_jpg_bytes(class_jpg, class_sub_dir, fname)

        # 車牌截圖：ID_車牌_yyyymmdd_hhmmss.jpg，存到 LPR/
        plate_jpg = state.get("best_plate_jpg")
        if plate_jpg and best_plate != "N/A":
            plate_safe = _sanitize_for_filename(best_plate)
            fname = f"{local_id}_{plate_safe}_{ts_for_name}.jpg"
            plate_img_rel = _save_jpg_bytes(plate_jpg, cfg.get("screenshot_dir_lpr", ""), fname)

    # 步驟 5: 對每個達門檻的 ROI 各 emit 一筆（同車多 ROI 共用同一組截圖路徑）
    tag = "[結算-強制]" if force else " "

    for roi_name, hit_count in triggered_rois.items():
        # save_output_db=false → 只印 log，不累積、不寫 DB（截圖已於步驟 4 獨立處理）
        if not cfg.get("save_output_db", True):
            print(f"{tag}[{cam_name}] ID={local_id}, 車號={best_plate}, 車種={cls_name}, "
                  f"ROI={roi_name}, 方向={direction}, 次數={hit_count}, "
                  f"時間軸={time_axis}, 時間點={create_time_str}  (DB 已停用)")
            continue

        pending_records[pad_index].append((
            device_code,
            cam_name,
            local_id,
            best_plate,
            cls_name,
            roi_name,
            direction,
            hit_count,
            time_axis,
            create_time_str,
            class_img_rel,
            plate_img_rel,
        ))

        print(f"{tag}[{cam_name}] ID={local_id}, 車號={best_plate}, 車種={cls_name}, "
              f"ROI={roi_name}, 方向={direction}, 次數={hit_count}, "
              f"時間軸={time_axis}, 時間點={create_time_str}")


# ==========================================
# 7. DB 寫入 (DB Flush)
# ==========================================

def flush_pending_to_db(pad_index):
    """
    把 pending_records[pad_index] 批次寫入 SQLite

    使用單一 transaction (BEGIN/COMMIT) 提升寫入效能；
    失敗時 ROLLBACK，pending 保留在記憶體等下次重試

    參數：
        pad_index (int): 哪一路 cam

    返回：
        int: 實際寫入筆數；無 pending 或無連線回傳 0
    """
    records = pending_records.get(pad_index, [])
    if not records:
        return 0

    conn = _db_conns.get(pad_index)
    if conn is None:
        # save_output_db=false 模式下不會走到這（_finalize_one 早就跳過 append）
        # 萬一有殘留 records 也清掉，避免 memory leak
        records.clear()
        return 0

    with _db_lock:
        try:
            conn.execute("BEGIN")
            conn.executemany(
                "INSERT INTO events "
                "(DeviceCode, CameraCode, TrackID, Plate, Class, ROI, Direction, HitCount, VideoTime, CreateTime, ClassImg, PlateImg) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                records
            )
            conn.execute("COMMIT")
            n = len(records)
            records.clear()
            return n
        except sqlite3.Error as e:
            try:
                conn.execute("ROLLBACK")
            except Exception:
                pass
            print(f"[ERROR] SQLite 寫入失敗 (pad_index={pad_index}): {e}")
            return 0


# ==========================================
# 8. 結束清理 (Shutdown Cleanup)
# ==========================================

def force_finalize_all():
    """
    程式結束前呼叫：強制結算所有殘留軌跡、flush 剩餘 pending、關閉所有 DB 連線

    處理流程：
    1. 對所有 track_history 內殘留的軌跡呼叫 _finalize_one(force=True)
       （這些車是程式結束時還在畫面內、還沒消失到 cleanup_frames 的）
    2. 對每路 cam 強制 flush 一次，確保 pending 都進 DB
    3. 關閉所有 DB 連線（WAL checkpoint 也會跟著做）
    4. 清空 track_history 釋放記憶體
    """
    print("\n[INFO] 開始執行強制結算...")

    # 步驟 1: 殘留軌跡逐一結算
    for m_key, state in list(track_history.items()):
        _finalize_one(m_key, state, force=True)

    # 步驟 2: 強制 flush 所有 pending
    for pad_index, cfg in SOURCE_CONFIGS.items():
        n = flush_pending_to_db(pad_index)
        if n > 0:
            db_path = _get_db_path(cfg, pad_index)
            print(f"[檔案儲存] {cfg.get('source_id')}：已強制寫入 {n} 筆剩餘資料到 {db_path}")

    # 步驟 3: 關閉所有 DB 連線
    for pad_index, conn in list(_db_conns.items()):
        try:
            conn.close()
        except Exception as e:
            print(f"[WARNING] 關閉 DB 連線失敗 (pad_index={pad_index}): {e}")
    _db_conns.clear()

    # 步驟 4: 釋放記憶體
    track_history.clear()