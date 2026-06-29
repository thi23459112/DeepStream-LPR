"""
全域設定載入器（車牌辨識 LPR 版）

主要功能：
1. 自動偵測專案根目錄 BASE_DIR，組出所有 DeepStream 設定檔的絕對路徑
2. 載入 ds_yaml/*.yaml，建立全域 SOURCE_CONFIGS 字典（每路 cam 一筆設定）
3. YAML source 智慧解析：支援相對路徑、絕對路徑、RTSP、HTTP、樣板等多種寫法
4. 載入 config_tracker_runtime.txt，決定追蹤器模式 (nvdcf / BoxMOT 系列)
5. 後處理 YAML 欄位：ROI 轉 numpy、RTSP 帳密編碼、影片 FPS 自動偵測
6. 截圖設定：依 save_screenshot 旗標決定資料夾根名（影片檔名 / source_id）並建立目錄
7. 暴露 LPR 三層 GIE 設定路徑常數給 main.py 使用：
   - INFER_CONFIG            : PGIE 車輛偵測
   - INFER_SEC_PLATE_CONFIG  : SGIE 車牌偵測
   - INFER_SEC_NUM_CONFIG    : SGIE 字元偵測
"""

import os
import sys
import glob
import cv2
import yaml
import urllib.parse
import configparser
import numpy as np


# ==========================================
# 1. 系統配置區 (System Configuration)
# ==========================================

# 自動偵測專案根目錄 (本檔位於 <project_root>/logic/config.py)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# YAML_DIR = f"{BASE_DIR}/ds_yaml"
YAML_DIR = os.environ.get("DS_YAML_DIR", f"{BASE_DIR}/ds_yaml")

# --- DeepStream 主要設定檔路徑 ---
INFER_CONFIG       = f"{BASE_DIR}/config_infer_primary_yolo11.txt"
TRACKER_CONFIG     = f"{BASE_DIR}/config_tracker_NvDCF_accuracy.yml"
PREPROCESS_CONFIG  = f"{BASE_DIR}/config_preprocess.txt"
ANALYTICS_CONFIG   = f"{BASE_DIR}/config_nvdsanalytics.txt"

# --- LPR 特有：兩層 SGIE 設定路徑 ---
INFER_SEC_PLATE_CONFIG = f"{BASE_DIR}/config_infer_secondary_plate.txt"
INFER_SEC_NUM_CONFIG   = f"{BASE_DIR}/config_infer_secondary_num.txt"

# --- 追蹤器執行期設定 (由 LPR_txt.py 產生) ---
TRACKER_RUNTIME_CONFIG = f"{BASE_DIR}/config_tracker_runtime.txt"

# --- 截圖根目錄 ---
SCREENSHOT_ROOT = f"{BASE_DIR}/screenshot"

# --- 支援的 URI scheme ---
_URI_SCHEMES = ("file://", "rtsp://", "rtsps://", "http://", "https://")

# --- Local ID 循環上限 ---
# 每路 cam 各自的 local_id 累積到此值後歸 1 重新計算（OSD 與 DB TrackID 同步循環）
# 撞號的紀錄靠 CreateTime 區分；查 DB 時記得帶時間範圍
LOCAL_ID_MAX = 999999


# ==========================================
# 2. YAML source 智慧解析 (Source URI Resolver)
# ==========================================

def _resolve_source_uri(raw):
    """
    把 YAML 的 source 欄位轉成 GStreamer 可用的合法 URI

    支援寫法：
        "videos/test3.mp4"               相對路徑 → 自動補 BASE_DIR 與 file:// 前綴
        "/abs/path/to/file.mp4"          絕對路徑 → 自動補 file:// 前綴
        "~/Videos/x.mp4"                 家目錄展開
        "${BASE_DIR}/videos/x.mp4"       樣板展開
        "file:///already/uri/x.mp4"      原樣使用
        "rtsp://user:pass@ip:port/path"  原樣使用
        "http://..." / "https://..."     原樣使用

    參數：
        raw (str): YAML 內原始字串

    返回：
        str: 解析後的 URI；空字串代表 raw 不合法
    """
    if not isinstance(raw, str) or not raw:
        return ""

    s = raw.strip()

    # 樣板展開
    if "${BASE_DIR}" in s:
        s = s.replace("${BASE_DIR}", BASE_DIR)

    # 已是合法 URI → 原樣回傳
    if s.startswith(_URI_SCHEMES):
        return s

    # 視為檔案路徑：展開家目錄、補絕對路徑
    if s.startswith("~"):
        s = os.path.expanduser(s)
    if not os.path.isabs(s):
        s = os.path.normpath(os.path.join(BASE_DIR, s))
    else:
        s = os.path.normpath(s)

    # 路徑容易拼錯，啟動時直接檢查存在
    if not os.path.exists(s):
        print(f"[WARNING] source 對應的檔案不存在：{s}")
        print(f"[WARNING]   原始 YAML 寫法：'{raw}'")
        print(f"[WARNING]   pipeline 啟動時很可能失敗，請確認路徑與檔名")

    return f"file://{s}"


def _parse_start_time(start_time_str):
    """
    將 YAML 字串解析成 datetime 物件

    參數：
        start_time_str (str): "YYYY-MM-DD HH:MM:SS" 格式字串

    返回：
        datetime | None: 解析成功回傳 datetime，失敗或空值回傳 None
    """
    if not start_time_str:
        return None
    try:
        from datetime import datetime
        return datetime.strptime(str(start_time_str), "%Y-%m-%d %H:%M:%S")
    except Exception as e:
        print(f"[WARNING] start_time 格式錯誤 ({start_time_str})，將忽略此欄位: {e}")
        return None


def _resolve_screenshot_dirname(source_uri, cam_name):
    """
    決定截圖資料夾的根名稱

    規則：
        檔案來源 → 用影片檔名去副檔名（如 忠C-20260512_145230.mp4 → 忠C-20260512_145230）
        RTSP/串流 → 用 source_id（如 camC）

    參數：
        source_uri (str): 已解析的 source URI
        cam_name (str): source_id

    返回：
        str: 資料夾根名稱
    """
    if source_uri.startswith("file://"):
        file_path = source_uri.replace("file://", "")
        base = os.path.basename(file_path)
        name, _ = os.path.splitext(base)
        return name if name else cam_name
    # RTSP / HTTP 串流用 source_id
    return cam_name


# ==========================================
# 3. YAML 載入與後處理 (YAML Loader)
# ==========================================

def load_dynamic_configs(yaml_dir):
    """
    讀取 ds_yaml/*.yaml，逐檔解析並補上衍生欄位

    處理流程：
    1. 掃描所有 .yaml 檔，依檔名排序（決定 pad_index 0..N-1）
    2. 每個 YAML 補上衍生欄位：
       - device_code              : 寫進 DB DeviceCode 欄位
       - cv_regions               : 多 ROI 多邊形轉成 {roi_name: numpy int32 array}
       - track_logic              : movement_threshold / min_roi_hits（給 probes.py 用）
       - save_output_db           : DB 寫入旗標（false 時純跑統計）
       - db_path / excel_path     : DB 檔絕對路徑
       - video_path               : 輸出影片絕對路徑
       - save_screenshot          : 截圖旗標
       - screenshot_dir_class     : 車種截圖資料夾絕對路徑
       - screenshot_dir_lpr       : 車牌截圖資料夾絕對路徑
       - source / stream_fps      : 解析後的 URI 與真實 FPS
       - is_file_source           : 來源是否為本地檔案
       - start_time_dt            : 影片首幀對應的真實時刻
       - rtsp_push                : RTSP 推流標準化設定 dict

    參數：
        yaml_dir (str): YAML 資料夾路徑

    返回：
        dict: {pad_index: cfg_dict, ...}
    """
    files = sorted(glob.glob(f"{yaml_dir}/*.yaml"))
    if not files:
        print(f"[ERROR] 找不到任何 YAML 檔於 {yaml_dir}")
        sys.exit(1)

    configs = {}
    for pad_index, f in enumerate(files):
        with open(f, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)

        cam_name = data.get("source_id", f"cam_{pad_index}")

        # 步驟 1: 裝置代碼
        device_cfg = data.get("device", {}) or {}
        data["device_code"] = str(device_cfg.get("code", "UNKNOWN"))

        # 步驟 2: 多 ROI 轉 numpy（給 probe 的 pointPolygonTest 用）
        regions_raw = data.get("geometry", {}).get("regions", {}) or {}
        cv_regions = {}
        for roi_name, pts in regions_raw.items():
            if pts and len(pts) >= 3:
                cv_regions[str(roi_name)] = np.array(pts, np.int32)
            else:
                print(f"[WARNING] {cam_name} 的 ROI '{roi_name}' 點數不足（{len(pts) if pts else 0}），略過")
        data["cv_regions"] = cv_regions
        if not cv_regions:
            print(f"[WARNING] {cam_name} 沒有任何有效的 ROI，將不會產生任何 DB 紀錄")

        # 步驟 3: track_logic 標準化（給 probes.py 用）
        # movement_threshold = Y 軸位移像素門檻（小於此值方向判定為 NA，不寫 DB）
        # min_roi_hits       = ROI 命中最少幀數（小於此值視為短暫飄過，不寫 DB）
        tl_cfg = data.get("track_logic", {}) or {}
        data["track_logic"] = {
            "movement_threshold": int(tl_cfg.get("movement_threshold", 30)),
            "min_roi_hits":       int(tl_cfg.get("min_roi_hits", 2)),
        }

        # 步驟 4: 輸出設定 - DB
        # 新欄位 output_db_dir / save_output_db；舊欄位 output_excel_dir 仍向下相容
        output_cfg = data.get("output", {}) or {}

        db_dir = output_cfg.get("output_db_dir")
        if not db_dir:
            db_dir = output_cfg.get("output_excel_dir", "output_db")
            if "output_excel_dir" in output_cfg:
                print(f"[INFO] {cam_name} 使用舊欄位 output_excel_dir，建議改為 output_db_dir")
        if not os.path.isabs(db_dir):
            db_dir = os.path.join(BASE_DIR, db_dir)

        save_db = bool(output_cfg.get("save_output_db", True))
        if save_db:
            os.makedirs(db_dir, exist_ok=True)

        data["save_output_db"] = save_db
        data["db_path"]    = os.path.join(db_dir, f"{cam_name}.db")
        # excel_path 鍵保留指向 .db，state_db.py 仍用此鍵推算路徑，向下相容
        data["excel_path"] = data["db_path"]

        # 步驟 5: 輸出設定 - 影片
        video_dir = output_cfg.get("output_video_dir", "output_video")
        if not os.path.isabs(video_dir):
            video_dir = os.path.join(BASE_DIR, video_dir)
        if output_cfg.get("save_output_video", False):
            os.makedirs(video_dir, exist_ok=True)
        data["video_path"] = os.path.join(video_dir, f"{cam_name}_output.mp4")

        # 步驟 6: 來源 URI 解析
        source_uri = _resolve_source_uri(data.get("source", ""))
        yaml_fps = data.get("stream_fps", 30.0)

        original = data.get("source", "")
        if original != source_uri:
            print(f"[INFO] {cam_name} source 解析: '{original}' → '{source_uri}'")

        # RTSP 帳密特殊字元安全編碼
        if source_uri.startswith("rtsp://"):
            try:
                parsed = urllib.parse.urlparse(source_uri)
                if parsed.username and parsed.password:
                    safe_username = urllib.parse.quote(urllib.parse.unquote(parsed.username))
                    safe_password = urllib.parse.quote(urllib.parse.unquote(parsed.password))
                    safe_netloc = f"{safe_username}:{safe_password}@{parsed.hostname}"
                    if parsed.port:
                        safe_netloc += f":{parsed.port}"
                    source_uri = urllib.parse.urlunparse((parsed.scheme, safe_netloc, parsed.path, parsed.params, parsed.query, parsed.fragment))
                    print(f"[INFO] RTSP URI 安全格式: {source_uri}")
            except Exception as e:
                print(f"[WARNING] 解析 RTSP URI 失敗: {e}")

        # 檔案模式：用 cv2 抓真實 FPS 覆寫 YAML 值
        if source_uri.startswith("file://"):
            file_path = source_uri.replace("file://", "")
            cap = cv2.VideoCapture(file_path)
            try:
                if cap.isOpened():
                    real_fps = cap.get(cv2.CAP_PROP_FPS)
                    if real_fps > 0:
                        yaml_fps = real_fps
            finally:
                cap.release()

        data["stream_fps"] = yaml_fps
        data["source"] = source_uri
        data["is_file_source"] = source_uri.startswith("file://")

        # 步驟 7: start_time 解析（僅檔案模式有效）
        data["start_time_dt"] = _parse_start_time(
            data.get("start_time") if data["is_file_source"] else None
        )

        # 步驟 8: 截圖設定
        # save_screenshot=true 時，依來源型態決定資料夾根名並建立 class/ 與 LPR/ 子目錄
        save_ss = bool(output_cfg.get("save_screenshot", False))
        data["save_screenshot"] = save_ss

        ss_root_name = _resolve_screenshot_dirname(source_uri, cam_name)
        ss_base = os.path.join(SCREENSHOT_ROOT, ss_root_name)
        data["screenshot_dir_class"] = os.path.join(ss_base, "class")
        data["screenshot_dir_lpr"]   = os.path.join(ss_base, "LPR")

        if save_ss:
            os.makedirs(data["screenshot_dir_class"], exist_ok=True)
            os.makedirs(data["screenshot_dir_lpr"], exist_ok=True)
            print(f"[INFO] {cam_name} 截圖已啟用，根目錄：{ss_base}")

        # 步驟 9: RTSP 推流設定標準化
        rtsp_cfg = data.get("rtsp_push", {}) or {}
        data["rtsp_push"] = {
            "enable":     bool(rtsp_cfg.get("enable", False)),
            "port":       int(rtsp_cfg.get("port", 8554)),
            "mount_path": str(rtsp_cfg.get("mount_path", cam_name)),
            "bitrate":    int(rtsp_cfg.get("bitrate", 4000000)),
            "encoder":    str(rtsp_cfg.get("encoder", "h264")).lower(),
        }

        configs[pad_index] = data

    return configs


# ==========================================
# 4. 追蹤器執行期設定 (Tracker Runtime Config)
# ==========================================

def load_tracker_runtime():
    """
    讀 config_tracker_runtime.txt 取得當前追蹤器模式

    檔案由 LPR_txt.py 產生，內容範例：
        [tracker]
        mode=bytetrack
        config=/abs/path/to/boxmot/configs/trackers/bytetrack.yaml

    返回：
        tuple: (mode, boxmot_config_path)
            mode (str): "nvdcf" 或 BoxMOT 追蹤器名稱
            boxmot_config_path (str | None): BoxMOT 模式時為設定檔絕對路徑，nvdcf 時為 None
    """
    if not os.path.exists(TRACKER_RUNTIME_CONFIG):
        print(f"[INFO] 找不到 {TRACKER_RUNTIME_CONFIG}，預設使用 nvdcf 模式")
        return "nvdcf", None

    parser = configparser.ConfigParser()
    try:
        parser.read(TRACKER_RUNTIME_CONFIG, encoding="utf-8")
    except Exception as e:
        print(f"[WARNING] 解析 {TRACKER_RUNTIME_CONFIG} 失敗：{e}，退回 nvdcf")
        return "nvdcf", None

    if not parser.has_section("tracker"):
        print(f"[WARNING] {TRACKER_RUNTIME_CONFIG} 缺 [tracker] 區塊，退回 nvdcf")
        return "nvdcf", None

    mode = parser.get("tracker", "mode", fallback="nvdcf").lower().strip()
    boxmot_cfg = parser.get("tracker", "config", fallback=None)

    if boxmot_cfg is not None:
        boxmot_cfg = boxmot_cfg.strip() or None

    return mode, boxmot_cfg


# ==========================================
# 5. 模組初始化 (Module Initialization)
# ==========================================

# 模組載入時自動執行：印路徑、載入 YAML、載入追蹤器模式
print(f"[INFO] [config.py] BASE_DIR 自動偵測 = {BASE_DIR}")

SOURCE_CONFIGS = load_dynamic_configs(YAML_DIR)

TRACKER_MODE, BOXMOT_TRACKER_CONFIG = load_tracker_runtime()
print(f"[INFO] 追蹤器模式：{TRACKER_MODE}")
if TRACKER_MODE != "nvdcf":
    print(f"[INFO] BoxMOT 追蹤器設定來源：{BOXMOT_TRACKER_CONFIG}")
