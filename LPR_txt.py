#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
DeepStream 設定檔自動產生器（車牌辨識 LPR 三層版）

主要功能：
1. 讀取 ds_yaml/*.yaml 各路 cam 設定，自動產生 6 份 DeepStream 組態檔
2. 三層推論架構：PGIE 車輛 → SGIE 車牌 → SGIE 字元
   - PGIE   : car 模型（多車種偵測）
   - SGIE 0 : plate 模型（對車輛做車牌偵測）
   - SGIE 1 : num 模型（對車牌做字元偵測，dynamic batch）
3. 多 ROI 支援：每路 cam 可定義任意數量 ROI，nvdsanalytics 自動畫線
4. YAML source 智慧解析：相對路徑、絕對路徑、RTSP、樣板都認得
5. 追蹤器執行期設定：依 YAML tracker.type 產生 main.py 啟動時讀的旗標檔

產生的設定檔：
    deepstream_app_config.txt          主應用設定（streammux / sink / 三層 GIE 串接）
    config_preprocess.txt              前處理（裁切 ROI、縮放、tensor 轉換）
    config_infer_primary_yolo11.txt    PGIE 車輛偵測（car_fp16.engine）
    config_infer_secondary_plate.txt   SGIE 車牌偵測（plate_fp16.engine）
    config_infer_secondary_num.txt     SGIE 字元偵測（num_fp16.engine，dynamic batch）
    config_nvdsanalytics.txt           ROI 區域繪製（多 ROI）
    config_tracker_runtime.txt         追蹤器執行期旗標（nvdcf / BoxMOT）

YAML 讀進來的關鍵欄位：
    weight_imgsz              car / plate 模型輸入解析度
    num_imgsz                 num 模型輸入解析度
    weight_batch_size         car / plate engine 的 max batch
    num_weight_batch_size     num engine 的 max batch
    detect.car_conf/iou       PGIE 信心值與 NMS IoU 閾值
    detect.plate_conf/iou     SGIE plate 閾值
    detect.num_conf/iou       SGIE num 閾值
    geometry.base_w/h         各路畫面原始解析度
    geometry.crop_points      裁切 ROI (preprocess 用)
    geometry.regions          多個計數 ROI (nvdsanalytics 用)
    display.show_roi          是否畫 ROI 黃線
    display.show_crop         是否畫裁切框
    source                    影片或串流 URI
    tracker.type              nvdcf | bytetrack | ocsort | ... (全域，只讀 cfgs[0])

重要對齊規則：
    - num 模型走 dynamic batch，必須有 force-implicit-batch-dim=0
      與 infer-dims=3;NUM_H;NUM_W，否則 DS 會把 engine 當靜態 batch 處理
    - output-buffer-pool-size 在多路、多 ROI 場景下必須拉大，否則會丟 buffer
    - plate 的 operate-on-class-ids 自動依 labels_car.txt 的行數產生 (0;1;...;N-1)
"""

import os
import sys
import glob
import yaml
import math
from typing import List, Dict, Any, Tuple


# ==========================================
# 1. 系統配置區 (System Configuration)
# ==========================================

# 自動偵測專案根目錄 (本檔位於 <project_root>/LPR_txt.py)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
YAML_DIR = f"{BASE_DIR}/ds_yaml"

# --- 標籤檔（位於 BASE_DIR 下） ---
CAR_LABEL_FILE   = "labels_car.txt"      # 車輛類別清單（多車種）
PLATE_LABEL_FILE = "labels_plate.txt"    # 車牌類別清單（通常只有 1 類）
NUM_LABEL_FILE   = "labels_num.txt"      # 字元類別清單（0-9, A-Z 共 36 類）

# --- 模型預設值（YAML 沒寫時的 fallback） ---
DEFAULT_WEIGHT_IMGSZ = 640
DEFAULT_NUM_IMGSZ    = 320
DEFAULT_WEIGHT_BATCH = 2
DEFAULT_NUM_BATCH    = 16

# --- 產生的設定檔絕對路徑 ---
APP_CONFIG             = f"{BASE_DIR}/deepstream_app_config.txt"
PREPROCESS_CONFIG      = f"{BASE_DIR}/config_preprocess.txt"
INFER_PRIMARY_CONFIG   = f"{BASE_DIR}/config_infer_primary_yolo11.txt"
INFER_PLATE_CONFIG     = f"{BASE_DIR}/config_infer_secondary_plate.txt"
INFER_NUM_CONFIG       = f"{BASE_DIR}/config_infer_secondary_num.txt"
ANALYTICS_CONFIG       = f"{BASE_DIR}/config_nvdsanalytics.txt"
TRACKER_RUNTIME_CONFIG = f"{BASE_DIR}/config_tracker_runtime.txt"

# --- 已實作的 BoxMOT 追蹤器白名單 ---
# A 級：純 motion-only，1×1 dummy frame 即可
# B 級：純 motion-only 但需要正確 H×W 的 zero frame
# C 級（需 ReID / 光流）目前未實作，會被擋下
SUPPORTED_BOXMOT_TRACKERS = [
    # A 級
    "bytetrack",
    "ocsort",
    "fasttracker",
    # B 級
    "sfsort",
    "cbiou",
]

# --- 支援的 URI scheme（與 logic/config.py 一致） ---
_URI_SCHEMES = ("file://", "rtsp://", "rtsps://", "http://", "https://")


# ==========================================
# 2. 通用輔助函式 (Utility Functions)
# ==========================================

def load_all_yamls(yaml_dir: str) -> List[Dict[str, Any]]:
    """
    讀取 yaml_dir 下所有 .yaml 檔（按檔名排序），回傳 list of dict

    參數：
        yaml_dir (str): YAML 資料夾路徑

    返回：
        list[dict]: 各路 cam 的設定字典（依檔名排序，對應 pad_index 0..N-1）
    """
    files = sorted(glob.glob(f"{yaml_dir}/*.yaml"))
    if not files:
        print(f"[ERROR] 找不到任何 YAML 檔案在：{yaml_dir}")
        sys.exit(1)

    cfgs = []
    for file_path in files:
        with open(file_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
            cfgs.append(data)
    return cfgs


def get_num_classes(label_filename: str) -> int:
    """
    讀 BASE_DIR/label_filename 的非空行數作為類別數

    參數：
        label_filename (str): 標籤檔檔名（相對 BASE_DIR）

    返回：
        int: 類別總數；檔案不存在或為空回傳 1
    """
    label_path = os.path.join(BASE_DIR, label_filename)
    if not os.path.exists(label_path):
        print(f"[WARNING] 標籤檔案不存在：{label_path}，使用預設值 1")
        return 1

    with open(label_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]
    if not lines:
        print(f"[WARNING] 標籤檔案 {label_path} 內容為空，使用預設值 1")
        return 1
    return len(lines)


def get_detect_thresholds(cfgs: List[Dict[str, Any]], model: str) -> Tuple[float, float]:
    """
    從 cfgs[0] 的 detect 區塊取出對應模型的 (conf, iou) 閾值

    所有 cam 共用同一組閾值（DS 配置檔本身不支援逐 stream 不同）

    參數：
        cfgs (list[dict]): 各路 cam 設定
        model (str): "car" / "plate" / "num"

    返回：
        tuple: (conf_thresh, iou_thresh)
    """
    detect = cfgs[0].get("detect", {})
    if model == "car":
        return detect.get("car_conf", 0.25), detect.get("car_iou", 0.45)
    if model == "plate":
        return detect.get("plate_conf", 0.25), detect.get("plate_iou", 0.45)
    return detect.get("num_conf", 0.25), detect.get("num_iou", 0.45)


def get_engine_batch(cfgs: List[Dict[str, Any]], model: str) -> int:
    """
    取對應模型的 engine batch size

    必須等於或小於 trtexec --maxShapes 上限，不然 DS 跑起來會報錯：
        car / plate → YAML weight_batch_size
        num         → YAML num_weight_batch_size

    參數：
        cfgs (list[dict]): 各路 cam 設定
        model (str): "car" / "plate" / "num"

    返回：
        int: batch size
    """
    cfg0 = cfgs[0]
    if model == "num":
        return int(cfg0.get("num_weight_batch_size", DEFAULT_NUM_BATCH))
    return int(cfg0.get("weight_batch_size", DEFAULT_WEIGHT_BATCH))


def get_engine_imgsz(cfgs: List[Dict[str, Any]], model: str) -> int:
    """
    取對應模型的 engine 輸入解析度

    此值必須跟 export ONNX 時的 --size、跟 trtexec --maxShapes 完全一致，
    不然 DS 跑起來會報 dimension mismatch：
        car / plate → YAML weight_imgsz
        num         → YAML num_imgsz

    參數：
        cfgs (list[dict]): 各路 cam 設定
        model (str): "car" / "plate" / "num"

    返回：
        int: imgsz
    """
    cfg0 = cfgs[0]
    if model == "num":
        return int(cfg0.get("num_imgsz", DEFAULT_NUM_IMGSZ))
    return int(cfg0.get("weight_imgsz", DEFAULT_WEIGHT_IMGSZ))


def crop_points_to_rect(points: List[List[int]]) -> Tuple[int, int, int, int]:
    """
    多邊形點位（YAML 寫的 crop_points）轉外接矩形 (x, y, w, h)

    DS preprocess 的 roi-params-src-N 只接受矩形不接受多邊形

    參數：
        points (list): 多邊形頂點 [[x1,y1], [x2,y2], ...]

    返回：
        tuple: (x, y, w, h)
    """
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    width  = max(1, int(max_x - min_x))
    height = max(1, int(max_y - min_y))
    return int(min_x), int(min_y), width, height


def resolve_muxer_size(cfgs: List[Dict[str, Any]]) -> Tuple[int, int]:
    """
    所有 cam 的 base_w / base_h 取最大值供 streammux 使用

    streammux 會把所有 stream 都縮放到這個尺寸

    參數：
        cfgs (list[dict]): 各路 cam 設定

    返回：
        tuple: (muxer_width, muxer_height)
    """
    max_w = max(cfg["geometry"]["base_w"] for cfg in cfgs)
    max_h = max(cfg["geometry"]["base_h"] for cfg in cfgs)
    return max_w, max_h


def compute_tiled_layout(num_sources: int) -> Tuple[int, int]:
    """
    依來源數量計算 tiled-display 的 (rows, cols)，越接近正方形越好

    參數：
        num_sources (int): cam 數量

    返回：
        tuple: (rows, cols)
    """
    cols = math.ceil(math.sqrt(num_sources))
    rows = math.ceil(num_sources / cols)
    return rows, cols


def is_live_stream(uri: str) -> bool:
    """判斷 URI 是否為即時串流 (RTSP/HTTP)，影響 streammux 的 live-source 屬性"""
    return uri.startswith(("rtsp://", "rtsps://", "http://", "https://"))


def _polygon_to_pts_string(points: List[List[int]]) -> str:
    """
    多邊形點位攤平成 nvdsanalytics 格式字串

    參數：
        points (list): [[x1,y1], [x2,y2], ...]

    返回：
        str: "x1;y1;x2;y2;..."
    """
    return ";".join(str(coord) for point in points for coord in point)


def _resolve_source_uri(raw: str) -> str:
    """
    YAML source 字串轉成 GStreamer 可用的合法 URI

    支援寫法：相對路徑、絕對路徑、~ 家目錄、${BASE_DIR} 樣板、已是 URI
    解析規則與 logic/config.py 完全一致，避免兩邊行為不同

    參數：
        raw (str): YAML 原始字串

    返回：
        str: 解析後 URI；raw 不合法時回傳空字串
    """
    if not isinstance(raw, str) or not raw:
        return ""

    s = raw.strip()
    if "${BASE_DIR}" in s:
        s = s.replace("${BASE_DIR}", BASE_DIR)

    if s.startswith(_URI_SCHEMES):
        return s

    if s.startswith("~"):
        s = os.path.expanduser(s)
    if not os.path.isabs(s):
        s = os.path.normpath(os.path.join(BASE_DIR, s))
    else:
        s = os.path.normpath(s)

    return f"file://{s}"


# ==========================================
# 3. 設定檔產生：preprocess
# ==========================================

def generate_preprocess_config(cfgs: List[Dict[str, Any]]) -> None:
    """
    產生 config_preprocess.txt（裁切 ROI、餵給 PGIE 的張量規格）

    處理流程：
    1. 共用 [property] 區塊：tensor 規格、normalization、custom lib
    2. [group-0] 對所有 src 套同一份 preprocess
    3. 每路 cam 各自的裁切 ROI 由 crop_points 轉外接矩形寫入

    重點：
        - network-input-shape 的 batch 對應 PGIE 的 weight_batch_size
        - network-input-shape 的 W/H 對應 weight_imgsz
        - draw-roi 由各 cam 的 display.show_crop 決定（任一為 true 就畫）

    參數：
        cfgs (list[dict]): 各路 cam 設定
    """
    camera_count = len(cfgs)
    engine_batch = get_engine_batch(cfgs, "car")
    imgsz = get_engine_imgsz(cfgs, "car")
    show_any_crop = any(cfg.get("display", {}).get("show_crop", False) for cfg in cfgs)

    lines = [
        "[property]",
        "enable=1",
        "target-unique-ids=1",
        "process-on-frame=1",
        "network-input-order=0",
        f"network-input-shape={engine_batch};3;{imgsz};{imgsz}",
        "network-color-format=0",
        "tensor-data-type=0",
        "tensor-name=input",
        f"processing-width={imgsz}",
        f"processing-height={imgsz}",
        "scaling-buf-pool-size=6",
        "tensor-buf-pool-size=6",
        "scaling-pool-memory-type=0",
        "scaling-pool-compute-hw=0",
        "scaling-filter=0",
        "maintain-aspect-ratio=1",
        "symmetric-padding=1",
        "custom-lib-path=/opt/nvidia/deepstream/deepstream/lib/gst-plugins/libcustom2d_preprocess.so",
        "custom-tensor-preparation-function=CustomTensorPreparation",
        "",
        "[user-configs]",
        "pixel-normalization-factor=0.003921568",   # 1/255，對應 YOLO 訓練時的 normalization
        "",
        "[group-0]",
        f"src-ids={';'.join(str(i) for i in range(camera_count))}",
        "process-on-roi=1",
        "custom-input-transformation-function=CustomAsyncTransformation",
        f"draw-roi={1 if show_any_crop else 0}",
        "roi-color=0;1;1;1",   # 青綠色（RGBA 0-1 範圍）
    ]

    # 為每路 cam 加入裁切矩形
    for i, cfg in enumerate(cfgs):
        crop_points = cfg["geometry"]["crop_points"]
        x, y, w, h = crop_points_to_rect(crop_points)
        lines.append(f"roi-params-src-{i}={x};{y};{w};{h}")

    with open(PREPROCESS_CONFIG, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# ==========================================
# 4. 設定檔產生：car PGIE
# ==========================================

def generate_primary_infer_config(cfgs: List[Dict[str, Any]]) -> None:
    """
    產生 config_infer_primary_yolo11.txt（車輛偵測，PGIE）

    car 模型用靜態 batch 即可（多路 cam 同步推論）
    類別數從 labels_car.txt 推
    閾值從 YAML detect.car_conf / car_iou

    參數：
        cfgs (list[dict]): 各路 cam 設定
    """
    batch_size = get_engine_batch(cfgs, "car")
    num_classes = get_num_classes(CAR_LABEL_FILE)
    conf_thresh, iou_thresh = get_detect_thresholds(cfgs, "car")

    content = f"""\
[property]
gpu-id=0
net-scale-factor=0.0039215697906911373
model-engine-file=car_fp16.engine
batch-size={batch_size}
network-mode=2
num-detected-classes={num_classes}
labelfile-path={CAR_LABEL_FILE}
interval=0
gie-unique-id=1
process-mode=1
network-type=0
cluster-mode=2
maintain-aspect-ratio=1
symmetric-padding=1
custom-lib-path=nvdsinfer_custom_impl_Yolo/libnvdsinfer_custom_impl_Yolo.so
parse-bbox-func-name=NvDsInferParseYolo

[class-attrs-all]
nms-iou-threshold={iou_thresh}
pre-cluster-threshold={conf_thresh}
topk=300
"""
    with open(INFER_PRIMARY_CONFIG, "w", encoding="utf-8") as f:
        f.write(content)


# ==========================================
# 5. 設定檔產生：plate SGIE
# ==========================================

def generate_plate_infer_config(cfgs: List[Dict[str, Any]]) -> None:
    """
    產生 config_infer_secondary_plate.txt（車牌偵測，SGIE）

    operate-on-class-ids 自動由 labels_car.txt 行數展開為 0;1;...;N-1，
    意思是「對 car 模型偵測到的所有車種都做車牌偵測」

    output-buffer-pool-size=16 是跑通的關鍵之一：
    多路 cam 同時有多輛車時，buffer 不夠會丟 plate ROI

    參數：
        cfgs (list[dict]): 各路 cam 設定
    """
    batch_size = get_engine_batch(cfgs, "plate")
    num_classes = get_num_classes(PLATE_LABEL_FILE)
    conf_thresh, iou_thresh = get_detect_thresholds(cfgs, "plate")

    # 自動展開 car 全部類別
    car_classes = get_num_classes(CAR_LABEL_FILE)
    operate_on_ids = ";".join(str(i) for i in range(car_classes))

    content = f"""\
[property]
gpu-id=0
net-scale-factor=0.0039215697906911373
model-engine-file=plate_fp16.engine
batch-size={batch_size}
network-mode=2
num-detected-classes={num_classes}
labelfile-path={PLATE_LABEL_FILE}
interval=0
gie-unique-id=2
operate-on-gie-id=1
operate-on-class-ids={operate_on_ids}
process-mode=2
scaling-compute-hw=1
network-type=0
cluster-mode=2
maintain-aspect-ratio=1
symmetric-padding=1
output-buffer-pool-size=16
custom-lib-path=nvdsinfer_custom_impl_Yolo/libnvdsinfer_custom_impl_Yolo.so
parse-bbox-func-name=NvDsInferParseYolo

[class-attrs-all]
nms-iou-threshold={iou_thresh}
pre-cluster-threshold={conf_thresh}
topk=100
"""
    with open(INFER_PLATE_CONFIG, "w", encoding="utf-8") as f:
        f.write(content)


# ==========================================
# 6. 設定檔產生：num SGIE (dynamic batch)
# ==========================================

def generate_num_infer_config(cfgs: List[Dict[str, Any]]) -> None:
    """
    產生 config_infer_secondary_num.txt（字元偵測，SGIE）

    這是跑通的核心設定，必須與 trtexec 編 engine 的設定完全對齊：
        - infer-dims=3;num_imgsz;num_imgsz   ← 從 YAML num_imgsz 取
        - force-implicit-batch-dim=0          ← dynamic batch 必須關掉 implicit
        - batch-size                          ← 從 YAML num_weight_batch_size 取
        - output-buffer-pool-size=32          ← 多路 × 多車 × 多字元時防止丟 buffer

    任何一項漏掉都會回到 batch=2 漏字元的舊問題

    參數：
        cfgs (list[dict]): 各路 cam 設定
    """
    batch_size = get_engine_batch(cfgs, "num")
    imgsz = get_engine_imgsz(cfgs, "num")
    num_classes = get_num_classes(NUM_LABEL_FILE)
    conf_thresh, iou_thresh = get_detect_thresholds(cfgs, "num")

    content = f"""\
[property]
gpu-id=0
net-scale-factor=0.0039215697906911373
model-engine-file=num_fp16.engine
batch-size={batch_size}
network-mode=2
num-detected-classes={num_classes}
labelfile-path={NUM_LABEL_FILE}
infer-dims=3;{imgsz};{imgsz}
force-implicit-batch-dim=0
interval=0
gie-unique-id=3
operate-on-gie-id=2
operate-on-class-ids=0
process-mode=2
scaling-compute-hw=1
network-type=0
cluster-mode=2
maintain-aspect-ratio=1
symmetric-padding=1
output-buffer-pool-size=32
custom-lib-path=nvdsinfer_custom_impl_Yolo/libnvdsinfer_custom_impl_Yolo.so
parse-bbox-func-name=NvDsInferParseYolo

[class-attrs-all]
nms-iou-threshold={iou_thresh}
pre-cluster-threshold={conf_thresh}
topk=100
"""
    with open(INFER_NUM_CONFIG, "w", encoding="utf-8") as f:
        f.write(content)


# ==========================================
# 7. 設定檔產生：nvdsanalytics
# ==========================================

def generate_analytics_config(cfgs: List[Dict[str, Any]], muxer_w: int, muxer_h: int) -> None:
    """
    產生 config_nvdsanalytics.txt（多 ROI 版本）

    多 ROI 機制：
        - YAML geometry.regions 是 dict：{roi_1: [...], roi_2: [...]}
        - 同一 [roi-filtering-stream-N] 內可放多個 roi-XXX=... 鍵
        - 每個鍵是一個獨立多邊形，nvdsanalytics 各自畫線
        - ROI 名稱會寫進 DB 的 ROI 欄位，與 probes.py 累積邏輯共用

    Fallback：未定義 regions → 用全畫面當預設 ROI，避免 nvdsanalytics 啟動報錯
    show_roi=False → 整個區塊 enable=0（但仍要產出，DS 要求 stream id 連續）

    參數：
        cfgs (list[dict]): 各路 cam 設定
        muxer_w (int): streammux 寬
        muxer_h (int): streammux 高
    """
    lines = [
        "[property]",
        "enable=1",
        f"config-width={muxer_w}",
        f"config-height={muxer_h}",
        "osd-mode=1",
        "display-font-size=12",
        ""
    ]

    for i, cfg in enumerate(cfgs):
        source_id = cfg.get("source_id", f"cam_{i}")
        show_roi = 1 if cfg.get("display", {}).get("show_roi", True) else 0
        regions = cfg.get("geometry", {}).get("regions", {}) or {}

        block = [
            f"[roi-filtering-stream-{i}]",
            f"enable={show_roi}",
        ]

        if regions:
            # 多 ROI：每個 region 寫一個 roi-{name}=... 鍵
            for roi_name, pts in regions.items():
                if not pts or len(pts) < 3:
                    print(f"[WARNING] {source_id} 的 ROI '{roi_name}' 點數不足({len(pts) if pts else 0})，略過")
                    continue
                pts_str = _polygon_to_pts_string(pts)
                block.append(f"roi-{roi_name}={pts_str}")
        else:
            # Fallback：用全畫面當預設 ROI
            print(f"[WARNING] {source_id} 沒有定義 regions，使用全畫面作為預設 ROI")
            pts_str = f"0;0;{muxer_w};0;{muxer_w};{muxer_h};0;{muxer_h}"
            block.append(f"roi-{source_id}={pts_str}")

        block.extend([
            "class-id=-1",
            "inverse-roi=0",
            ""
        ])

        lines.extend(block)

    with open(ANALYTICS_CONFIG, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# ==========================================
# 8. 設定檔產生：tracker runtime
# ==========================================

def get_tracker_mode(cfgs: List[Dict[str, Any]]) -> str:
    """
    讀 cfgs[0] 的 tracker.type，回傳追蹤器模式字串

    全域設定：所有 cam 共用一條 pipeline，只看第一個 YAML 的 tracker 區塊
    向下相容：未寫 tracker 區塊 → 預設 "nvdcf"

    參數：
        cfgs (list[dict]): 各路 cam 設定

    返回：
        str: 小寫化的 tracker 名稱（"nvdcf" / "bytetrack" / ...）
    """
    tracker_cfg = cfgs[0].get("tracker", {}) or {}
    mode = str(tracker_cfg.get("type", "nvdcf")).lower().strip()
    return mode


def generate_tracker_runtime_config(cfgs: List[Dict[str, Any]]) -> str:
    """
    產生 config_tracker_runtime.txt 給 main.py 啟動時讀取

    產出格式（INI，main.py 用 configparser 解析）：
        [tracker]
        mode=<nvdcf | bytetrack | ...>
        config=/abs/path/to/boxmot/configs/trackers/<mode>.yaml   # BoxMOT 模式才有

    BoxMOT 微調規則：
        - 不在 ds_yaml/*.yaml 暴露追蹤器內部參數
        - 想調整參數直接編輯 boxmot/configs/trackers/<mode>.yaml

    參數：
        cfgs (list[dict]): 各路 cam 設定

    返回：
        str: tracker mode 字串，供上層印 log

    例外：
        ValueError: tracker.type 不在白名單內
        FileNotFoundError: BoxMOT 設定檔不存在（boxmot/ 未複製）
    """
    mode = get_tracker_mode(cfgs)

    # 白名單檢查
    if mode != "nvdcf" and mode not in SUPPORTED_BOXMOT_TRACKERS:
        raise ValueError(
            f"未支援的 tracker.type='{mode}'。\n"
            f"  可用值：nvdcf, {', '.join(SUPPORTED_BOXMOT_TRACKERS)}\n"
            f"  C 級追蹤器（botsort / strongsort / deepocsort / hybridsort / "
            f"imprassoc / boosttrack）需要真實 BGR 像素（CMC 光流 / ReID），\n"
            f"  目前未實作 NVMM→CPU 拷貝邏輯，故不支援。"
        )

    lines = [
        "# 由 LPR_txt.py 自動產生，請勿手動編輯",
        "# 來源：ds_yaml/*.yaml 內 tracker.type（讀 cfgs[0]）",
        "# BoxMOT 微調請直接編輯 boxmot/configs/trackers/<mode>.yaml",
        "",
        "[tracker]",
        f"mode={mode}",
    ]

    # BoxMOT 模式：附上對應的設定檔絕對路徑
    if mode in SUPPORTED_BOXMOT_TRACKERS:
        config_path = f"{BASE_DIR}/boxmot/configs/trackers/{mode}.yaml"
        if not os.path.exists(config_path):
            raise FileNotFoundError(
                f"找不到 BoxMOT 追蹤器設定：{config_path}\n"
                f"  請確認 boxmot/ 已複製到專案根目錄。"
            )
        lines.append(f"config={config_path}")

    with open(TRACKER_RUNTIME_CONFIG, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    return mode


# ==========================================
# 9. 設定檔產生：deepstream_app_config
# ==========================================

def generate_deepstream_app_config(cfgs: List[Dict[str, Any]], muxer_w: int, muxer_h: int) -> None:
    """
    產生 deepstream_app_config.txt（整合所有元件）

    依序組裝：sources / streammux / preprocess / pgie / tracker /
              sgie0 (plate) / sgie1 (num) / nvds-analytics / sink / tiled-display

    type 自動判斷：rtsp/http/https → 4 (live)，其他 → 3 (file)
    live-source 自動：任一來源是 live → 1，否則 0
    streammux batch-size 自動 = cam 數量

    參數：
        cfgs (list[dict]): 各路 cam 設定
        muxer_w (int): streammux 寬
        muxer_h (int): streammux 高
    """
    num_sources = len(cfgs)
    rows, cols = compute_tiled_layout(num_sources)

    # tiled-display 顯示尺寸（不影響推論，純顯示）
    display_width = 1280
    display_height = 720

    # 任一來源是 live 串流就把 streammux 設為 live-source 模式
    live_source = 0
    for cfg in cfgs:
        if is_live_stream(_resolve_source_uri(cfg.get("source", ""))):
            live_source = 1
            break

    lines = [
        "[application]",
        "enable-perf-measurement=1",
        "perf-measurement-interval-sec=1",
        "",
        "[tiled-display]",
        "enable=1",
        f"rows={rows}",
        f"columns={cols}",
        f"width={display_width}",
        f"height={display_height}",
        "gpu-id=0",
        "nvbuf-memory-type=0",
        ""
    ]

    # 每路 cam 一個 [sourceN]
    for i, cfg in enumerate(cfgs):
        uri = _resolve_source_uri(cfg.get("source", f"videos/source_{i}.mp4"))
        # type 3 = 本地檔案、type 4 = 即時串流
        source_type = 4 if is_live_stream(uri) else 3

        lines.extend([
            f"[source{i}]",
            "enable=1",
            f"type={source_type}",
            f"uri={uri}",
            "num-sources=1",
            "gpu-id=0",
            "cudadec-memtype=0",
            ""
        ])

    # 固定區段：sink / osd / streammux / preprocess / 三層 GIE / tracker / analytics
    lines.extend([
        "[sink0]",
        "enable=1",
        "type=2",                  # nveglglessink（顯示視窗）
        "sync=0",
        "qos=0",
        "gpu-id=0",
        "nvbuf-memory-type=0",
        "",
        "[osd]",
        "enable=1",
        "gpu-id=0",
        "border-width=2",
        "text-size=15",
        "text-color=1;1;1;1",
        "text-bg-color=0;0;0;1",
        "font=Serif",
        "display-text=1",
        "display-bbox=1",
        "",
        "[streammux]",
        "gpu-id=0",
        f"live-source={live_source}",
        f"batch-size={num_sources}",     # streammux batch 一定等於 cam 數
        "batched-push-timeout=40000",
        f"width={muxer_w}",
        f"height={muxer_h}",
        "enable-padding=0",
        "nvbuf-memory-type=0",
        "",
        "[pre-process]",
        "enable=1",
        f"config-file={PREPROCESS_CONFIG}",
        "",
        "[primary-gie]",
        "enable=1",
        "gpu-id=0",
        "gie-unique-id=1",
        "nvbuf-memory-type=0",
        f"config-file={INFER_PRIMARY_CONFIG}",
        "input-tensor-meta=1",            # PGIE 從 nvdspreprocess 拿張量
        "",
        "[tracker]",
        "enable=1",
        "tracker-width=640",
        "tracker-height=384",
        "ll-lib-file=/opt/nvidia/deepstream/deepstream/lib/libnvds_nvmultiobjecttracker.so",
        f"ll-config-file={BASE_DIR}/config_tracker_NvDCF_accuracy.yml",
        "gpu-id=0",
        "display-tracking-id=1",
        "",
        "[secondary-gie0]",
        "enable=1",
        "gpu-id=0",
        "gie-unique-id=2",
        "nvbuf-memory-type=0",
        "operate-on-gie-id=1",            # 對 PGIE 結果做車牌偵測
        f"config-file={INFER_PLATE_CONFIG}",
        "",
        "[secondary-gie1]",
        "enable=1",
        "gpu-id=0",
        "gie-unique-id=3",
        "nvbuf-memory-type=0",
        "operate-on-gie-id=2",            # 對 plate 結果做字元偵測
        f"config-file={INFER_NUM_CONFIG}",
        "",
        "[nvds-analytics]",
        "enable=1",
        f"config-file={ANALYTICS_CONFIG}",
        "",
        "[tests]",
        "file-loop=0"
    ])

    with open(APP_CONFIG, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# ==========================================
# 10. 主程式 (Main Entry Point)
# ==========================================

def main() -> None:
    """
    主流程：載入 YAML → 印解析結果 → 依序產生 7 份設定檔（LPR 三層版）

    處理流程：
    1. 自動偵測 BASE_DIR 並印出
    2. 載入所有 YAML 並依檔名排序
    3. 算出 streammux 尺寸、三個 engine 的參數
    4. 印每路 cam 的 source 解析結果與 ROI 數量（方便確認）
    5. 依序產生 7 個設定檔
    6. 印產生結果摘要與 tracker 模式提示
    """
    print(f"[INFO] 自動偵測到專案路徑 BASE_DIR = {BASE_DIR}")
    print("正在載入 YAML 設定檔...")
    cfgs = load_all_yamls(YAML_DIR)
    print(f"已載入 {len(cfgs)} 個攝影機設定")

    # 步驟 1: 算 streammux 尺寸與三個 engine 參數
    muxer_w, muxer_h = resolve_muxer_size(cfgs)
    print(f"Streammux 輸出尺寸: {muxer_w} x {muxer_h}")

    car_batch   = get_engine_batch(cfgs, "car")
    plate_batch = get_engine_batch(cfgs, "plate")
    num_batch   = get_engine_batch(cfgs, "num")
    car_imgsz   = get_engine_imgsz(cfgs, "car")
    num_imgsz   = get_engine_imgsz(cfgs, "num")
    print(f"Engine 設定 — car/plate: batch={car_batch}, imgsz={car_imgsz}")
    print(f"           — num:       batch={num_batch}, imgsz={num_imgsz}")

    # 步驟 2: 印每路 cam 的 source 解析結果與 ROI
    for i, cfg in enumerate(cfgs):
        source_id = cfg.get("source_id", f"cam_{i}")
        raw_source = cfg.get("source", "")
        resolved = _resolve_source_uri(raw_source)
        regions = cfg.get("geometry", {}).get("regions", {}) or {}

        print(f"  [{source_id}] source='{raw_source}' → {resolved}")
        if regions:
            print(f"           ROI 數量: {len(regions)} → {list(regions.keys())}")
        else:
            print(f"           未定義 ROI，將使用全畫面")

    # 步驟 3: 依序產生 7 個設定檔
    generate_preprocess_config(cfgs)
    generate_primary_infer_config(cfgs)
    generate_plate_infer_config(cfgs)
    generate_num_infer_config(cfgs)
    generate_analytics_config(cfgs, muxer_w, muxer_h)
    generate_deepstream_app_config(cfgs, muxer_w, muxer_h)
    tracker_mode = generate_tracker_runtime_config(cfgs)

    # 步驟 4: 印摘要
    print("\n[DONE] 所有設定檔產生完畢！")
    print(f"  - {APP_CONFIG}")
    print(f"  - {PREPROCESS_CONFIG}")
    print(f"  - {INFER_PRIMARY_CONFIG}")
    print(f"  - {INFER_PLATE_CONFIG}")
    print(f"  - {INFER_NUM_CONFIG}")
    print(f"  - {ANALYTICS_CONFIG}")
    print(f"  - {TRACKER_RUNTIME_CONFIG}  (tracker mode = {tracker_mode})")

    # BoxMOT 模式額外提示
    if tracker_mode != "nvdcf":
        print(f"\n  ⚠ tracker.type = '{tracker_mode}' (BoxMOT)")
        print(f"     main.py 啟動後將 *跳過* nvtracker，改在 pgie.src 探針裡用 BoxMOT 接管")
        print(f"     BoxMOT 微調請直接編輯：boxmot/configs/trackers/{tracker_mode}.yaml")


if __name__ == "__main__":
    main()