"""
DeepStream Probe 探針集合（車牌辨識 LPR 版）

主要功能：
1. 兩種追蹤器模式共用同一份軌跡狀態邏輯：
   - NvDCF (nvdcf)  → tracker_src_pad_buffer_probe，掛在 tracker.src
   - BoxMOT 系列    → boxmot_pgie_src_probe，掛在 pgie.src
2. BoxMOT 模式接管偵測：清空 PGIE obj_meta → 餵給 BoxMOT → 用追蹤結果重建 obj_meta
3. 車輛軌跡狀態維護：start_y、方向 IN/OUT、多 ROI 命中累積、車種投票、車牌投票
4. 消失時結算：物件 missing_frames 達門檻 → 呼叫 _finalize_one 寫 DB
5. LPR 三層獨有：
   - expand_plate_probe   ：把車牌框上下左右擴 10%，給下游字元偵測留邊界
   - assemble_plate_probe ：每幀組裝車牌字串、寫進 plate_votes、更新 OSD
6. 截圖（面積最大選幀，車種與車牌同幀）：
   - 車牌框「面積」刷新歷史最大時，同一幀同時截兩張（車最靠近鏡頭、車牌最大那刻）
     車牌框 → state["best_plate_jpg"]
     車輛框 → state["best_class_jpg"]
   - 車輛框用「車輛框 ∪ 車牌框」聯集，且下緣再加車牌高度 30% padding，
     確保車輛框照片一定含完整車牌（補 DeepStream 車輛框下緣常差幾 px 的問題）
   - 車輛框 obj_meta 直接從 assemble_plate_probe 所在的同一 frame_meta 撈（uid=1），
     不跨探針傳遞，保證與車牌框同幀同步
   - fallback：整段都沒辨識到車牌時，退用「車輛框面積最大」那幀
   結算時由 state_db.py 寫檔
7. OSD 視覺化：bbox 用車種色（不變紅）、ID 標籤、左上角即時 FPS 顯示
"""

import os
import time
import cv2
import numpy as np
from collections import Counter, deque
from gi.repository import Gst
import pyds

from logic.color import get_class_color, CLASS_MAP, NUM_MAP
from logic.config import SOURCE_CONFIGS
from logic.state_db import (
    get_local_id, _finalize_one, flush_pending_to_db,
    track_history, pending_records, last_flush_times,
    fps_streams, local_id_maps
)


# ==========================================
# 1. 系統配置區 (System Configuration)
# ==========================================

# --- 模組執行期狀態 ---
g_last_fps_print_time = time.time()          # 上次印 FPS 報告的時間戳

# --- LPR 三層的 unique_component_id 對應 ---
# nvtracker / BoxMOT 重建 → 1（車輛）
# SGIE plate           → 2（車牌）
# SGIE num             → 3（字元）
_UID_VEHICLE = 1
_UID_PLATE   = 2
_UID_CHAR    = 3

# --- 截圖相關 ---
# Object Encoder context（由 main.py 啟動時透過 set_obj_enc_context 注入）
g_enc_ctx = None

# 截圖編碼暫存資料夾（用 /tmp，Jetson 上通常是 tmpfs 記憶體檔案系統）
# Object Encoder 以 saveImg 存到這裡後立刻讀回成 bytes 並刪檔，等同記憶體進出
_TMP_DIR = "/tmp/lpr_crop_tmp"

# JPEG 編碼品質
_JPEG_QUALITY = 70

# 聯集車輛框時，下緣額外往下延伸的比例（以車牌框高度為基準）
# 補 DeepStream 車輛框下緣常比車牌底部高幾 px 的問題，確保截圖含完整車牌
_PLATE_BOTTOM_PAD_RATIO = 0.30

# Object Encoder 失敗時只印一次警告，避免洗版
_enc_warned = False


def set_obj_enc_context(ctx):
    """
    由 main.py 在啟動時注入 Object Encoder context

    參數：
        ctx: pyds.nvds_obj_enc_create_context() 的回傳值；None 表示不啟用截圖
    """
    global g_enc_ctx
    g_enc_ctx = ctx
    if ctx is not None:
        os.makedirs(_TMP_DIR, exist_ok=True)
        print(f"[INFO] [probes] Object Encoder context 已就緒，暫存目錄：{_TMP_DIR}")


# ==========================================
# 2. 截圖編碼輔助 (Crop Encoding Helper)
# ==========================================

def _encode_crop_to_bytes(gst_buffer, obj_meta, frame_meta, tmp_name):
    """
    用 Object Encoder 對指定 obj_meta 的框裁切編碼成 JPEG，回傳 bytes

    處理流程（GPU 裁切，不落 CPU 整張 frame）：
    1. 設定 NvDsObjEncUsrArgs（saveImg=True、quality=70、輸出暫存檔路徑）
    2. nvds_obj_enc_process 對該框做 GPU 裁切 + JPEG 編碼 + 寫暫存檔
    3. nvds_obj_enc_finish 等待編碼完成
    4. 讀回暫存檔成 bytes
    5. 無論成功或失敗，finally 確保暫存檔被刪除（避免 /tmp tmpfs 長期累積殘檔）

    參數：
        gst_buffer: GStreamer buffer（process 需要其 hash）
        obj_meta (pyds.NvDsObjectMeta): 要裁切的框（車輛框或車牌框）
        frame_meta (pyds.NvDsFrameMeta): 該框所在的 frame meta
        tmp_name (str): 暫存檔名（含 .jpg），放在 _TMP_DIR 底下

    返回：
        bytes | None: JPEG 位元組；失敗回傳 None
    """
    global _enc_warned

    if g_enc_ctx is None:
        return None

    tmp_path = os.path.join(_TMP_DIR, tmp_name)
    data = None
    try:
        enc_args = pyds.NvDsObjEncUsrArgs()
        enc_args.saveImg = True
        enc_args.attachUsrMeta = False
        enc_args.quality = _JPEG_QUALITY
        enc_args.fileNameImg = tmp_path

        pyds.nvds_obj_enc_process(g_enc_ctx, enc_args, hash(gst_buffer), obj_meta, frame_meta)
        pyds.nvds_obj_enc_finish(g_enc_ctx)

        if os.path.exists(tmp_path) and os.path.getsize(tmp_path) > 0:
            with open(tmp_path, "rb") as f:
                data = f.read()
    except Exception as e:
        if not _enc_warned:
            print(f"[WARNING] [probes] Object Encoder 編碼失敗（之後不再重複此警告）：{e}")
            print(f"[WARNING]   若截圖全黑或格式錯，可能需要在 pipeline 加 nvvideoconvert 轉 RGBA")
            _enc_warned = True
    finally:
        # 無論成功失敗都清暫存檔，避免編碼失敗時殘檔累積在 /tmp（tmpfs 記憶體）
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass

    return data


# ==========================================
# 3. 共用核心邏輯 (Shared Tracking Logic)
# ==========================================

def _process_tracked_frame(gst_buffer, frame_meta, current_frame_objects, pad_index, cfg):
    """
    遍歷 frame_meta 內所有 obj_meta，維護軌跡狀態、調整 OSD 顯示、（啟用時）車種截圖 fallback

    處理流程：
    1. 過濾追蹤器輸出與無效 ID（只處理 unique_component_id=1）
    2. 對每個物件計算 bbox 底部中心點 (cx, cy)
    3. 首次出現 → 初始化軌跡狀態（含 plate_votes 與截圖欄位）
    4. 多 ROI 命中判斷 → 累加 roi_hits[roi_name] + class_votes
    5. 車種截圖 fallback：用「車輛框面積最大」那幀補一張（整段沒車牌時才會用到）
    6. 方向判斷（軌跡共用）：用 cy 與 start_y 的 Y 軸位移判斷 IN/OUT
    7. OSD 視覺化：bbox 永遠用車種色（不變紅）、ID 標籤

    參數：
        gst_buffer: GStreamer buffer（截圖編碼用）
        frame_meta (pyds.NvDsFrameMeta): 當前幀的 meta
        current_frame_objects (set): 本幀出現的 (pad_index, obj_id) 集合
        pad_index (int): 哪一路 cam
        cfg (dict): 該路 cam 的 YAML 設定
    """
    cv_regions = cfg.get("cv_regions", {})
    movement_threshold = cfg.get("track_logic", {}).get("movement_threshold", 30)
    save_ss = cfg.get("save_screenshot", False)

    l_obj = frame_meta.obj_meta_list
    while l_obj is not None:
        try:
            obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)
        except StopIteration:
            break

        # 步驟 1: 只處理車輛（unique_component_id=1），略過車牌（2）與字元（3）
        if obj_meta.unique_component_id != _UID_VEHICLE:
            l_obj = l_obj.next
            continue

        obj_id = obj_meta.object_id
        if obj_id == -1:
            l_obj = l_obj.next
            continue

        unique_key = (pad_index, obj_id)
        current_frame_objects.add(unique_key)
        local_id = get_local_id(pad_index, obj_id)

        # 步驟 2: 計算 bbox 底部中心點（車輛地面位置）
        cx = int(obj_meta.rect_params.left + (obj_meta.rect_params.width / 2))
        cy = int(obj_meta.rect_params.top + obj_meta.rect_params.height)

        # 步驟 3: 初始化軌跡狀態（首次出現）
        if unique_key not in track_history:
            track_history[unique_key] = {
                "start_y":             cy,                # 起始 Y，判斷方向用
                "missing_frames":      0,
                "direction":           "NA",             # IN / OUT / NA
                "class_votes":         Counter(),        # 車種投票
                "plate_votes":         Counter(),        # 車牌字串投票（LPR 獨有）
                "last_frame_num":      frame_meta.frame_num,
                "roi_hits":            {},               # 多 ROI 各自累計 {roi_name: count}
                "best_class_jpg":      None,             # 車種截圖 JPEG bytes（與車牌同幀，聯集框）
                "best_plate_jpg":      None,             # 車牌截圖 JPEG bytes
                "best_plate_area":     0,                # 車牌截圖選幀依據：車牌框面積高水位
                "fallback_class_jpg":  None,             # 車種 fallback 截圖（車輛框面積最大幀）
                "fallback_class_area": 0,                # fallback 車輛框面積高水位
            }

        state = track_history[unique_key]
        state["missing_frames"] = 0
        state["last_frame_num"] = frame_meta.frame_num

        # 取得車輛框 rect_params（步驟 5 截圖面積、步驟 7 OSD 都會用到）
        r = obj_meta.rect_params

        # 步驟 4: 多 ROI 命中判斷（命中時累加 roi_hits 與 class_votes）
        for roi_name, polygon in cv_regions.items():
            if cv2.pointPolygonTest(polygon, (cx, cy), False) >= 0:
                state["roi_hits"][roi_name] = state["roi_hits"].get(roi_name, 0) + 1
                state["class_votes"][obj_meta.class_id] += 1

        # 步驟 5: 車種截圖 fallback（只在整段沒車牌時，結算才會用到這張）
        # 選幀依據：車輛框面積最大那幀（車最靠近鏡頭、車身最完整）
        # 主要車種截圖在 assemble_plate_probe 跟車牌同幀截（聯集框，含車牌）
        if save_ss:
            veh_area = float(r.width) * float(r.height)
            if veh_area > state["fallback_class_area"]:
                jpg = _encode_crop_to_bytes(
                    gst_buffer, obj_meta, frame_meta,
                    tmp_name=f"{pad_index}_{obj_id}_class_fb.jpg",
                )
                if jpg:
                    state["fallback_class_jpg"] = jpg
                    state["fallback_class_area"] = veh_area

        # 步驟 6: 方向判斷（軌跡共用，沿用原版邏輯）
        # 首次定向後就固定，不再翻轉（避免抖動造成方向反覆切換）
        if state["direction"] == "NA":
            dy = cy - state["start_y"]
            if dy > movement_threshold:
                state["direction"] = "IN"      # Y 增加 → 向下移動
            elif dy < -movement_threshold:
                state["direction"] = "OUT"     # Y 減少 → 向上移動

        # 步驟 7: OSD 視覺化（bbox 永遠用車種色，不變紅）
        cls_id = obj_meta.class_id
        cls_name = CLASS_MAP.get(cls_id, f"Class_{cls_id}")
        color = get_class_color(cls_id)

        r.border_width = 4
        r.border_color.set(*color)
        r.has_bg_color = 0

        txt = obj_meta.text_params
        txt.display_text = f"ID:{local_id} {cls_name}"
        txt.font_params.font_name = "Serif Bold"
        txt.font_params.font_size = 14
        txt.font_params.font_color.set(1.0, 1.0, 1.0, 1.0)
        txt.set_bg_clr = 1
        txt.text_bg_clr.set(*color)

        text_h = int(14 * 1.4)
        txt.x_offset = max(0, int(r.left) + 0)
        txt.y_offset = max(0, int(r.top + r.height) - text_h - 10)

        l_obj = l_obj.next


def _post_frame_housekeeping(current_frame_objects):
    """
    每幀結束時的清理工作（兩種模式共用）

    處理流程：
    1. 偵測消失的軌跡（連續 missing_frames >= cleanup_frames）
       → 呼叫 _finalize_one 結算（含截圖寫檔）→ 從 track_history 移除
    2. 每 30 秒印一次 FPS 效能報告
    3. 依 flush_interval_seconds 定期把 pending 寫進 SQLite DB

    參數：
        current_frame_objects (set): 本幀出現的 (pad_index, obj_id) 集合
    """
    global g_last_fps_print_time

    # 步驟 1: 消失軌跡 → 結算 → 清理
    missing_keys = set(track_history.keys()) - current_frame_objects
    for m_key in missing_keys:
        pad_index, obj_id = m_key
        cfg = SOURCE_CONFIGS.get(pad_index, {})
        track_history[m_key]["missing_frames"] += 1
        cleanup_frames = cfg.get("session", {}).get("cleanup_frames", 30)

        if track_history[m_key]["missing_frames"] >= cleanup_frames:
            # 結算後才刪
            _finalize_one(m_key, track_history[m_key], force=False)
            del track_history[m_key]
            if obj_id in local_id_maps[pad_index]:
                del local_id_maps[pad_index][obj_id]

    # 步驟 2: 每 30 秒印 FPS
    current_time = time.time()
    if current_time - g_last_fps_print_time >= 30:
        print("\n" + "=" * 35)
        print(f"[{time.strftime('%H:%M:%S')}] 即時處理效能報告 (FPS)：")
        for sid, stats in sorted(fps_streams.items()):
            c_name = SOURCE_CONFIGS[sid].get("source_id", f"cam_{sid}")
            print(f" • {c_name.ljust(10)}: {stats['current_fps']:.2f} FPS")
        print("=" * 35 + "\n")
        g_last_fps_print_time = current_time

    # 步驟 3: 定期 flush 到 SQLite DB
    for pad_index, cfg in SOURCE_CONFIGS.items():
        flush_interval = cfg.get("session", {}).get("flush_interval_seconds", 30)
        if current_time - last_flush_times[pad_index] >= flush_interval:
            flush_pending_to_db(pad_index)
            last_flush_times[pad_index] = current_time


def _update_fps(pad_index):
    """
    更新指定 pad 的即時 FPS 統計（兩種模式共用）

    使用滑動視窗（30 幀）計算瞬時 FPS

    參數：
        pad_index (int): 哪一路 cam
    """
    if "timestamps" not in fps_streams[pad_index]:
        fps_streams[pad_index]["timestamps"] = deque(maxlen=30)
    now = time.time()
    q = fps_streams[pad_index]["timestamps"]
    q.append(now)
    if len(q) > 1:
        fps_streams[pad_index]["current_fps"] = (len(q) - 1) / (q[-1] - q[0])


# ==========================================
# 4. NvDCF 模式探針 (NvDCF Tracker Probe)
# ==========================================

def tracker_src_pad_buffer_probe(pad, info, u_data):
    """
    NvDCF 模式專用探針：掛在 tracker.src

    obj_meta 由 nvtracker 提供，已包含有效的 object_id；
    本探針只負責 FPS 統計 + 軌跡狀態更新 + 收尾清理。

    參數：
        pad, info, u_data: GStreamer probe 標準參數

    返回：
        Gst.PadProbeReturn.OK
    """
    gst_buffer = info.get_buffer()
    if not gst_buffer:
        return Gst.PadProbeReturn.OK

    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
    l_frame = batch_meta.frame_meta_list
    current_frame_objects = set()

    while l_frame is not None:
        try:
            frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
        except StopIteration:
            break

        pad_index = frame_meta.pad_index
        cfg = SOURCE_CONFIGS.get(pad_index)
        if not cfg:
            l_frame = l_frame.next
            continue

        _update_fps(pad_index)
        _process_tracked_frame(gst_buffer, frame_meta, current_frame_objects, pad_index, cfg)

        l_frame = l_frame.next

    _post_frame_housekeeping(current_frame_objects)
    return Gst.PadProbeReturn.OK


# ==========================================
# 5. BoxMOT 模式探針 (BoxMOT Tracker Probe)
# ==========================================

def boxmot_pgie_src_probe(pad, info, u_data):
    """
    BoxMOT 模式專用探針：掛在 pgie.src

    處理流程：
    1. 從 obj_meta 抽出所有 PGIE 偵測框（全車種都收，不過濾）
    2. 清空 frame_meta 內所有 PGIE obj_meta（回到 pool）
    3. 把偵測框餵給 BoxMOT，拿回追蹤結果（含 id、可能不同的框）
    4. 用 BoxMOT 輸出重建 obj_meta（框、id、conf、class）
       重建後的 obj_meta 會繼續流經下游 SGIE plate / SGIE num，
       讓 LPR 三層架構能正常運作
    5. 走和 NvDCF 模式同一份的軌跡狀態邏輯

    重要前提：本探針必須在 pgie.src，下游不可有 nvtracker
              （否則 nvtracker 會覆寫掉我們重建的 meta）

    參數：
        pad, info, u_data: GStreamer probe 標準參數

    返回：
        Gst.PadProbeReturn.OK
    """
    # lazy import 避免 nvdcf 模式啟動時也載入 boxmot
    from logic.boxmot_adapter import track as boxmot_track

    gst_buffer = info.get_buffer()
    if not gst_buffer:
        return Gst.PadProbeReturn.OK

    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
    l_frame = batch_meta.frame_meta_list
    current_frame_objects = set()

    while l_frame is not None:
        try:
            frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
        except StopIteration:
            break

        pad_index = frame_meta.pad_index
        cfg = SOURCE_CONFIGS.get(pad_index)
        if not cfg:
            l_frame = l_frame.next
            continue

        _update_fps(pad_index)

        # 步驟 1: 抽出所有 PGIE 偵測框（LPR 全車種都收，不做類別過濾）
        dets_list = []
        obj_metas_to_remove = []   # 先蒐集要刪的，等迴圈結束再 remove 才安全

        l_obj = frame_meta.obj_meta_list
        while l_obj is not None:
            try:
                obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)
            except StopIteration:
                break

            # pgie.src 用 detector_bbox_info 最準確（nvtracker 介入後才有 tracker_bbox_info）
            try:
                det_box = obj_meta.detector_bbox_info.org_bbox_coords
                x1 = float(det_box.left)
                y1 = float(det_box.top)
                x2 = float(det_box.left + det_box.width)
                y2 = float(det_box.top + det_box.height)
            except Exception:
                # 萬一拿不到 detector_bbox_info，退回 rect_params
                r = obj_meta.rect_params
                x1 = float(r.left)
                y1 = float(r.top)
                x2 = float(r.left + r.width)
                y2 = float(r.top + r.height)

            conf = float(obj_meta.confidence) if obj_meta.confidence > 0 else 0.5
            cls = int(obj_meta.class_id)

            # LPR 全車種都收
            dets_list.append([x1, y1, x2, y2, conf, cls])

            obj_metas_to_remove.append(obj_meta)
            l_obj = l_obj.next

        # 步驟 2: 清空 frame_meta 內所有 obj_meta
        # 從 frame 移除後 obj_meta 自動回 pool，下面重新申請即可
        for om in obj_metas_to_remove:
            pyds.nvds_remove_obj_meta_from_frame(frame_meta, om)

        # 步驟 3: 餵給 BoxMOT 取得追蹤結果
        if dets_list:
            dets = np.asarray(dets_list, dtype=np.float32)
        else:
            dets = np.empty((0, 6), dtype=np.float32)

        # A/B 級追蹤器不需要 frame；C 級才需要從 NVMM 拷貝
        tracks = boxmot_track(pad_index, dets, frame=None)

        # 步驟 4: 用 BoxMOT 輸出重建 obj_meta
        # tracks 格式：[x1, y1, x2, y2, id, conf, cls, det_ind]，shape=(M, 8)
        # 重建後會繼續流經下游 SGIE plate / SGIE num，三層 LPR 鏈不中斷
        for tr in tracks:
            x1, y1, x2, y2 = float(tr[0]), float(tr[1]), float(tr[2]), float(tr[3])
            tid = int(tr[4])
            conf = float(tr[5])
            cls = int(tr[6])

            new_obj = pyds.nvds_acquire_obj_meta_from_pool(batch_meta)
            if new_obj is None:
                # pool 滿了就跳過這個 track（極少發生）
                continue

            # unique_component_id=1 與 nvtracker 預設一致，
            # 讓下游 _process_tracked_frame 與 SGIE 都能正確辨識
            new_obj.unique_component_id = _UID_VEHICLE
            new_obj.class_id = cls
            new_obj.object_id = tid
            new_obj.confidence = conf
            new_obj.obj_label = CLASS_MAP.get(cls, f"Class_{cls}")

            # rect_params 用 BoxMOT 自己給的框
            r = new_obj.rect_params
            r.left = x1
            r.top = y1
            r.width = max(1.0, x2 - x1)
            r.height = max(1.0, y2 - y1)
            r.border_width = 4
            r.has_bg_color = 0
            r.border_color.set(*get_class_color(cls))   # 預設車種色，後面 _process 還會覆寫一次

            pyds.nvds_add_obj_meta_to_frame(frame_meta, new_obj, None)

        # 步驟 5: 走和 nvdcf 一樣的軌跡狀態邏輯（含車種截圖 fallback）
        _process_tracked_frame(gst_buffer, frame_meta, current_frame_objects, pad_index, cfg)

        l_frame = l_frame.next

    _post_frame_housekeeping(current_frame_objects)
    return Gst.PadProbeReturn.OK


# ==========================================
# 6. 車牌框擴張 (Plate Expansion)
# ==========================================

def expand_plate_probe(pad, info, u_data):
    """
    LPR 特有探針：掛在 sgie_plate.src

    把 SGIE plate 偵測到的車牌框上下左右各擴 10%，
    這樣下游 SGIE num 抓 ROI 做字元偵測時才有足夠邊界，
    不會因為車牌框太緊導致邊緣字元被切掉。
    車牌截圖（assemble_plate_probe）也是裁這個擴張後的框。

    處理流程：
    1. 走訪所有 obj_meta，只處理 unique_component_id=2（車牌）
    2. 框長寬各放大 20%（每邊各加 10%）
    3. 邊界限制：避免擴到畫面外
    4. 設定 OSD 樣式：綠色邊框、隱藏預設標籤（等 assemble_plate_probe 寫車牌字串）

    參數：
        pad, info, u_data: GStreamer probe 標準參數

    返回：
        Gst.PadProbeReturn.OK
    """
    gst_buffer = info.get_buffer()
    if not gst_buffer:
        return Gst.PadProbeReturn.OK

    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
    l_frame = batch_meta.frame_meta_list

    while l_frame is not None:
        try:
            frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
        except StopIteration:
            break

        frame_w = frame_meta.source_frame_width
        frame_h = frame_meta.source_frame_height

        l_obj = frame_meta.obj_meta_list
        while l_obj is not None:
            try:
                obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)
            except StopIteration:
                break

            # 只處理車牌物件（unique_component_id=2，來自 SGIE plate）
            if obj_meta.unique_component_id == _UID_PLATE:
                rect = obj_meta.rect_params
                w = rect.width
                h = rect.height

                dw = w * 0.10    # 水平各擴 10%
                dh = h * 0.10    # 垂直各擴 10%

                rect.left   -= dw
                rect.top    -= dh
                rect.width  += 2.0 * dw
                rect.height += 2.0 * dh

                # 邊界限制：避免超出畫面
                rect.left = max(0.0, rect.left)
                rect.top  = max(0.0, rect.top)
                if rect.left + rect.width > frame_w:
                    rect.width = frame_w - rect.left
                if rect.top + rect.height > frame_h:
                    rect.height = frame_h - rect.top

                # OSD 樣式：綠色邊框（辨識中），標籤等 assemble_plate_probe 寫
                rect.border_width = 3
                rect.border_color.set(0.0, 1.0, 0.0, 1.0)

                obj_meta.text_params.set_bg_clr = 0
                obj_meta.text_params.font_params.font_size = 0

            l_obj = l_obj.next
        l_frame = l_frame.next

    return Gst.PadProbeReturn.OK


# ==========================================
# 7. 車牌字串組裝 (Plate String Assembly)
# ==========================================

def _frame_chars_to_string(chars, char_nms_iou=0.5):
    """
    把一幀內歸屬同一車牌的字元清單，做 NMS + 由左到右排序 + 串成字串

    處理流程：
    1. 提取所有字元的座標、信心度、類別 ID
    2. 用 cv2.dnn.NMSBoxes 做 NMS（去掉重疊字元）
    3. 依水平中心點由左到右排序
    4. 透過 NUM_MAP 把類別 ID 轉成字元，拼成完整車牌字串

    參數：
        chars (list[dict]): 字元清單，每個 dict 含 x1/y1/x2/y2/score/cls_id
        char_nms_iou (float): NMS IoU 門檻

    返回：
        str: 組裝好的車牌字串（如 "ABC-1234"）；無字元時回傳空字串
    """
    if not chars:
        return ""

    # 步驟 1: 提取座標、信心度、類別 ID
    x1 = np.array([c["x1"] for c in chars])
    y1 = np.array([c["y1"] for c in chars])
    x2 = np.array([c["x2"] for c in chars])
    y2 = np.array([c["y2"] for c in chars])
    scores = np.array([c["score"] for c in chars])
    cls_ids = np.array([c["cls_id"] for c in chars])

    # 步驟 2: NMS 去重疊
    boxes_xywh = np.stack([x1, y1, x2 - x1, y2 - y1], axis=1).tolist()
    idxs = cv2.dnn.NMSBoxes(boxes_xywh, scores.tolist(), 0.0, char_nms_iou)
    if idxs is None or len(idxs) == 0:
        return ""

    idxs = np.array(idxs).flatten()
    fx1 = x1[idxs]
    fx2 = x2[idxs]
    fcls = cls_ids[idxs]

    # 步驟 3: 由左到右排序
    order = np.argsort((fx1 + fx2) / 2.0)

    # 步驟 4: 類別 ID → 字元，拼成字串
    return "".join(NUM_MAP.get(int(fcls[i]), "") for i in order)


def assemble_plate_probe(pad, info, u_data):
    """
    LPR 核心探針：掛在 sgie_num.src

    每幀組裝車牌字串、累積 plate_votes、更新 OSD、（啟用時）車牌+車種同幀截圖

    處理流程：
    1. 走訪 obj_meta 分流成「車輛框 vehicles」+「車牌框 plates」+「字元框 chars」
       （車輛框直接從本 frame 撈 uid=1，與車牌同幀同步，不跨探針傳遞）
    2. 字元 → 車輛配對：字元中心點落在車身內 + 重疊面積最大
    3. 車牌 → 車輛配對：同邏輯
    4. 對每台車組裝該幀車牌字串 → 加一票到 state["plate_votes"]
    5. 為每張車牌更新 OSD 文字；車牌框「面積」刷新歷史最大時，
       同一幀同時截車牌框 + 車輛框（聯集框 + 下緣 padding，確保車輛框必含完整車牌）

    參數：
        pad, info, u_data: GStreamer probe 標準參數

    返回：
        Gst.PadProbeReturn.OK
    """
    gst_buffer = info.get_buffer()
    if not gst_buffer:
        return Gst.PadProbeReturn.OK

    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
    l_frame = batch_meta.frame_meta_list

    while l_frame is not None:
        try:
            frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
        except StopIteration:
            break

        pad_idx = frame_meta.pad_index
        cfg = SOURCE_CONFIGS.get(pad_idx, {})
        save_ss = cfg.get("save_screenshot", False)
        frame_w = frame_meta.source_frame_width
        frame_h = frame_meta.source_frame_height

        # 步驟 1: 走訪 obj_meta，分流成 vehicles / plates / chars
        # ⭐ 車輛框直接從本 frame 撈（uid=1），與車牌、字元同一個 frame_meta，
        #    保證同幀同步，不依賴跨探針的 module 暫存（那在多 queue pipeline 不可靠）
        vehicles = {}   # v_id → {"obj": obj_meta, 車輛框座標}
        plates = []
        chars = []

        l_obj = frame_meta.obj_meta_list
        while l_obj is not None:
            try:
                obj = pyds.NvDsObjectMeta.cast(l_obj.data)
            except StopIteration:
                break

            uid = obj.unique_component_id
            r = obj.rect_params
            x1 = float(r.left)
            y1 = float(r.top)
            x2 = x1 + float(r.width)
            y2 = y1 + float(r.height)

            if uid == _UID_VEHICLE:
                v_id = obj.object_id
                if v_id != -1:
                    vehicles[v_id] = {
                        "obj": obj,
                        "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                    }
            elif uid == _UID_PLATE:
                plates.append({
                    "obj": obj,
                    "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                    "area": float(r.width) * float(r.height),   # 車牌框面積（選幀依據）
                })
            elif uid == _UID_CHAR:
                # 隱藏字元本身的 OSD（不顯示個別字元框）
                r.border_width = 0
                r.has_bg_color = 0
                txt = obj.text_params
                txt.set_bg_clr = 0
                txt.font_params.font_size = 0
                txt.display_text = ""

                chars.append({
                    "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                    "score": float(obj.confidence),
                    "cls_id": int(obj.class_id),
                })
            l_obj = l_obj.next

        # 把 vehicles 整理成 list 供配對用（用本 frame 的車輛框，非 track_history 的 last_v_box）
        active_vehicles = [
            {"v_id": v_id, "x1": v["x1"], "y1": v["y1"], "x2": v["x2"], "y2": v["y2"]}
            for v_id, v in vehicles.items()
        ]

        # 步驟 2: 字元 → 車輛配對（中心點在車身內 + 重疊面積最大）
        vehicle_chars = {}   # v_id → list of chars
        for c in chars:
            ccx = (c["x1"] + c["x2"]) / 2.0
            ccy = (c["y1"] + c["y2"]) / 2.0
            best_vid = None
            best_overlap = 0.0
            for v in active_vehicles:
                if not (v["x1"] <= ccx <= v["x2"] and v["y1"] <= ccy <= v["y2"]):
                    continue
                ix1 = max(c["x1"], v["x1"])
                iy1 = max(c["y1"], v["y1"])
                ix2 = min(c["x2"], v["x2"])
                iy2 = min(c["y2"], v["y2"])
                ov = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
                if ov > best_overlap:
                    best_overlap = ov
                    best_vid = v["v_id"]
            if best_vid is not None:
                vehicle_chars.setdefault(best_vid, []).append(c)

        # 步驟 3: 車牌框 → 車輛配對（同邏輯，用於 OSD 顯示與截圖時知道屬於哪台車）
        plate_to_vehicle = {}   # plates 內 index → v_id
        for pi, p in enumerate(plates):
            pcx = (p["x1"] + p["x2"]) / 2.0
            pcy = (p["y1"] + p["y2"]) / 2.0
            best_vid = None
            best_overlap = 0.0
            for v in active_vehicles:
                if not (v["x1"] <= pcx <= v["x2"] and v["y1"] <= pcy <= v["y2"]):
                    continue
                ix1 = max(p["x1"], v["x1"])
                iy1 = max(p["y1"], v["y1"])
                ix2 = min(p["x2"], v["x2"])
                iy2 = min(p["y2"], v["y2"])
                ov = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
                if ov > best_overlap:
                    best_overlap = ov
                    best_vid = v["v_id"]
            plate_to_vehicle[pi] = best_vid

        # 步驟 4: 對每台車組裝該幀車牌字串、累積 plate_votes
        for v_id, chars_for_v in vehicle_chars.items():
            v_key = (pad_idx, v_id)
            if v_key not in track_history:
                continue
            plate_str = _frame_chars_to_string(chars_for_v, char_nms_iou=0.5)
            if plate_str:
                state = track_history[v_key]
                if "plate_votes" not in state:
                    state["plate_votes"] = Counter()
                state["plate_votes"][plate_str] += 1

        # 步驟 5: 為每張車牌更新 OSD 文字 + 面積最大時同幀截車牌框與車輛框
        for pi, p in enumerate(plates):
            plate_obj = p["obj"]
            v_id = plate_to_vehicle.get(pi)

            plate_str = ""
            if v_id is not None and v_id in vehicle_chars:
                plate_str = _frame_chars_to_string(vehicle_chars[v_id], char_nms_iou=0.5)

            # 5-1: OSD 文字（顯示該幀辨識到的車牌號碼）
            txt = plate_obj.text_params
            r = plate_obj.rect_params

            if plate_str:
                txt.display_text = plate_str
                txt.font_params.font_name = "Serif Bold"
                txt.font_params.font_size = 13
                txt.font_params.font_color.set(1.0, 1.0, 1.0, 1.0)   # 白字
                txt.set_bg_clr = 1
                txt.text_bg_clr.set(0.0, 0.0, 0.0, 1.0)              # 黑底
                txt.x_offset = int(r.left)
                txt.y_offset = max(0, int(r.top + r.height))
            else:
                # 沒辨識到車牌就不顯示文字
                txt.set_bg_clr = 0
                txt.font_params.font_size = 0

            # 5-2: 面積最大選幀截圖 — 該車的車牌框面積刷新歷史最大時：
            #      車牌框 → 直接截
            #      車輛框 → 用「車輛框 ∪ 車牌框」聯集 + 下緣加 padding 後截，
            #              確保車輛框照片含完整車牌（補 DS 車輛框下緣常差幾 px）
            #      車輛框 obj 從本 frame 的 vehicles 取，與車牌同幀，必定對得上
            #      只在成功組出車牌字串的幀才納入，避免拿到沒辨識的大框
            if save_ss and v_id is not None and plate_str:
                v_key = (pad_idx, v_id)
                if v_key in track_history:
                    state = track_history[v_key]
                    plate_area = p["area"]
                    if plate_area > state.get("best_plate_area", 0):
                        # 截車牌框
                        plate_jpg = _encode_crop_to_bytes(
                            gst_buffer, plate_obj, frame_meta,
                            tmp_name=f"{pad_idx}_{v_id}_plate.jpg",
                        )

                        # 同幀截車輛框（聯集車牌框 + 下緣 padding）
                        class_jpg = None
                        veh_entry = vehicles.get(v_id)
                        if veh_entry is not None:
                            veh_obj = veh_entry["obj"]
                            vr = veh_obj.rect_params
                            # 暫存車輛原始框（截完還原，不影響 OSD）
                            o_left, o_top, o_w, o_h = vr.left, vr.top, vr.width, vr.height

                            # 車牌框高度，用來算下緣額外 padding
                            plate_h = p["y2"] - p["y1"]
                            bottom_pad = plate_h * _PLATE_BOTTOM_PAD_RATIO

                            # 聯集框：車輛框與車牌框的最小外接矩形，下緣再加 padding
                            u_x1 = min(float(o_left), p["x1"])
                            u_y1 = min(float(o_top), p["y1"])
                            u_x2 = max(float(o_left + o_w), p["x2"])
                            u_y2 = max(float(o_top + o_h), p["y2"]) + bottom_pad

                            # 邊界裁切（避免超出畫面）
                            u_x1 = max(0.0, u_x1)
                            u_y1 = max(0.0, u_y1)
                            u_x2 = min(float(frame_w), u_x2)
                            u_y2 = min(float(frame_h), u_y2)

                            # 套用聯集框
                            vr.left   = u_x1
                            vr.top    = u_y1
                            vr.width  = max(1.0, u_x2 - u_x1)
                            vr.height = max(1.0, u_y2 - u_y1)

                            # 截圖
                            class_jpg = _encode_crop_to_bytes(
                                gst_buffer, veh_obj, frame_meta,
                                tmp_name=f"{pad_idx}_{v_id}_class.jpg",
                            )

                            # 還原車輛原始框
                            vr.left, vr.top, vr.width, vr.height = o_left, o_top, o_w, o_h

                        # 車牌圖成功才更新（確保面積高水位與圖一致）
                        if plate_jpg:
                            state["best_plate_jpg"] = plate_jpg
                            state["best_plate_area"] = plate_area
                            if class_jpg:
                                state["best_class_jpg"] = class_jpg

        l_frame = l_frame.next

    return Gst.PadProbeReturn.OK


# ==========================================
# 8. 每路畫面 OSD 探針 (Per-Cam FPS Overlay)
# ==========================================

def per_cam_osd_probe(pad, info, pad_index):
    """
    每路 nvosd.sink 上的 OSD 探針：左上角畫即時 FPS 文字

    兩種追蹤器模式共用，由該路 cam 的 display.show_fps_overlay 決定是否顯示

    參數：
        pad, info: GStreamer probe 標準參數
        pad_index (int): 哪一路 cam（由 main.py 用 add_probe 帶入）

    返回：
        Gst.PadProbeReturn.OK
    """
    gst_buffer = info.get_buffer()
    if not gst_buffer:
        return Gst.PadProbeReturn.OK

    cfg = SOURCE_CONFIGS.get(pad_index)
    if not cfg:
        return Gst.PadProbeReturn.OK

    show_fps = cfg.get("display", {}).get("show_fps_overlay", True)

    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
    l_frame = batch_meta.frame_meta_list
    while l_frame is not None:
        try:
            frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
        except StopIteration:
            break

        display_meta = pyds.nvds_acquire_display_meta_from_pool(batch_meta)
        display_meta.num_labels = 0
        display_meta.num_lines = 0
        display_meta.num_rects = 0
        display_meta.num_circles = 0

        if show_fps and pad_index in fps_streams:
            display_meta.num_labels = 1
            txt_params = display_meta.text_params[0]
            txt_params.display_text = f"FPS: {fps_streams[pad_index]['current_fps']:.1f}"
            txt_params.x_offset = 5
            txt_params.y_offset = 5
            txt_params.font_params.font_name = "Serif Bold"
            txt_params.font_params.font_size = 25
            txt_params.font_params.font_color.set(0.0, 1.0, 0.0, 1.0)
            txt_params.set_bg_clr = 1
            txt_params.text_bg_clr.set(0.0, 0.0, 0.0, 0.8)

        pyds.nvds_add_display_meta_to_frame(frame_meta, display_meta)
        l_frame = l_frame.next

    return Gst.PadProbeReturn.OK