#!/usr/bin/env python3
"""
DeepStream 7.1 車牌辨識 LPR 主程式（三層架構）

主要功能：
1. 三層推論架構：PGIE 車輛 → SGIE 車牌 → SGIE 字元
2. 依 TRACKER_MODE 動態組裝 pipeline：
   - nvdcf  → PGIE → nvtracker → SGIE plate → SGIE num → analytics
   - BoxMOT → PGIE → SGIE plate → SGIE num → analytics（跳過 nvtracker）
3. 條件式掛載追蹤探針：
   - nvdcf  → tracker_src_pad_buffer_probe 掛在 tracker.src
   - BoxMOT → boxmot_pgie_src_probe 掛在 pgie.src
4. LPR 三層探針（兩種模式共用）：
   - expand_plate_probe   → sgie_plate.src
   - assemble_plate_probe → sgie_num.src
5. 截圖功能：任一 cam 啟用 save_screenshot 時建立 Object Encoder context，
   注入 probes 後在得票刷新時做 GPU 裁切編碼，結束前銷毀 context
6. 多路 cam 各自的下游分支：本地預覽 / 影片存檔 / RTSP 推流
7. RTSP server：把每路 cam 的 udpsink 註冊成獨立 mount_path
8. 安全結束機制：Q 鍵 / Ctrl+C 觸發 EOS，等影片封裝完成才退出
"""

import sys
import time
import select
import termios
import tty

import gi
gi.require_version('Gst', '1.0')
gi.require_version('GstRtspServer', '1.0')
from gi.repository import GLib, Gst, GstRtspServer

import pyds

from logic.color import load_labels, CLASS_MAP
from logic.config import (
    SOURCE_CONFIGS, INFER_CONFIG, TRACKER_CONFIG,
    PREPROCESS_CONFIG, ANALYTICS_CONFIG,
    INFER_SEC_PLATE_CONFIG, INFER_SEC_NUM_CONFIG,
    TRACKER_MODE, BOXMOT_TRACKER_CONFIG,
)
from logic.state_db import initialize_state_managers, force_finalize_all
from logic.pipeline import (
    cb_newpad, cb_source_setup, make_elm,
    _build_display_sink, setup_cam_branch,
)
from logic.probes import (
    tracker_src_pad_buffer_probe,
    boxmot_pgie_src_probe,
    expand_plate_probe,
    assemble_plate_probe,
    per_cam_osd_probe,
    set_obj_enc_context,
)


# ==========================================
# 1. 全域狀態 (Global State)
# ==========================================

g_loop          = None     # GLib 主迴圈
g_pipeline      = None     # GStreamer 主 pipeline
g_eos_triggered = False    # EOS 是否已發送（避免重複觸發）
g_rtsp_server   = None     # RTSP server 引用（持有避免被 GC）
g_obj_enc_ctx   = None     # Object Encoder context（截圖用，結束前銷毀）


# ==========================================
# 2. 結束與訊息處理 (Lifecycle Callbacks)
# ==========================================

def force_quit_loop():
    """
    EOS 超時的強制退出 fallback

    用於：發送 EOS 後等待 8 秒影片仍未封裝完成時強制 quit，
          避免無限卡住

    返回：
        bool: False 讓 GLib timeout 不再重複觸發
    """
    global g_loop
    print("\n[WARNING] 等待影片封裝逾時，強制退出管線！")
    if g_loop and g_loop.is_running():
        g_loop.quit()
    return False


def keyboard_cb(fd, condition):
    """
    終端機按鍵處理：按 Q 觸發 EOS 安全退出

    處理流程：
    1. 讀一個字元
    2. 若是 q/Q 且尚未觸發 EOS → 發送 EOS event + 啟動 8 秒 timeout 保險

    參數：
        fd, condition: GLib io_add_watch 標準參數

    返回：
        bool: True 持續監聽；False 移除監聽（已觸發 EOS 後）
    """
    global g_eos_triggered, g_pipeline, g_loop

    ch = sys.stdin.read(1)
    if ch in ('q', 'Q') and not g_eos_triggered:
        g_eos_triggered = True
        print("\n[INFO] 收到 'Q' 鍵，正在安全發送 EOS 訊號 (等待影片寫入)...")
        if g_pipeline:
            g_pipeline.send_event(Gst.Event.new_eos())
            GLib.timeout_add_seconds(8, force_quit_loop)
        return False
    return True


def bus_call(bus, message, loop):
    """
    GStreamer bus 訊息處理

    處理策略：
        EOS               → 正常結束主迴圈
        RTSP 相關錯誤     → 印 WARNING 但不退出，等待自動重連
        其它嚴重錯誤      → 印 ERROR 並退出

    參數：
        bus, message: GStreamer 標準參數
        loop (GLib.MainLoop): 要操作的主迴圈

    返回：
        bool: True 繼續接收訊息
    """
    t = message.type

    if t == Gst.MessageType.EOS:
        print("[INFO] 影像串流結束 (EOS 處理完畢)，準備安全退出...")
        loop.quit()

    elif t == Gst.MessageType.ERROR:
        err, debug = message.parse_error()
        err_msg = str(err).lower()

        # RTSP 訊號錯誤不退出，等 uridecodebin 自動重連
        if ("rtsp" in err_msg or "timeout" in err_msg
                or "resource not found" in err_msg or "could not read" in err_msg):
            print(f"[WARNING] RTSP 串流不穩或中斷: {err}。系統保持運行，等待自動重連...")
        else:
            print(f"[ERROR] 嚴重管線錯誤: {err}: {debug}")
            loop.quit()

    return True


# ==========================================
# 3. Pipeline 輔助 (Pipeline Helpers)
# ==========================================

def _enlarge_queue(q, max_buffers=400):
    """
    放寬 queue 容量，避免下游處理偶發較慢時被反壓

    參數：
        q (Gst.Element): queue 元件
        max_buffers (int): 緩衝區最大數量
    """
    q.set_property("max-size-buffers", max_buffers)
    q.set_property("max-size-bytes", 0)
    q.set_property("max-size-time", 0)


def _start_rtsp_server(rtsp_routes):
    """
    啟動 GstRtspServer 並把每路 cam 的 udpsink 端口註冊成 mount_path

    處理流程：
    1. 依 port 分組（同 port 不同 mount_path 共用一台 server）
    2. 為每個 port 建一個 RTSPServer
    3. 為每路 cam 建一個 RTSPMediaFactory，套用對應的 udpsrc + rtp{h264,h265}pay

    參數：
        rtsp_routes (list[dict]): 每路 cam 的推流設定
            {pad_index, udp_port, port, mount_path, encoder}

    返回：
        list | None: 啟動的 RTSPServer 列表（None 表示無推流需求）
    """
    if not rtsp_routes:
        return None

    # 步驟 1: 依 port 分組
    routes_by_port = {}
    for r in rtsp_routes:
        routes_by_port.setdefault(r["port"], []).append(r)

    # 步驟 2: 每個 port 啟一台 RTSP server
    servers = []
    for port, routes in routes_by_port.items():
        server = GstRtspServer.RTSPServer()
        server.set_service(str(port))
        mounts = server.get_mount_points()

        # 步驟 3: 每路 cam 註冊一個 mount_path
        for r in routes:
            udp_port   = r["udp_port"]
            encoder    = r["encoder"]
            mount_path = "/" + r["mount_path"].lstrip("/")

            # encoding-name 必須對齊客戶端，否則 SDP 不匹配連不上
            enc_name = "H265" if encoder == "h265" else "H264"

            launch_str = (
                f"( udpsrc port={udp_port} caps=\"application/x-rtp, "
                f"media=video, clock-rate=90000, encoding-name={enc_name}, payload=96\" "
                f"! rtp{encoder}depay ! rtp{encoder}pay name=pay0 pt=96 )"
            )

            factory = GstRtspServer.RTSPMediaFactory()
            factory.set_launch(launch_str)
            factory.set_shared(True)   # 多客戶端可同時連同一 mount
            mounts.add_factory(mount_path, factory)

            print(f"[INFO] RTSP 推流註冊: rtsp://<本機IP>:{port}{mount_path} "
                  f"(encoder={encoder}, udp_port={udp_port})")

        server.attach(None)
        servers.append(server)

    return servers


def _init_obj_encoder_if_needed():
    """
    若任一路 cam 啟用 save_screenshot，建立 Object Encoder context 並注入 probes

    處理流程：
    1. 掃描 SOURCE_CONFIGS，檢查是否有任何 cam 開啟截圖
    2. 有 → 建立 context（gpu_id=0）、呼叫 set_obj_enc_context 注入
       無 → 注入 None（probes 截圖邏輯會自動 no-op）

    返回：
        Object Encoder context | None
    """
    need_ss = any(cfg.get("save_screenshot", False) for cfg in SOURCE_CONFIGS.values())

    if not need_ss:
        print("[INFO] 無 cam 啟用截圖，跳過 Object Encoder 初始化")
        set_obj_enc_context(None)
        return None

    try:
        ctx = pyds.nvds_obj_enc_create_context(0)   # gpu_id=0
        print("[INFO] Object Encoder context 建立成功（截圖功能啟用）")
        set_obj_enc_context(ctx)
        return ctx
    except Exception as e:
        print(f"[WARNING] Object Encoder context 建立失敗，截圖功能停用：{e}")
        set_obj_enc_context(None)
        return None


# ==========================================
# 4. 主程式 (Main)
# ==========================================

def main():
    """
    主程式進入點

    處理流程：
    1. 印追蹤器模式並（BoxMOT 模式）初始化 tracker instance
    2. 初始化 Object Encoder（若有 cam 啟用截圖）
    3. 建立 GStreamer pipeline 與 streammux
    4. 為每路 cam 建立 uridecodebin 來源
    5. 建立共用推論元件（preprocess / pgie / sgie_plate / sgie_num / analytics）
    6. 依 TRACKER_MODE 條件式建立 nvtracker 與連結 pipeline 中段
    7. 條件式掛載追蹤探針 + 永遠掛 expand_plate_probe / assemble_plate_probe
    8. 建立 demux 並為每路 cam 組下游分支
    9. 啟動 RTSP server（若有 cam 啟用推流）
    10. 進入 GLib 主迴圈，等待 Q 鍵或 EOS 退出
    """
    global g_loop, g_pipeline, g_eos_triggered, g_rtsp_server, g_obj_enc_ctx

    # ---- 步驟 1: 印追蹤器模式 + (BoxMOT) 初始化 tracker instance ----
    if TRACKER_MODE == "nvdcf":
        print("[INFO] 初始化 DeepStream LPR 三層架構：PGIE → NvDCF → SGIE plate → SGIE num → Analytics")
    else:
        print(f"[INFO] 初始化 DeepStream LPR 三層架構：PGIE → {TRACKER_MODE} (BoxMOT) → SGIE plate → SGIE num → Analytics")
        print(f"[INFO]   pipeline 將跳過 nvtracker，追蹤交由 pgie.src 上的 BoxMOT 探針處理")

        from logic.boxmot_adapter import initialize_boxmot_trackers
        initialize_boxmot_trackers()

    # ---- 步驟 2: 初始化 Object Encoder（截圖用） ----
    Gst.init(None)
    g_obj_enc_ctx = _init_obj_encoder_if_needed()

    # ---- 步驟 3: 建立 pipeline 與 streammux ----
    g_pipeline = Gst.Pipeline.new("lpr-pipeline")

    num_sources = len(SOURCE_CONFIGS)

    # 任一 cam 開啟 show_window 就建立本地預覽 sink
    show_window = any(
        cfg.get("display", {}).get("show_window", True)
        for cfg in SOURCE_CONFIGS.values()
    )

    streammux = make_elm("nvstreammux", "Stream-muxer")
    streammux.set_property("width", 1920)
    streammux.set_property("height", 1080)
    streammux.set_property("batch-size", num_sources)
    streammux.set_property("batched-push-timeout", 70000)
    streammux.set_property("live-source", 1)
    streammux.set_property("nvbuf-memory-type", 0)
    g_pipeline.add(streammux)

    # ---- 步驟 4: 每路 cam 建一個 uridecodebin ----
    for pad_index, cfg in SOURCE_CONFIGS.items():
        source = make_elm("uridecodebin", f"uri-decode-bin-{pad_index}")
        source.set_property("uri", cfg["source"])
        source.connect("pad-added", cb_newpad, {"streammux": streammux, "pad_index": pad_index})
        source.connect("source-setup", cb_source_setup, None)
        g_pipeline.add(source)

    # ---- 步驟 5: 共用推論元件（含 LPR 三層的 SGIE plate / num） ----
    q1           = make_elm("queue", "q1")
    q2           = make_elm("queue", "q2")
    q3           = make_elm("queue", "q3")
    q_sgie_plate = make_elm("queue", "q_sgie_plate")
    q_sgie_num   = make_elm("queue", "q_sgie_num")
    q_analytics  = make_elm("queue", "q_analytics")
    q4           = make_elm("queue", "q4")
    _enlarge_queue(q_sgie_plate, max_buffers=400)
    _enlarge_queue(q_sgie_num,   max_buffers=400)
    _enlarge_queue(q_analytics,  max_buffers=200)

    preprocess = make_elm("nvdspreprocess", "preprocess")
    preprocess.set_property("config-file", PREPROCESS_CONFIG)

    pgie = make_elm("nvinfer", "primary-inference")
    pgie.set_property("config-file-path", INFER_CONFIG)
    pgie.set_property("input-tensor-meta", True)

    sgie_plate = make_elm("nvinfer", "secondary-plate")
    sgie_plate.set_property("config-file-path", INFER_SEC_PLATE_CONFIG)

    sgie_num = make_elm("nvinfer", "secondary-num")
    sgie_num.set_property("config-file-path", INFER_SEC_NUM_CONFIG)

    analytics = make_elm("nvdsanalytics", "analytics")
    analytics.set_property("config-file", ANALYTICS_CONFIG)

    # ---- 步驟 6: 依 TRACKER_MODE 條件式建立 nvtracker + 連結 pipeline 中段 ----
    tracker = None
    if TRACKER_MODE == "nvdcf":
        tracker = make_elm("nvtracker", "tracker")
        tracker.set_property("ll-config-file", TRACKER_CONFIG)
        tracker.set_property(
            "ll-lib-file",
            "/opt/nvidia/deepstream/deepstream/lib/libnvds_nvmultiobjecttracker.so",
        )
        tracker.set_property("tracker-width", 640)
        tracker.set_property("tracker-height", 384)

    pipeline_elements = [
        q1, preprocess, q2, pgie, q3,
        q_sgie_plate, sgie_plate, q_sgie_num, sgie_num,
        q_analytics, analytics, q4,
    ]
    if tracker is not None:
        pipeline_elements.append(tracker)
    for elm in pipeline_elements:
        g_pipeline.add(elm)

    # 共用前段：streammux → q1 → preprocess → q2 → pgie → q3
    streammux.link(q1)
    q1.link(preprocess)
    preprocess.link(q2)
    q2.link(pgie)
    pgie.link(q3)

    if TRACKER_MODE == "nvdcf":
        # 原流程：pgie → q3 → nvtracker → q_sgie_plate → sgie_plate → q_sgie_num → sgie_num → q_analytics
        q3.link(tracker)
        tracker.link(q_sgie_plate)
        print("[INFO] Pipeline 中段：pgie → q3 → nvtracker → q_sgie_plate → sgie_plate "
              "→ q_sgie_num → sgie_num → q_analytics → analytics → q4")
    else:
        # BoxMOT 流程：跳過 nvtracker，pgie.src probe 重建 obj_meta 後直接餵給 SGIE
        q3.link(q_sgie_plate)
        print(f"[INFO] Pipeline 中段：pgie → q3 → q_sgie_plate → sgie_plate → q_sgie_num → sgie_num "
              f"→ q_analytics → analytics → q4  ({TRACKER_MODE} 模式，已跳過 nvtracker)")

    q_sgie_plate.link(sgie_plate)
    sgie_plate.link(q_sgie_num)
    q_sgie_num.link(sgie_num)
    sgie_num.link(q_analytics)
    q_analytics.link(analytics)
    analytics.link(q4)

    # ---- 步驟 7: 條件式掛載追蹤探針 + 永遠掛 LPR 三層探針 ----

    # 7-1: 追蹤探針（二選一）
    if TRACKER_MODE == "nvdcf":
        tracker.get_static_pad("src").add_probe(
            Gst.PadProbeType.BUFFER, tracker_src_pad_buffer_probe, 0
        )
        print("[INFO] 已掛載探針：tracker_src_pad_buffer_probe → tracker.src")
    else:
        pgie.get_static_pad("src").add_probe(
            Gst.PadProbeType.BUFFER, boxmot_pgie_src_probe, 0
        )
        print(f"[INFO] 已掛載探針：boxmot_pgie_src_probe → pgie.src ({TRACKER_MODE})")

    # 7-2: LPR 三層探針（永遠掛，兩種追蹤模式都需要）
    sgie_plate.get_static_pad("src").add_probe(
        Gst.PadProbeType.BUFFER, expand_plate_probe, 0
    )
    print("[INFO] 已掛載探針：expand_plate_probe → sgie_plate.src")

    sgie_num.get_static_pad("src").add_probe(
        Gst.PadProbeType.BUFFER, assemble_plate_probe, 0
    )
    print("[INFO] 已掛載探針：assemble_plate_probe → sgie_num.src")

    # ---- 步驟 8: demux 並為每路 cam 組下游分支 ----
    demux = make_elm("nvstreamdemux", "demuxer")
    g_pipeline.add(demux)
    q4.link(demux)

    display_streammux = _build_display_sink(g_pipeline, num_sources) if show_window else None

    # 收集所有啟用 RTSP 推流的路，等下批次註冊到 RTSP server
    rtsp_routes = []
    for pad_index, cfg in SOURCE_CONFIGS.items():
        udp_port = setup_cam_branch(
            g_pipeline, pad_index, cfg, demux, display_streammux, per_cam_osd_probe
        )
        if udp_port is not None:
            rtsp_routes.append({
                "pad_index":  pad_index,
                "udp_port":   udp_port,
                "port":       cfg["rtsp_push"]["port"],
                "mount_path": cfg["rtsp_push"]["mount_path"],
                "encoder":    cfg["rtsp_push"]["encoder"],
            })

    # ---- 步驟 9: 啟動 RTSP server ----
    if rtsp_routes:
        g_rtsp_server = _start_rtsp_server(rtsp_routes)
        print(f"[INFO] 共 {len(rtsp_routes)} 條 RTSP 推流就緒")
    else:
        print("[INFO] 無 cam 啟用 RTSP 推流，跳過 RTSP server 啟動")

    # ---- 步驟 10: 鍵盤監聽 + 主迴圈 ----
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        GLib.io_add_watch(fd, GLib.PRIORITY_DEFAULT, GLib.IOCondition.IN, keyboard_cb)
        print("\n[INFO] 💡 提示：在終端機按下 'q' 鍵即可優雅退出並存檔...\n")

        g_loop = GLib.MainLoop()
        bus = g_pipeline.get_bus()
        bus.add_signal_watch()
        bus.connect("message", bus_call, g_loop)

        g_pipeline.set_state(Gst.State.PLAYING)
        g_loop.run()

    except KeyboardInterrupt:
        # Ctrl+C：發送 EOS 後再跑一次迴圈等影片寫完
        print("\n[INFO] 收到 Ctrl+C，準備發送 EOS...")
        if not g_eos_triggered:
            g_eos_triggered = True
            g_pipeline.send_event(Gst.Event.new_eos())
            GLib.timeout_add_seconds(8, force_quit_loop)
        try:
            g_loop.run()
        except KeyboardInterrupt:
            print("\n[INFO] 強制終止！")
            pass

    finally:
        # 還原終端機設定、flush DB（含截圖寫檔）、停 pipeline、銷毀 Object Encoder
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        force_finalize_all()
        g_pipeline.set_state(Gst.State.NULL)

        # 截圖 context 在結算寫檔完成後才銷毀
        if g_obj_enc_ctx is not None:
            try:
                pyds.nvds_obj_enc_destroy_context(g_obj_enc_ctx)
                print("[INFO] Object Encoder context 已銷毀")
            except Exception as e:
                print(f"[WARNING] 銷毀 Object Encoder context 失敗：{e}")


if __name__ == '__main__':
    initialize_state_managers()
    main()