# logic/pipeline.py
"""
pipeline.py
-----------
GStreamer pipeline 元件建構與分支邏輯。

對每路 cam，setup_cam_branch 依以下 YAML 欄位動態組裝分支：
    - output.save_output_video → 寫檔分支
    - display.show_window      → 本地預覽分支（合併到 tile 視窗）
    - rtsp_push.enable         → RTSP 推流分支（中央主機可遠端接收）

三個分支彼此獨立，任意組合（皆開、皆關、只開其中一個）。

Tile 佈局：
    自動依 cam 數量計算 rows × cols，每格保持 16:9。
"""

import sys
from gi.repository import Gst
import os
from logic.config import SOURCE_CONFIGS


def cb_newpad(decodebin, decoder_src_pad, data):
    """uridecodebin 對外動態 pad 出現時，鏈到 streammux 的對應 sink_N。"""
    caps = decoder_src_pad.get_current_caps()
    if not caps:
        caps = decoder_src_pad.query_caps()

    if caps.get_structure(0).get_name().find("video") != -1:
        sinkpad = data["streammux"].get_request_pad(f"sink_{data['pad_index']}")
        if not sinkpad.is_linked():
            decoder_src_pad.link(sinkpad)


def cb_source_setup(decodebin, source_element, user_data):
    """RTSP 來源出現時自動注入防斷線參數。"""
    if source_element.get_name().startswith("rtspsrc"):
        print(f"[INFO] 偵測到 RTSP 來源，注入防斷線參數: {source_element.get_name()}")
        source_element.set_property("protocols", 4)
        source_element.set_property("latency", 200)
        source_element.set_property("timeout", 5000000)
        source_element.set_property("drop-on-latency", True)


def make_elm(gst_type, name):
    """建立 GStreamer 元件，失敗則整支程式退出。"""
    elm = Gst.ElementFactory.make(gst_type, name)
    if not elm:
        sys.exit(f"[ERROR] 無法建立 element: {gst_type} ({name})")
    return elm


# ==========================================
# 自動依 cam 數決定 tile 佈局
# ==========================================
def _get_tile_layout(num_sources):
    """
    根據 cam 總數回傳 (rows, cols, total_width, total_height)。
    每格保持 16:9，視窗總寬固定 1920，總高隨列數變動。

    對應表：
        1     → 1×1  (1920×1080)
        2     → 1×2  (1920× 540)
        3~4   → 2×2  (1920×1080)
        5~6   → 2×3  (1920× 720)
        7~9   → 3×3  (1920×1080)
        >9    → 4×4  (1920×1080)
    """
    if num_sources == 1:
        rows, cols = 1, 1
    elif num_sources == 2:
        rows, cols = 1, 2
    elif num_sources <= 4:
        rows, cols = 2, 2
    elif num_sources <= 6:
        rows, cols = 2, 3
    elif num_sources <= 9:
        rows, cols = 3, 3
    else:
        rows, cols = 4, 4

    total_width = 1920
    cell_w = total_width // cols
    cell_h = int(cell_w * 9 / 16)
    total_height = cell_h * rows
    return rows, cols, total_width, total_height


# ==========================================
# 寫檔分支：file 來源
# ==========================================
def _build_save_branch_for_file(pipeline, pad_index, video_path, source_fps):
    """
    建立寫檔分支(本地檔案來源版)：
        nvvideoconvert → videorate → capsfilter(NV12, framerate=N/1)
        → nvv4l2h264enc → h264parse → qtmux → filesink

    ⭐ 修復「Buffer has no PTS」錯誤：
        - 插入 videorate 並用 capsfilter 明確指定 framerate，
          videorate 會以固定 framerate 重新產生 PTS，避免下游 qtmux 拒收。
        - qtmux 加 dts-method=1（從 PTS 推 DTS）作為雙保險。

    回傳起點 element(用於上游 link)。
    """
    i = pad_index

    nvvidconv_s = make_elm("nvvideoconvert", f"convertor-save-{i}")
    nvvidconv_s.set_property("nvbuf-memory-type", 0)

    # ⭐ 新增：videorate 確保 framerate / PTS 規律
    videorate = make_elm("videorate", f"videorate-save-{i}")

    cap_filter = make_elm("capsfilter", f"cap-filter-save-{i}")
    fps_int = int(round(source_fps))
    if fps_int <= 0:
        fps_int = 30
    caps = Gst.Caps.from_string(
        f"video/x-raw(memory:NVMM), format=NV12, framerate={fps_int}/1"
    )
    cap_filter.set_property("caps", caps)

    encoder = make_elm("nvv4l2h264enc", f"encoder-{i}")
    encoder.set_property("bitrate", 4000000)
    encoder.set_property("preset-level", 1)
    encoder.set_property("insert-sps-pps", 1)
    encoder.set_property("profile", 0)
    encoder.set_property("maxperf-enable", 1)
    encoder.set_property("iframeinterval", fps_int)

    parser = make_elm("h264parse", f"h264-parser-{i}")

    muxer = make_elm("qtmux", f"muxer-{i}")
    # ⭐ 新增：DTS 由 PTS 推導(ASC)，避免 qtmux 對缺失 DTS 過度敏感
    muxer.set_property("dts-method", 1)

    filesink = make_elm("filesink", f"filesink-{i}")
    filesink.set_property("location", video_path)
    filesink.set_property("async", False)
    filesink.set_property("sync", False)

    for elm in [nvvidconv_s, videorate, cap_filter, encoder, parser, muxer, filesink]:
        pipeline.add(elm)

    nvvidconv_s.link(videorate)
    videorate.link(cap_filter)
    cap_filter.link(encoder)
    encoder.link(parser)
    parser.link(muxer)
    muxer.link(filesink)

    return nvvidconv_s


# ==========================================
# 寫檔分支：RTSP 來源（需要 videorate 把 FPS 固定）
# ==========================================
def _build_save_branch_for_rtsp(pipeline, pad_index, video_path, source_fps):
    """建立寫檔分支（RTSP 串流版），插入 videorate 穩定 FPS。"""
    i = pad_index

    nvvidconv_s = make_elm("nvvideoconvert", f"convertor-save-{i}")
    nvvidconv_s.set_property("nvbuf-memory-type", 0)

    videorate = make_elm("videorate", f"videorate-{i}")
    cap_filter = make_elm("capsfilter", f"cap-filter-save-{i}")

    fps_int = int(round(source_fps))
    if fps_int <= 0:
        fps_int = 30

    caps = Gst.Caps.from_string(f"video/x-raw(memory:NVMM), format=NV12, framerate={fps_int}/1")
    cap_filter.set_property("caps", caps)

    encoder = make_elm("nvv4l2h264enc", f"encoder-{i}")
    encoder.set_property("bitrate", 4000000)
    encoder.set_property("preset-level", 1)
    encoder.set_property("insert-sps-pps", 1)
    encoder.set_property("profile", 0)
    encoder.set_property("maxperf-enable", 1)
    encoder.set_property("iframeinterval", fps_int)

    parser = make_elm("h264parse", f"h264-parser-{i}")
    muxer = make_elm("qtmux", f"muxer-{i}")
    muxer.set_property("dts-method", 1)
    filesink = make_elm("filesink", f"filesink-{i}")
    filesink.set_property("location", video_path)
    filesink.set_property("async", False)
    filesink.set_property("sync", False)

    for elm in [nvvidconv_s, videorate, cap_filter, encoder, parser, muxer, filesink]:
        pipeline.add(elm)

    nvvidconv_s.link(videorate)
    videorate.link(cap_filter)
    cap_filter.link(encoder)
    encoder.link(parser)
    parser.link(muxer)
    muxer.link(filesink)

    return nvvidconv_s


# ==========================================
# ⭐ RTSP 推流分支（選項 A：每路獨立）
# ==========================================
def _build_rtsp_push_branch(pipeline, pad_index, rtsp_cfg):
    """
    建立 RTSP 推流分支。輸出 RTP/UDP 封包到 localhost 的隨機 port，
    再由 GstRtspServer（在 main.py 內啟動）對外提供 RTSP 服務。

    分支結構：
        nvvideoconvert → capsfilter(NV12) → nvv4l2{h264/h265}enc
        → h{264/5}parse → rtp{264/5}pay → udpsink(host=127.0.0.1, port=N)

    參數：
        pipeline   : GstPipeline 物件
        pad_index  : 第幾路 cam（0, 1, ...）
        rtsp_cfg   : 從 YAML 的 rtsp_push 區塊解析後的 dict
                     必含 enable / port / mount_path / bitrate / encoder

    回傳：
        起點 element (nvvideoconvert)，由呼叫端 link 上游；
        以及內部使用的 udp port（給 main.py 的 RTSP server 設定 SDP 用）。
    """
    i = pad_index
    bitrate = rtsp_cfg["bitrate"]
    encoder_type = rtsp_cfg["encoder"]  # "h264" 或 "h265"

    # 為了讓多路推流共用同一個 RTSP server port（例如 8554），
    # 每路在內部用一個獨立的 UDP loopback port 串資料給 server。
    # 規則：5400 + pad_index（5400, 5401, 5402...）足夠分配，不會與常用 port 衝突。
    udp_port = 5400 + i

    # === 元件建立 ===
    nvvidconv_r = make_elm("nvvideoconvert", f"convertor-rtsp-{i}")
    nvvidconv_r.set_property("nvbuf-memory-type", 0)

    cap_filter = make_elm("capsfilter", f"cap-filter-rtsp-{i}")
    cap_filter.set_property("caps", Gst.Caps.from_string("video/x-raw(memory:NVMM), format=NV12"))

    if encoder_type == "h265":
        encoder = make_elm("nvv4l2h265enc", f"encoder-rtsp-{i}")
        parser = make_elm("h265parse", f"parser-rtsp-{i}")
        rtp_pay = make_elm("rtph265pay", f"rtppay-{i}")
        rtp_pay.set_property("pt", 96)
    else:  # h264 預設
        encoder = make_elm("nvv4l2h264enc", f"encoder-rtsp-{i}")
        parser = make_elm("h264parse", f"parser-rtsp-{i}")
        rtp_pay = make_elm("rtph264pay", f"rtppay-{i}")
        rtp_pay.set_property("pt", 96)

    encoder.set_property("bitrate", bitrate)
    encoder.set_property("preset-level", 1)
    encoder.set_property("insert-sps-pps", 1)
    encoder.set_property("profile", 0)
    encoder.set_property("maxperf-enable", 1)
    # 推流要低延遲，iframe 間隔短一點（每秒至少一個 keyframe）
    encoder.set_property("iframeinterval", 30)

    rtp_pay.set_property("config-interval", 1)  # SPS/PPS 每秒重送一次（中途接入也能解碼）

    udp_sink = make_elm("udpsink", f"udpsink-rtsp-{i}")
    udp_sink.set_property("host", "127.0.0.1")
    udp_sink.set_property("port", udp_port)
    udp_sink.set_property("async", False)
    udp_sink.set_property("sync", False)
    # ⭐ 推流分支不可被反壓影響主推論 pipeline，故 max-lateness=0 + qos
    udp_sink.set_property("qos", False)

    # === 加入 pipeline 並串接 ===
    for elm in [nvvidconv_r, cap_filter, encoder, parser, rtp_pay, udp_sink]:
        pipeline.add(elm)

    nvvidconv_r.link(cap_filter)
    cap_filter.link(encoder)
    encoder.link(parser)
    parser.link(rtp_pay)
    rtp_pay.link(udp_sink)

    return nvvidconv_r, udp_port


# ==========================================
# 本地顯示分支
# ==========================================
def _build_display_sink(pipeline, num_sources, has_live_source=False):
    """
    本地 tile 預覽視窗：
        streammux → tiler → nvvideoconvert → nvegltransform → nveglglessink
    回傳 streammux 作為各路顯示分支共用 sink_N 接點。

    ⭐ live-source 必須與主線 streammux 一致：
        - 檔案模式 (live-source=0)：依 buffer PTS 跑
        - RTSP 模式 (live-source=1)：依 wall clock 跑
        否則下游 streammux 會等永遠不會到的 buffer → 整個 pipeline 卡死。
    """
    rows, cols, total_w, total_h = _get_tile_layout(num_sources)
    print(f"[INFO] Tile 佈局: {rows}x{cols}, 視窗尺寸: {total_w}x{total_h}")

    streammux2 = make_elm("nvstreammux", "Stream-muxer-display")
    streammux2.set_property("batch-size", num_sources)  # 新舊版 mux 皆支援

    if os.environ.get("USE_NEW_NVSTREAMMUX") == "yes":
        # 新版 mux：不接受 width/height/live-source 等舊屬性，改用 config_mux.txt。
        # 來源皆 1080p、真正拼接由 tiler 完成，故新版 mux 不縮放也不影響顯示結果。
        _mux_cfg = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config_mux.txt"
        )
        if os.path.exists(_mux_cfg):
            streammux2.set_property("config-file-path", _mux_cfg)
        else:
            print(f"[WARNING] 找不到 {_mux_cfg}，顯示用 mux 將用內建預設值")
    else:
        # 舊版 mux：維持原本設定
        streammux2.set_property("width", 1920)
        streammux2.set_property("height", 1080)
        streammux2.set_property("batched-push-timeout", 10000)
        # ⭐ live-source 依主線是否有 live 來源決定，不可寫死 1
        streammux2.set_property("live-source", 1 if has_live_source else 0)
        streammux2.set_property("nvbuf-memory-type", 0)

    tiler = make_elm("nvmultistreamtiler", "nvtiler-display")
    tiler.set_property("rows", rows)
    tiler.set_property("columns", cols)
    tiler.set_property("width", total_w)
    tiler.set_property("height", total_h)

    q_d1 = make_elm("queue", "q-display-1")

    nvvidconv = make_elm("nvvideoconvert", "convertor-display")
    nvvidconv.set_property("nvbuf-memory-type", 0)

    q_d2 = make_elm("queue", "q-display-2")
    transform = make_elm("nvegltransform", "nvegl-transform-display")
    q_d3 = make_elm("queue", "q-display-3")

    sink = make_elm("nveglglessink", "nvvideo-renderer-display")
    sink.set_property("sync", False)
    # ⭐ 顯示分支不要做 QoS drop，避免主線被反壓
    sink.set_property("qos", False)

    for elm in [streammux2, tiler, q_d1, nvvidconv, q_d2, transform, q_d3, sink]:
        pipeline.add(elm)

    streammux2.link(tiler)
    tiler.link(q_d1)
    q_d1.link(nvvidconv)
    nvvidconv.link(q_d2)
    q_d2.link(transform)
    transform.link(q_d3)
    q_d3.link(sink)

    return streammux2


# ==========================================
# 每路 cam 分支組裝（動態組合 save / show / rtsp_push）
# ==========================================
def setup_cam_branch(pipeline, pad_index, cfg, demux, display_streammux, osd_probe_callback):
    """
    為單路 cam 建立完整下游分支。
    上游：demux.src_{pad_index}
    下游分支由三個 YAML 旗標決定：
        save (output.save_output_video) : 寫檔
        show (display.show_window)      : 本地預覽
        rtsp (rtsp_push.enable)         : RTSP 推流

    流程：
        demux.src_N → queue → nvvideoconvert → caps(RGBA) → nvdsosd → tee
            tee ├─→ save 分支 (可選)
                ├─→ show 分支 (可選，鏈到外部 display_streammux)
                └─→ rtsp 分支 (可選，內部 udpsink 給 RTSP server)
            若三者皆關，後接 fakesink 吞掉 buffer。

    回傳：
        若有 RTSP 分支則回傳 (udp_port)，供 main.py 註冊到 RTSP server；
        否則回傳 None。
    """
    i = pad_index
    src_pad = demux.get_request_pad(f"src_{i}")

    # === 共用前段：queue → RGBA 轉色 → OSD ===
    q_cam = make_elm("queue", f"q-cam-{i}")

    nvvidconv_osd = make_elm("nvvideoconvert", f"conv_osd_{i}")
    nvvidconv_osd.set_property("nvbuf-memory-type", 0)

    caps_osd = make_elm("capsfilter", f"caps_osd_{i}")
    caps_osd.set_property("caps", Gst.Caps.from_string("video/x-raw(memory:NVMM), format=RGBA"))

    nvosd_i = make_elm("nvdsosd", f"nvosd-{i}")
    nvosd_i.set_property("process-mode", 2)

    for elm in [q_cam, nvvidconv_osd, caps_osd, nvosd_i]:
        pipeline.add(elm)

    src_pad.link(q_cam.get_static_pad("sink"))
    q_cam.link(nvvidconv_osd)
    nvvidconv_osd.link(caps_osd)
    caps_osd.link(nvosd_i)

    # OSD 探針（畫 FPS 等等）
    nvosd_i.get_static_pad("sink").add_probe(
        Gst.PadProbeType.BUFFER,
        lambda pad, info, idx=i: osd_probe_callback(pad, info, idx),
        0
    )

    # === 三個分支開關 ===
    cam_save = cfg.get("output", {}).get("save_output_video", False)
    cam_show = cfg.get("display", {}).get("show_window", True)
    cam_rtsp = cfg.get("rtsp_push", {}).get("enable", False)
    is_file = cfg.get("is_file_source", False)

    # 計算本路啟用的下游分支數，決定要不要插 tee
    enabled_branches = sum([cam_save, cam_show, cam_rtsp])
    rtsp_port = None  # 若有推流，回傳給 main.py 用

    # ---------- 0 個分支：直接 fakesink ----------
    if enabled_branches == 0:
        fake = make_elm("fakesink", f"fake-{i}")
        fake.set_property("sync", False)
        fake.set_property("async", False)
        pipeline.add(fake)
        nvosd_i.link(fake)
        return None

    # ---------- 1 個分支：不用 tee，直接 link ----------
    if enabled_branches == 1:
        if cam_save:
            if is_file:
                # ⭐ 修改：補上 cfg["stream_fps"] 給 videorate 用
                nvosd_i.link(_build_save_branch_for_file(pipeline, i, cfg["video_path"], cfg["stream_fps"]))
            else:
                nvosd_i.link(_build_save_branch_for_rtsp(pipeline, i, cfg["video_path"], cfg["stream_fps"]))
        elif cam_show:
            _link_show_branch(pipeline, i, nvosd_i, display_streammux)
        elif cam_rtsp:
            entry, rtsp_port = _build_rtsp_push_branch(pipeline, i, cfg["rtsp_push"])
            nvosd_i.link(entry)
        return rtsp_port

    # ---------- 2 個以上分支：用 tee 分流 ----------
    tee = make_elm("tee", f"tee-{i}")
    pipeline.add(tee)
    nvosd_i.link(tee)

    if cam_save:
        q_s = make_elm("queue", f"q-s-{i}")
        pipeline.add(q_s)
        tee.link(q_s)
        if is_file:
            # ⭐ 修改：補上 cfg["stream_fps"] 給 videorate 用
            q_s.link(_build_save_branch_for_file(pipeline, i, cfg["video_path"], cfg["stream_fps"]))
        else:
            q_s.link(_build_save_branch_for_rtsp(pipeline, i, cfg["video_path"], cfg["stream_fps"]))

    if cam_show:
        _link_show_branch(pipeline, i, tee, display_streammux)

    if cam_rtsp:
        q_r = make_elm("queue", f"q-rtsp-{i}")
        # 推流分支特別重要：開 leaky=downstream 避免推流卡時反壓影響本地推論主線
        q_r.set_property("leaky", 2)  # 2 = downstream
        q_r.set_property("max-size-buffers", 30)
        pipeline.add(q_r)
        tee.link(q_r)
        entry, rtsp_port = _build_rtsp_push_branch(pipeline, i, cfg["rtsp_push"])
        q_r.link(entry)

    return rtsp_port


def _link_show_branch(pipeline, i, upstream, display_streammux):
    """
    建立本地顯示子分支：queue → nvvideoconvert → display_streammux.sink_N
    upstream 可能是 nvosd（單一分支）或 tee（多分支）。
    """
    q_d = make_elm("queue", f"q-d-{i}")
    nv_d = make_elm("nvvideoconvert", f"nv-d-{i}")
    nv_d.set_property("nvbuf-memory-type", 0)

    pipeline.add(q_d)
    pipeline.add(nv_d)

    upstream.link(q_d)
    q_d.link(nv_d)
    nv_d.get_static_pad("src").link(display_streammux.get_request_pad(f"sink_{i}"))
