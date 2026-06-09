# Jetson + Conda 環境解決 `gi` 與 `pyds` 模組問題

## 環境資訊

| 項目 | 內容 |
|------|------|
| 裝置 | NVIDIA Jetson |
| 系統 | Ubuntu 22.04 |
| 虛擬環境 | Conda（環境名稱：`tracking`） |
| Python | 3.10（cp310） |
| 架構 | aarch64（ARM64） |
| DeepStream | 7.1（`/opt/nvidia/deepstream/deepstream-7.1`） |
| 專案 | DeepStream-LPR |

---

## 問題一：`ModuleNotFoundError: No module named 'gi'`

`gi`（GObject Introspection）屬於系統層級套件。Jetson + DeepStream 環境中，系統層級通常已內建 `gi`，問題只在於 conda 環境預設會隔離系統套件，因此真正要做的是把系統的 `dist-packages` 連結進 conda 環境。

### 步驟 1：在 conda 環境中建立 `.pth` 檔案連結系統套件

這個做法讓 conda 環境啟動時，自動把系統的 `dist-packages` 加入搜尋路徑。

```bash
SITE_PKG=$(python3 -c "import site; print(site.getsitepackages()[0])")
echo "/usr/lib/python3/dist-packages" > "$SITE_PKG/system_gi.pth"
echo "已寫入：$SITE_PKG/system_gi.pth"
```

本次實際寫入位置：

```
/home/nvidia/miniconda3/envs/tracking/lib/python3.10/site-packages/system_gi.pth
```

### 步驟 2：驗證

```bash
python3 -c "import gi; print(gi.__file__)"
```

成功時會指向系統路徑：

```
/usr/lib/python3/dist-packages/gi/__init__.py
```

進一步測試 GStreamer（DeepStream 會用到）：

```bash
python3 -c "import gi; gi.require_version('Gst', '1.0'); from gi.repository import Gst; Gst.init(None); print('GStreamer OK')"
```

> ⚠️ **注意事項**
> `.pth` 做法會讓系統 `dist-packages` 下的**所有**套件對該 conda 環境可見，偶爾可能與 conda 內已安裝的同名套件產生版本衝突。若日後某套件行為異常，可回想是否與此有關。

---

## 問題二：`ModuleNotFoundError: No module named 'pyds'`

`pyds` 是 DeepStream 的 Python bindings，**不在**系統 `dist-packages` 裡，而是隨 DeepStream SDK 提供（通常為預先編譯好的 wheel 檔）。

### 步驟 1：確認 DeepStream 版本與 pyds wheel 位置

```bash
ls /opt/nvidia/deepstream/

find /opt/nvidia/deepstream/ -name "pyds*.so" 2>/dev/null
find /opt/nvidia/deepstream/ -name "pyds*.whl" 2>/dev/null
```

本次找到的 wheel：

```
/opt/nvidia/deepstream/deepstream-7.1/sources/deepstream_python_apps/pyds-1.2.0-cp310-cp310-linux_aarch64.whl
```

> 💡 wheel 檔名中的 `cp310` 代表 Python 3.10、`linux_aarch64` 代表 ARM64（Jetson）。需與 conda 環境的 Python 版本相符才能直接安裝；相符時不必重新編譯。

### 步驟 2：在 conda 環境中安裝 pyds

確認仍在 `(tracking)` 環境，然後：

```bash
pip install /opt/nvidia/deepstream/deepstream-7.1/sources/deepstream_python_apps/pyds-1.2.0-cp310-cp310-linux_aarch64.whl
```

### 步驟 3：驗證

```bash
python3 -c "import pyds; print('pyds OK')"
```

---

## 快速重現（懶人包）

在 `(tracking)` conda 環境中，依序執行：

```bash
# 1. 連結系統 gi 到 conda 環境
SITE_PKG=$(python3 -c "import site; print(site.getsitepackages()[0])")
echo "/usr/lib/python3/dist-packages" > "$SITE_PKG/system_gi.pth"

# 2. 安裝 pyds（依實際 DeepStream 版本與 wheel 路徑調整）
pip install /opt/nvidia/deepstream/deepstream-7.1/sources/deepstream_python_apps/pyds-1.2.0-cp310-cp310-linux_aarch64.whl

# 3. 驗證
python3 -c "import gi; gi.require_version('Gst', '1.0'); from gi.repository import Gst; Gst.init(None); print('gi / GStreamer OK')"
python3 -c "import pyds; print('pyds OK')"
```

---

## 常見後續問題

- **`ValueError: Namespace Gst not available`（或其他 namespace）**
  代表缺少對應的 typelib。確認需要哪個 namespace 後，安裝對應的 `gir1.2-*` 套件即可。例如 GStreamer：
  ```bash
  sudo apt install gir1.2-gstreamer-1.0 gir1.2-gst-plugins-base-1.0 -y
  ```

- **更換 conda 環境或重建環境後 `gi` 又消失**
  `.pth` 檔是寫在特定環境的 `site-packages` 內，新環境需重新執行上面的步驟 2。

- **pyds wheel 的 Python 版本不符**
  若 conda 環境改用其他 Python 版本（例如 3.8），則需找到對應 `cp38` 的 wheel，或從 `deepstream_python_apps` 原始碼自行編譯。