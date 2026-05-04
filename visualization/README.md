# 订单时空可视化

本目录存放**可视化脚本**与**输出 HTML**（默认写入 `output/`）。

## 依赖

- **3D（推荐）**：仅需仓库根目录 `requirements.txt`（含 `pandas`、`h3`），无额外包。
- **2D Plotly**：再安装：

```bash
.venv\Scripts\pip install -r visualization\requirements.txt
```

## 3D：H3 六边形挤出 + 时间轴（deck.gl）

与参考图一致：**深色底图 + 3D 柱高与颜色均表示订单量/预测值 + 底部直方图与时间窗**。

```bash
.venv\Scripts\python visualization\scripts\build_spacetime_deck_3d.py
.venv\Scripts\python visualization\scripts\build_spacetime_deck_3d.py --date 2015-08-29 --window-hours 1
```

请使用仓库内 **`.venv` 的 Python`** 运行构建脚本；若用系统 **Python 3.14** 等环境，可能出现 `numpy` DLL 无法加载（与脚本无关，需换已安装好 numpy 的解释器）。

| 参数 | 说明 |
|------|------|
| `--predictions` | 面板 CSV，默认 `models/xgboost_hourly_v3_h3/test_predictions.csv` |
| `--date` | 单日 `YYYY-MM-DD`；省略为数据中最后一天 |
| `--window-hours` | 地图上同时展示的小时槽数（1–6） |
| `--output` | 输出 HTML 路径，默认 `visualization/output/order_spacetime_3d.html` |

打开生成文件需 **能访问外网**（unpkg / esm.sh / jsdelivr、CARTO 瓦片）。import map 已默认走 **unpkg**（部分网络下 jsdelivr 对 `@deck.gl/mapbox` 会 **400**）。**@luma.gl** 与 **@deck.gl/core** 同为 **9.0.x**（`~9.0.27`），勿混用 9.3.x。

**若地图区域全黑、图例/直方图也为空**：多半是 **deck / MapLibre 的 ES 模块未加载成功**（国内 jsdelivr 不稳定、或 `file://` 打开）。请改用 **`http://127.0.0.1:端口/.../order_spacetime_3d.html`** 打开，并看 F12 控制台；失败时页面 `#map` 内会显示红色错误说明。

若 `file://` 下 ES 模块受限，请用本地 HTTP，例如：

```bash
.venv\Scripts\python -m http.server 8765 --directory .
```

浏览器访问：`http://127.0.0.1:8765/visualization/output/order_spacetime_3d.html`。

## 2D：Plotly 散点 + 帧动画

无 deck 时可用（非 3D 挤出）：

```bash
.venv\Scripts\python visualization\scripts\spacetime_plotly.py
```

| 参数 | 说明 |
|------|------|
| `--predictions` | 面板 CSV |
| `--date` | 单日 |
| `--metric` | `count` 或 `prediction` |
| `--mapbox-token` | 可选 |

输出：`visualization/output/order_spacetime.html`。
