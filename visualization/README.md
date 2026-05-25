# 订单时空可视化

本目录存放**可视化脚本**与**输出 HTML**（默认写入 `output/`）。

## 依赖

- 仅需仓库根目录 `requirements.txt`（含 `pandas`、`h3`），无额外包。

## 3D：H3 六边形挤出 + 时间轴（deck.gl）

生成带真实底图的3D可视化，包含：
- 3D六边形柱状图（高度和颜色表示订单量/预测值）
- 底部时间直方图与播放控制
- MapLibre 底图（OSM 瓦片）

```bash
.venv\Scripts\python visualization\scripts\build_spacetime_deck_3d.py
.venv\Scripts\python visualization\scripts\build_spacetime_deck_3d.py --date 2015-08-29 --window-hours 1
```

### 参数说明

| 参数 | 说明 |
|------|------|
| `--predictions` | 面板 CSV，默认 `models/xgboost_hourly_v3_h3/test_predictions.csv` |
| `--date` | 单日 `YYYY-MM-DD`；省略为数据中最后一天 |
| `--window-hours` | 地图上同时展示的小时槽数（1–6） |
| `--output` | 输出 HTML 路径，默认 `visualization/output/order_spacetime_3d.html` |

### 使用方式

1. 运行构建脚本生成HTML文件
2. 使用本地HTTP服务器打开（不支持`file://`协议）：

```bash
.venv\Scripts\python -m http.server 8000 --directory visualization/output
```

3. 在浏览器中访问：`http://localhost:8000/order_spacetime_3d.html`

### 界面功能

- **顶部工具栏**：切换实测/预测模式、图例说明
- **底部控制栏**：
  - 时间轴滑块：拖动选择时间
  - 播放控制：上一步、播放/暂停、下一步、重置
  - 速度调节：调整播放速度
- **3D/2D切换**：点击右上角按钮切换视角

## 文件结构

```
visualization/
├── output/              # 生成的HTML文件
│   ├── order_spacetime_3d.html    # 3D可视化输出
│   └── order_spacetime.html       # 2D可视化输出（可选）
├── scripts/             # 构建脚本
│   ├── build_spacetime_deck_3d.py # 3D可视化构建脚本
│   └── spacetime_plotly.py        # 2D Plotly可视化脚本
├── templates/           # HTML模板
│   └── spacetime_deck_3d.html     # 3D可视化模板
├── proxy_server.py      # 代理服务器（解决CDN跨域问题）
├── requirements.txt     # 2D可视化额外依赖
└── README.md            # 本说明文件
```

## 注意事项

- 需要网络访问以加载 deck.gl、MapLibre 和 OSM 瓦片
- 建议使用现代浏览器（Chrome、Firefox、Edge）
- 首次加载可能需要几秒时间下载依赖库
