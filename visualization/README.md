# 订单时空可视化

本目录存放 3D/2D 可视化脚本与生成结果。当前已经生成的 3D 展示版本是 2014 年 6 月最后一周（06-24 到 06-30）的逐日 HTML，默认输出到 `output/`。

## 依赖

- 构建脚本依赖仓库根目录 `requirements.txt` 中的 `pandas`、`h3` 等基础包。
- 3D 页面通过 CDN 加载 `h3-js`、`deck.gl` 和 `maplibre-gl`，当前脚本已切换为 `jsdelivr`，避免部分浏览器对 `unpkg` 的 Tracking Prevention 限制。
- 3D 底图使用 MapLibre 的矢量深色样式，浏览器需要联网加载地图样式和瓦片。

## 3D：H3 六边形挤出 + 时间轴（deck.gl）

当前 3D 可视化包含：
- 3D 六边形柱状图，柱高和颜色分别表示订单量/预测值
- 底部时间直方图与播放控制
- 可交互的 MapLibre 矢量底图，不是静态图片

### 生成命令

```bash
python3 visualization/scripts/build_spacetime_deck_3d.py --date 2014-06-30 --output visualization/output/order_spacetime_3d_2014-06-30.html
```

如果要一次生成六月最后一周的全部版本：

```bash
for d in 2014-06-24 2014-06-25 2014-06-26 2014-06-27 2014-06-28 2014-06-29 2014-06-30; do
  python3 visualization/scripts/build_spacetime_deck_3d.py --date "$d" --output "visualization/output/order_spacetime_3d_${d}.html"
done
```

### 参数说明

| 参数 | 说明 |
|------|------|
| `--predictions` | 面板 CSV，默认 `models/xgboost_hourly_v3_h3/test_predictions.csv` |
| `--date` | 单日 `YYYY-MM-DD`；省略时使用数据中的最后一天 |
| `--window-hours` | 地图上同时展示的小时槽数，当前推荐保持 `1` |
| `--output` | 输出 HTML 路径，默认 `visualization/output/order_spacetime_3d.html` |

### 使用方式

1. 先运行构建脚本生成 HTML 文件。
2. 使用本地 HTTP 服务器打开生成的 HTML，避免 `file://` 协议导致资源加载受限：

```bash
python3 -m http.server 8000 --directory visualization/output
```

3. 在浏览器中访问例如：

```text
http://127.0.0.1:8000/order_spacetime_3d_2014-06-30.html
```

### 当前可用文件

```text
visualization/output/order_spacetime_3d_2014-06-24.html
visualization/output/order_spacetime_3d_2014-06-25.html
visualization/output/order_spacetime_3d_2014-06-26.html
visualization/output/order_spacetime_3d_2014-06-27.html
visualization/output/order_spacetime_3d_2014-06-28.html
visualization/output/order_spacetime_3d_2014-06-29.html
visualization/output/order_spacetime_3d_2014-06-30.html
```

### 界面功能

- 顶部区域：切换实测/预测模式、查看图例
- 地图区域：3D 六边形柱与底图联动显示
- 底部控制栏：时间轴滑块、播放/暂停、快进/重置、速度调节
- 右上角按钮：在 2D 视角和 3D 视角之间切换

## 文件结构

```text
visualization/
├── output/              # 生成的 HTML 文件
│   ├── order_spacetime_3d.html
│   └── order_spacetime_3d_2014-06-24.html ~ order_spacetime_3d_2014-06-30.html
├── scripts/             # 构建脚本
│   ├── build_spacetime_deck_3d.py
│   └── spacetime_plotly.py
├── templates/           # HTML 模板
│   └── spacetime_deck_3d.html
├── proxy_server.py      # 代理服务器（仅在特定环境下需要）
└── README.md            # 本说明文件
```

## 注意事项

- 第一次打开时会加载 CDN 脚本和地图样式，网络较慢时会有短暂空白。
- 如果浏览器仍提示 Tracking Prevention，优先确认页面是通过本地 HTTP 服务器访问的，而不是直接双击 HTML 文件。
