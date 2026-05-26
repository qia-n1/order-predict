# NYC Taxi Demand Forecasting

本项目基于纽约市 2015 年黄色出租车行程记录，构建了一套区域级小时粒度的订单需求预测系统。数据来源于纽约市政府开放数据平台（NYC Open Data），经过空间网格划分和特征工程后，以 XGBoost 作为核心预测模型，经历了三个迭代版本的演进。

整个工作流从原始行程记录出发，将每条上车记录映射到 Uber H3 六边形网格中，再按小时聚合为「区域-时段-订单量」的面板数据。在此基础上叠加时间编码、历史统计、天气数据和节假日信息等多维特征，训练梯度提升树模型来预测未来某个六边形区域在某个小时内的出租车需求量。

## 数据

原始数据通过 `src/get_data.py` 从纽约市政府 Socrata API 拉取，默认抽取 10 万条行程记录，保存为 `data/NYC_YellowTaxi_2015_100k.csv`。每条记录包含上车时间、上车经纬度等字段。随后 `src/feature_engineering.py` 对原始记录进行预处理：利用 H3 库将经纬度编码为六边形网格索引，并生成一系列时间特征。时间编码采用分箱后的正余弦变换（`time_sin/cos`、`day_sin/cos`），使得时间在跨越午夜或周末衔接时能平滑过渡，而非产生跳变。处理后的数据保存为 `data/NYC_YellowTaxi_2015_100k_features.csv`。

## 模型版本

### V1：基线 XGBoost

V1（`src/train_xgboost.py`）建立了完整的训练框架。原始行程记录先按 H3 网格和小时聚合为面板数据，再构造滞后特征（lag-1/2/24/168）和滑动窗口均值（rolling mean 3/6/24）作为历史需求信号，配合小时、星期、周末等时间编码和网格中心坐标，以 `reg:squarederror` 为目标函数训练 XGBoost。为应对大量零需求样本导致的分布偏斜，V1 引入了分层样本权重机制——零需求行权重为 1，非零行为 3，高需求行可设置 6 或 10，迫使模型在高需求区域投入更多拟合注意力。数据集按时间顺序切分为 70/15/15 的训练、验证、测试集，验证集用于 early stopping，测试集仅做最终评估。

### V2：外部特征增强

V2（`src/train_xgboost_v2.py`）在 V1 的基础上做了两方面扩展。一是增加了外部数据：通过 Open-Meteo 历史天气 API 获取逐小时的温度、体感温度、降水量、云量、风速、湿度和天气代码，并将天气类型展开为独热编码（晴/多云/雾/毛毛雨/雨/雪/雷），同时衍生出 `is_precipitating`、`is_hot`、`is_humid` 等布尔标记；利用美国联邦假日日历生成节假日标识。二是增加了空间上下文特征：对每个 H3 网格计算其上级分辨率（parent resolution）的聚合统计量，包括 parent 区域的总需求滞后和均值滑动窗口，从而让模型感知更大尺度的空间趋势。特征总量从 V1 的 20 个扩展到 56 个，在测试集 RMSE 和 MAE 上均有小幅改善。

### V3：深度嵌入 + XGBoost 两阶段

V3（`src/train_xgboost_v3.py`）引入了深度学习作为特征提取前端。整体是一个两阶段有监督方案：第一阶段训练一个 CNN + GRU + Attention 模型直接预测订单量，CNN 以目标网格周围 5x5 邻域的需求热力图为输入学习空间相关性，GRU 以过去 24 小时的历史序列为输入捕捉时间依赖，外部特征通过独立的 MLP 编码，三路嵌入经 Attention 加权融合为一个 32 维的 `fusion_embedding`。第一阶段训练完成后，冻结深度模型，将融合嵌入向量提取出来，与 V2 的全部人工特征拼接，作为第二阶段 XGBoost 的输入。这种做法保留了 XGBoost 在表格数据上的稳定性，同时利用深度网络自动学习到的高阶时空表示来补充人工特征的不足。从实验结果看，V3 在整体 PCC 上有所提升，但在稀疏区域的 MAE 表现与 V2 持平，说明深度嵌入对高需求区域的增益更为明显。

## H3 分辨率消融实验

`src/run_h3_resolution_ablation.py` 对分辨率 6 到 9 进行了系统消融。分辨率越高，六边形越小、网格数越多，空间信息越精细，但零需求比例也随之升高，数据稀疏性加剧。实验结果保存在 `models/h3_resolution_ablation/` 下，包括每个分辨率的模型、指标和训练曲线。从消融结果可以看到，分辨率 8（边长约 0.46 km，网格数约 400 级别）在精度和稀疏性之间取得了较好的平衡，因此被选为默认配置。

## 进度

目前可视化之前的工作已经完成，所有结果都在models文件夹下，需要展示的图片在models/picture文件夹。

## 最终交付物

以下文件是当前仓库的最终展示入口，适合直接用于汇报或页面说明：

- [models/picture/jtjm_model_results_dashboard_dark.png](models/picture/jtjm_model_results_dashboard_dark.png) - H3 结果总览仪表盘，包含地图、图例和时间轴。
- [visualization/output/order_spacetime_3d_2014-06-30.html](visualization/output/order_spacetime_3d_2014-06-30.html) - 3D 订单时空可视化入口，使用六月最后一周的逐日 HTML 版本，可通过本地 HTTP 服务器打开。
- [models/picture/h3_resolution_ablation_table.png](models/picture/h3_resolution_ablation_table.png) - H3 分辨率消融表。
- [models/picture/h3_resolution_sparsity_tradeoff.png](models/picture/h3_resolution_sparsity_tradeoff.png) - 分辨率与稀疏性权衡图。
- [models/picture/h3_resolution_ablation_trends.png](models/picture/h3_resolution_ablation_trends.png) - 分辨率指标趋势图。
- [models/picture/h3_v1_v2_v3_comparison_table.png](models/picture/h3_v1_v2_v3_comparison_table.png) - V1/V2/V3 对比表。
- [models/picture/v2_h3_feature_category_contribution.png](models/picture/v2_h3_feature_category_contribution.png) - 特征类别贡献环图。
- [models/picture/v2_h3_feature_group_breakdown.png](models/picture/v2_h3_feature_group_breakdown.png) - 特征组分解图。
- [models/picture/v2_h3_feature_importance_top15.png](models/picture/v2_h3_feature_importance_top15.png) - V2-H3 Top 15 特征重要性。

如需重新生成仪表盘，可运行：

```bash
python3 src/visualize_model_results.py
```

如需查看 3D 订单时空可视化，先启动本地服务：

```bash
python3 -m http.server 8000 --directory visualization/output
```

然后访问：

```text
http://127.0.0.1:8000/order_spacetime_3d_2014-06-30.html
```