# Version 3: CNN/RNN Attention Embedding + XGBoost

## 1. 方案目标

当前项目中的 XGBoost 模型主要依赖人工设计的统计特征，例如 `lag_1`、`lag_24`、`rolling_mean_24`、`time_sin`、`time_cos` 等。Version 3 的目标是在 XGBoost 前增加一个深度特征提取器，用 CNN、RNN 和 Attention 自动学习时空表示，再将学习到的 embedding 作为 XGBoost 的输入特征。

该方案不是无监督学习，而是一个两阶段的有监督特征学习流程：

1. 第一阶段：训练 CNN/RNN/Attention 深度模型预测区域小时订单数 `order_count`，并用验证集指标进行 early stopping。
2. 第二阶段：冻结深度模型，提取中间层 embedding，将 embedding 输入 XGBoost，再次训练最终预测模型。

最终预测目标仍然是：

```text
某个 geohash 区域，在某个小时内的出租车订单数量 order_count
```

## 2. 数据组织方式

原始订单数据先聚合为时空面板数据：

```text
geohash + time_slot -> order_count
```

其中：

- `geohash` 表示城市空间网格区域。
- `time_slot` 表示按小时取整后的时间段。
- `order_count` 表示该区域该小时内的上车订单数量。

例如：

```text
geohash = dr5ru0
time_slot = 2015-08-28 09:00:00
order_count = 15
```

含义是：`dr5ru0` 区域在 `2015-08-28 09:00-09:59` 这一小时内有 15 个上车订单。

## 3. 数据划分

为了避免时间泄漏，训练、验证和测试集应按时间顺序划分，而不是随机划分：

```text
前 70% 时间段：训练集
中间 15% 时间段：验证集
最后 15% 时间段：测试集
```

该划分同时用于两个阶段：

- 深度特征提取器使用训练集训练，验证集 early stopping，测试集只做最终评估。
- XGBoost 使用训练集 embedding 训练，验证集 embedding 调参或 early stopping，测试集 embedding 做最终测试。

测试集不能参与 CNN/RNN/Attention 的训练停止，也不能参与 XGBoost 的调参。

## 4. 模型整体结构

Version 3 的整体结构如下：

```text
空间输入 -> CNN 模块 -> 空间 embedding
时间输入 -> RNN 模块 -> 时间 embedding
外部特征 -> 全连接模块 -> 外部 embedding

空间 embedding + 时间 embedding + 外部 embedding
        -> Attention 融合
        -> fusion embedding
        -> 第一阶段预测头 -> order_count

冻结深度模型后：

fusion embedding + 可选基础特征
        -> XGBoost
        -> order_count
```

## 5. CNN 空间特征提取模块

CNN 模块只负责提取空间特征。

输入可以构造成目标 geohash 周围的空间邻域矩阵，例如 `5 x 5` 或 `7 x 7` 的区域订单热力图：

```text
[
  [0, 1, 3, 2, 0],
  [1, 4, 8, 6, 2],
  [0, 5, 目标区域, 7, 3],
  [0, 2, 6, 4, 1],
  [0, 1, 2, 1, 0]
]
```

CNN 通过卷积层学习相邻区域之间的空间相关性，例如商圈扩散、车站周边需求溢出、相邻区域同步增长等。

输出为：

```text
spatial_embedding
```

## 6. RNN 时间特征提取模块

RNN 模块只负责提取时间特征。

输入为目标区域过去一段时间的订单序列，例如过去 24 小时或过去 168 小时：

```text
[x(t-24), x(t-23), ..., x(t-1)]
```

其中 `x(t-i)` 表示目标 geohash 区域在历史第 `i` 个小时的订单数。

RNN 可以选择：

- GRU
- LSTM
- 简化 RNN

建议优先使用 GRU 或 LSTM，因为它们更适合学习较长时间跨度中的周期性和趋势变化。

输出为：

```text
temporal_embedding
```

## 7. 外部特征全连接模块

外部特征不进入 CNN 或 RNN，而是单独通过全连接层编码。

可加入的外部特征包括：

- 天气类型 one-hot 编码
- 温度
- 降雨量
- 是否节假日
- 是否周末
- 星期几
- 小时
- 交通状态
- 体育、文娱活动标识

外部特征经过 MLP 后输出：

```text
external_embedding
```

该模块的作用是将天气、节假日、工作日等非时空网格特征转化为可与时空 embedding 融合的向量表示。

## 8. Attention 融合模块

Attention 模块用于融合三类信息：

```text
spatial_embedding
temporal_embedding
external_embedding
```

它的作用是让模型自动学习不同场景下各类特征的重要性。例如：

- 早晚高峰可能更依赖时间特征。
- 大型活动或商圈扩散可能更依赖空间特征。
- 雨雪天气、节假日可能更依赖外部特征。

Attention 融合后得到：

```text
fusion_embedding
```

该向量是深度模型学习到的高层时空需求表示，也是后续输入 XGBoost 的核心特征。

## 9. 第一阶段训练方式

第一阶段训练完整的深度预测模型：

```text
CNN + RNN + 外部 MLP + Attention + 预测头 -> order_count
```

训练标签为真实订单数：

```text
y = order_count
```

因此第一阶段是有监督学习，不是无监督学习。

推荐损失函数：

- `MAE`
- `MSE`
- `Poisson NLL`

由于订单量是计数值，`Poisson NLL` 与任务形式更匹配；如果为了与现有 XGBoost 评估方式保持一致，也可以直接使用 `MAE` 或 `MSE`。

## 10. 第一阶段停止指标

CNN/RNN/Attention 的训练停止应基于验证集，而不是测试集。

推荐 early stopping 指标：

```text
valid_mae
```

或：

```text
valid_rmse
```

训练流程：

```text
每个 epoch 后计算验证集 MAE/RMSE
如果连续 patience 轮没有提升，则停止训练
恢复验证集指标最优的模型权重
```

推荐参数：

```text
patience = 10 到 20
max_epochs = 100 到 300
```

如果使用 `Poisson NLL` 作为损失函数，也可以监控：

```text
valid_poisson_nll
```

但最终报告仍建议包含 `MAE` 和 `RMSE`，方便与现有 XGBoost 模型对比。

## 11. 第二阶段 embedding 提取

第一阶段训练完成后，去掉最后的预测头，保留到 `fusion_embedding` 为止的部分：

```text
CNN + RNN + 外部 MLP + Attention -> fusion_embedding
```

然后分别对训练集、验证集、测试集生成 embedding：

```text
train_embeddings.csv
valid_embeddings.csv
test_embeddings.csv
```

每一行 embedding 对应一个样本：

```text
geohash, time_slot, emb_1, emb_2, ..., emb_n, order_count
```

可以选择只输入 embedding，也可以拼接少量基础特征：

```text
embeddings + hour + weekday + weekend + center_lat + center_lon
```

## 12. 第二阶段 XGBoost 训练

XGBoost 的输入从原来的人工统计特征改为：

```text
fusion_embedding + 可选基础特征
```

监督标签仍然是：

```text
order_count
```

推荐 XGBoost 参数保持与当前项目类似：

```text
objective = count:poisson
eval_metric = rmse, mae
eta = 0.05
max_depth = 6 到 8
subsample = 0.8
colsample_bytree = 0.8
```

训练停止仍然使用验证集 early stopping：

```text
early_stopping_rounds = 30
```

最终只在测试集上报告一次结果。

## 13. 与当前版本的区别

当前 XGBoost 版本主要依赖人工特征：

```text
lag_1, lag_2, lag_24, lag_168
rolling_mean_3, rolling_mean_6, rolling_mean_24
time_sin, time_cos, day_sin, day_cos
geohash_mean_count
```

Version 3 的主要区别是：

```text
用 CNN 自动学习空间相关性
用 RNN 自动学习时间依赖
用 MLP 编码天气、节假日等外部维度
用 Attention 融合空间、时间、外部信息
用融合后的 embedding 作为 XGBoost 输入
```

因此 Version 3 并不是完全不需要设计输入，而是将人工设计重点从“统计特征”转移为“输入结构”：

- CNN 需要定义空间邻域范围。
- RNN 需要定义历史窗口长度。
- 外部特征需要定义可用变量。
- Attention 负责学习不同特征源的权重。

## 14. 推荐实验对比

为了验证 Version 3 是否有效，建议至少对比以下模型：

| 版本 | 输入特征 | 模型 |
| --- | --- | --- |
| Baseline | 人工统计特征 | XGBoost |
| V3-A | CNN/RNN/Attention embedding | XGBoost |
| V3-B | embedding + 少量基础特征 | XGBoost |
| V3-C | CNN/RNN/Attention 端到端 | 深度模型预测头 |

评估指标：

```text
MAE
RMSE
```

如果需要衡量趋势一致性，也可以增加：

```text
PCC
```

## 15. 总结

Version 3 采用“两阶段有监督特征学习 + XGBoost”的方案：

1. 先训练 CNN/RNN/Attention 深度模型，让它基于真实订单数学习有预测能力的时空 embedding。
2. 再冻结深度模型，提取 `fusion_embedding`。
3. 最后用 XGBoost 基于 embedding 预测区域小时订单量。

该方案保留了 XGBoost 在表格预测上的稳定性，同时引入 CNN、RNN 和 Attention 自动学习空间、时间和外部因素的高层表示，可以减少对人工 lag、rolling mean 等统计特征的依赖。
