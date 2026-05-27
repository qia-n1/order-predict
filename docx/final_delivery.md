# 最终交付索引

本文件用于汇总当前仓库的最终交付内容，方便汇报、展示和继续维护。

## 1. 主要成果图

- [H3 结果总览仪表盘](../models/picture/jtjm_model_results_dashboard_dark.png)
- [H3 分辨率消融表](../models/picture/h3_resolution_ablation_table.png)
- [H3 分辨率与稀疏性权衡图](../models/picture/h3_resolution_sparsity_tradeoff.png)
- [H3 分辨率趋势图](../models/picture/h3_resolution_ablation_trends.png)
- [V1/V2/V3 对比表](../models/picture/h3_v1_v2_v3_comparison_table.png)
- [V2-H3 特征类别贡献环图](../models/picture/v2_h3_feature_category_contribution.png)
- [V2-H3 特征组分解图](../models/picture/v2_h3_feature_group_breakdown.png)
- [V2-H3 特征重要性 Top 15](../models/picture/v2_h3_feature_importance_top15.png)

## 2. 关键报告文档

- [feature_engineering.md](feature_engineering.md)
- [model_design.md](model_design.md)
- [version3.md](version3.md)
- [visualization.md](visualization.md)

## 3. 可重复生成脚本

- [src/visualize_model_results.py](../src/visualize_model_results.py)
- [src/run_h3_resolution_ablation.py](../src/run_h3_resolution_ablation.py)
- [src/train_xgboost.py](../src/train_xgboost.py)
- [src/train_xgboost_v2.py](../src/train_xgboost_v2.py)
- [src/train_xgboost_v3.py](../src/train_xgboost_v3.py)

## 4. 使用建议

- 论文或汇报封面优先使用 H3 结果总览仪表盘。
- 方法章节优先引用 feature_engineering / model_design / version3。
- 实验章节优先引用分辨率消融、模型对比和特征重要性图片。

## 5. 快速预览

重新生成总览图：

```bash
python3 src/visualize_model_results.py
```

查看最终图片目录：

```bash
ls models/picture
```
