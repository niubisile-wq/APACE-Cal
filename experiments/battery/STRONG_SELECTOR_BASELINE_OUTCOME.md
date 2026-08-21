# 强选择器基线实验结果

## 目的

在冻结的固定候选池协议下，补充 farthest-first/k-center 和 coverage 类选择器，检验 APACE-Cal 是否只胜过较弱的普通 active selector。该实验不修改 APACE-Cal 冻结方法，也不覆盖任何外部盲测结果。

## 协议

- 六个开发域，H=10/20/50，K=1/3/5/10；
- 每个设置 100 个 episode；
- 每个 episode 使用 70% acquisition pool 和共同的 30% test pool；
- 选择器只读取 protocol/early-curve 特征，不读取寿命标签；
- 所有 selector 使用同一个 log-mean 校准预测器和相同的标签预算；
- 比较 random、protocol/curve/hybrid medoid、protocol/curve/hybrid k-center。

结果文件：`batterylife_strong_selector_baselines.json`。

## 关键宏平均 MAPE

| 设置 | random | protocol medoid | curve medoid | hybrid medoid | protocol k-center | curve k-center | hybrid k-center |
|---|---:|---:|---:|---:|---:|---:|---:|
| H10/K3 | 56.380 | 53.022 | 55.145 | 54.311 | 57.378 | **52.743** | 52.925 |
| H20/K3 | 56.544 | **53.840** | 55.731 | 54.137 | 59.679 | 67.588 | 65.314 |
| H50/K3 | 56.465 | **51.270** | 55.742 | 54.845 | 56.659 | 54.684 | 54.165 |
| H50/K5 | 54.275 | 53.724 | 55.331 | 55.161 | 51.522 | **48.044** | 48.463 |
| H50/K10 | 51.171 | 47.993 | 47.191 | 48.521 | 48.925 | **45.555** | 47.704 |

相对于 random，selector 的逐设置域级胜负次数（72 个域/设置组合）为：

- protocol medoid：28 胜 / 9 平 / 35 负；
- curve medoid：32 胜 / 9 平 / 31 负；
- hybrid medoid：31 胜 / 9 平 / 32 负；
- protocol k-center：36 胜 / 9 平 / 27 负；
- curve k-center：30 胜 / 9 平 / 33 负；
- hybrid k-center：29 胜 / 9 平 / 34 负。

## 解释和限制

这些结果证明现代 coverage selector 并非无效：在部分设置下可以优于 random。但其表现随域、H、K 明显变化，不能稳定控制最差域退化；这正是 APACE-Cal 安全门控需要解决的问题。

该诊断使用统一 log-mean predictor，不能直接替代 APACE-Cal 的完整 predictor-coupled E2 rankboard。正文应把它作为强 selector 对照/补充材料，并在后续统一 predictor 接口后再做最终主表比较。
