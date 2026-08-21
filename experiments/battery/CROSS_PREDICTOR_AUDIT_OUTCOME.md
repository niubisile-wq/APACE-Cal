# 跨预测器 APACE-Cal 审计

## 协议

- H50/K3；六个开发域；100 个 episode；
- 每个 episode 中 baseline 和 router 使用相同随机支持/路由支持；
- 逐个评估冻结局部校准接口中的所有 predictor；
- 不修改 APACE-Cal，不用于外部盲测确认。

## 宏平均结果

| predictor | baseline MAPE | router MAPE | 平均相对下降 |
|---|---:|---:|---:|
| logmean | 59.426 | 51.881 | 5.40% |
| w0.125/bw0.5 | 52.353 | 31.670 | 18.09% |
| w0.5/bw0.5 | 49.003 | 25.591 | 21.19% |
| w1/bw0.5 | 47.463 | 22.969 | 22.32% |
| w2/bw0.5 | 47.034 | 22.043 | 22.65% |
| winf/bw0.5 | 48.041 | 24.676 | 20.47% |
| support median | 70.635 | 65.728 | 6.53% |
| binary LOO w0.5 | 55.646 | 37.068 | 14.79% |
| binary LOO w2 | 53.487 | 40.980 | 9.91% |

其余 bandwidth=1/2 的 kernel 也保持正向，但增益较小。

## 结论

路由支持相对于相同 predictor 的随机支持在多数 predictor 上保持正向，因此 APACE-Cal 的支持选择并非只对单个 predictor 有效；但增益大小明显依赖 predictor，不能声称“任意 predictor 即插即用”。论文应将方法定义为“与局部校准接口协同的支持选择和安全路由”，并把跨 predictor 结果放入补充材料或鲁棒性表。

结果文件：`batterylife_cross_predictor_audit.json`。
