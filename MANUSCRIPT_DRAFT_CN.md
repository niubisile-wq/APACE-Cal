# APACE-Cal 论文初稿骨架（实验结果驱动）

## 暂定题目

**跨数据集电池寿命预测中的非对称、协议感知少样本目标域校准**

英文：*Asymmetric Protocol-Aware Few-Shot Target-Domain Calibration for Cross-Dataset Battery Life Prediction*

## 摘要（结果版草稿）

跨实验室、跨测试协议的电池寿命预测通常面临目标域标注昂贵、早期曲线信息有限和协议分布偏移同时存在的问题。本文提出 APACE-Cal，一种带安全门控、与局部校准接口协同的条件式目标域校准方法。方法根据协议差异、早期退化曲线和两者一致性，在主动支持集与严格匹配的随机回退之间做标签盲决策。在六个开发域、三个早期观测窗口、四个总标签预算和 100 个固定池 episode 中，H=50/K=3 的宏平均 MAPE 从 48.54% 降至 22.94%，冻结统计协议下相对下降 21.45%，层级 bootstrap 95% CI 为 [0.86%, 76.66%]；三个主窗口均无域级退化。冻结后的 SNU 和 27-cell MathWorks 外部审计分别取得 30.92% 和 47.99% 的 H50/K3 相对下降；NA-ion、MATR 和 HUST+XJTU 则按规则安全回退且不退化。结果支持“有条件主动校准”而非“无条件主动选样”的结论。

术语说明：本文最终方法名统一为 **APACE-Cal**；“v2”仅用于冻结实验脚本、JSON 和复现文件名，表示冻结核心版本，不代表另一个待比较的方法。

## 1. 引言要回答的问题

1. 在跨数据集电池寿命预测中，少量目标域寿命标签应该如何分配，才能同时利用协议信息和早期曲线信息？
2. 如何在目标域不可识别或协议偏移风险高时，自动禁止主动策略，避免负迁移？
3. 一个冻结、标签盲的支持集/路由规则，能否在独立外部数据上保持安全性？

本文不声称提出新的寿命预测骨干、不声称提供概率预测区间，也不声称 active route 在所有新域都必然有效。

## 2. 方法

### 2.1 问题定义

给定目标域电芯集合、每个电芯前 H 个循环的观测和总预算 K 个完整寿命标签；先冻结目标域 acquisition/test 划分，再从 acquisition pool 中选择 K 个支持电芯，使用其寿命标签校准外层选择的预测器。测试电芯寿命标签只在最后评估阶段打开。

### 2.2 结构化协议—曲线表示

协议向量由五维实验条件构成；曲线向量由 H 循环早期退化特征构成。各模态单独进行 robust scaling，并形成协议距离 `D_p`、曲线距离 `D_c` 以及组合距离 `sqrt(D_p² + w D_c²)`。

### 2.3 非对称路由与安全回退

- `D_p=0`：强制 fallback；
- `0.30≤D_p<0.60`：中等协议离散度全部 fallback；
- `0<D_p<0.30` 且 `K∈{3,5,10}`：使用 `w=2` 的设施选择和固定带宽 `0.5` 的 log-life 核校准；
- `D_p≥0.60`：只有 `K=3` 启用主动选择；若 `ρ<0.35`，使用支持寿命中位数，否则使用 `w=0.5`、带宽 `0.5` 的 log-life 核校准；
- `D_p≥0.60` 且 `K≥5`：强制 fallback；
- `K=1`：始终 fallback，避免单标签条件下把采样噪声误判为可识别结构。

所有路由阈值、权重和预测器选择均在开发域 LODO 结果上冻结，目标测试寿命标签不参与 acquisition、router 或 predictor selection。

## 3. 实验协议

六个开发域为 CALB、HNEI、MICH_EXP、CALCE、MICH 和 SNL。设置 H∈{10,20,50}、K∈{1,3,5,10}，每个设置 100 个确定性 episode。主比较为 matched random-K fixed-pool baseline；补充比较包括 nested fixed-pool、普通主动采样、sequential GPR active learning、medoid/k-center 强选择器、PBT 接口、四种经典骨干、模态消融、no-rho、always-active、输入缺失/噪声、支持稳定性和 EOL cycle 成本。

主要统计端点为六域宏平均 MAPE，主终点为 H=50/K=3；报告域级改善/持平/退化计数、层级 bootstrap 95% CI、配对 cell Wilcoxon 检验和 Holm 校正。

## 4. 结果（当前冻结证据）

主结果表和图见 `experiments/battery/PAPER_EXPERIMENT_TABLES.md` 与 `experiments/battery/paper_figures/`。六域 H=50/K=3 为 48.5352→22.9422%；H=10/K=3 为 43.2863→24.2773%；H=20/K=3 为 44.4487→22.4691%。所有 12 个 H×K 设置均无相对 baseline 退化域，K=1 的逐位相同是预冻结安全回退而非增益。

强基线实验表明 APACE-Cal 的优势不只是来自弱随机基线；nested fixed-pool、普通 active selector、sequential GPR active learning 以及 medoid/k-center coverage selector 均呈现明显域依赖性。sequential GPR 在 K=3 的 H10/H20/H50 上分别由 matched-random GPR 的 49.48/51.00/56.20% 变为 66.31/66.67/61.77%，说明标准后验驱动的逐次获取也会发生负迁移；该比较使用其自身匹配的 GPR 基线，不与 APACE-Cal 的 MAPE 做跨预测器直接排名。PBT 原生全局偏置接口出现负结果，而残差核接口在开发域出现正向诊断，说明“支持集选择”和“预测器校准接口”必须共同设计，不能把方法声称成任意黑盒骨干插件。

## 4.1 强选择器、风险收益和跨预测器审计

新增的 fixed-pool 强选择器审计比较 protocol/curve/hybrid medoid 与
farthest-first/k-center。结果显示 coverage selector 在部分 H/K 设置优于
random，但胜负随域变化，不能稳定控制最差域退化；因此 APACE-Cal 的核心
贡献应表述为安全门控，而不是几何选择器普遍优于所有 baseline。

H50/K3 的跨预测器审计在相同支持集上逐一评估冻结 kernel、logmean、median
和 binary-LOO 接口。router 相对 matched random 在大多数 predictor 上保持
正向，但增益大小依赖 predictor，故本文不声称任意黑盒 predictor 即插即用。
层级风险收益审计显示三种主窗口均为 3/3 域改善、0 域退化，但逐电芯仍有
少量大幅退化行；正式统计对象应为域/episode 层级，并采用 cluster bootstrap
和预先声明的非劣效性 margin。

同一冻结协议的多指标重跑保存了逐预测残差。H50/K3 的 MAE 为
189.2812→141.2802，RMSE 为 250.2722→153.1716，sMAPE 为
29.4453→22.3307；H10/H20 的 MAPE 与冻结主 JSON 逐位一致。多指标结果放在
补充材料，主终点仍预先定义为宏平均 MAPE。

## 5. 外部盲测

SNU Dataset 1 在方法冻结和预标签 manifest 固定后开标签，H=50/K=3 为 0.4030345→0.2783977%，83/2/5 改善/持平/退化；由于寿命 CV 仅约 1.27%，只作为机制确认。NA-ion 有 34 个电芯，协议离散度恒为零，12 个设置全部 fallback，0/34/0 且逐位一致；这是安全性证据，不是 active 增益证据。补充的 BatteryLife 标准化外部审计包含 MATR 169-cell 和 HUST+XJTU 100-cell 完整标签队列：两者均按预冻结规则全部安全回退，分别为 0/169/0 和 0/100/0，逐位与 baseline 一致。MATR 的协议离散度为 0.4337，落入中等异质性 abstention 区间；因此不能把该大样本审计表述为 active 增益确认。另一个 HUST+Tongji+XJTU pooled manifest 因 Tongji 22 个处理电芯缺少官方 life label，未事后过滤，登记为数据契约失败。

UConn-MathWorks LFP/Gr 的 27-cell 外部 manifest 在寿命标签打开前冻结，H=10/20/50、K=3 分别取得 37.72%、38.41% 和 47.99% 的相对 MAPE 下降；K=1/5/10 按冻结规则回退。该结果与 SNU 一起构成严格外部 active confirmation，但样本量仍不足以支持所有化学体系上的普适 active 增益。

Zenodo 21700 Expt4/Expt5 通过 Range 方式完成了前50周期预标签和 episode 冻结；两者均未满足共同80% EOL。其 dataset-local 90% EOL 结果仅作为 nonblind exploratory，不计入独立 active confirmation。

## 6. 讨论与限制

结果支持条件式策略：当协议异质性和曲线结构提供足够可识别性时主动选择支持集，否则退回 matched baseline。完整压力测试显示曲线缺失和标签噪声下总体方向较稳定，但严重协议缺失会使主动增益消失；E7 小扰动审计还显示硬阈值在零/中等离散度边界附近可能发生 route flip。因此方法应被表述为带安全门控的条件式校准器，而不是扰动不变的路由分类器。限制包括：独立 active 确认主要来自 SNU 和 27-cell MathWorks 外部审计，而 169-cell MATR 与 100-cell HUST+XJTU 标准化队列均验证了安全回退而非 active 增益；Stroebl 大型外部档案尚未在本地完整读取；没有报告 90/95% 预测区间；并行墙钟成本只用 cycle proxy 表示；未声称任意深度骨干都可直接使用原生 APACE-Cal 校准器。

外部候选合同审计进一步检查了 UL-PUR、修正后的 SDU、SNU Dataset 2、RWTH 和
Luh–Blank。当前没有新的候选同时满足严格标签合同和 active route，因此没有为
扩大样本而放宽信息屏障、统一 EOL 或使用旧版非盲结果。已有 cycle-cost 仅是
支持寿命的 serial/parallel cycle proxy，不代表真实 wall-clock 或能量节省。

## 7. 可复现性

冻结方法与结果由 `batterylife_verify_apace_v2_artifacts.py` 验证；扩展实验包由 `verify_experiment_execution_artifacts.py` 验证。当前最终检查分别为 13/13 和 118/118 通过；扩展检查已覆盖 sequential GPR 和 MathWorks K3 表格口径。所有开发结果、外部 manifest、失败接口和负结果均保留，不覆盖冻结文件。
