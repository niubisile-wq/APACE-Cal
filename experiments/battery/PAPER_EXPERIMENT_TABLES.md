# APACE-Cal 论文实验主表（由冻结 JSON 汇总）

## 表 1：六域开发集固定总预算实验

数值为六个目标域宏平均 MAPE（%），每个电芯只产生一个预测，100 个
episode seeds。箭头为 matched baseline → APACE-Cal v2。方法只改变目标
域 K 个已完成电芯的选择；外层预测器在其余五个开发域上选择。

| 早期循环 H | K=1 | K=3 | K=5 | K=10 |
|---:|---:|---:|---:|---:|
| 10 | 78.0306 → 78.0306 | 43.2863 → **24.2773** | 33.8536 → **23.6949** | 27.0315 → **20.9349** |
| 20 | 74.1345 → 74.1345 | 44.4487 → **22.4691** | 33.6597 → **22.9002** | 28.7175 → **21.2384** |
| 50 | 80.0138 → 80.0138 | 48.5352 → **22.9422** | 31.0330 → **21.5097** | 23.5591 → **18.6239** |

12 个 H×K 设置均无相对 baseline 退化域；K=1 的逐位相同来自预先冻结
的不可识别性安全回退，不应写成主动选样增益。

## 表 2：七域开发/复核集 K=3 审计

MATR 已在 v1 盲测中开启标签，故这里只能作为“已见失败域的 v2 修复审计”，
不能当作独立确认。七域宏 MAPE（%）：

| H | baseline | v2 | 改善/持平/退化域 |
|---:|---:|---:|---:|
| 10 | 42.812 | 26.165 | 4 / 3 / 0 |
| 20 | 43.817 | 24.618 | 4 / 3 / 0 |
| 50 | 47.156 | 24.958 | 4 / 3 / 0 |

## 表 3：冻结后 SNU 动态工况独立确认

SNU Dataset 1 在方法冻结、预标签 manifest 固定后才读取寿命标签。唯一触发
主动路由的设置为 H=50,K=3：

| 设置 | baseline MAPE | APACE-Cal v2 MAPE | 相对改善 | 逐电芯结果 |
|---|---:|---:|---:|---:|
| SNU H50/K3 | 0.4030345% | **0.2783977%** | **30.9246%** | 83 改善 / 2 持平 / 5 退化 |

配对 cell bootstrap 的绝对 MAPE 改善 95% CI 为 [0.101787, 0.148415]
个百分点；相对改善 CI 为 [17.4557%, 65.4244%]。SNU 寿命分布 CV 约 1.27%，
所以该结果只作跨工况机制确认，不单独支撑宽寿命范围的实用性结论。

其余 11 个 SNU H×K 设置严格走回退分支，baseline 与方法逐位相同；这验证
了安全回退和信息屏障，但不应被计入主动方法的成功次数。

## 表 4：rho 路由机制消融（六域开发集）

去掉 rho 路由只把阈值设为 0，其余全部相同。K=3 的宏 MAPE（%）：

| H | 冻结 v2 | no-rho | 结论 |
|---:|---:|---:|---|
| 10 | 24.2773 | 24.0456 | 宏平均接近，不能声称 rho 带来总平均增益 |
| 20 | 22.4691 | 22.5603 | 去路由略差 |
| 50 | 22.9422 | 22.9422 | 路由未改变触发分支 |

逐域结果显示 rho 的作用是风险控制/分支选择，而非一个必然提高宏平均的
附加模块；该诚实表述应保留在论文的消融和局限部分。

## 表 5：NA-ion 独立盲测安全回退审计

NA-ion 共 34 个电芯，H∈{10,20,50}、K∈{1,3,5,10} 的 12 个设置全部在
预标签阶段测得协议离散度为 0，因此按冻结规则不触发 active route。每个设置
均为 0 改善 / 34 持平 / 0 退化，baseline 与方法 MAPE 逐位一致（配对 p=1）。
该表证明安全回退在新化学体系上有效，不应被解释为 active calibration gain。

## 结果文件与复现

- 冻结主结果：`batterylife_asymmetric_cohort_router_v2.json`
- 当前逐字节复跑：`batterylife_apace_v2_current_rerun.json`
- SNU 盲测：`snu_dynamic_dataset1_blind_eval.json`
- SNU 统计：`snu_dynamic_dataset1_blind_stats.json`
- rho 消融：`batterylife_apace_v2_no_rho_ablation.json`
- NA-ion 盲测：`batterylife_naion_blind_eval_v2.json`
- 论文图：`paper_figures/fig_main_heatmaps.pdf`、`paper_figures/fig_primary_ci.pdf`

## 表 6：sequential GPR 主动学习强基线

该基线使用与自身匹配的 random-GPR 支持初始化和相同固定池/总 K 协议，随后按
GPR 后验逐次获取。K=3 的宏 MAPE（%）为：

| H | matched-random GPR | sequential GPR | 方向 |
|---:|---:|---:|---|
| 10 | 49.4792 | 66.3126 | 退化 |
| 20 | 50.9977 | 66.6708 | 退化 |
| 50 | 56.2016 | 61.7700 | 退化 |

该结果用于证明标准后验驱动主动学习也会出现跨域负迁移。由于预测器和随机基线
均与 APACE-Cal 主表不同，不进行跨预测器的绝对 MAPE 排名。

## 表 7：MathWorks 严格外部 active confirmation

UConn-MathWorks LFP/Gr 共 27 个电芯，prelabel screening 和 episode identities
在寿命标签打开前冻结。K=3 的结果为：

| H | baseline MAPE | APACE-Cal MAPE | 相对改善 |
|---:|---:|---:|---:|
| 10 | 24.5717% | 15.3021% | 37.7247% |
| 20 | 25.8916% | 15.9459% | 38.4126% |
| 50 | 28.7249% | 14.9412% | 47.9850% |

K=1/5/10 按冻结规则与 matched baseline 逐位相同。该结果与 SNU 一起构成严格
外部 active confirmation，但 27-cell 样本量不足以支持所有化学体系的普适结论。

## 表 8：多指标主窗口补充审计

H50/K3 的 MAE 为 189.2812 → 141.2802，RMSE 为 250.2722 → 153.1716，
sMAPE 为 29.4453 → 22.3307。H10/H20/H50 的 MAPE 与冻结主 JSON 逐位复现。
层级 bootstrap、selector 强基线、跨 predictor 和外部候选合同结果分别见：

- `MULTIMETRIC_HIERARCHICAL_STATS_OUTCOME.md`
- `STRONG_SELECTOR_BASELINE_OUTCOME.md`
- `CROSS_PREDICTOR_AUDIT_OUTCOME.md`
- `EXTERNAL_CANDIDATE_CONTRACT_OUTCOME.md`

21700 Expt4/Expt5 的 90% EOL 结果均为 nonblind exploratory，不进入本表的独立
外部确认统计。
