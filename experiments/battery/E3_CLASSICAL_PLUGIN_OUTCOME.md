# E3：四类本地骨干插件结果

脚本：`batterylife_apace_classical_backbone_plugin.py`  
结果：`batterylife_apace_classical_backbone_plugin.json`  
状态：开发域跨骨干插件诊断；不修改冻结 APACE-Cal v2。

## 统一协议

对每一个 held-out 目标域，模型只在其余五个开发域训练。目标域只用前 H
特征做无标签路由，K 个支持 EOL 标签只用于残差校准；random-K 与 APACE-K
共享 split、测试池、seed 和模型输出。运行的四个源模型为 Ridge、Random
Forest、ExtraTrees、GradientBoosting；所有 model×calibrator×arm 都输出。

跨域 log-life 外推的固定数值护栏为源域最大 life：预测 log 值先裁剪到
`[0, log(max(source life))]`，该上界在读取目标标签前确定。第一次未加护栏的
运行因 exp overflow 被终止，不进入结果表；修正版全程无 warning/NaN。

## 关键 K=3 结果

下面是残差局部核（固定 `w=2,bw=0.5`）的六域宏平均 MAPE，random → APACE：

| H | Ridge | Random Forest | ExtraTrees | GradientBoosting |
|---:|---:|---:|---:|---:|
| 10 | 584.36 → 52.50 | 41.16 → **29.69** | 42.68 → **25.47** | 45.35 → **29.22** |
| 20 | 2485.77 → 33.84 | 40.39 → **26.66** | 41.85 → **22.25** | 43.69 → **28.15** |
| 50 | 2497.62 → 646.29 | 45.81 → **25.20** | 46.38 → **19.92** | 48.39 → **21.88** |

Ridge 的 random 残差核在若干域出现极端外推，因此不作为成功骨干；其 APACE
值也不能掩盖这一失败。RF/ExtraTrees/GBR 在 K=3 三个主动域上均显示正向
支持选择收益，但这仍是开发域证据，不是独立确认。

## 正确解释

1. APACE 的无标签支持队列具有跨传统骨干的可迁移信号；
2. 骨干输出必须配套局部残差校准，简单 global median log-bias 会失败；
3. 不能把该插件直接写成冻结 v2 的“任意骨干保证”；
4. 若论文保留插件结果，正文应把它作为方法可插拔性/机制支持，主方法仍是
   冻结的 APACE-Cal v2；若要升级为正式新接口，必须重新冻结并做新的盲域确认。

