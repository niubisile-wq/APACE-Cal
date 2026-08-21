# E4 获取模态消融

脚本：`batterylife_apace_v2_modality_acquisition_ablations.py`  
结果：`batterylife_apace_v2_protocol_only_ablation.json`、
`batterylife_apace_v2_curve_only_ablation.json`

只替换 facility-location 获取使用的距离，冻结 v2 的回退门、rho 分支、预测器、
外层选择、split 和 100 seeds 全部不变。

| H | 冻结 v2 | protocol-only | curve-only |
|---:|---:|---:|---:|
| 10/K3 | 24.277 | 26.442 | **23.247** |
| 20/K3 | **22.469** | 25.474 | 25.138 |
| 50/K3 | **22.942** | 30.369 | 23.861 |

逐域安全性也必须同时看：protocol-only 在 H10/H20/H50 K3 分别出现 1 个域
恶化，最坏相对变化 +25.71%、+27.80%、+8.69%；curve-only 三个 H 的 K3
均为 0 个退化域，但宏平均不总是最优。

## 判定

- 协议和早期曲线各自都携带有效信号；
- 单一模态会在某些 H 上胜出，但在另一些 H/域上失去稳健性；
- APACE 的多模态路由不是“固定拼接后必胜”，而是根据协议离散度和 rho
  决定何时启用哪类信息；
- 该消融支持“协议感知 + 早期曲线一致性”的机制叙述，但不应宣称每个 H
  上多模态宏平均严格优于所有单模态。

