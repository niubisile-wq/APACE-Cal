# E3：官方 PBT 固定总 K 插件结果

日期：2026-08-20 UTC  
脚本：`batterylife_apace_pbt_fixed_pool_plugin.py`  
结果：`batterylife_apace_pbt_fixed_pool_plugin.json`

## 协议

固定官方 PBT 三检查点集成，只改变 K 个目标支持电芯的身份：

- random-K：同 APACE 使用相同 70/30 acquisition/test split；
- APACE-K：使用冻结 v2 的无标签路由；
- 两组都使用同一个 PBT 预测和同一个 log-life median-bias 校准器；
- test EOL 只在支持身份冻结后读取；
- H=10/20/50，K=1/3/5/10，100 个共同 episode；
- PBT 覆盖完整的 CALB（27 cells）和 HNEI（14 cells）。

## 观察结果

HNEI 的 `D_p` 落在冻结中间回退区，所有设置 APACE 与 random 逐位相同。
CALB 主动场景的结果如下（MAPE，random → APACE）：

| H | K=3 | K=5 | K=10 |
|---:|---:|---:|---:|
| 10 | 216.775 → 233.600 | 225.489 → 238.954 | 239.928 → 241.894 |
| 20 | 216.777 → 238.053 | 228.281 → 246.142 | 235.264 → 243.648 |
| 50 | 221.582 → 241.694 | 237.335 → **226.973** | 239.638 → **234.932** |

H50/K3 的 APACE 退化为 20.11 个百分点，不能被忽略。H50/K5、K10
出现改善，但未形成一致的跨 H/K 优势。

## 判定

该实验**不支持**“APACE 支持选择可直接插入任意黑箱骨干并保持收益”的强主张。
它支持一个更精确的结论：APACE-Cal v2 的支持选择、路由和冻结的 local
log-life calibrator 是一个耦合方法；更换为 PBT 的固定 log-bias 接口后，
支持代表性不保证仍然对应 PBT 的残差代表性。

该负结果不是方法冻结失败：

1. 没有修改 v2；
2. 没有使用 CALB test EOL 做选择；
3. 没有删掉退化 H/K；
4. HNEI 的回退逐位相同，验证了安全分支；
5. CALB 的退化揭示了跨骨干适配需要单独定义残差校准器，不能事后包装成已解决。

后续可在开发域做一次预登记的接口诊断（PBT residual-kernel、support median、
random-K outer-selected calibrator），但不得把其中最有利的接口追加回冻结 v2，
也不得在未重新冻结和独立确认的情况下写成 APACE 原始方法结果。

