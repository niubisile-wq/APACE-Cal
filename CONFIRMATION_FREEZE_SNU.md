# SNU 动态工况独立确认冻结

日期：2026-08-20 UTC

方法参数完全沿用 `METHOD_FREEZE_V2.md`，不因 ISU-ILCC 的标签—窗口技术冲突而调整。新的确认数据为 Mendeley Data DOI `10.17632/npjy7vdgky.1` 中的 Dataset 1；Dataset 2 仅用于在冻结 Dataset 1 manifest 前验证跨格式解析器和 evaluator。

官方原始包大小 `2,939,686,597` bytes，SHA-256 `2c58920e663fd089297ec2678a5a8ff791b6737466b9420bc2129c0cd2cde7ff`。

跨格式规则固定为：仅使用数值 `TotCycle=1..H`；3 Ah 标称容量、25°C 环境和 10 s 采样间隔取自官方元数据；每周期统计映射到既有 56 维签名；协议向量仍是温度、SOC 下/上界、前 H 周期充电 C-rate 中位数、放电 C-rate 中位数。EOL 唯一定义为完整 CSV 的最大数值 `TotCycle`。

Dataset 1 预标签读取必须在遇到首个数值 `TotCycle>50` 的行立即停止。只有独立 evaluate 模式允许扫到 EOF。任何 Dataset 1 文件尾在 manifest 写盘并哈希前均不得读取。

确认门槛沿用 v2：主动设置任一 H 不得恶化；至少一个主动设置改善 ≥10%；所有回退设置逐位相同；完整披露全部 H×K。

## 干跑与预标签筛查

Dataset 2 的 12 个电芯通过完整 evaluator 干跑，全部 12 个 H×K 设置按中等协议离散度回退，且方法与基线对每个电芯逐位相同。干跑结果 SHA-256 为 `425d466a2835e3657033bb242e8bc63517a8ce20c7ee00cca92007865c3fd825`。

Dataset 2 的基线 MAPE 仅约 0.38%–0.53%，提示以最大 `TotCycle` 为标签可能接近实验计划长度而非充分展开的退化寿命；该适用性风险在 Dataset 1 揭盲前预先披露。若 Dataset 1 同样近常数，结果不得夸大为一般寿命预测证据。

Dataset 1 的早期数据筛查发现 90 个电芯，只有 H50/K3 触发主动分支：H10/H20/H50 的 `D_p=0.48448/0.44721/0.64743`，H50 的 `ρ=0.25009`，故预定使用三设施支持中位数；其余 11 个设置回退。筛查没有读取任何 `TotCycle>50` 行。

## 正式冻结哈希

- SNU 跨格式盲测链脚本：`b195ddadb59fe8fe0f57232a9f30c941929149db166e2e1ed02b4784beda8e99`
- APACE-Cal v2 方法脚本：`438a0eeff5091e7bc65c2ed79eafafc9100e007c23800630c36adf90f6f9549b`
- v1 方法依赖：`6a41c9b965559056d74a0c22bb641d52ee1bc5cc9e69da04a48f9308d6a38e1f`
- 六域开发结果：`101c0d9b3161fb32c3ac80df8efc2a08089caddea82facfaba68bfbe323bbd65`

本记录现转为正式冻结。后续只允许以相同脚本重建 Dataset 1 manifest 并执行一次 evaluate；任何代码或规则修改都会使哈希检查失败。
