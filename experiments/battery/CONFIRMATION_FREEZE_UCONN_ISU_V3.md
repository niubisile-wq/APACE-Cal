# UConn-ISU-ILCC v3 一次性外部确认冻结

冻结时间：2026-08-20 UTC。该文件建立在预标签 manifest 已生成之后；此后才允许读取 RPT 寿命信息。

## 输入与预标签证据

- 数据集：UConn-ISU-ILCC LFP/Gr，64 个电芯、11 组工况。
- 官方 EOL 定义：约 80% SOH；RPT 中的寿命信息在预标签阶段未读取。
- cycling_part1.zip、cycling_part2.zip、rpt_data.zip 均通过 `unzip -t` CRC 检查。
- 预标签 manifest：`/autodl-fs/data/battery_external_uconn_isu/uconn_isu_prelabel_manifest_v3.json`
- manifest SHA-256：`4188ce0682652149fbeaecbf68f847df371f06c4dfab2dc2281cd265488e2b23`
- 完整 episode manifest：`/autodl-fs/data/battery_external_uconn_isu/uconn_isu_manifest_v3.json`
- 完整 episode manifest SHA-256：`a8b14771d838a5f9a4e2849853b530965e8ba22558e5b63d823c725b98e7ce95`
- 预标签脚本 SHA-256：`680f586b3d5c86f4cebdec2309c014c7f90c71b62b31b078302ddc87ee986782`
- v3 稳定性候选脚本 SHA-256：`943fe63437db4cf8ee43b99635a528f4d2a5ac045c7c6b39a6f1591f0c6584ac`
- 完整 manifest 构建器 SHA-256：`86b58c7a2297082ad6822d74cc43c70d0a90700586b6cca5b562c164402b265e`

## 预标签结果

- H10/H20/H50 均有 64 个电芯的前50周期曲线。
- 协议离散度均为 `0.6026378174`，触发高离散度 K=3 active 分支。
- 协议/曲线 Spearman rho：H10=`0.6855454`、H20=`0.6345006`、H50=`0.6385690`。
- K=5/10 按冻结规则回退；K=1 按冻结规则回退。
- v3 完整 manifest 中仅 H10/K3、H20/K3、H50/K3 为 active，其他9个设置为回退。

## 揭盲后固定规则

- H∈{10,20,50}，K∈{1,3,5,10}，100 个共同 episode，70/30 acquisition/test。
- v3 稳定性拒识门：100 次无标签1%真实变化维度扰动，active route 翻转率不得超过5%；协议覆盖率低于90%时回退。
- 预测器、距离权重、带宽、外层LODO选择和支持身份规则均不因UConn结果修改。
- 一次性读取 RPT 中的 `Num Cycles`/官方寿命字段，生成完整结果；无论结果正负均保留。

该外部域在冻结后不得再调参。若触发的active设置出现退化，结果仍作为确认失败记录，不能用其反调方法。
