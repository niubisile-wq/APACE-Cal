# 新公开外部数据源侦察

本轮额外检查了公开数据源，筛选条件为：可访问的 cell-level 数据、前 50 周期曲线、寿命/EOL 标签、协议元数据，以及能够在标签打开前冻结 manifest。

| 数据源 | 侦察结果 | 当前决策 |
|---|---|---|
| [商用 21700 综合老化数据（Zenodo 10637533）](https://zenodo.org/records/10637533) | 有 raw/processed timeseries、性能汇总和多种实验，但完整压缩包约 6–12 GB/实验，且需重新确认统一 cell-level EOL 与前 50 周期接口 | 作为后续候选，不在当前实例盲目下载几十 GB |
| [Degradation path indicators（Zenodo 15755725）](https://zenodo.org/records/15755725) | 48 个商用电芯；公开描述显示每个电芯只有约 70 个 aging cycles 和 checkup/脉冲数据，不能直接满足完整 cycle-life/EOL 合同 | 排除当前 active confirmation |
| [VUB relaxation dataset（Zenodo 10899830）](https://zenodo.org/records/10899830) | 数据记录为 restricted access，当前 API 无公开 files | 无法在当前权限下冻结盲测 |
| BatteryLife 官方数据集 | 已完成 HUST/XJTU、MATR、SDU、Tongji 等合同筛查 | 现有队列已登记，新增候选均无合格 active route |
| UConn-ILCC NMC/Gr | 44 cells，EOL=65%，但当前标签成员/预标签流程已单独登记，不能重复包装为新 blind active confirmation | 保留现有非盲/预标签审计结论 |

数据源侦察并未产生可立即执行的第三个严格 active blind 队列；因此没有以下载不完整、EOL 不统一或 restricted 数据冒充外部验证。
