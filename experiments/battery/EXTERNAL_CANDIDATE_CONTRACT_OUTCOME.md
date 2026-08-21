# 外部候选数据合同审计

本审计在任何新增标签打开前完成，仅读取冻结 prelabel manifest、协议路由和输入合同，不读取或改变标签值。

| 候选 | 预标签路由 | 合同状态 | 决策 |
|---|---|---|---|
| UL-PUR | 1200/1200 fallback-medium | 有效 | 无 active route，不能形成新增 active confirmation |
| SDU 旧 manifest | 900 active / 300 fallback | 无效 | 标签成员合同后来修正，旧结果不可用 |
| SDU 修正 manifest | 1200/1200 fallback-medium | 有效 | 不能形成新增 active confirmation |
| SNU Dataset 2 | 1200/1200 fallback-medium | 有效 | 只能作为安全回退审计 |
| RWTH | 旧版 adaptive 结果 | 不兼容 | 不是冻结 APACE-Cal v2 blind chain |
| Luh–Blank | 无前 50 周期曲线/统一 EOL | 不满足输入合同 | 技术排除 |
| Zenodo 21700 Expt4 | 8 cells，协议离散度 0.2236，K=3/5/10 触发 active；但共同 80% EOL 不存在 | 标签合同失败 | 仅保留非盲 90% EOL 探索，不计入外部确认 |
| Zenodo 21700 Expt5 | 8 cells，完整预冻结 episode manifest，K=3/5/10 触发 active；D/E 未达共同 80% EOL | 标签合同失败 | 90% EOL 探索正向但 nonblind，不计入外部确认 |

当前候选中没有同时满足“标签合同有效 + 严格 prelabel 冻结 + 至少一个 active route”的新增数据集。因此没有把任何候选强行打开成外部 active 结果。

机器审计结果：`batterylife_external_candidate_contract_audit.json`。
