# UConn-ILCC NMC/Gr 外部候选预标签结果

该候选在任何 RPT/寿命字段未读取的情况下完成了 ZIP CRC、前50周期可用性和协议预标签。

- 44 个电芯在 H10/H20/H50 均有早期曲线；
- 协议离散度 `D_p=0.509373`；
- H10/H20/H50 的 K=1/3/5/10 全部落入冻结的中等离散度 fallback；
- 因无任何 active 设置，不读取 RPT 寿命标签，不执行 evaluator；
- 该候选仅验证安全回退，不作为主动增益确认。

机器清单：`/autodl-fs/data/battery_external_uconn_ilcc_nmc/uconn_nmc_manifest_v3.json`。
