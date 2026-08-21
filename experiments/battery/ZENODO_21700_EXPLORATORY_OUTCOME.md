# Zenodo 21700 Expt4 exploratory outcome

## Prelabel probe

通过 HTTP Range 只读取 Expt 4 ZIP 中每个电芯 cycle-summary 的前 50 行，未下载 11.4 GB 压缩包。8 个电芯覆盖 10/25/40°C，协议离散度为 0.2236；冻结规则在 K=3/5/10 触发 active_w2，K=1 fallback。

## 标签合同审计

预标签 manifest 冻结后才读取小型 performance-summary 文件。8 个电芯均达到 dataset-local 90% SoH，但只有 2/8 电芯达到 80% SoH；所有电芯的记录在约 4653 ageing cycles 截止。因此不存在共同的 80% EOL，不能使用最大观测周期伪造寿命标签。

## 非盲探索（不属于外部确认）

使用每个电芯首次 SoH≤90% 的 ageing cycles 作为 dataset-local exploratory label，结果如下：

| 设置 | 相对 MAPE 变化 |
|---|---:|
| H10/K3 | -15.49% |
| H10/K5 | -1.31% |
| H20/K3 | -23.73% |
| H20/K5 | +9.75% |
| H50/K3 | -8.84% |
| H50/K5 | +12.98% |

正负结果混合，且该标签定义是在性能摘要打开后才确定的；所以这组结果不能计入独立盲测，也不能写入主论文 active confirmation。它只证明该源并非一个可以无条件加入论文的干净外部验证集。

结果文件：

- `batterylife_zenodo_21700_expt4_prelabel.json`
- `batterylife_zenodo_21700_postlabel_audit.json`
- `batterylife_zenodo_21700_exploratory_90eol.json`
