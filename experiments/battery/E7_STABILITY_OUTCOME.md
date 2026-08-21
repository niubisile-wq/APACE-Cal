# E7 支持队列稳定性

脚本：`batterylife_apace_support_stability.py`  
数据：`batterylife_apace_cost_audit.json` 中 100 个共同 episode 的支持身份。

H=10/20/50、K=3 在六个开发域平均两两 Jaccard：

| H | random-K | APACE-K |
|---:|---:|---:|
| 10 | 0.1125 | **0.2880** |
| 20 | 0.1100 | **0.2819** |
| 50 | 0.1100 | **0.2913** |

APACE 选择的支持集合比随机支持更稳定，但 Jaccard 仍远小于 1，说明它不是
固定挑同一批电芯；它会随 acquisition pool 和 tie seed 变化，同时保留更强的
协议/曲线覆盖。该结果支持“稳定但非硬编码”的解释，不应把稳定性夸大成唯一性。

