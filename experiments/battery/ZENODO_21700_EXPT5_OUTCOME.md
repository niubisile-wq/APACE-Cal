# Zenodo 21700 Expt5 outcome

## Strict prelabel stage

Expt5 is the standard-cycle control experiment. The prelabel script accessed only
the ZIP central directory and the first 50 rows of each of 8 cycle-summary files
through HTTP Range requests; the complete 10.4 GB archive was not downloaded.
The frozen manifest contains all 100 episode acquisition/test/support identities
for H=10/20/50 and K=1/3/5/10. Protocol dispersion is 0.2236 and K=3/5/10
route to `active_w2`.

## EOL contract

After the manifest was frozen, performance summaries were opened. All 8 cells
reach 90% SoH, but cells D and E stop at 0.8103 and 0.8161 SoH, so a common 80%
EOL is not observed. The strict 80%-EOL evaluation is therefore not eligible.

## Nonblind 90%-EOL exploration

Using the first ageing-cycle point with SoH≤90% as a dataset-local exploratory
label gives large but unvalidated gains:

| Setting | Relative MAPE reduction |
|---|---:|
| H10/K3 | 75.28% |
| H10/K5 | 36.82% |
| H20/K3 | 66.46% |
| H20/K5 | 29.81% |
| H50/K3 | 79.17% |
| H50/K5 | 20.15% |

These values are explicitly **nonblind exploratory** because the 90% EOL choice
was made after the performance summaries were inspected. They cannot be added to
the independent active-confirmation count. They do justify prioritising a future
clean re-download with a pre-registered 90% EOL contract.

Artifacts:

- `batterylife_zenodo_21700_expt5_prelabel.json`
- `batterylife_zenodo_21700_expt5_postlabel_audit.json`
- `batterylife_zenodo_21700_expt5_exploratory_90eol.json`
