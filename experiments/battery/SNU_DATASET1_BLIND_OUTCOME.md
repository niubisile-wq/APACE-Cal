# SNU dynamic Dataset 1 blind outcome

Date: 2026-08-20 UTC

## Frozen chain

- Public source: Mendeley Data DOI `10.17632/npjy7vdgky.1`.
- Archive: 2,939,686,597 bytes; SHA-256
  `2c58920e663fd089297ec2678a5a8ff791b6737466b9420bc2129c0cd2cde7ff`.
- Cross-format and method freeze: `CONFIRMATION_FREEZE_SNU.md`, SHA-256
  `def4183ec017c12bb0c3bf2f8b1100f0c0961d3b8abb5e485e8316075f69a4c5`.
- Final prelabel manifest SHA-256:
  `a5115c810617f069a6245a2dcfa44c27ddc0e7c558138f2b81edd1813d783095`.
- The prelabel reader stopped every CSV before reading any row with numeric
  `TotCycle>50`; EOL was not derived until the frozen evaluator ran.
- Only H50/K3 was active (`D_p=0.647426`, rho `0.250092`), routing to three
  facilities plus support median. The other 11 settings were frozen fallbacks.

## One-shot result

- H50/K3 strong baseline MAPE: `0.4030345%`.
- H50/K3 APACE-Cal MAPE: `0.2783977%`.
- Relative MAPE reduction: `30.9246%`.
- Improved / tied / worse cells: `83 / 2 / 5` of 90.
- Two-sided paired cell Wilcoxon: `p=1.7648e-15`.
- All 11 fallback settings were bit-identical to the matched baseline for all
  90 cells.
- Blind result JSON SHA-256:
  `655fd2f8bbc829d738635cc6159a66d039069a84dbd8f3507c7ec1f25fa1af7d`.

The result passes the frozen confirmation red lines: no active-setting
degradation, active relative improvement above 10%, and exact safe fallback.

## Interpretation limit

The absolute baseline MAPE is already only 0.403%. Dataset 2 dry-run and the
Dataset 1 result indicate that maximum numeric `TotCycle` is tightly clustered
near the planned profile length. This experiment therefore confirms the
unlabeled routing and non-degradation mechanism under a cross-format dynamic
protocol shift, but it is weak evidence for large practical improvement on a
broad-life target. The paper must report both the 30.92% relative reduction and
the 0.1246 percentage-point absolute reduction, and must not use this domain
alone to claim broad real-world utility.
