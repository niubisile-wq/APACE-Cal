# APACE-Cal

Code and reproducibility materials for **Safety-Gated Protocol-Aware Few-Shot Calibration for Cross-Dataset Battery Lifetime Prediction**.

APACE-Cal treats target-domain support acquisition as a selective decision. Using label-blind protocol dispersion, early degradation-curve structure and their concordance, the frozen policy chooses either:

- an active branch with structure-aware representative support; or
- an abstention branch that returns the matched-random fixed-pool baseline.

## Repository contents

- `experiments/battery/`: battery lifetime experiments, controls, ablations, robustness audits and configuration files.
- `paper/`: manuscript source and BibTeX references.
- `figures/`: paper figure sources and submission figures.
- `requirements-battery-experiments.lock`: experiment environment dependencies.
- `Dockerfile.battery-experiments`: container recipe for the experiment environment.

## Data

Datasets are not redistributed in this repository. Please obtain them from their original providers and configure the local data paths expected by the experiment scripts.

## Reproducibility

The primary policy is evaluated under fixed-pool episodes with fixed support budgets and frozen routing rules. The experiment configuration files document the corresponding settings. Results and manifests generated from private execution environments are intentionally not included unless they contain no restricted or identifying data.

## Citation

If you use this code, please cite the associated manuscript:

```bibtex
@article{liu_apacecal,
  title   = {Safety-Gated Protocol-Aware Few-Shot Calibration for Cross-Dataset Battery Lifetime Prediction},
  author  = {Liu, Zixuan and Xiong, Wei},
  journal = {Journal of Energy Storage},
  year    = {2026}
}
```

## License

License terms will be added before archival release. Until then, please contact the authors before redistributing substantial portions of the code.
