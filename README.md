# Physics-informed hybrid inverse recipe reconstruction
[![DOI](https://zenodo.org/badge/1455045202.svg)](https://doi.org/10.5281/zenodo.24157856)

Code for the manuscript *A Physics-Informed Hybrid Inverse Machine Learning Framework for High-Dimensional Sparse Recipe Reconstruction*.

> **No industrial data.** The production records used in the paper are under a non-disclosure agreement. `data/` holds a synthetic dataset with the same raw-file layout, so the whole pipeline runs, but its results do not reproduce the paper's numbers.

## Repository layout

```
piml-color/
├── configs/                 paths.yaml
├── data/                    synthetic data
│   ├── raw/                 recipe, cmc, production, pantones
│   ├── interim/             cleaned tables
│   ├── processed/           master table, features, targets
│   ├── splits/              cross-validation folds
│   ├── global_naming/       dye-to-hue mapping
│   └── manifests/           input/output hashes per stage
├── scripts/                 pipeline (00–06) and experiments (60–70)
├── src/piml_color_recipe/
│   ├── preprocessing/       auditing, cleaning, target matching, splits
│   ├── features/            spectral, dye, recipe and process features
│   ├── inverse/             Kubelka–Munk physics
│   └── utils/
└── outputs/                 results (not versioned)
```

## Setup

Python 3.11.

```
pip install -r requirements.txt
pip install -r requirements-optional.txt   # only for scripts 61 and 62
```

## Usage

Run every script from the repository root.

**1. Data pipeline.** Optional: `data/` is already built.

```
python scripts/make_synthetic_raw_data.py   # --seed N for a different dataset
python scripts/00_audit_raw_data.py
python scripts/01_clean_all.py
python scripts/02_build_master_table.py
python scripts/03_build_features.py
python scripts/04_make_validation_splits.py
python scripts/06_build_dye_hue_mapping.py
```

**2. Experiments.** Run script 60 first; the other scripts read its results.

| Script | Purpose |
|---|---|
| `60_physics_inverse_experiments.py --exp <id>` | `e00b` baselines, `e01` KM forward fit, `e02` exhaustive KM matching, `e16` final method and ablation |
| `61_attention_hybrid_experiments.py` | Attention and FT-Transformer comparisons |
| `62_nsga2_feature_selection.py` | Nested NSGA-II feature selection |
| `64_uncertainty_check.py` | Amount intervals and coverage |
| `68_statistics.py` | Descriptive statistics |
| `69_review_checks.py` | Calibration, identifiability and strata checks |
| `70_review_experiments.py --only B S R` | Blend control, KM simulation, repeated CV |

**3. Figures and tables.** `63_paper_figures.py` and `65_paper_tables.py` need the results of script 60 (`e00b e02 e08 e09 e12 e16 t01`) and script 61.

## Citation

If you use this code, please cite the preprint:

Md. Muqtadir Fuad, Md. Istiak Adnan, Md. Atiqur Rahman and Ridwan Al Aziz. *A Physics-Informed Hybrid Inverse Machine Learning Framework for High-Dimensional Sparse Recipe Reconstruction.* arXiv:2610.14097, 2026. https://arxiv.org/abs/2610.14097

```bibtex
@misc{fuad2026physics,
  title         = {A Physics-Informed Hybrid Inverse Machine Learning Framework for High-Dimensional Sparse Recipe Reconstruction},
  author        = {Fuad, Md. Muqtadir and Adnan, Md. Istiak and Rahman, Md. Atiqur and Aziz, Ridwan Al},
  year          = {2026},
  eprint        = {2610.14097},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG},
  doi           = {10.48550/arXiv.2610.14097},
  url           = {https://arxiv.org/abs/2610.14097}
}
```

## License

MIT. See [LICENSE](LICENSE).
