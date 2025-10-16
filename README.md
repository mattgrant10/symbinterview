# symbinterview

**Symbolic Machine Learning vs Deep Learning framework**

This project provides an educational pipeline comparing traditional (black‑box) ML methods
(Random Forest, XGBoost, Neural Network) against symbolic regression (PySR) on tennis match
data. It covers data loading, feature engineering, model training, equation discovery, and
side-by-side comparison.

## Quickstart

Clone the repo and install dependencies:

```bash
conda env create -f environment.yml
conda activate symbinterviews
# Optional: if using Poetry
poetry install
poetry shell
```

Run the analysis (by default it looks for Data1_ATP_symbint in the repo root):

```bash
python tennis_ml_comparison.py \
    [--data-dir Data1_ATP_symbint] \
    --start-year 2017 --end-year 2023 \
    [--quiet | --debug]
```

## Directory structure

```
.
├── Data1_ATP_symbint/         # raw Excel data by year
├── environment.yml            # Conda environment spec
├── pyproject.toml             # Poetry project spec
├── tennis_ml_comparison.py    # core script with both pipeline and CLI
├── symbolic_vs_deep_notebook.ipynb  # interactive exploration notebook
└── discovered_equations.txt   # output file for PySR equations (generated)
```

## License

Specify license here (e.g., MIT, Apache 2.0).