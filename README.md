# ABDA Transport Delays

Bayesian regression analysis comparing four probabilistic models for predicting public transport departure delays. Implemented in **Stan** via **CmdStanPy**, evaluated using **PSIS-LOO cross-validation**.

> This is a refactored version of a university project (Applied Bayesian Data Analysis, TU Dortmund). The original single-file notebook is preserved in `notebooks/original_analysis.ipynb`. This version restructures the analysis into a proper Python package with modular source files and a reproducible pipeline.

---

## The Problem

Public transport delay data has two properties that break standard regression:
1. **A spike at zero** — many trips depart exactly on time
2. **A heavy right tail** — a smaller number of trips are very late

We compare four Bayesian models that handle these properties differently.

---

## Models

| Model | Type | Key Idea |
|-------|------|----------|
| **A** | Student-t regression | Robust baseline on log-transformed delay |
| **B** | Two-regime mixture | Captures minor delays vs major disruptions |
| **C** | Hurdle lognormal | Explicitly separates zero vs positive delays |
| **D** | Hierarchical Student-t | Route-level partial pooling |

### Winner: Model B
The two-component mixture model achieved the best PSIS-LOO score. Real transport delays likely come from two distinct processes — routine operational variance and actual disruption events. The mixture captures this naturally.

| Model | elpd_LOO | ΔLOOIC |
|-------|----------|--------|
| **Model B** | -6255.69 | 0.00 |
| Model C | -6601.47 | 691.55 |
| Model D | -6729.17 | 946.97 |
| Model A | -6742.13 | 972.88 |

---

## Project Structure

```
abda-transport-delays/
├── main.py                  ← run this to execute the full pipeline
├── requirements.txt
├── Dockerfile
├── src/
│   ├── data_loader.py       ← loads dataset from Kaggle or local CSV
│   ├── preprocessing.py     ← feature engineering and design matrices
│   ├── models.py            ← fits all four Stan models
│   ├── diagnostics.py       ← R-hat, LOO, Pareto-k checks
│   └── visualisation.py     ← all plots
├── models/                  ← Stan model files (.stan)
├── notebooks/
│   └── original_analysis.ipynb  ← original notebook for reference
├── data/raw/                ← place dataset here (see below)
└── results/figures/         ← all plots saved here after running
```

---

## Setup

```bash
git clone https://github.com/avi-0106/abda-transport-delays.git
cd abda-transport-delays

python -m venv venv
source venv/Scripts/activate  # Windows
# source venv/bin/activate    # macOS/Linux

pip install -r requirements.txt
```

### Windows users only
Before installing CmdStan, install Strawberry Perl which provides the C++ build tools required to compile Stan models:
1. Download and install from https://strawberryperl.com/
2. Restart your terminal after installing
3. Then run:
```bash
python -m cmdstanpy.install_cmdstan
```

### macOS/Linux
```bash
python -m cmdstanpy.install_cmdstan
```

---

## Dataset

Download from Kaggle and place in `data/raw/`:
[Public Transport Delays with Weather and Events](https://www.kaggle.com/datasets/khushikyad001/public-transport-delays-with-weather-and-events)

## Run

```bash
# Auto-download from Kaggle
python main.py

# Use local CSV
python main.py --csv data/raw/transport_delays.csv

# Force recompile Stan models
python main.py --recompile
```

---

## Key Concepts

**PSIS-LOO** — measures how well the model predicts unseen data. Higher elpd = better.

**Pareto-k** — reliability check for LOO. Values below 0.5 = trustworthy estimates.

**Partial pooling** — each route gets its own intercept, shrunk toward the global mean. Sparse routes borrow strength from the full dataset.

---

## Authors
Abhishek Mishra · Siddhant Mishra  
Applied Bayesian Data Analysis — TU Dortmund University