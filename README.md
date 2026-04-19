# ABDA Transport Delays

Bayesian regression analysis comparing four probabilistic models for predicting public transport departure delays. Implemented in **Stan** via **CmdStanPy**, evaluated using **PSIS-LOO cross-validation**.

> This is a refactored version of a university project (Advanced Bayesian Data Analysis, TU Dortmund). The original single-file notebook is preserved in `notebooks/original_analysis.ipynb`. This version restructures the analysis into a proper Python package with modular source files and a reproducible pipeline.

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

---

## Project Structure