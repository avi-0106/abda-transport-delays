"""
main.py - End-to-end pipeline for the Bayesian transport delay analysis.

Run from the project root:
    python main.py

Or with a local CSV:
    python main.py --csv data/raw/transport_delays.csv

Or force recompile Stan models:
    python main.py --recompile
"""
import argparse
from pathlib import Path
import arviz as az
from src.data_loader import load_dataset
from src.preprocessing import build_model_data
from src import models as mdl
from src import diagnostics as diag
from src import visualisation as viz

FIGURES_DIR = Path("results/figures")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
az.rcParams["stats.ci_prob"] = 0.95


def run(local_csv=None, force_recompile=False):

    # 1. Load & preprocess
    df = load_dataset(local_csv)
    data = build_model_data(df)

    # 2. EDA
    viz.eda_overview(data.delay, data.y,
                     save_path=FIGURES_DIR / "eda_overview.png")

    # 3. Fit all models
    print("\n--- Fitting models (takes a few minutes) ---")
    fitted = mdl.fit_all_models(data, force_recompile=force_recompile)

    # 4. Convergence diagnostics
    for name, idata in fitted.items():
        diag.check_convergence(idata, name)

    # 5. Posterior predictive checks
    viz.plot_ppc_delay(fitted["Model A"], data.delay,
                       "Model A — Student-t regression", zero_mode="approx",
                       save_path=FIGURES_DIR / "ppc_model_a.png")
    viz.plot_ppc_delay(fitted["Model B"], data.delay,
                       "Model B — Mixture regression", zero_mode="approx",
                       save_path=FIGURES_DIR / "ppc_model_b.png")
    viz.plot_ppc_delay(fitted["Model C"], data.delay,
                       "Model C — Hurdle lognormal", zero_mode="exact",
                       save_path=FIGURES_DIR / "ppc_model_c.png")
    viz.plot_ppc_delay(fitted["Model D"], data.delay,
                       "Model D — Hierarchical", zero_mode="approx",
                       save_path=FIGURES_DIR / "ppc_model_d.png")

    for name, idata in fitted.items():
        viz.plot_ppc_kde(idata, name,
                         save_path=FIGURES_DIR / f"ppc_kde_{name.replace(' ', '_').lower()}.png")

    viz.plot_trace(fitted["Model A"], ["alpha", "sigma", "nu"],
                   "Model A — trace", save_path=FIGURES_DIR / "trace_model_a.png")
    viz.plot_trace(fitted["Model B"], ["alpha1", "delta", "sigma1", "sigma2", "w"],
                   "Model B — trace", save_path=FIGURES_DIR / "trace_model_b.png")
    viz.plot_trace(fitted["Model C"], ["alpha_z", "alpha_d", "sigma_d"],
                   "Model C — trace", save_path=FIGURES_DIR / "trace_model_c.png")
    viz.plot_trace(fitted["Model D"], ["alpha", "tau_route", "sigma", "nu"],
                   "Model D — trace", save_path=FIGURES_DIR / "trace_model_d.png")

    # 6. Model comparison
    print("\n--- Model comparison (PSIS-LOO) ---")
    loo_table, pareto_table = diag.loo_comparison(fitted)
    print("\nLOO Table:")
    print(loo_table.to_string(index=False))
    print("\nPareto-k Summary:")
    print(pareto_table.to_string(index=False))

    viz.plot_loo_weights(fitted,
                         save_path=FIGURES_DIR / "loo_weights.png")
    viz.plot_zero_share(fitted, data.delay,
                        save_path=FIGURES_DIR / "zero_share.png")

    loo_results = {n: az.loo(idata, pointwise=True) for n, idata in fitted.items()}
    viz.plot_pareto_k(loo_results,
                      save_path=FIGURES_DIR / "pareto_k.png")
    viz.plot_ccdf(fitted,
                  save_path=FIGURES_DIR / "ccdf_tail.png")
    viz.plot_energy(fitted,
                    save_path=FIGURES_DIR / "energy.png")

    print("\n--- Done! Figures saved to results/figures/ ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default=None)
    parser.add_argument("--recompile", action="store_true")
    args = parser.parse_args()
    run(local_csv=args.csv, force_recompile=args.recompile)