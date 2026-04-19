"""
diagnostics.py - MCMC health checks and model comparison.
"""
import warnings
import arviz as az
import numpy as np
import pandas as pd


def check_convergence(idata, model_name=""):
    """Check R-hat and ESS for convergence. R-hat should be close to 1.0."""
    summary = az.summary(idata, filter_vars="like",
                         var_names=["alpha", "sigma", "nu", "alpha1", "delta",
                                    "w", "sigma1", "sigma2", "alpha_z", "alpha_d",
                                    "sigma_d", "tau_route"])
    cols = ["mean", "sd"] + [c for c in summary.columns if c.startswith("hdi_")] + ["r_hat", "ess_bulk"]
    summary = summary[[c for c in cols if c in summary.columns]]
    if model_name:
        print(f"\n--- Convergence: {model_name} ---")
    bad = summary[summary["r_hat"] > 1.01] if "r_hat" in summary.columns else pd.DataFrame()
    if not bad.empty:
        warnings.warn(f"{model_name}: {len(bad)} parameter(s) with R-hat > 1.01")
    else:
        print("  All R-hat values <= 1.01 ✓")
    return summary


def loo_comparison(model_dict):
    """
    Run PSIS-LOO for each model.
    Returns a comparison table and a Pareto-k diagnostic table.
    """
    loo_results = {}
    for name, idata in model_dict.items():
        try:
            loo_results[name] = az.loo(idata, pointwise=True)
        except Exception as e:
            warnings.warn(f"LOO failed for {name}: {e}")

    rows = []
    for name, loo in loo_results.items():
        elpd = float(loo.elpd_loo)
        se = float(loo.se)
        rows.append({"Model": name, "elpd_loo": elpd, "SE": se,
                     "LOOIC": -2.0 * elpd, "SE(LOOIC)": 2.0 * se})

    loo_table = (pd.DataFrame(rows)
                 .sort_values("LOOIC", ascending=True)
                 .reset_index(drop=True))
    loo_table["ΔLOOIC"] = (loo_table["LOOIC"] - loo_table.loc[0, "LOOIC"]).round(2)
    loo_table = loo_table.round({"elpd_loo": 2, "SE": 2, "LOOIC": 2, "SE(LOOIC)": 2})

    pareto_rows = []
    for name in loo_table["Model"]:
        if name not in loo_results:
            continue
        k = np.asarray(loo_results[name].pareto_k).ravel()
        n = len(k)
        pareto_rows.append({
            "Model": name, "n_obs": n,
            "k>0.5 (%)": round(100.0 * np.mean(k > 0.5), 2),
            "k>0.7 (%)": round(100.0 * np.mean(k > 0.7), 2),
            "k>1.0 (%)": round(100.0 * np.mean(k > 1.0), 2),
        })

    return loo_table, pd.DataFrame(pareto_rows)


def top_coefficients(fit, var_name, names, top_n=12):
    """Return top-N most influential predictors by absolute posterior mean."""
    draws = fit.stan_variable(var_name)
    q05, q50, q95 = np.quantile(draws, [0.05, 0.50, 0.95], axis=0)
    df = pd.DataFrame({"feature": names, "mean": draws.mean(axis=0),
                       "median": q50, "q05": q05, "q95": q95})
    order = np.argsort(np.abs(df["mean"].to_numpy()))[::-1]
    return df.iloc[order].head(top_n).reset_index(drop=True)


def route_effects_summary(fit, route_levels, top_n=8):
    """Summarise route-level intercept adjustments from Model D."""
    draws = fit.stan_variable("a_route")
    q05, q50, q95 = np.quantile(draws, [0.05, 0.50, 0.95], axis=0)
    df = pd.DataFrame({"route": route_levels, "mean": draws.mean(axis=0),
                       "median": q50, "q05": q05, "q95": q95})
    order = np.argsort(np.abs(df["mean"].to_numpy()))[::-1]
    return df.iloc[order].head(top_n).reset_index(drop=True)