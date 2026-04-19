"""
visualisation.py - All plots for the transport delay analysis.
"""
import math
import arviz as az
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import PercentFormatter
from scipy.stats import gaussian_kde

PALETTE = {
    "Model A": "#c67c2f",
    "Model B": "#8b9fb5",
    "Model C": "#0f766e",
    "Model D": "#7c4d8b",
    "observed": "#111827",
    "predictive": "#0f766e",
    "band": "#6bb8a9",
    "hist": "#e9d7aa",
    "zero_obs": "#c67c2f",
    "zero_pred": "#0f766e",
}

STYLE = {
    "figure.facecolor": "#f6f1e8",
    "axes.facecolor": "#fcfaf5",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.15,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "axes.labelsize": 11,
    "legend.frameon": False,
}

_rng = np.random.default_rng(42)


def _apply_style(ax):
    ax.set_facecolor(STYLE["axes.facecolor"])
    ax.grid(alpha=0.15, linewidth=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _density_band(y_rep, x, lower=0.10, upper=0.90, max_draws=250):
    idx = _rng.choice(y_rep.shape[0], size=min(max_draws, y_rep.shape[0]), replace=False)
    dens = []
    for i in idx:
        d = y_rep[i]
        d = d[np.isfinite(d)]
        if len(d) > 5 and np.std(d) > 1e-8:
            dens.append(gaussian_kde(d)(x))
    if not dens:
        return np.zeros_like(x), np.zeros_like(x), np.zeros_like(x)
    dens = np.vstack(dens)
    return np.median(dens, axis=0), np.quantile(dens, lower, axis=0), np.quantile(dens, upper, axis=0)


def eda_overview(delay, y, save_path=None):
    """Three-panel EDA: raw delay, log-transformed delay, zero vs positive share."""
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.5),
                                 gridspec_kw={"width_ratios": [1.2, 1.2, 0.8]})
        axes[0].hist(delay, bins=40, color="#e0bf7a", edgecolor="white", linewidth=0.8)
        axes[0].set_title("Raw departure delay")
        axes[0].set_xlabel("Minutes")
        axes[0].set_ylabel("Number of trips")

        axes[1].hist(y, bins=40, color="#74a57f", edgecolor="white", linewidth=0.8)
        axes[1].set_title("Log-transformed delay  [y = log(1 + delay)]")
        axes[1].set_xlabel("y = log1p(delay_minutes)")
        axes[1].set_ylabel("Number of trips")

        zero_share = np.mean(delay == 0)
        bars = axes[2].bar(["On time", "Delayed"], [zero_share, 1 - zero_share],
                           color=[PALETTE["zero_obs"], PALETTE["zero_pred"]], width=0.55)
        axes[2].yaxis.set_major_formatter(PercentFormatter(1.0))
        axes[2].set_title("On-time vs delayed share")
        for bar, val in zip(bars, [zero_share, 1 - zero_share]):
            axes[2].text(bar.get_x() + bar.get_width() / 2, val + 0.01,
                         f"{val:.1%}", ha="center", va="bottom", fontsize=11)
        for ax in axes:
            _apply_style(ax)
        fig.suptitle("Observed delay distribution — EDA", fontsize=15, fontweight="bold", y=1.02)
        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, bbox_inches="tight", dpi=150)
        plt.close(fig)


def plot_ppc_delay(idata, delay, title, zero_mode="approx", save_path=None):
    """Posterior predictive check on the delay scale."""
    y_rep_log = np.asarray(idata.posterior_predictive["y_rep"]).reshape(-1, len(delay))
    delay_rep = np.maximum(np.expm1(y_rep_log), 0.0)

    if zero_mode == "exact":
        pred_zero = np.mean(np.isclose(y_rep_log, 0.0), axis=1)
        ylabel = "Share of exact zeros"
    else:
        pred_zero = np.mean(delay_rep < 0.5, axis=1)
        ylabel = "Share of trips <0.5 min delay"

    obs_zero = float(np.mean(delay == 0))
    z_q10, z_q90 = np.quantile(pred_zero, [0.10, 0.90])
    pos_obs = delay[delay > 0]
    x = np.linspace(0, float(np.quantile(pos_obs, 0.99)), 450)
    d_med, d_lo, d_hi = _density_band(delay_rep, x)

    with plt.rc_context(STYLE):
        fig = plt.figure(figsize=(13, 5.5), facecolor=STYLE["figure.facecolor"])
        gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 2.4], wspace=0.22)
        ax0, ax1 = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])

        bars = ax0.bar(["Observed", "Predicted"], [obs_zero, pred_zero.mean()],
                       color=[PALETTE["zero_obs"], PALETTE["zero_pred"]], width=0.6)
        ax0.errorbar(1, pred_zero.mean(),
                     yerr=[[pred_zero.mean() - z_q10], [z_q90 - pred_zero.mean()]],
                     fmt="none", ecolor="#184e47", elinewidth=2, capsize=6)
        ax0.yaxis.set_major_formatter(PercentFormatter(1.0))
        ax0.set_ylim(0, max(obs_zero, z_q90) * 1.3)
        ax0.set_title("Zero-delay mass fit")
        ax0.set_ylabel(ylabel)
        for bar, val in zip(bars, [obs_zero, pred_zero.mean()]):
            ax0.text(bar.get_x() + bar.get_width() / 2, val + 0.005,
                     f"{val:.1%}", ha="center", va="bottom", fontsize=11)
        _apply_style(ax0)

        ax1.hist(pos_obs, bins=28, density=True, color=PALETTE["hist"],
                 edgecolor="white", linewidth=0.8, alpha=0.9, label="Observed")
        ax1.fill_between(x, d_lo, d_hi, color=PALETTE["band"], alpha=0.3, label="80% predictive band")
        ax1.plot(x, d_med, color=PALETTE["predictive"], linewidth=2.5, label="Median prediction")
        ax1.set_xlabel("Delay (minutes)")
        ax1.set_ylabel("Density")
        ax1.set_title("Positive-delay density fit")
        ax1.legend()
        _apply_style(ax1)

        fig.suptitle(title, fontsize=15, fontweight="bold", y=0.99)
        if save_path:
            fig.savefig(save_path, bbox_inches="tight", dpi=150)
        plt.close(fig)


def plot_ppc_kde(idata, title, save_path=None):
    """KDE posterior predictive check on the log scale."""
    y_obs = np.asarray(idata.observed_data["y"]).ravel()
    y_rep = np.asarray(idata.posterior_predictive["y_rep"]).reshape(-1, len(y_obs))
    merged = np.concatenate([y_obs, y_rep.ravel()])
    x_max = float(np.quantile(merged[np.isfinite(merged)], 0.995))
    x = np.linspace(0, max(0.25, x_max), 500)
    d_med, d_lo, d_hi = _density_band(y_rep, x)
    obs_kde = gaussian_kde(y_obs)(x)

    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(9.5, 5.0), facecolor=STYLE["figure.facecolor"])
        ax.hist(y_obs, bins=28, density=True, color="#ead39c",
                edgecolor="white", linewidth=0.7, alpha=0.65, label="Observed histogram")
        ax.fill_between(x, d_lo, d_hi, color=PALETTE["band"], alpha=0.28, label="80% predictive band")
        ax.plot(x, d_med, color=PALETTE["predictive"], linewidth=2.5, label="Median prediction")
        ax.plot(x, obs_kde, color="#9a3412", linewidth=2.2, linestyle="--", label="Observed KDE")
        ax.set_xlim(0, x_max)
        ax.set_xlabel("y = log1p(delay_minutes)")
        ax.set_ylabel("Density")
        ax.set_title(f"{title} — log-scale KDE check")
        ax.legend()
        _apply_style(ax)
        if save_path:
            fig.savefig(save_path, bbox_inches="tight", dpi=150)
        plt.close(fig)


def plot_trace(idata, var_names, title, last_draws=500, figsize=(14, 8), save_path=None):
    """MCMC trace plots — chains should overlap like a hairy caterpillar."""
    if not var_names:
        return
    idata_slice = idata.sel(draw=slice(-last_draws, None))
    with plt.rc_context(STYLE):
        az.plot_trace(idata_slice, var_names=var_names)
        fig = plt.gcf()
        fig.suptitle(title, fontsize=15, fontweight="bold", y=0.995)
        fig.subplots_adjust(top=0.92, hspace=0.52, wspace=0.20)
        if save_path:
            fig.savefig(save_path, bbox_inches="tight", dpi=150)
        plt.close(fig)


def plot_energy(model_dict, save_path=None):
    """Energy diagnostic grid — one panel per model."""
    names = list(model_dict.keys())
    n = len(names)
    n_cols = 2 if n > 1 else 1
    n_rows = math.ceil(n / n_cols)
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(n_rows, n_cols,
                                 figsize=(7.0 * n_cols, 4.8 * n_rows), squeeze=False)
        for ax, name in zip(axes.ravel(), names):
            energy = np.asarray(model_dict[name].sample_stats["energy"])  # (chain, draw)
            marginal = (energy - energy.mean()).ravel()
            transition = np.diff(energy, axis=1).ravel()
            x = np.linspace(min(marginal.min(), transition.min()),
                            max(marginal.max(), transition.max()), 300)
            kde_m = gaussian_kde(marginal)(x)
            ax.fill_between(x, kde_m, alpha=0.25, color="#0f766e")
            ax.plot(x, kde_m, color="#0f766e", linewidth=2, label="Marginal energy")
            if np.std(transition) > 1e-8:
                ax.plot(x, gaussian_kde(transition)(x), color="#c67c2f",
                        linewidth=2, label="Energy transition")
            ax.set_xlabel("Centered energy")
            ax.legend(fontsize=9)
            ax.set_title(name, fontsize=13, fontweight="bold")
            _apply_style(ax)
        for ax in axes.ravel()[n:]:
            ax.axis("off")
        fig.suptitle("Energy diagnostics", fontsize=14, fontweight="bold", y=0.995)
        fig.subplots_adjust(top=0.90, hspace=0.40, wspace=0.26)
        if save_path:
            fig.savefig(save_path, bbox_inches="tight", dpi=150)
        plt.close(fig)


def plot_loo_weights(model_dict, save_path=None):
    """LOO stacking weights — higher means better predictive contribution."""
    cmp = az.compare(model_dict)
    weights = cmp["weight"].sort_values(ascending=True)
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(7, 4), facecolor=STYLE["figure.facecolor"])
        colors = [PALETTE.get(n, "#888") for n in weights.index]
        ax.barh(weights.index, weights.values, color=colors)
        ax.set_xlabel("Stacking weight")
        ax.set_title("LOO stacking weights\n(higher = better predictive contribution)")
        for i, (name, val) in enumerate(weights.items()):
            ax.text(val + 0.005, i, f"{val:.2f}", va="center", fontsize=10)
        _apply_style(ax)
        if save_path:
            fig.savefig(save_path, bbox_inches="tight", dpi=150)
        plt.close(fig)


def plot_zero_share(model_dict, delay, save_path=None):
    """Predicted vs observed zero-delay share per model."""
    labels, means = [], []
    for name, idata in model_dict.items():
        y_rep = np.asarray(idata.posterior_predictive["y_rep"]).reshape(-1, len(delay))
        delay_rep = np.maximum(np.expm1(y_rep), 0.0)
        if name == "Model C":
            share = np.mean(np.isclose(y_rep, 0.0), axis=1).mean()
        else:
            share = np.mean(delay_rep < 0.5, axis=1).mean()
        labels.append(name)
        means.append(float(share))

    obs_zero = float(np.mean(delay == 0))
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(7, 4), facecolor=STYLE["figure.facecolor"])
        ax.bar(labels, means, color=[PALETTE.get(n, "#888") for n in labels], width=0.55)
        ax.axhline(obs_zero, color="#4b3a2a", linestyle="--", linewidth=1.8,
                   label=f"Observed ({obs_zero:.1%})")
        ax.yaxis.set_major_formatter(PercentFormatter(1.0))
        ax.set_title("Predicted vs observed zero-delay share")
        ax.legend()
        _apply_style(ax)
        if save_path:
            fig.savefig(save_path, bbox_inches="tight", dpi=150)
        plt.close(fig)


def plot_pareto_k(loo_results, save_path=None):
    """Pareto-k histogram grid — most values should be below 0.5."""
    names = list(loo_results.keys())
    n = len(names)
    n_cols = 2 if n > 1 else 1
    n_rows = math.ceil(n / n_cols)
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(n_rows, n_cols,
                                 figsize=(7.4 * n_cols, 4.4 * n_rows), squeeze=False)
        for ax, name in zip(axes.ravel(), names):
            k = np.asarray(loo_results[name].pareto_k).ravel()
            ax.hist(k, bins=40, color="#8b9fb5", alpha=0.85, edgecolor="white")
            ax.axvline(0.5, color="#0f766e", linestyle="--", linewidth=1.5, label="k=0.5")
            ax.axvline(0.7, color="#c67c2f", linestyle="--", linewidth=1.5, label="k=0.7")
            ax.axvline(1.0, color="#9f1239", linestyle="--", linewidth=1.5, label="k=1.0")
            ax.set_title(f"{name} — {100.0 * np.mean(k > 0.7):.1f}% with k > 0.7")
            ax.set_xlabel("Pareto-k")
            ax.set_ylabel("Count")
            ax.legend(fontsize=9)
            _apply_style(ax)
        for ax in axes.ravel()[n:]:
            ax.axis("off")
        fig.suptitle("Pareto-k diagnostics (most should be below 0.5)",
                     fontsize=14, fontweight="bold", y=0.995)
        fig.subplots_adjust(top=0.90, hspace=0.35, wspace=0.26)
        if save_path:
            fig.savefig(save_path, bbox_inches="tight", dpi=150)
        plt.close(fig)


def plot_ccdf(model_dict, save_path=None):
    """Tail CCDF on log y-axis — shows how well each model fits long delays."""
    def _ccdf(samples, x_grid):
        s = np.sort(np.asarray(samples, dtype=float))
        return 1.0 - np.searchsorted(s, x_grid, side="right") / max(len(s), 1)

    names = list(model_dict.keys())
    n = len(names)
    n_cols = 2 if n > 1 else 1
    n_rows = math.ceil(n / n_cols)
    local_rng = np.random.default_rng(123)

    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(n_rows, n_cols,
                                 figsize=(7.4 * n_cols, 4.8 * n_rows), squeeze=False)
        for ax, name in zip(axes.ravel(), names):
            idata = model_dict[name]
            y_obs = np.asarray(idata.observed_data["y"]).ravel()
            y_rep = np.asarray(idata.posterior_predictive["y_rep"]).reshape(-1, len(y_obs))
            merged = np.concatenate([y_obs, y_rep.ravel()])
            x_max = float(np.quantile(merged[np.isfinite(merged)], 0.995))
            x = np.linspace(0, max(0.25, x_max), 400)
            obs_ccdf = _ccdf(y_obs[np.isfinite(y_obs)], x)
            draw_idx = local_rng.choice(y_rep.shape[0], size=min(250, y_rep.shape[0]), replace=False)
            ccdfs = [_ccdf(y_rep[d][np.isfinite(y_rep[d])], x)
                     for d in draw_idx if np.isfinite(y_rep[d]).sum() > 2]
            ccdfs = np.vstack(ccdfs) if ccdfs else np.array([obs_ccdf])
            eps = 1e-6
            ax.fill_between(x, np.maximum(np.quantile(ccdfs, 0.10, axis=0), eps),
                            np.maximum(np.quantile(ccdfs, 0.90, axis=0), eps),
                            color=PALETTE["band"], alpha=0.30, label="80% band")
            ax.plot(x, np.maximum(np.median(ccdfs, axis=0), eps),
                    color=PALETTE["predictive"], linewidth=2.2, label="Predicted")
            ax.plot(x, np.maximum(obs_ccdf, eps),
                    color=PALETTE["observed"], linewidth=2.0, label="Observed")
            ax.set_yscale("log")
            ax.set_title(name)
            ax.set_xlabel("y = log1p(delay)")
            ax.set_ylabel("P(Y > y)")
            ax.legend(fontsize=9)
            _apply_style(ax)
        for ax in axes.ravel()[n:]:
            ax.axis("off")
        fig.suptitle("Tail CCDF — how well each model fits long delays",
                     fontsize=14, fontweight="bold", y=0.995)
        fig.subplots_adjust(top=0.90, hspace=0.36, wspace=0.26)
        if save_path:
            fig.savefig(save_path, bbox_inches="tight", dpi=150)
        plt.close(fig)