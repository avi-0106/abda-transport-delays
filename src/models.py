"""
models.py - Compiles and fits the four Stan models.
"""
from pathlib import Path
import arviz as az
import numpy as np
import xarray as xr
from cmdstanpy import CmdStanModel
from src.preprocessing import ModelData

MODELS_DIR = Path("models")

SAMPLE_ARGS = dict(
    chains=4,
    parallel_chains=4,
    iter_warmup=1000,
    iter_sampling=1000,
    seed=42,
    adapt_delta=0.99,
    max_treedepth=15,
)

def _load_model(stan_path, force_recompile=False):
    exe_path = stan_path.with_suffix(".exe")
    if exe_path.exists() and not force_recompile:
        return CmdStanModel(exe_file=str(exe_path))
    return CmdStanModel(stan_file=str(stan_path))

def _divergence_count(fit):
    try:
        return int(np.sum(fit.method_variables()["divergent__"]))
    except Exception:
        return 0

def _reshape_draws(fit, var_name):
    draws = fit.stan_variable(var_name)
    return draws.reshape(fit.chains, -1, draws.shape[-1])

def _make_idata(fit, observed_y, feature_vars=None, feature_names=None,
                feature_coord_name="feature", extra_coords=None, extra_dims=None):
    coords = {"obs_id": np.arange(len(observed_y))}
    dims = {"y": ["obs_id"], "log_lik": ["obs_id"]}
    if feature_vars and feature_names:
        coords[feature_coord_name] = feature_names
        for var in feature_vars:
            dims[var] = [feature_coord_name]
    if extra_coords:
        coords.update(extra_coords)
    if extra_dims:
        dims.update(extra_dims)
    idata = az.from_cmdstanpy(
        posterior=fit, log_likelihood="log_lik",
        observed_data={"y": observed_y}, coords=coords, dims=dims,
    )
    y_rep_data = _reshape_draws(fit, "y_rep")
    n_chains, n_draws, n_obs = y_rep_data.shape
    idata.posterior_predictive = xr.Dataset({
        "y_rep": xr.DataArray(y_rep_data, dims=["chain", "draw", "obs_id"],
                              coords={"obs_id": np.arange(n_obs)})
    })
    return idata

def _adjust_loglik(idata, y):
    observed = xr.DataArray(y, dims=["obs_id"], coords={"obs_id": np.arange(len(y))})
    ll_name = next(iter(idata.log_likelihood.data_vars))
    idata.log_likelihood[ll_name] = idata.log_likelihood[ll_name] - observed
    return idata

def fit_model_a(data, force_recompile=False):
    """Model A: Single Student-t regression on log1p(delay)."""
    print("\n=== Fitting Model A: Student-t regression ===")
    model = _load_model(MODELS_DIR / "model_a_student_t_regression.stan", force_recompile)
    fit = model.sample(
        data={"N": data.N, "K": data.K, "X": data.X, "y": data.y, "y_bar": data.y_bar},
        inits={"alpha": data.y_bar, "beta": np.zeros(data.K), "sigma": 0.5, "nu_minus2": 10.0},
        **SAMPLE_ARGS,
    )
    return _make_idata(fit, data.y, feature_vars=["beta"], feature_names=data.feature_names)

def fit_model_b(data, force_recompile=False):
    """Model B: Two-regime Student-t mixture regression."""
    print("\n=== Fitting Model B: Two-regime mixture regression ===")
    model = _load_model(MODELS_DIR / "model_b_mixture_regression.stan", force_recompile)
    fit = model.sample(
        data={"N": data.N, "K": data.K, "X": data.X, "y": data.y, "y_bar": data.y_bar},
        inits={"alpha1": data.y_bar, "beta1": np.zeros(data.K), "delta": 0.3,
               "sigma1": 0.4, "sigma2": 0.8, "w": 0.5, "nu1_minus2": 10.0, "nu2_minus2": 10.0},
        **SAMPLE_ARGS,
    )
    return _make_idata(fit, data.y, feature_vars=["beta1"], feature_names=data.feature_names)

def fit_model_c(data, force_recompile=False):
    """Model C: Hurdle lognormal regression."""
    print("\n=== Fitting Model C: Hurdle lognormal regression ===")
    model = _load_model(MODELS_DIR / "model_c_hurdle_lognormal.stan", force_recompile)
    fit = model.sample(
        data={"N": data.N, "K": data.K, "X": data.X, "delay": data.delay,
              "z": data.z.astype(int), "log_delay_pos_mean": data.log_delay_pos_mean,
              "logit_positive_rate": data.logit_positive_rate},
        inits={"alpha_z": data.logit_positive_rate, "beta_z": np.zeros(data.K),
               "alpha_d": data.log_delay_pos_mean, "beta_d": np.zeros(data.K), "sigma_d": 0.6},
        **SAMPLE_ARGS,
    )
    return _make_idata(fit, data.y, feature_vars=["beta_z", "beta_d"], feature_names=data.feature_names)

def fit_model_d(data, force_recompile=False):
    """Model D: Hierarchical Student-t with route-level partial pooling."""
    print("\n=== Fitting Model D: Hierarchical Student-t regression ===")
    model = _load_model(MODELS_DIR / "model_d_hierarchical_route.stan", force_recompile)
    sample_args = SAMPLE_ARGS.copy()
    init_d = {"alpha": data.y_bar, "beta": np.zeros(data.K_hier), "sigma": 0.5,
              "nu_minus2": 10.0, "tau_route": 0.3, "z_route": np.zeros(data.J_route)}
    stan_data = {"N": data.N, "K": data.K_hier, "J_route": data.J_route,
                 "X": data.X_hier, "route_id": data.route_idx, "y": data.y}
    fit = model.sample(data=stan_data, inits=init_d, **sample_args)
    n_div = _divergence_count(fit)
    if n_div > 0:
        print(f"  {n_div} divergences — refitting with stricter settings.")
        sample_args.update({"adapt_delta": 0.995, "max_treedepth": 16})
        fit = model.sample(data=stan_data, inits=init_d, **sample_args)
    idata = _make_idata(fit, data.y, feature_vars=["beta"],
                        feature_names=data.feature_names_hier,
                        extra_coords={"route": data.route_levels},
                        extra_dims={"z_route": ["route"], "a_route": ["route"]})
    return _adjust_loglik(idata, data.y)

def fit_all_models(data, force_recompile=False):
    """Fit all four models and return as a dict."""
    return {
        "Model A": fit_model_a(data, force_recompile),
        "Model B": fit_model_b(data, force_recompile),
        "Model C": fit_model_c(data, force_recompile),
        "Model D": fit_model_d(data, force_recompile),
    }