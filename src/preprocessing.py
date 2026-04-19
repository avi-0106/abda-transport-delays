"""
preprocessing.py - Transforms raw data into model-ready arrays.
"""
from dataclasses import dataclass
import numpy as np
import pandas as pd

DELAY_CAP = 300

_DELAY_CANDIDATES = [
    "actual_departure_delay_min", "departure_delay_min",
    "delay_min", "delay_minutes", "Delay_Minutes", "Delay",
]

@dataclass
class ModelData:
    y: np.ndarray
    delay: np.ndarray
    z: np.ndarray
    X: np.ndarray
    X_hier: np.ndarray
    feature_names: list
    feature_names_hier: list
    route_idx: np.ndarray
    route_levels: list
    N: int
    K: int
    K_hier: int
    J_route: int
    y_bar: float
    positive_rate: float
    logit_positive_rate: float
    log_delay_pos_mean: float

def _find_delay_col(df):
    for c in _DELAY_CANDIDATES:
        if c in df.columns:
            return c
    raise KeyError(f"No delay column found. Columns: {df.columns.tolist()}")

def _zscore(series):
    values = pd.to_numeric(series, errors="coerce").astype(float)
    sd = values.std(ddof=0)
    if not np.isfinite(sd) or sd == 0:
        return pd.Series(np.zeros(len(values)), index=series.index)
    return (values - values.mean()) / sd

def build_model_data(df):
    """Transforms raw DataFrame into a ModelData object ready for Stan."""
    delay_col = _find_delay_col(df)
    print(f"Using delay column: '{delay_col}'")

    mdf = df.copy()
    raw = pd.to_numeric(mdf[delay_col], errors="coerce").replace([np.inf, -np.inf], np.nan)
    mdf["delay_minutes"] = raw.clip(lower=0, upper=DELAY_CAP)

    for col in ["transport_type", "route_id", "origin_station",
                "destination_station", "weather_condition", "season", "weekday"]:
        mdf[col] = mdf[col].fillna("Unknown").astype(str)
    mdf["event_type"] = mdf["event_type"].fillna("None").astype(str).replace("nan", "None")

    time_source = mdf["scheduled_departure"].fillna(mdf["time"])
    clock = pd.to_datetime(time_source, format="%H:%M:%S", errors="coerce")
    mdf["hour_float"] = clock.dt.hour + clock.dt.minute / 60.0
    mdf["hour_sin"] = np.sin(2 * np.pi * mdf["hour_float"] / 24.0)
    mdf["hour_cos"] = np.cos(2 * np.pi * mdf["hour_float"] / 24.0)
    mdf["has_event"] = (mdf["event_type"] != "None").astype(int)

    scaled_numeric = ["temperature_C", "precipitation_mm", "wind_speed_kmh",
                      "humidity_percent", "event_attendance_est", "traffic_congestion_index"]
    binary_cols = ["holiday", "peak_hour", "has_event"]
    cyclic_cols = ["hour_sin", "hour_cos"]
    categorical_cols = ["transport_type", "weather_condition", "season", "weekday",
                        "route_id", "origin_station", "destination_station"]

    keep = ["delay_minutes", "hour_float"] + scaled_numeric + binary_cols + cyclic_cols + categorical_cols
    adf = mdf[keep].dropna().reset_index(drop=True)
    adf["y"] = np.log1p(adf["delay_minutes"])

    for col in scaled_numeric:
        adf[col] = _zscore(adf[col])
    for col in binary_cols:
        adf[col] = adf[col].astype(int)

    X_num = adf[scaled_numeric + binary_cols + cyclic_cols].astype(float)
    X_cat = pd.get_dummies(adf[categorical_cols], drop_first=True, dtype=float)
    X_df = pd.concat([X_num, X_cat], axis=1)

    feature_names = X_df.columns.tolist()
    X = X_df.to_numpy(dtype=float)

    route_dummy_cols = [c for c in X_df.columns if c.startswith("route_id_")]
    X_hier_df = X_df.drop(columns=route_dummy_cols)
    feature_names_hier = X_hier_df.columns.tolist()
    X_hier = X_hier_df.to_numpy(dtype=float)

    route_levels = sorted(adf["route_id"].unique().tolist())
    route_lookup = {name: idx + 1 for idx, name in enumerate(route_levels)}
    route_idx = adf["route_id"].map(route_lookup).to_numpy(dtype=int)

    delay = adf["delay_minutes"].to_numpy(dtype=float)
    y = adf["y"].to_numpy(dtype=float)
    z = (delay > 0).astype(int)
    positive_rate = float(z.mean())

    N, K = X.shape
    _, K_hier = X_hier.shape

    print(f"Ready: N={N:,}  K={K}  K_hier={K_hier}  routes={len(route_levels)}")
    return ModelData(
        y=y, delay=delay, z=z, X=X, X_hier=X_hier,
        feature_names=feature_names, feature_names_hier=feature_names_hier,
        route_idx=route_idx, route_levels=route_levels,
        N=N, K=K, K_hier=K_hier, J_route=len(route_levels),
        y_bar=float(y.mean()), positive_rate=positive_rate,
        logit_positive_rate=float(np.log(positive_rate / (1 - positive_rate))),
        log_delay_pos_mean=float(np.log(delay[delay > 0]).mean()),
    )