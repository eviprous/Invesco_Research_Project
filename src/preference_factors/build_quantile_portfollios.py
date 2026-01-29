import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from tqdm import tqdm
import statsmodels.api as sm


def compute_rolling_betas(
        returns: pd.DataFrame, 
        factor_returns: pd.DataFrame,  ## factor we are sorting by EBC or CW-EBC
        rolling_window: int =36, 
        min_obs: int =10
        ):
    """2 pass multi factor Fama-MacBeth"""

    asset_cols = returns.columns
    combined = returns.join(factor_returns, how="inner")

    # Prepare betas containers as float from the start
    betas_dict = {
        f: pd.DataFrame(index=pd.Index([], name="date"), columns=asset_cols, dtype=float)
        for f in factor_returns.columns
    }
    beta_dates = []

    # ---------- First pass: rolling time-series ----------
    for end_date in tqdm(combined.index[rolling_window-1:-1], desc="Time-series betas"):
        pos = combined.index.get_loc(end_date)
        next_date = combined.index[pos + 1]

        start_idx = pos - (rolling_window - 1)
        window_dates = combined.index[start_idx: start_idx + rolling_window]

        # Build window, drop any rows where either y or X has NaN
        x_win = factor_returns.loc[window_dates]
        for asset in asset_cols:
            y_win = returns.loc[window_dates, asset]
            data = pd.concat([y_win, x_win], axis=1).dropna()
            if len(data) < min_obs:
                for f in factor_returns.columns:
                    betas_dict[f].loc[next_date, asset] = np.nan
                continue

            Y = pd.to_numeric(data[asset], errors="coerce").astype(float)
            X = sm.add_constant(
                data[factor_returns.columns].apply(pd.to_numeric, errors="coerce").astype(float),
                has_constant="add"
            )

            # Guard against singular matrix if too few distinct X rows
            if X.shape[0] <= X.shape[1] or np.linalg.matrix_rank(X) < X.shape[1]:
                for f in factor_returns.columns:
                    betas_dict[f].loc[next_date, asset] = np.nan
                continue

            res = sm.OLS(Y, X).fit()

            for f in factor_returns.columns:
                betas_dict[f].loc[next_date, asset] = res.params.get(f, np.nan)

        beta_dates.append(next_date)

    return betas_dict

##### functinos to create and name the quantiles

def quantile_labels(factor_name: str, n_q: int):
    if n_q == 3:
        return {
            1: f"{factor_name}_Low",
            2: f"{factor_name}_Mid",
            3: f"{factor_name}_High",
        }
    elif n_q == 2:
        return {
            1: f"{factor_name}_Low",
            2: f"{factor_name}_High",
        }
    else:
        return {
            i: f"{factor_name}_Q{i}" for i in range(1, n_q + 1)
        }


def assign_quantiles(series: pd.Series, n_q: int) -> pd.Series:
    """
    Assign quantiles so that highest values get highest quantile.
    Returns integers 1,...,n_q
    """
    try:
        q = pd.qcut(series, n_q, labels=False)
    except Exception:
        ranks = series.rank(method="first")
        q = pd.cut(ranks, bins=n_q, labels=False)

    return (n_q - q).astype(int)

def relabel_quantile_index(index: pd.MultiIndex, n_q1: int, n_q2: int, factor_1_name: str, factor_2_name: str):
    """
    Convert (q1, q2) integer index to labeled strings.
    """
    q1_labels = quantile_labels(factor_1_name, n_q1)
    q2_labels = quantile_labels(factor_2_name, n_q2)

    new_index = pd.MultiIndex.from_tuples(
        [
            (date, q1_labels[q1], q2_labels[q2])
            for date, q1, q2 in index
        ],
        names=["date", factor_1_name, factor_2_name]
    )

    return new_index



##############################################
# Two-way quantile portfolios
##############################################

def build_two_way_quantile_portfolios(
    returns: pd.DataFrame,
    betas_1: pd.DataFrame,
    betas_1_name: str,
    betas_2: pd.DataFrame,
    betas_2_name: str,
    n_q1: int = 3,
    n_q2: int = 3,
    min_assets_per_cell: int = 3
):
    """
    Build two-way sorted equal-weighted portfolios.

    Returns both numeric and labeled versions.
    """

    dates_common = (
        returns.index
        .intersection(betas_1.index)
        .intersection(betas_2.index)
    )

    port_rows = []
    quantile_map = []

    for date in tqdm(sorted(dates_common), desc="Forming quantile portfolios"):
        df = pd.DataFrame({
            "beta1": betas_1.loc[date],
            "beta2": betas_2.loc[date],
            "ret": returns.loc[date]
        }).dropna()

        if df.empty:
            continue

        df["q1"] = assign_quantiles(df["beta1"], n_q1)
        df["q2"] = np.nan

        for i in range(1, n_q1 + 1):
            mask = df["q1"] == i
            if mask.sum() > 0:
                df.loc[mask, "q2"] = assign_quantiles(
                    df.loc[mask, "beta2"], n_q2
                )

        df["q2"] = df["q2"].astype(int)

        for i in range(1, n_q1 + 1):
            for j in range(1, n_q2 + 1):
                stocks = df.index[(df.q1 == i) & (df.q2 == j)].tolist()
                quantile_map.append({
                    "date": date,
                    "q1": i,
                    "q2": j,
                    "stocks": stocks
                })

                cell = df[(df.q1 == i) & (df.q2 == j)]
                n_assets = cell.shape[0]

                ret_ew = (
                    cell["ret"].mean()
                    if n_assets >= min_assets_per_cell
                    else np.nan
                )

                port_rows.append({
                    "date": date,
                    "q1": i,
                    "q2": j,
                    "ret_ew": ret_ew,
                    "n_assets": n_assets
                })

    port_long = (
        pd.DataFrame(port_rows)
        .set_index(["date", "q1", "q2"])
        .sort_index()
    )

    port_wide = (
        port_long["ret_ew"]
        .unstack(["q1", "q2"])
        .sort_index(axis=1)
    )
    port_wide.columns = [f"Q{q1}_Q{q2}" for q1, q2 in port_wide.columns]

    quantile_map_df = (
        pd.DataFrame(quantile_map)
        .set_index(["date", "q1", "q2"])
        .sort_index()
    )

    # ---- labeled versions (presentation only) ----
    port_long_labeled = port_long.copy()
    port_long_labeled.index = relabel_quantile_index(
        port_long.index, n_q1, n_q2,betas_1_name, betas_2_name
    )

    port_wide_labeled = (
        port_long_labeled["ret_ew"]
        .unstack([betas_1_name, betas_2_name])
    )
    port_wide_labeled.columns = [
        f"{q1}_{q2}" for q1, q2 in port_wide_labeled.columns
    ]

    quantile_map_labeled = quantile_map_df.copy()
    quantile_map_labeled.index = relabel_quantile_index(
        quantile_map_df.index, n_q1, n_q2, betas_1_name, betas_2_name
    )

    return {
        "wide_numeric": port_wide,
        "long_numeric": port_long,
        "quantile_map_numeric": quantile_map_df,
        "wide_labeled": port_wide_labeled,
        "long_labeled": port_long_labeled,
        "quantile_map_labeled": quantile_map_labeled
    }


