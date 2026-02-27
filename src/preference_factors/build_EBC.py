import pandas as pd
import statsmodels.api as sm
import numpy as np
from scipy.optimize import minimize

from pathlib import Path
from tqdm.auto import tqdm
from joblib import Parallel, delayed

def _solve_ebc_for_row(row, betas_row):
    valid_betas = betas_row.replace(0.0, np.nan).dropna()
    if valid_betas.empty:
        return row, None

    beta_vals = valid_betas.values
    n = len(beta_vals)
    idx = valid_betas.index

    def objective(w):
        return np.sum((w * beta_vals - 1/n) ** 2)

    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
    bounds = [(0, 1)] * n

    X0 = 1 / np.abs(beta_vals)
    X0 /= X0.sum()

    result = minimize(objective, X0, bounds=bounds, constraints=constraints)
    if not result.success:
        return row, None

    return row, pd.Series(result.x, index=idx)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"

def compute_rolling_market_betas_and_alphas(asset_ret_df, factors_df, window, beta_mkt=True):
    """
    Compute rolling CAPM betas using:
    (R_i - RF) = alpha + beta * (MKT_RF)
    """

    factors = {"MKT_RF": beta_mkt}
    selected_factors = [f for f, include in factors.items() if include]

    dates = asset_ret_df.index[window:]
    tickers = asset_ret_df.columns

    alphas_df = pd.DataFrame(index=dates, columns=tickers, dtype=float)
    betas_df = pd.DataFrame(index=dates, columns=tickers, dtype=float)

    for i in tqdm(
        range(window, len(asset_ret_df)),
        desc="Rolling OLS CAPM",
        total=len(asset_ret_df) - window
    ):

        date = asset_ret_df.index[i]

        # Rolling window
        Y_win = asset_ret_df.iloc[i - window:i].astype(float)                 # raw returns
        X_win = factors_df[selected_factors].iloc[i - window:i].astype(float) # MKT_RF
        rf_win = factors_df["RF"].iloc[i - window:i].astype(float)

        # Align indices
        Y_win, X_win = Y_win.align(X_win, join="inner", axis=0)
        rf_win = rf_win.loc[Y_win.index]

        # ---- Proper CAPM: subtract RF ----
        Y_win = Y_win.sub(rf_win, axis=0)

        # Drop rows with NaNs in factors
        valid_rows = ~X_win.isna().any(axis=1)
        X_win = X_win.loc[valid_rows]
        Y_win = Y_win.loc[valid_rows]

        if len(X_win) < window:
            continue

        # Drop assets with NaNs
        valid_cols = ~Y_win.isna().any(axis=0)
        Y_win = Y_win.loc[:, valid_cols]

        if Y_win.shape[1] == 0:
            continue

        X = sm.add_constant(X_win, has_constant="add").values
        Y = Y_win.values

        XtX = X.T @ X

        # Condition check (important for early instability)
        if np.linalg.matrix_rank(XtX) < XtX.shape[0]:
            continue

        B = np.linalg.solve(XtX, X.T @ Y)

        alphas_df.loc[date, Y_win.columns] = B[0, :]
        betas_df.loc[date, Y_win.columns] = B[1, :]

    return alphas_df, betas_df

def equal_beta_contribution_weights_parallel(betas):
    '''Calculate the weights for the portfolio to have an equal beta contribution on all assets.'''

    # Run per-row optimizations in parallel
    results = Parallel(n_jobs=-1)(
        delayed(_solve_ebc_for_row)(row, betas.loc[row])
        for row in tqdm(betas.index, desc="EBC optimization (parallel)")
    )

    # Assemble results
    full_weights = pd.DataFrame(0.0, index=betas.index, columns=betas.columns)

    for row, weights in results:
        if weights is not None:
            full_weights.loc[row, weights.index] = weights

    return full_weights



def equal_beta_contribution_weights(betas):
    '''Calculate the weights for the portfolio to have an equal beta contribution on all assets.'''
    full_weights = pd.DataFrame(index=betas.index, columns=betas.columns)
    #for row in betas.index:
    for row in tqdm(betas.index, desc="EBC optimization"):

        
        beta_row = betas.loc[row]
        valid_betas = beta_row.replace(0.0, np.nan).dropna()
        if valid_betas.empty:
            continue

        beta_vals = valid_betas.values
        n = len(beta_vals)
        idx = valid_betas.index

        def objective(w):
            contribution = w * beta_vals
            return np.sum((contribution - 1/n) ** 2)
        
        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
        bounds = [(0, 1) for _ in range(n)] ## long only

        X0 = 1 / np.abs(beta_vals)
        X0 = X0 / np.sum(X0)

        result = minimize(objective, X0, bounds=bounds, constraints=constraints)
        if not result.success:
            continue

        weights = pd.Series(result.x, index=idx)

        full_weights.loc[row, weights.index] = weights
    full_weights = full_weights.fillna(0.0)
    return full_weights

def compute_EBC_betas_old(full_weights, betas):
    '''Compute the EBC betas given the full weights and betas.'''
    EBC_betas = pd.DataFrame(index=full_weights.index, columns=betas.columns)
    for row in full_weights.index:
        weight_row = full_weights.loc[row]
        beta_row = betas.loc[row]
        EBC_beta_row = weight_row * beta_row
        EBC_betas.loc[row] = EBC_beta_row
    return EBC_betas

def compute_EBC_betas(full_weights, betas):
    """
    Compute EBC beta contributions using aligned weights and betas.
    """

    # Align safely
    full_weights, betas = full_weights.align(
        betas,
        join="inner",
        axis=0
    )

    # Vectorized multiplication
    EBC_betas = full_weights * betas

    return EBC_betas


def compute_EBC_returns_old(asset_returns, full_weights):
    '''Compute the EBC returns given the asset returns and full weights.'''
    EBC_returns = pd.Series(index=full_weights.index)
    for row in full_weights.index:
        weight_row = full_weights.loc[row]
        asset_return_row = asset_returns.loc[row]
        EBC_returns.loc[row] = (weight_row * asset_return_row).sum()
    return EBC_returns

def compute_EBC_returns(asset_returns, full_weights):
    """
    Compute EBC returns using aligned weights and asset returns.
    """
    # Align automatically (safe)
    full_weights, asset_returns = full_weights.align(
        asset_returns,
        join="inner",
        axis=0
    )

    # Vectorized portfolio return
    EBC_returns = (full_weights * asset_returns).sum(axis=1)

    return EBC_returns



def build_EBC_dataset_monthly_old(monthly_asset_returns, monthly_factors, window= 36):
    alphas, betas = compute_rolling_market_betas_and_alphas(monthly_asset_returns, monthly_factors, window)
    full_weights = equal_beta_contribution_weights_parallel(betas)

    #Rebalance quarterly
    weights_rebal = full_weights.resample("Q").last()

    # Shift once to avoid look-ahead bias
    weights_rebal = weights_rebal.shift(1)

    # forward-fill to monthly
    weights_monthly = weights_rebal.reindex(monthly_asset_returns.index, method="ffill")

    weights_monthly, monthly_asset_returns = weights_monthly.align(
    monthly_asset_returns,
    join="inner",
    axis=0
    )

    EBC_returns = compute_EBC_returns(monthly_asset_returns, weights_monthly)

    #EBC_returns = compute_EBC_returns(monthly_asset_returns, full_weights)
    EBC_betas = compute_EBC_betas(weights_monthly, betas)
    EBC_returns = EBC_returns.to_frame(name="EBC_returns_monthly")
    return EBC_returns, weights_monthly, EBC_betas

def build_EBC_dataset_monthly(monthly_asset_returns, monthly_factors, window=36):

    # ---- Convert PeriodIndex → DatetimeIndex ONLY if needed ----
    if isinstance(monthly_asset_returns.index, pd.PeriodIndex):
        monthly_asset_returns = monthly_asset_returns.copy()
        monthly_factors = monthly_factors.copy()

        monthly_asset_returns.index = monthly_asset_returns.index.to_timestamp()
        monthly_factors.index = monthly_factors.index.to_timestamp()

    alphas, betas = compute_rolling_market_betas_and_alphas(
        monthly_asset_returns, monthly_factors, window
    )

    full_weights = equal_beta_contribution_weights_parallel(betas)

    if isinstance(full_weights.index, pd.PeriodIndex):
        full_weights.index = full_weights.index.to_timestamp()

    # ---- Quarterly Rebalancing ----
    weights_rebal = full_weights.resample("Q").last()

    # Shift once to avoid look-ahead bias
    weights_rebal = weights_rebal.shift(1)

    # Forward-fill to monthly
    weights_monthly = weights_rebal.reindex(
        monthly_asset_returns.index,
        method="ffill"
    )

    EBC_returns = compute_EBC_returns(
        monthly_asset_returns,
        weights_monthly
    )

    EBC_betas = compute_EBC_betas(weights_monthly, betas)

    EBC_returns = EBC_returns.to_frame(name="EBC_returns_monthly")

    return EBC_returns, weights_monthly, EBC_betas



def build_EBC_dataset_daily(daily_asset_returns, daily_factors, window= 252):
    alphas, betas = compute_rolling_market_betas_and_alphas(daily_asset_returns, daily_factors, window)
    
    full_weights = equal_beta_contribution_weights_parallel(betas)
    #full_weights = equal_beta_contribution_weights(betas)

    # Rebalance Monthly
    weights_rebal = full_weights.resample("M").last()
    #weights_rebal = weights_rebal.shift(1) --> we do NOT want to shift one month
    weights_rebal = weights_rebal.shift(1, freq="D")


    # forward-fill to daily
    weights_daily = weights_rebal.reindex(daily_asset_returns.index, method="ffill")

    weights_daily, daily_asset_returns = weights_daily.align(
    daily_asset_returns,
    join="inner",
    axis=0
    )

    EBC_returns = compute_EBC_returns(daily_asset_returns, weights_daily)

    EBC_betas = compute_EBC_betas(weights_daily, betas)

    EBC_returns = EBC_returns.to_frame(name="EBC_returns_daily")
    return EBC_returns, weights_daily, EBC_betas
