import pandas as pd
import statsmodels.api as sm
import numpy as np
from scipy.optimize import minimize

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

def compute_rolling_market_betas_and_alphas(asset_ret_df, factors_df, window, beta_mkt = True):
    '''Compute the rolling betas and alphas for each stock in asset_ret_df against the factors in factors_df.'''
    factors = {'MKT_RF': beta_mkt}
    selected_factors = [factor for factor, include in factors.items() if include]
    results = {"alpha": []}
    for factor in selected_factors:
        results[f"beta_{factor}"] = []

    index = []
    for ticker in asset_ret_df.columns:
        for i in range(window, len(asset_ret_df)):
            ret_window = asset_ret_df[ticker].iloc[i - window:i]
            factor_window = factors_df[selected_factors].iloc[i - window:i]
            X = sm.add_constant(factor_window)
            Y = ret_window
            model = sm.OLS(Y, X).fit()
            results["alpha"].append(model.params["const"])
            for factor in selected_factors:
                results[f"beta_{factor}"].append(model.params[factor])
                index.append(asset_ret_df.index[i])
    results_df = pd.DataFrame(results, index=index)
    return results_df


def equal_beta_contribution_weights(betas):
    '''Calculate the weights for the portfolio to have an equal beta contribution on all assets.'''
    full_weights = pd.DataFrame(index=betas.index, columns=betas.columns)
    for row in betas.index:
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
        bounds = [(0, 1) for _ in range(n)]

        X0 = 1 / np.abs(beta_vals)
        X0 = X0 / np.sum(X0)

        result = minimize(objective, X0, bounds=bounds, constraints=constraints)
        if not result.success:
            continue

        weights = pd.Series(result.x, index=idx)

        full_weights.loc[row, weights.index] = weights
    full_weights = full_weights.fillna(0.0)
    return full_weights

def compute_EBC_betas(full_weights, betas):
    '''Compute the EBC betas given the full weights and betas.'''
    EBC_betas = pd.DataFrame(index=full_weights.index, columns=betas.columns)
    for row in full_weights.index:
        weight_row = full_weights.loc[row]
        beta_row = betas.loc[row]
        EBC_beta_row = weight_row * beta_row
        EBC_betas.loc[row] = EBC_beta_row
    return EBC_betas

def compute_EBC_returns(asset_returns, full_weights):
    '''Compute the EBC returns given the asset returns and full weights.'''
    EBC_returns = pd.Series(index=full_weights.index)
    for row in full_weights.index:
        weight_row = full_weights.loc[row]
        asset_return_row = asset_returns.loc[row]
        EBC_returns.loc[row] = (weight_row * asset_return_row).sum()
    return EBC_returns


def build_EBC_dataset_monthly(monthly_asset_returns, monthly_factors, window):
    betas = compute_rolling_market_betas_and_alphas(monthly_asset_returns, monthly_factors, window)
    full_weights = equal_beta_contribution_weights(betas)
    EBC_returns = compute_EBC_returns(monthly_asset_returns, full_weights)
    print("Sanity check: EBC betas")
    EBC_betas = compute_EBC_betas(full_weights, betas)
    print(EBC_betas.head())
    return EBC_returns


def build_EBC_dataset_daily(daily_asset_returns, daily_factors, window):
    betas = compute_rolling_market_betas_and_alphas(daily_asset_returns, daily_factors, window)
    full_weights = equal_beta_contribution_weights(betas)
    EBC_returns = compute_EBC_returns(daily_asset_returns, full_weights)
    print("Sanity check: EBC betas")
    EBC_betas = compute_EBC_betas(full_weights, betas)
    print(EBC_betas.head())
    return EBC_returns


def build_EBC_dataset(
    returns_file: str,
    ff_factors_file: str,
    window,
    frequency: str = 'monthly',
) -> pd.DataFrame:
    
    if frequency not in {"daily", "monthly"}:
        raise ValueError("frequency must be 'daily' or 'monthly'")
    
    # ------------------
    # Load data
    # ------------------
    returns = pd.read_csv(
    DATA_PROCESSED_DIR / returns_file,
    index_col=0,
    parse_dates=True
    )


    ff_factors = pd.read_csv(
        DATA_PROCESSED_DIR / ff_factors_file,
        index_col=0,
        parse_dates=True
    )

    # Ensure numeric
    rets = returns.apply(pd.to_numeric, errors="coerce")

    if frequency == "monthly":
        rets.index = rets.index.to_period("M")
        ff_factors.index = ff_factors.index.to_period("M")

    # Align returns and factors on common dates
    rets, ff_factors = rets.align(ff_factors, join="inner", axis=0)

    if frequency == "monthly":
        return build_EBC_dataset_monthly(rets,ff_factors,window)
    else:
        return build_EBC_dataset_daily(rets, ff_factors,window)