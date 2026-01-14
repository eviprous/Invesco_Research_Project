import pandas as pd
import statsmodels.api as sm
from src.preference_factors.build_preference_dataset import (
    build_preference_factor_dataset
)

def compute_rolling_betas_and_alphas(excess_ret_df, factors_df, ticker, window = 90, beta_mkt = True, beta_smb = False, beta_hml = False, beta_mom = False, beta_rmw = False, beta_cma = False):
    '''Compute the rolling betas and alphas for each stock in excess_ret_df against the factors in factors_df.'''
    factors = {"MKT-RF": beta_mkt, "SMB": beta_smb, "HML": beta_hml, "MOM": beta_mom, "RMW": beta_rmw, "CMA": beta_cma}
    selected_factors = [factor for factor, include in factors.items() if include]
    results = {"alpha": []}
    for factor in selected_factors:
        results[f"beta_{factor}"] = []
    index = []
    for i in range(window, len(excess_ret_df)):
        ret_window = excess_ret_df[ticker].iloc[i - window:i]
        factor_window = factors_df[selected_factors].iloc[i - window:i]
        X = sm.add_constant(factor_window)
        Y = ret_window
        model = sm.OLS(Y, X).fit()
        results["alpha"].append(model.params["const"])
        for factor in selected_factors:
            results[f"beta_{factor}"].append(model.params[factor])
            index.append(excess_ret_df.index[i])
            results_df = pd.DataFrame(results, index=index)
    return results_df

def equal_beta_contribution_weights():
    '''Calculate the weights for the portfolio to have an equal beta contribution on all assets.'''
    betas = compute_rolling_betas_and_alphas(....)
    valid_betas = betas.dropna(axis=0, how="all")
    valid_betas = valid_betas[valid_betas!= 0]
    return #weights

def compute_EBC_returns():

    return

def build_EBC_dataset_monthly():
    df_monthly = build_preference_factor_dataset(
    returns_file="sp500_returns_monthly_with_tickers.csv",
    #returns_file="sp500_returns_with_tickers_old.csv",
    market_caps_file="sp500_market_caps_monthly.csv",
    #market_caps_file="sp500_market_caps_old.csv",
    ff_factors_file="ff_factors_monthly.csv",
    frequency="monthly",
    )
    #also add EBC columns
    EBC_returns = compute_EBC_returns
    #merge with df monthly

    #create EBC - CW


def build_EBC_dataset_daily():
    returns
