import pandas as pd
import statsmodels.api as sm

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

def equal_beta_contribution(betas):
    '''Calculate the weights for the portfolio to have an equal beta contribution on all assets.'''
    valid_betas = betas.dropna(axis=0, how="all")
    valid_betas = valid_betas[valid_betas!= 0]
    