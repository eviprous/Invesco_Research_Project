import numpy as np
import pandas as pd
import statsmodels.api as sm

def run_time_series_regressions(portrets_wide, factors, f1_name, f2_name, frequency="monthly"):
    results = {}

    # Adaptive Parameters
    # Daily: 20 lags (~1 month), Min 60 obs (~3 months)
    # Monthly: 6 lags (~0.5 year), Min 12 obs (1 year)
    hac_lags = 20 if frequency == "daily" else 6
    min_obs = 60 if frequency == "daily" else 12

    for p in portrets_wide:
        y = portrets_wide[p].dropna()
        if len(y) < min_obs:
            results[p] = None
            continue

        # Ensure factors align with the available portfolio returns
        X = sm.add_constant(factors.loc[y.index])

        try:
            # ACADEMIC FIX: Use Newey-West (HAC) standard errors
            m = sm.OLS(y, X).fit(cov_type='HAC', cov_kwds={'maxlags': hac_lags})
        except Exception:
            results[p] = None
            continue

        results[p] = {
            "alpha": m.params.get("const", np.nan),
            "alpha_t": m.tvalues.get("const", np.nan),
            "betaEBC": m.params.get(f1_name, np.nan),
            "betaCap_EBC": m.params.get(f2_name, np.nan),
            "betaEBC_t": m.tvalues.get(f1_name, np.nan),
            "betaCap_EBC_t": m.tvalues.get(f2_name, np.nan),
            "rsq": m.rsquared,
            "nobs": int(m.nobs),
        }

    return pd.DataFrame.from_dict(results, orient="index")

def run_fama_macbeth(portrets_wide, beta_table, frequency="monthly"):
    """
    Second Pass: Monthly cross-sectional regressions.
    Standard Fama-MacBeth (1973) methodology.
    """
    gamma_rows = []
    # Loop over every unique date present in the portfolio returns
    for date, row in portrets_wide.iterrows():
        R_t = row.dropna()
        if len(R_t) < 3: # Need enough assets to run OLS -> constant + 2 factors
            continue
            
        # Match the Betas for the assets that exist on this specific date
        X = sm.add_constant(beta_table.loc[R_t.index])
        
        try:
            # The cross-sectional OLS for month 'date'
            res = sm.OLS(R_t, X).fit()
            
            res_dict = {
                "date": date, 
                "alpha": res.params.iloc[0],           # The Intercept (g0)
                "lambda_EBC": res.params.iloc[1],      # The first factor premium (g1)
                "lambda_Cap_EBC": res.params.iloc[2]   # The second factor premium (g2)
            }
            gamma_rows.append(res_dict)
        except Exception:
            continue

    # Convert to DataFrame
    gammas = pd.DataFrame(gamma_rows).set_index("date").dropna()

    if gammas.empty:
        raise ValueError("No valid Fama-MacBeth regressions were estimated.")

    mean_lambdas = gammas.mean()

    # --- Frequency-robust inference ---
    if frequency == "daily":
        # HAC over time-series of gammas
        T = len(gammas)
        lags = 20  # ~1 month of daily autocorrelation

        se_dict = {}
        t_dict = {}

        for col in gammas.columns:
            y = gammas[col]
            X = sm.add_constant(pd.Series(1, index=y.index))
            res = sm.OLS(y, X).fit(cov_type='HAC', cov_kwds={'maxlags': lags})
            #se_dict[col] = res.bse["const"]
            #t_dict[col] = res.tvalues["const"]
            se_dict[col] = res.bse.iloc[0]
            t_dict[col] = res.tvalues.iloc[0]

        se_lambdas = pd.Series(se_dict)
        t_stats = pd.Series(t_dict)

    else:
        # Standard FM
        T = len(gammas)
        se_lambdas = gammas.std(ddof=1) / np.sqrt(T)
        t_stats = mean_lambdas / se_lambdas

    return pd.DataFrame({
        "lambda_mean": mean_lambdas,
        "lambda_se": se_lambdas,
        "lambda_fm_tstat": t_stats,
    })

