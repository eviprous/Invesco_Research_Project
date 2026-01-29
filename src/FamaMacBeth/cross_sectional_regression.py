import numpy as np
import pandas as pd
import statsmodels.api as sm

def run_time_series_regressions(portrets_wide, factors, f1_name, f2_name):
    results = {}

    for p in portrets_wide:
        y = portrets_wide[p].dropna()
        if len(y) < 12:
            results[p] = None
            continue

        X = sm.add_constant(factors.loc[y.index], has_constant="add")

        try:
            m = sm.OLS(y, X).fit()
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


def run_fama_macbeth(portrets_wide, beta_table, factors):
    gamma_rows = []

    for date, row in portrets_wide.iterrows():
        R_t = row.dropna()
        if len(R_t) < 3:
            gamma_rows.append({"date": date, "alpha": np.nan,
                               "lambdaEBC": np.nan, "lambda_Cap_EBC": np.nan})
            continue

        X = sm.add_constant(beta_table.loc[R_t.index])

        try:
            res = sm.OLS(R_t.values, X.values).fit()
            gamma_rows.append({
                "date": date,
                "alpha": res.params[0],
                "lambdaEBC": res.params[1],
                "lambda_Cap_EBC": res.params[2],
            })
        except Exception:
            gamma_rows.append({"date": date, "alpha": np.nan,
                               "lambdaEBC": np.nan, "lambda_Cap_EBC": np.nan})

    gammas = pd.DataFrame(gamma_rows).set_index("date").dropna()

    T = len(gammas)
    mean = gammas.mean()
    se = gammas.std(ddof=1) / np.sqrt(T)

    return pd.DataFrame({
        "lambda_mean": mean,
        "lambda_se": se,
        "lambda_fm_tstat": mean / se,
    })
