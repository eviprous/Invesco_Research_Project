import numpy as np
import pandas as pd

def pricing_errors(portrets_wide, beta_table, ts_summary, fm_table):
    g0 = fm_table.loc["alpha", "lambda_mean"]
    g1 = fm_table.loc["lambdaEBC", "lambda_mean"]
    g2 = fm_table.loc["lambda_Cap_EBC", "lambda_mean"]

    rows = []
    mean_r = portrets_wide.mean()

    for p in beta_table.index:
        b1, b2 = beta_table.loc[p]
        rows.append({
            "portfolio": p,
            "mean_return": mean_r[p],
            "betaEBC": b1,
            "betaCap_EBC": b2,
            "alpha_ts": ts_summary.loc[p, "alpha"],
            "alpha_fm": mean_r[p] - (g0 + g1*b1 + g2*b2),
        })

    return pd.DataFrame(rows).set_index("portfolio")


def build_grids(pricing_df, n_q1, n_q2):
    mean_grid = pd.DataFrame(np.nan, index=range(1, n_q1+1), columns=range(1, n_q2+1))
    alpha_grid = mean_grid.copy()

    for p, r in pricing_df.iterrows():
        q1, q2 = map(int, p.replace("Q", "").split("_"))
        mean_grid.loc[q1, q2] = r["mean_return"]
        alpha_grid.loc[q1, q2] = r["alpha_fm"]

    return mean_grid, alpha_grid
