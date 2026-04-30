import numpy as np
import pandas as pd
import statsmodels.api as sm
from tqdm import tqdm


def compute_rolling_betas(
    asset_returns: pd.DataFrame,
    factor_returns: pd.DataFrame,
    rolling_window: int = 36,
    min_obs: int = 10,
):
    """
    Compute rolling factor betas for a panel of assets.
    """
    # if factor_returns.shape[1] != 1:
    #     raise ValueError("factor_returns must contain exactly ONE column")

    # Align indices — use intersection to avoid column name conflicts
    # (factor name may match a ticker, e.g. "EW" is an S&P 500 ticker)
    common_idx = asset_returns.index.intersection(factor_returns.index).sort_values()
    combined = pd.DataFrame(index=common_idx)
    tickers = asset_returns.columns
    factor_name = factor_returns.columns[0]

    # initializing empty betas dataframe with tickers for columns
    betas_df = pd.DataFrame(index=pd.Index([], name="date"),
                        columns=tickers,
                        dtype=float)


    # ---------- Rolling time-series regressions ----------
    for end_date in tqdm(combined.index[rolling_window - 1:- 1], desc="Estimating rolling betas" ):
        end_idx = combined.index.get_loc(end_date)
        next_date = combined.index[end_idx + 1]

        start_idx = end_idx - (rolling_window -1)
        window_dates = combined.index[start_idx : end_idx + 1]

        X_win = factor_returns.loc[window_dates].astype(float)
        Y_win = asset_returns.loc[window_dates, tickers].astype(float)

        # drop rows with any NaNs in factor
        valid_rows = ~X_win.isna().any(axis=1)
        X_win = X_win.loc[valid_rows]
        Y_win = Y_win.loc[valid_rows]

        if len(X_win) < min_obs:
            betas_df.loc[next_date, :] = np.nan
            continue

        # drop tickers with any NaNs in this window
        valid_cols = ~Y_win.isna().any(axis=0)
        Y_win = Y_win.loc[:, valid_cols]
        if Y_win.shape[1] == 0:
            betas_df.loc[next_date, :] = np.nan
            continue

        X = sm.add_constant(X_win[[factor_name]], has_constant="add").values  # T x 2
        Y = Y_win.values  # T x N

        XtX = X.T @ X
        if np.linalg.matrix_rank(XtX) < XtX.shape[0]:
            betas_df.loc[next_date, Y_win.columns] = np.nan
            continue

        B = np.linalg.solve(XtX, X.T @ Y)  # 2 x N

        # B[0] = alpha, B[1] = beta for factor_name
        betas_df.loc[next_date, Y_win.columns] = B[1, :]

    # clip the most extreme betas 0.5% of tails

    stacked = betas_df.stack(dropna=True)
    lower = stacked.quantile(0.005)
    upper = stacked.quantile(0.995)
    betas_df = betas_df.clip(lower, upper)



    return betas_df

