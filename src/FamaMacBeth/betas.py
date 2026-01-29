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
    if factor_returns.shape[1] != 1:
        raise ValueError("factor_returns must contain exactly ONE column")

    # Align once
    combined = asset_returns.join(factor_returns, how="inner")
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

        X_win = factor_returns.loc[window_dates]

        for asset in tickers:
            y_win = asset_returns.loc[window_dates, asset]
            data = pd.concat([y_win, X_win], axis=1).dropna()

            if len(data) < min_obs:
                betas_df.loc[next_date, asset] = np.nan
                continue

            Y = data[asset].astype(float)
            X = sm.add_constant(data[[factor_name]].astype(float), has_constant="add")

            # Singular matrix guard
            if X.shape[0] <= X.shape[1] or np.linalg.matrix_rank(X) < X.shape[1]:
                betas_df.loc[next_date, asset] = np.nan
                continue

            res = sm.OLS(Y, X).fit()

            betas_df.loc[next_date, asset] = res.params.get(factor_name)

    return betas_df

