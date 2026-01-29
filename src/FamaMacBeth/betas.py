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

    Parameters
    ----------
    asset_returns : DataFrame
        Index = date, columns = tickers, values = returns
    factor_returns : DataFrame
        Index = date, columns = factor names
    rolling_window : int
        Window length for time-series regressions
    min_obs : int
        Minimum observations required to estimate betas

    Returns
    -------
    betas_dict : dict[str, DataFrame]
        One DataFrame per factor:
        index = date (t+1, portfolio formation date)
        columns = tickers
    """

    # Align once
    combined = asset_returns.join(factor_returns, how="inner")
    tickers = asset_returns.columns
    factors = factor_returns.columns

    betas_dict = {
        f: pd.DataFrame(index=pd.Index([], name="date"),
                        columns=tickers,
                        dtype=float)
        for f in factors
    }

    # ---------- Rolling time-series regressions ----------
    for end_idx in tqdm(
        range(rolling_window - 1, len(combined.index) - 1),
        desc="Estimating rolling betas",
    ):
        end_date = combined.index[end_idx]
        next_date = combined.index[end_idx + 1]

        window_dates = combined.index[end_idx - rolling_window + 1 : end_idx + 1]

        X_win = factor_returns.loc[window_dates]

        for asset in tickers:
            y_win = asset_returns.loc[window_dates, asset]
            data = pd.concat([y_win, X_win], axis=1).dropna()

            if len(data) < min_obs:
                for f in factors:
                    betas_dict[f].loc[next_date, asset] = np.nan
                continue

            Y = data[asset].astype(float)
            X = sm.add_constant(data[factors].astype(float), has_constant="add")

            # Singular matrix guard
            if X.shape[0] <= X.shape[1] or np.linalg.matrix_rank(X) < X.shape[1]:
                for f in factors:
                    betas_dict[f].loc[next_date, asset] = np.nan
                continue

            res = sm.OLS(Y, X).fit()

            for f in factors:
                betas_dict[f].loc[next_date, asset] = res.params.get(f, np.nan)

    return betas_dict
