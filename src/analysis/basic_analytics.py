import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt


from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

def get_excess_returns(
    df: pd.DataFrame,
    portfolio_col: str,
    rf_col: str = "RF"
) -> pd.Series:
    """
    Compute excess returns for a portfolio.
    """
    return df[portfolio_col] - df[rf_col]

# ------------------------------------------------------
# Define functions for all the regressions
# ------------------------------------------------------

def _run_factor_regression(
    y: pd.Series,
    X: pd.DataFrame
) -> pd.DataFrame:
    """
    Run OLS regression and return clean summary stats.
    """
    X = sm.add_constant(X)
    model = sm.OLS(y, X).fit()

    out = pd.DataFrame({
        "coef": model.params,
        "t_stat": model.tvalues
    })

    out.loc["R2", "coef"] = model.rsquared
    out.loc["R2", "t_stat"] = np.nan

    return out


def run_capm(
    df: pd.DataFrame,
    portfolio_col: str,
    mkt_col: str = "MKT_RF",
    rf_col: str = "RF"
) -> pd.DataFrame:
    """
    CAPM regression:
    (R_p - R_f) = alpha + beta * MKT_RF
    """
    y = get_excess_returns(df, portfolio_col, rf_col)
    X = df[[mkt_col]]

    return _run_factor_regression(y, X)


def run_factor_model(
    df: pd.DataFrame,
    portfolio_col: str,
    rf_col: str = "RF",
    factors: list[str] | None = None
) -> pd.DataFrame:
    """
    Fama-French 5-factor + Momentum regression.
    If factors empty then default is FF5 + Momentum.
    """
    if factors is None:
        factors = ["MKT_RF", "SMB", "HML", "RMW", "CMA", "MOM"]

    y = get_excess_returns(df, portfolio_col, rf_col)
    X = df[factors]

    return _run_factor_regression(y, X)

# ------------------------------------------------------
# Define functions for log cummulative returns
# ------------------------------------------------------
def compute_log_cumulative_returns(
    df: pd.DataFrame,
    columns: list[str]
) -> pd.DataFrame:
    """
    Compute log cumulative returns for selected return series.
    Matches old implementation exactly.
    """
    returns = df[columns].dropna()
    return np.log1p(returns).cumsum()

def rolling_factor_betas(
    df: pd.DataFrame,
    portfolio_col: str,
    factors: list[str],
    rf_col: str = "RF",
    window: int = 36,
    plot: bool = True
) -> pd.DataFrame:
    """
    Compute (and optionally plot) rolling alpha and factor betas.
    """

    y = df[portfolio_col] - df[rf_col]
    X = df[factors]

    results = []

    for i in range(window, len(df)):
        y_win = y.iloc[i - window:i]
        X_win = X.iloc[i - window:i]

        X_win = sm.add_constant(X_win)
        model = sm.OLS(y_win, X_win).fit()

        row = model.params.to_dict()
        row["R2"] = model.rsquared
        results.append(row)

    betas = pd.DataFrame(results, index=df.index[window:])
    betas = betas.rename(columns={"const": "alpha"})

    # -------- plot --------
    if plot:
        cols_to_plot = ["alpha"] + factors
        betas[cols_to_plot].plot(
            figsize=(14, 6),
            title=f"{window}-Period Rolling Alpha & Betas"
        )
        plt.axhline(0, color="black", linewidth=1)
        plt.grid(True)
        plt.show()

    return betas



    
# ------------------------------------------------------
# Define functions for plotting returns
# ------------------------------------------------------
def plot_cumulative_returns(
    df: pd.DataFrame,
    columns: list[str],
    title: str | None = None,
    start_value: float = 1.0
):
    """
    Plot cumulative returns for selected series.
    """
    cum = compute_log_cumulative_returns(df, columns)

    fig, ax = plt.subplots(figsize=(10, 6))
    cum.plot(ax=ax)

    ax.set_xlabel("Date")
    ax.set_ylabel("Log Cumulative Return")

    if title:
        ax.set_title(title)

    ax.grid(True)
    ax.legend(title="Portfolio")

    plt.tight_layout()
    plt.show()

