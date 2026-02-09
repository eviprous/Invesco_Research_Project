from pathlib import Path
import numpy as np
import pandas as pd
from tqdm.auto import tqdm



def load_monthly_inputs(data_raw_dir):
    data_raw_dir = Path(data_raw_dir)
    sp_500 = pd.read_csv(
        data_raw_dir / "sp500_returns_monthly_with_tickers.csv",
        index_col=0,
        parse_dates=True,
    )
    sp_500.index = sp_500.index.to_period("M").to_timestamp()

    sp_caps = pd.read_csv(
        data_raw_dir / "sp500_market_caps_monthly.csv",
        index_col=0,
        parse_dates=True,
    )
    sp_caps.index = sp_caps.index.to_period("M").to_timestamp()

    rf = pd.read_csv(
        data_raw_dir / "ff_factors_monthly.csv",
        index_col=0,
        parse_dates=True,
    )
    rf.index = rf.index.to_period("M").to_timestamp()
    rf = rf[["RF"]]

    return sp_500, sp_caps, rf

def load_daily_inputs(data_raw_dir):
    data_raw_dir = Path(data_raw_dir)
    sp_500 = pd.read_csv(
        data_raw_dir / "sp500_returns_daily_with_tickers.csv",
        index_col=0,
        parse_dates=True,
    )

    sp_caps = pd.read_csv(
        data_raw_dir / "sp500_market_caps_daily.csv",
        index_col=0,
        parse_dates=True,
    )

    rf = pd.read_csv(
        data_raw_dir / "ff_factors_daily.csv",
        index_col=0,
        parse_dates=True,
    )
    rf = rf[["RF"]]

    return sp_500, sp_caps, rf


def backtest_quantile_portfolios_old(members_df, returns_df, caps_df, n_q1, n_q2, weight="cap"):
    members_df = members_df.copy()
    members_df["month"] = members_df.index.get_level_values("date").to_period("M")
    months = members_df["month"].sort_values().unique().to_timestamp()

    all_portfolio_returns = pd.DataFrame(index=months)

    for q1 in range(1, n_q1 + 1):
        for q2 in range(1, n_q2 + 1):
            portfolio_returns = []

            for month in months:
                tickers = members_df.loc[
                    (members_df["month"] == month.to_period("M"))
                    & (members_df["q1"] == q1)
                    & (members_df["q2"] == q2)
                ].index.get_level_values("ticker").unique()

                if len(tickers) == 0 or month not in returns_df.index or month not in caps_df.index:
                    portfolio_returns.append(np.nan)
                    continue

                ret = returns_df.loc[month, tickers]
                cap = caps_df.loc[month, tickers]
                df = pd.concat([ret, cap], axis=1, keys=["ret", "cap"]).dropna()

                if df.empty:
                    portfolio_returns.append(np.nan)
                    continue

                if weight == "cap":
                    weights = df["cap"] / df["cap"].sum()
                else:
                    weights = pd.Series(1.0 / len(df), index=df.index)

                portfolio_returns.append((weights * df["ret"]).sum())

            all_portfolio_returns[f"Q{q1}_Q{q2}"] = portfolio_returns

    return all_portfolio_returns

def backtest_quantile_portfolios(
    members_df,
    returns_df,
    caps_df,
    n_q1,
    n_q2,
    weight="cap",
    frequency="monthly",   # "monthly" or "daily"
):
    """
    Backtest 3x3 (or general nxn) portfolios formed monthly.

    - Sorting is ALWAYS monthly
    - Returns can be monthly or daily
    - Holdings are fixed within each month
    """

    members_df = members_df.copy()

    # --- Monthly formation period ---
    members_df["month"] = members_df.index.get_level_values("date").to_period("M")
    months = members_df["month"].sort_values().unique().to_timestamp()

    # --- Output index ---
    if frequency == "monthly":
        out_index = months
    elif frequency == "daily":
        out_index = returns_df.index
    else:
        raise ValueError("frequency must be 'monthly' or 'daily'")

    all_portfolio_returns = pd.DataFrame(index=out_index)

    # --- Loop over quantile portfolios ---
    #for q1 in range(1, n_q1 + 1):
        #for q2 in range(1, n_q2 + 1):
    for q1 in tqdm(range(1, n_q1 + 1), desc="Q1 buckets"):
        for q2 in tqdm(range(1, n_q2 + 1), desc="Q2 buckets", leave=False):


            if frequency == "monthly":
                portfolio_returns = []

            for month in months:
                # --- Get constituents for this month ---
                tickers = members_df.loc[
                    (members_df["month"] == month.to_period("M")) &
                    (members_df["q1"] == q1) &
                    (members_df["q2"] == q2)
                ].index.get_level_values("ticker").unique()

                if len(tickers) == 0:
                    if frequency == "monthly":
                        portfolio_returns.append(np.nan)
                    continue

                # ==========================
                # MONTHLY RETURNS
                # ==========================
                if frequency == "monthly":

                    if month not in returns_df.index or month not in caps_df.index:
                        portfolio_returns.append(np.nan)
                        continue

                    ret = returns_df.loc[month, tickers]
                    cap = caps_df.loc[month, tickers]

                    df = pd.concat([ret, cap], axis=1, keys=["ret", "cap"]).dropna()

                    if df.empty:
                        portfolio_returns.append(np.nan)
                        continue

                    if weight == "cap":
                        weights = df["cap"] / df["cap"].sum()
                    else:
                        weights = pd.Series(1.0 / len(df), index=df.index)

                    portfolio_returns.append((weights * df["ret"]).sum())

                # ==========================
                # DAILY RETURNS
                # ==========================
                elif frequency == "daily":

                    daily_dates = returns_df.index[
                        returns_df.index.to_period("M") == month.to_period("M")
                    ]

                    for day in daily_dates:
                        if day not in returns_df.index or day not in caps_df.index:
                            all_portfolio_returns.loc[day, f"Q{q1}_Q{q2}"] = np.nan
                            continue

                        ret = returns_df.loc[day, tickers]
                        cap = caps_df.loc[day, tickers]

                        df = pd.concat([ret, cap], axis=1, keys=["ret", "cap"]).dropna()

                        if df.empty:
                            all_portfolio_returns.loc[day, f"Q{q1}_Q{q2}"] = np.nan
                            continue

                        if weight == "cap":
                            weights = df["cap"] / df["cap"].sum()
                        else:
                            weights = pd.Series(1.0 / len(df), index=df.index)

                        all_portfolio_returns.loc[day, f"Q{q1}_Q{q2}"] = (
                            weights * df["ret"]
                        ).sum()

            # --- Assign monthly portfolio ---
            if frequency == "monthly":
                all_portfolio_returns[f"Q{q1}_Q{q2}"] = portfolio_returns

    return all_portfolio_returns



def compute_excess_returns(portfolio_returns, rf, start_date=None):
    common_index = portfolio_returns.index.intersection(rf.index)
    portfolio_returns = portfolio_returns.loc[common_index]
    excess = portfolio_returns.subtract(rf.loc[common_index, "RF"], axis=0)
    if start_date is not None:
        excess = excess.loc[start_date:]
    return excess


def count_assets_by_cell(members_df):
    counts_by_month = (
        members_df.reset_index()
        .groupby(["date", "q1", "q2"])["ticker"]
        .nunique()
        .rename("n_assets")
        .reset_index()
    )
    avg_counts = (
        counts_by_month.groupby(["q1", "q2"])["n_assets"].mean().unstack()
    )
    min_counts = (
        counts_by_month.groupby(["q1", "q2"])["n_assets"].min().unstack()
    )
    return counts_by_month, avg_counts, min_counts
