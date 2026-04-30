from pathlib import Path
import numpy as np
import pandas as pd
from tqdm.auto import tqdm



def load_monthly_inputs(data_raw_dir, excess=True):
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

    sp_500, rf = sp_500.align(rf, join="inner", axis=0)
    if excess:
        sp_500 = sp_500.sub(rf["RF"], axis=0)

    # Ensure market caps match the filtered dates of the returns
    sp_caps = sp_caps.loc[sp_500.index]

    return sp_500, sp_caps, rf

def load_daily_inputs(data_raw_dir, excess=True):
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

    sp_500, rf = sp_500.align(rf, join="inner", axis=0)
    if excess:
        sp_500 = sp_500.sub(rf["RF"], axis=0)

    # Ensure market caps match the filtered dates of the returns
    sp_caps = sp_caps.loc[sp_500.index]

    return sp_500, sp_caps, rf


def backtest_quantile_portfolios(
    members_df,
    returns_df,
    caps_df,
    n_q1,
    n_q2,
    cap_weights
):
    """
    Backtest quantile portfolios using members from build_double_sorted_members.
    For each hold_date in members_df, computes portfolio return using current-period
    caps (floating weights, not fixed at formation). This matches the original
    build_double_sorted_portfolios behavior.
    """
    all_hold_dates = members_df.index.get_level_values("date").unique().sort_values()
    all_portfolio_returns = pd.DataFrame(index=returns_df.index)

    for q1 in tqdm(range(1, n_q1 + 1), desc="Q1 buckets"):
        for q2 in range(1, n_q2 + 1):
            port_label = f"Q{q1}_Q{q2}"
            column_rets = pd.Series(index=returns_df.index, dtype=float)

            for hold_date in all_hold_dates:
                tickers = members_df.loc[
                    (members_df.index.get_level_values("date") == hold_date) &
                    (members_df["q1"] == q1) &
                    (members_df["q2"] == q2)
                ].index.get_level_values("ticker").unique()

                if len(tickers) == 0:
                    continue

                try:
                    ret_s = returns_df.loc[hold_date, tickers].dropna()
                except KeyError:
                    continue

                if len(ret_s) == 0:
                    continue

                if cap_weights:
                    try:
                        cap_s = caps_df.loc[hold_date, ret_s.index].dropna()
                    except KeyError:
                        continue
                    common = ret_s.index.intersection(cap_s.index)
                    if len(common) == 0 or cap_s[common].sum() == 0:
                        continue
                    w = cap_s[common] / cap_s[common].sum()
                    column_rets.loc[hold_date] = (w * ret_s[common]).sum()
                else:
                    column_rets.loc[hold_date] = ret_s.mean()

            all_portfolio_returns[port_label] = column_rets

    return all_portfolio_returns.dropna(how='all')


