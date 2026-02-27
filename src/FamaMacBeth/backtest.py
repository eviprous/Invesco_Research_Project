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

    # Subtract RF from every ticker column
    sp_500, rf = sp_500.align(rf, join="inner", axis=0)
    sp_500 = sp_500.sub(rf["RF"], axis=0)

    # Ensure market caps match the filtered dates of the returns
    sp_caps = sp_caps.loc[sp_500.index]

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

    # Subtract RF from every ticker column
    sp_500, rf = sp_500.align(rf, join="inner", axis=0)
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
    weight="cap",
    frequency="monthly",
):
    """
    Universal Backtester:
    Works for any formation frequency (Daily, Monthly, Quarterly) 
    and any return frequency (Daily, Monthly).
    """
    members_df = members_df.copy()
    
    # All dates where a portfolio formation occurred
    formation_dates = members_df.index.get_level_values("date").unique().sort_values()
    
    # Initialize output container
    all_portfolio_returns = pd.DataFrame(index=returns_df.index)

    for q1 in tqdm(range(1, n_q1 + 1), desc="Q1 buckets"):
        for q2 in range(1, n_q2 + 1):
            port_label = f"Q{q1}_Q{q2}"
            
            # We will fill this column period-by-period
            column_rets = pd.Series(index=returns_df.index, dtype=float)

            for i, start_date in enumerate(formation_dates):
                # 1. Get the stocks chosen at this formation date
                tickers = members_df.loc[
                    (members_df.index.get_level_values("date") == start_date) &
                    (members_df["q1"] == q1) &
                    (members_df["q2"] == q2)
                ].index.get_level_values("ticker").unique()

                if len(tickers) == 0:
                    continue

                # 2. Identify the holding period
                # From T+1 (day after formation) until the next formation date (inclusive)
                if i + 1 < len(formation_dates):
                    end_date = formation_dates[i+1]
                    holding_days = returns_df.index[(returns_df.index >= start_date) & (returns_df.index < end_date)]
                else:
                    holding_days = returns_df.index[returns_df.index >= start_date]

                if len(holding_days) == 0:
                    continue

                # 3. Calculate returns for every step in the holding window
                for hold_date in holding_days:
                    try:
                        ret_s = returns_df.loc[hold_date, tickers]
                        cap_s = caps_df.loc[hold_date, tickers]
                        
                        df_step = pd.concat([ret_s, cap_s], axis=1, keys=["ret", "cap"]).dropna()
                        
                        if df_step.empty:
                            continue

                        if weight == "cap":
                            w = df_step["cap"] / df_step["cap"].sum()
                        else:
                            w = 1.0 / len(df_step)

                        column_rets.loc[hold_date] = (w * df_step["ret"]).sum()
                    except KeyError:
                        continue # Date missing in returns/caps
            
            all_portfolio_returns[port_label] = column_rets

    return all_portfolio_returns.dropna(how='all')


