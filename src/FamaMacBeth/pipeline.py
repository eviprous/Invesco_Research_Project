import sys
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm

project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

import src.FamaMacBeth.cross_sectional_regression as fmb_csr
import src.FamaMacBeth.pricing as fmb_pricing
import src.FamaMacBeth.backtest as fmb_backtest
import pandas as pd

############################## Load inputs and make them excess ##############################

def load_monthly_inputs(data_raw_dir, excess = False):
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

    ff = pd.read_csv(
        data_raw_dir / "ff_factors_monthly.csv",
        index_col=0,
        parse_dates=True,
    )
    ff.index = ff.index.to_period("M").to_timestamp()
    #rf = rf[["RF"]]

    # Subtract RF from every ticker column
    common_index = sp_500.index.intersection(ff.index)

    sp_500 = sp_500.loc[common_index]
    ff = ff.loc[common_index]
    if excess:
        sp_500 = sp_500.sub(ff["RF"], axis=0)

    # Ensure market caps match the filtered dates of the returns
    sp_caps = sp_caps.loc[sp_500.index]

    return sp_500, sp_caps, ff

def load_daily_inputs(data_raw_dir, excess = True):
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

    ff = pd.read_csv(
        data_raw_dir / "ff_factors_daily.csv",
        index_col=0,
        parse_dates=True,
    )
    #rf = rf[["RF"]]

    # Subtract RF from every ticker column
    sp_500, ff = sp_500.align(ff, join="inner", axis=0)
    if excess:
        sp_500 = sp_500.sub(ff["RF"], axis=0)

    # Ensure market caps match the filtered dates of the returns
    sp_caps = sp_caps.loc[sp_500.index]

    return sp_500, sp_caps, ff

############################## Assign Quantiles Function ##############################

def assign_quantiles(series, nq, highest_is_1=True):
    """Assign quantiles 1..nq to a Series (nullable Int)."""
    s = pd.to_numeric(series.copy(), errors="coerce")
    nonan = s.dropna()

    if nonan.empty:
        return pd.Series(index=s.index, dtype="Int64")

    try:
        q = pd.qcut(nonan, nq, labels=False, duplicates="drop") + 1
        q = pd.Series(q, index=nonan.index).astype("Int64")
    except Exception:
        ranks = nonan.rank(method="first")
        q = np.ceil(ranks / len(ranks) * nq).astype(int)
        q = pd.Series(q, index=nonan.index).astype("Int64")

    if highest_is_1:
        q = (nq + 1 - q).astype("Int64")

    return q.clip(1, nq).reindex(s.index)


############################## get members ##############################

def build_double_sorted_members(
    beta_EBC_df,
    beta_CW_EBC_df,
    returns_df,
    n_q1,
    n_q2,
    formation_frequency,
):
    """
    Constructs ONLY portfolio membership.
    
    Output:
    members: MultiIndex DataFrame indexed by (date, ticker)
             columns = ['q1', 'q2']
    """

    # --- Standardize Indices ---
    if isinstance(beta_EBC_df.index, pd.PeriodIndex):
        beta_EBC_df = beta_EBC_df.copy()
        beta_EBC_df.index = beta_EBC_df.index.to_timestamp()

    if isinstance(beta_CW_EBC_df.index, pd.PeriodIndex):
        beta_CW_EBC_df = beta_CW_EBC_df.copy()
        beta_CW_EBC_df.index = beta_CW_EBC_df.index.to_timestamp()

    if isinstance(returns_df.index, pd.PeriodIndex):
        returns_df = returns_df.copy()
        returns_df.index = returns_df.index.to_timestamp()

    # --- Common Dates ---
    all_dates = (
        beta_EBC_df.index
        .intersection(beta_CW_EBC_df.index)
        .intersection(returns_df.index)
    )

    # --- Formation Dates ---
    if formation_frequency == "daily":
        formation_dates = all_dates

    elif formation_frequency == "monthly":
        formation_dates = (
            pd.Series(all_dates)
            #.groupby(pd.Series(all_dates).dt.to_period("M"))
            .groupby(all_dates.to_period("M"))
            .last()
            .values
        )

    elif formation_frequency == "quarterly":
        formation_dates = (
            pd.Series(all_dates)
            #.groupby(pd.Series(all_dates).dt.to_period("Q"))
            .groupby(all_dates.to_period("Q"))
            .last()
            .values
        )

    else:
        raise ValueError("formation_frequency must be daily/monthly/quarterly")

    dates = pd.DatetimeIndex(formation_dates).sort_values()
    ret_dates = returns_df.index

    members_rows = []

    # --- Main Loop ---
    for i, date in enumerate(dates):

        try:
            beta1 = beta_EBC_df.loc[date]
            beta2 = beta_CW_EBC_df.loc[date]

            # Hold AFTER formation date up to and including next formation date
            if i + 1 < len(dates):
                next_formation = dates[i + 1]
                holding_dates = ret_dates[
                    (ret_dates > date) & (ret_dates <= next_formation)
                ]
            else:
                holding_dates = ret_dates[ret_dates > date]

            if len(holding_dates) == 0:
                continue

        except KeyError:
            continue

        # --- Cross-Section ---
        df_chars = pd.DataFrame({
            "beta1": beta1,
            "beta2": beta2
        }).dropna()

        if df_chars.empty:
            continue

        # First sort
        df_chars["q1"] = assign_quantiles(df_chars["beta1"], n_q1)

        # Second sort within q1
        df_chars["q2"] = pd.Series(index=df_chars.index, dtype="Int64")

        for q in range(1, n_q1 + 1):
            mask = df_chars["q1"] == q
            if mask.any():
                df_chars.loc[mask, "q2"] = assign_quantiles(
                    df_chars.loc[mask, "beta2"],
                    n_q2
                )

        # --- Expand membership across holding window ---
        for hold_date in holding_dates:
            tmp = df_chars[["q1", "q2"]].copy()
            tmp["date"] = hold_date
            tmp["ticker"] = tmp.index
            members_rows.append(tmp.reset_index(drop=True))

    # --- Final Output ---
    members = (
        pd.concat(members_rows, ignore_index=True)
        .set_index(["date", "ticker"])
        .sort_index()
    )

    return members


############################## Run full pipeline through backtest ##############################
def run_full_factor_pipeline(
    beta_EBC_df, # 1
    beta_CW_EBC_df, # 2
    returns_df,
    caps_df,
    EBC_returns_df,
    Cap_EBC_returns_df,
    cap_weights,
    n_q1=3,
    n_q2=3,
    frequency="monthly",
    formation_frequency="quarterly",
    min_assets_per_cell=3,
    f1_name="EBC",
    f2_name="CW-EBC"
):
    members = build_double_sorted_members(
        beta_EBC_df,
        beta_CW_EBC_df,
        returns_df,
        n_q1,
        n_q2,
        formation_frequency)

    portrets_wide = fmb_backtest.backtest_quantile_portfolios(
       members,
       returns_df,
       caps_df,
       n_q1,
       n_q2,
       cap_weights
    )


    factors = pd.concat([EBC_returns_df, Cap_EBC_returns_df], axis=1).dropna()
    
    if frequency == "monthly":
        if isinstance(factors.index, pd.PeriodIndex):
            factors.index = factors.index.to_timestamp()

    if factors.shape[1] != 2:
        raise ValueError("Expected two single-column Series/DataFrames for factors.")
    
    factors.columns = [f1_name, f2_name]
    common_index = portrets_wide.index.intersection(factors.index)

    portrets_wide = portrets_wide.loc[common_index]
    factors = factors.loc[common_index]


    ts_summary = fmb_csr.run_time_series_regressions(portrets_wide, factors, f1_name, f2_name, frequency=frequency)
    beta_table = ts_summary[["betaEBC", "betaCap_EBC"]].dropna()

    fm_table = fmb_csr.run_fama_macbeth(portrets_wide[beta_table.index], beta_table, frequency=frequency)
    pricing_df = fmb_pricing.pricing_errors(portrets_wide, beta_table, ts_summary, fm_table)
    mean_grid, alpha_grid = fmb_pricing.build_grids(pricing_df, n_q1, n_q2)

    return {
        #"portrets": portrets,
        "portrets_wide": portrets_wide,
        "members": members,
        #"quantile_map": qmap,
        "ts_summary": ts_summary,
        "fm_table": fm_table,
        "pricing": pricing_df,
        "mean_grid": mean_grid,
        "alpha_grid": alpha_grid,
    }
