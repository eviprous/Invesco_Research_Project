from calendar import month
import pandas as pd
import numpy as np
from scipy.optimize import minimize

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

######################################################################
################ Helper Functions for all datasets ###################
######################################################################

def build_ff_dataset(
        ff_factors_file: str,
        frequency: str = 'monthly'
) -> pd.DataFrame:
    """
    function that reads and returns a dataset only with the FF factors
    """
    if frequency not in {'monthly', 'daily'}:
        raise ValueError("frequency must be 'daily' or 'monthly'")
    
    ff_factors = pd.read_csv(
        DATA_RAW_DIR / ff_factors_file,
        index_col=0,
        parse_dates=True
    )

    if frequency == "monthly":
        ff_factors.index = ff_factors.index.to_period("M")

    return ff_factors.dropna()

def build_returns_dataset(
        returns_file: str,
        frequency: str = 'monthly'
) -> pd.DataFrame:
    """
    function that reads and returns a dataset only with the returns
    """
    
    if frequency not in {"daily", "monthly"}:
        raise ValueError("frequency must be 'daily' or 'monthly'")

    # ------------------
    # Load data
    # ------------------

    returns = pd.read_csv(
    DATA_RAW_DIR / returns_file,
    index_col=0,
    parse_dates=True
    )

    returns = returns.apply(pd.to_numeric, errors="coerce")

    if frequency == "monthly":
        returns.index = returns.index.to_period("M")
    return returns

def build_market_cap_dataset(
        market_caps_file: str,
        frequency: str = 'monthly'
) -> pd.DataFrame:
    """
    function that reads and retuyrns a dataset only with market caps
    """

    if frequency not in {"daily", "monthly"}:
        raise ValueError("frequency must be 'daily' or 'monthly'")

    market_caps = pd.read_csv(
        DATA_RAW_DIR / market_caps_file,
        index_col=0,
        parse_dates=True
    )

    # Convert market caps to dollars
    market_caps = market_caps * 1000.0

    # Ensure numeric
    market_caps = market_caps.apply(pd.to_numeric, errors="coerce")

    if frequency == "monthly":
        market_caps.index = market_caps.index.to_period("M")

    return market_caps


##############################################################
################ Functions for EBC dataset ###################
##############################################################
import src.preference_factors.build_EBC as ebc

def build_EBC_dataset(
    returns_file: str,
    ff_factors_file: str,
    frequency: str = 'monthly'):
    """
    function that returns EBC returns, weights and betas for the frequency chosen
    """
    if frequency not in {"daily", "monthly"}:
        raise ValueError("frequency must be 'daily' or 'monthly'")
    
    # ------------------
    # Load data
    # ------------------
    returns = build_returns_dataset(returns_file, frequency)
    ff_factors = build_ff_dataset(ff_factors_file,frequency)
    # Align returns and factors on common dates
    returns, ff_factors = returns.align(ff_factors, join="inner", axis=0)

    if frequency == "monthly":
        return ebc.build_EBC_dataset_monthly(returns,ff_factors)
    else:
        return ebc.build_EBC_dataset_daily(returns, ff_factors)
    
##################################################################
################ Functions for EW & CW dataset ###################
##################################################################

def build_cw_ew_dataset(
    returns_file: str,
    market_caps_file: str,
    ff_factors_file: str,
    frequency: str = "monthly"
):
    """
    Build CW and EW portfolios only (no EBC), and merge with FF factors.
    """

    if frequency not in {"daily", "monthly"}:
        raise ValueError("frequency must be 'daily' or 'monthly'")

    # ------------------
    # Load data
    # ------------------
    returns = build_returns_dataset(returns_file, frequency)
    market_caps = build_market_cap_dataset(market_caps_file, frequency)

    # Align columns (assets)
    common_assets = returns.columns.intersection(market_caps.columns)
    returns = returns[common_assets]
    market_caps = market_caps[common_assets]

    # Ensure numeric
    returns = returns.apply(pd.to_numeric, errors="coerce")
    market_caps = market_caps.apply(pd.to_numeric, errors="coerce")

    # ------------------
    # CW (CORRECT VERSION WITH LAG)
    # ------------------
    lagged_caps = market_caps.shift(1)

    # Remove assets without returns
    lagged_caps = lagged_caps.where(returns.notna())

    # Normalize weights each period
    cap_weights = lagged_caps.div(lagged_caps.sum(axis=1), axis=0)

    ret_cw = (cap_weights * returns).sum(axis=1)

    # ------------------
    # EW (clean version)
    # ------------------
    active = returns.notna()
    n_active = active.sum(axis=1)

    ew_weights = active.div(n_active, axis=0)

    ret_ew = (ew_weights * returns).sum(axis=1)

    # ------------------
    # Merge with FF
    # ------------------
    merged = pd.DataFrame({
        "CW": ret_cw,
        "EW": ret_ew
    }).dropna()

    return merged

#####################################################################
################ Functions for preference dataset ###################
#####################################################################

def build_all_dataset(
    returns_file: str,
    market_caps_file: str,
    ff_factors_file: str,
    frequency: str = "monthly"
):
    """
    Build CW, EW, and EBC portfolios and compute preference spreads.
    Returns both excess-only dataset and dataset including FF factors.
    """

    if frequency not in {"daily", "monthly"}:
        raise ValueError("frequency must be 'daily' or 'monthly'")

    # -------------------------------------------------
    # 1. Build CW & EW (RAW returns, already lagged properly)
    # -------------------------------------------------
    cw_ew_df = build_cw_ew_dataset(
        returns_file,
        market_caps_file,
        ff_factors_file,
        frequency
    )

    # -------------------------------------------------
    # 2. Build EBC (RAW returns)
    # -------------------------------------------------
    df_EBC_returns, df_EBC_weights, df_EBC_beta_contributions = build_EBC_dataset(
        returns_file,
        ff_factors_file,
        frequency
    )

    ret_ebc = df_EBC_returns.iloc[:, 0]

    # Ensure monthly PeriodIndex consistency
    if frequency == "monthly":
        if not isinstance(ret_ebc.index, pd.PeriodIndex):
            ret_ebc.index = ret_ebc.index.to_period("M")


    # -------------------------------------------------
    # 3. Align all portfolios (SAFE ALIGNMENT)
    # -------------------------------------------------

    # Ensure consistent index types
    if frequency == "monthly":
        if isinstance(cw_ew_df.index, pd.PeriodIndex) and not isinstance(ret_ebc.index, pd.PeriodIndex):
            ret_ebc = ret_ebc.copy()
            ret_ebc.index = ret_ebc.index.to_period("M")

        if isinstance(ret_ebc.index, pd.PeriodIndex) and not isinstance(cw_ew_df.index, pd.PeriodIndex):
            cw_ew_df = cw_ew_df.copy()
            cw_ew_df.index = cw_ew_df.index.to_period("M")

    # Explicit alignment
    cw_ew_df, ret_ebc = cw_ew_df.align(ret_ebc, join="inner", axis=0)

    merged = cw_ew_df.copy()
    merged["EBC"] = ret_ebc

    # -------------------------------------------------
    # 4. Merge with FF factors
    # -------------------------------------------------
    ff_factors = build_ff_dataset(ff_factors_file, frequency)

    full_df = merged.join(ff_factors, how="inner").dropna()

    # -------------------------------------------------
    # 5. Convert to excess returns (consistent space)
    # -------------------------------------------------
    full_df["CW_RF"]  = full_df["CW"]  - full_df["RF"]
    full_df["EW_RF"]  = full_df["EW"]  - full_df["RF"]
    full_df["EBC_RF"] = full_df["EBC"] - full_df["RF"]

    # Preference spreads in excess space
    full_df["CW-EW"]  = full_df["CW_RF"]  - full_df["EW_RF"]
    full_df["CW-EBC"] = full_df["CW_RF"]  - full_df["EBC_RF"]

    # -------------------------------------------------
    # 6. Final cleaned dataset (excess-only)
    # -------------------------------------------------
    excess_df = full_df[
        ["CW_RF", "EW_RF", "EBC_RF", "CW-EW", "CW-EBC"]
    ]

    # -------------------------------------------------
    # 7. Save
    # -------------------------------------------------
    save_dir = DATA_PROCESSED_DIR / frequency
    save_dir.mkdir(parents=True, exist_ok=True)

    if frequency == "monthly":
        df_EBC_weights.to_csv(save_dir / "monthly_EBC_weights.csv")
        df_EBC_beta_contributions.to_csv(save_dir / "monthly_EBC_beta_contributions.csv")
        df_EBC_returns.to_csv(save_dir / "monthly_EBC_raw_returns.csv")

        excess_df.to_csv(save_dir / "monthly_preference_excess_returns.csv")
        full_df.to_csv(save_dir / "monthly_preference_full_dataset.csv")

    else:
        df_EBC_weights.to_csv(save_dir / "daily_EBC_weights.csv")
        df_EBC_beta_contributions.to_csv(save_dir / "daily_EBC_beta_contributions.csv")
        df_EBC_returns.to_csv(save_dir / "daily_EBC_raw_returns.csv")

        excess_df.to_csv(save_dir / "daily_preference_excess_returns.csv")
        full_df.to_csv(save_dir / "daily_preference_full_dataset.csv")

    return excess_df, full_df






def build_all_dataset_old(
    returns_file: str,
    market_caps_file: str,
    ff_factors_file: str,
    frequency: str = 'monthly'
) -> pd.DataFrame:  
    """
    function that reads returns and market cap files and creates df with preferences CW-EW and CW-EBC
    """ 
    if frequency not in {"daily", "monthly"}:
        raise ValueError("frequency must be 'daily' or 'monthly'")

    # ------------------
    # Load data
    # ------------------

    returns = build_returns_dataset(returns_file, frequency)
    market_caps = build_market_cap_dataset(market_caps_file, frequency)

    # Align assets
    common_assets = returns.columns.intersection(market_caps.columns)
    returns = returns[common_assets]
    market_caps = market_caps[common_assets]

    # ------------------
    # CW portfolio
    # ------------------
    cap_weights = market_caps.div(market_caps.sum(axis=1), axis=0)
    ret_cw = (returns * cap_weights).sum(axis=1)

    # ------------------
    # EW portfolio
    # ------------------
    active = market_caps > 0.0
    n_active = active.sum(axis=1)
    ew_weights = active.div(n_active, axis=0)
    ret_ew = (returns * ew_weights).sum(axis=1)

    # ------------------
    # EBC portfolio -> only EBC returns
    # ------------------
    df_EBC_returns, df_EBC_weights, df_EBC_beta_contributions = build_EBC_dataset(returns_file, ff_factors_file, frequency)

    if frequency == "monthly":
        df_EBC_returns.index = df_EBC_returns.index.to_period("M")
        
    # Ensure alignment
    ret_ebc = df_EBC_returns.iloc[:, 0]
    ret_cw, ret_ebc = ret_cw.align(ret_ebc, join="inner")
    ret_ew = ret_ew.loc[ret_cw.index]


    # ------------------
    # Preference portfolio
    # ------------------
    merged_df = pd.DataFrame(
        {
            "CW": ret_cw,
            "EW": ret_ew,
            "EBC": ret_ebc
        }
    ).dropna()

    ff_factors = build_ff_dataset(ff_factors_file,frequency)
    full_merged_df = merged_df.join(ff_factors, how="inner").dropna()

    # We transform CW and EW into excess space to match EBC
    full_merged_df["CW"] = full_merged_df["CW"] - full_merged_df["RF"]
    full_merged_df["EW"] = full_merged_df["EW"] - full_merged_df["RF"]
    full_merged_df["EBC"] = full_merged_df["EBC"] - full_merged_df["RF"]

    # Since everything is now excess, the RF cancels out perfectly
    full_merged_df["CW-EW"] = full_merged_df["CW"] - full_merged_df["EW"]
    full_merged_df["CW-EBC"] = full_merged_df["CW"] - full_merged_df["EBC"]

    # 6. Final cleanup for saving
    merged_df = full_merged_df[["CW", "EW", "EBC", "CW-EW", "CW-EBC"]]

    save_dir = DATA_PROCESSED_DIR / frequency
    if frequency == "monthly":
        # save individual datasets
        df_EBC_weights.to_csv(save_dir / "monthly_EBC_weights.csv")
        df_EBC_beta_contributions.to_csv(save_dir / "monthly_EBC_beta_contributions.csv")
        df_EBC_returns.to_csv(save_dir / "monthly_EBC_excess_returns.csv")
        merged_df.to_csv(save_dir / "monthly_preference_excess_returns.csv")
        full_merged_df.to_csv(save_dir / "monthly_preference_excess_returns_and_factors.csv")

    elif frequency == "daily":
        df_EBC_weights.to_csv(save_dir / "daily_EBC_weights.csv")
        df_EBC_beta_contributions.to_csv(save_dir / "daily_EBC_beta_contributions.csv")
        df_EBC_returns.to_csv(save_dir / "daily_EBC_excess_returns.csv")
        merged_df.to_csv(save_dir / "daily_preference_excess_returns.csv")
        full_merged_df.to_csv(save_dir / "daily_preference_excess_returns_and_factors.csv")


    return merged_df, full_merged_df


