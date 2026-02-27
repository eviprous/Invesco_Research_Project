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

#####################################################################
################ Functions for preference dataset ###################
#####################################################################

def build_all_dataset(
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