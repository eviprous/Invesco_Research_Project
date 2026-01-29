import pandas as pd
import numpy as np
from scipy.optimize import minimize

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"

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



##############################################################
################ Functions for Quantile dataset ###################
##############################################################
import src.preference_factors.build_quantile_portfollios as bq

def build_quantile_portfolios(
        returns_file: str,
        factor_returns,
        frequency,
        rolling_window = 36,
        min_obs = 10,
        n_q1=3,
        n_q2=3,
        min_assets_per_cell = 10
):
    if frequency not in {"daily", "monthly"}:
        raise ValueError("frequency must be 'daily' or 'monthly'")
    
    factor_names = list(factor_returns.columns)
    if len(factor_names) != 2:
        raise ValueError("Please make sure your factors_returns only include two columns")
    
    returns = build_returns_dataset(returns_file, frequency)
    
    betas_dict = bq.compute_rolling_betas(
        returns,
        factor_returns,
        rolling_window,
        min_obs
    )

    betas_1_name = factor_names[0]
    betas_1 = betas_dict[betas_1_name]

    betas_2_name = factor_names[1]
    betas_2 = betas_dict[betas_2_name]

    quantile_results = bq.build_two_way_quantile_portfolios(
        returns,
        betas_1,
        betas_1_name,
        betas_2,
        betas_2_name,
        n_q1,
        n_q2,
        min_assets_per_cell
    )

    return quantile_results


#####################################################################
################ Functions for preference dataset ###################
#####################################################################

def build_preference_dataset(
    returns_file: str,
    market_caps_file: str,
    ff_factors_file: str,
    frequency: str = 'monthly',
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
    ebc_returns,_,_ = build_EBC_dataset(returns_file, ff_factors_file, frequency)
    # Ensure alignment
    ret_ebc = ebc_returns.iloc[:, 0]
    ret_cw, ret_ebc = ret_cw.align(ret_ebc, join="inner")
    ret_ew = ret_ew.loc[ret_cw.index]


    # ------------------
    # Preference portfolio
    # ------------------
    ret_cw_ew = ret_cw - ret_ew
    ret_cw_ebc = ret_cw - ret_ebc

    portfolios = pd.DataFrame(
        {
            "CW": ret_cw,
            "EW": ret_ew,
            "EBC": ret_ebc,
            "CW-EW": ret_cw_ew,
            "CW-EBC": ret_cw_ebc
        }
    )

    return portfolios.dropna()



def build_preference_factor_dataset(
    returns_file: str,
    market_caps_file: str,
    ff_factors_file: str,
    frequency: str = 'monthly',
) -> pd.DataFrame:
    """
    Function that reads returns, market caps, and ff files and combines them to create preferences and factors in one dataset
    used in all regressions
    """

    if frequency not in {"daily", "monthly"}:
        raise ValueError("frequency must be 'daily' or 'monthly'")

    # ------------------
    # Load data
    # ------------------

    preference_df = build_preference_dataset(returns_file, market_caps_file,ff_factors_file,frequency)

    ff_factors = build_ff_dataset(ff_factors_file,frequency)

    # ------------------
    # Merge with factors
    # ------------------
    df = preference_df.join(ff_factors, how="inner")

    return df.dropna()
