import pandas as pd

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

def build_preference_factor_dataset(
    returns_file: str,
    market_caps_file: str,
    ff_factors_file: str,
    frequency: str = 'monthly',
) -> pd.DataFrame:
    """
    Build CW, EW, and Preference (CW−EW) portfolios and merge with FF factors.

    Parameters
    ----------
    returns_path : str
        Path to CSV with asset returns (date × ticker).
    market_caps_path : str
        Path to CSV with market caps (date × ticker), in thousands.
    ff_factors_path : str
        Path to CSV with FF factors.
    frequency : {"daily", "monthly"}

    Returns
    -------
    pd.DataFrame
        Regression-ready dataset.
    """

    if frequency not in {"daily", "monthly"}:
        raise ValueError("frequency must be 'daily' or 'monthly'")

    # ------------------
    # Load data
    # ------------------

    returns = pd.read_csv(
    DATA_PROCESSED_DIR / returns_file,
    index_col=0,
    parse_dates=True
    )

    market_caps = pd.read_csv(
        DATA_PROCESSED_DIR / market_caps_file,
        index_col=0,
        parse_dates=True
    )

    ff_factors = pd.read_csv(
        DATA_PROCESSED_DIR / ff_factors_file,
        index_col=0,
        parse_dates=True
    )

    # Convert market caps to dollars
    market_caps = market_caps * 1000.0

    # Ensure numeric
    rets = returns.apply(pd.to_numeric, errors="coerce")
    caps = market_caps.apply(pd.to_numeric, errors="coerce")

    # Align assets
    common_assets = rets.columns.intersection(caps.columns)
    rets = rets[common_assets]
    caps = caps[common_assets]

    # ------------------
    # CW portfolio
    # ------------------
    cap_weights = caps.div(caps.sum(axis=1), axis=0)
    ret_cw = (rets * cap_weights).sum(axis=1)

    # ------------------
    # EW portfolio
    # ------------------
    active = caps > 0.0
    n_active = active.sum(axis=1)
    ew_weights = active.div(n_active, axis=0)
    ret_ew = (rets * ew_weights).sum(axis=1)

    # ------------------
    # Preference portfolio
    # ------------------
    ret_pref = ret_cw - ret_ew

    portfolios = pd.DataFrame(
        {
            "CW": ret_cw,
            "EW": ret_ew,
            "CW-EW": ret_pref,
        }
    )

    # ------------------
    # Date alignment
    # ------------------
    if frequency == "monthly":
        portfolios.index = portfolios.index.to_period("M")
        ff_factors.index = ff_factors.index.to_period("M")

    # ------------------
    # Merge with factors
    # ------------------
    df = portfolios.join(ff_factors, how="inner")

    return df.dropna()
