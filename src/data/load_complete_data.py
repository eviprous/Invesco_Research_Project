import pandas as pd
import wrds
from typing import Tuple
from pathlib import Path

# -------------------------------------------------
# Filters applied (all are standard academic filters):
#
#   1. shrcd IN (10, 11)     — ordinary common shares only
#                              excludes ADRs, ETFs, REITs, closed-end funds
#   2. exchcd IN (1, 2, 3)   — NYSE, AMEX, NASDAQ only
#                              excludes OTC pink sheets and foreign exchanges
#   3. abs(prc) > 1          — price above $1 (penny stock filter)
#
# Universe: all ordinary US common shares listed on major exchanges with price > $1,
# covering ~5,000-8,000 stocks at any point in time vs. the fixed 500 of the S&P 500.
# -------------------------------------------------


# -------------------------------------------------
# Core WRDS pulls
# -------------------------------------------------

def load_crsp_full(conn, start, end) -> pd.DataFrame:
    """
    Pull returns, price, shares outstanding, share code, and exchange code
    from CRSP monthly file for the full universe (no index membership filter).
    """
    return conn.raw_sql(f"""
        SELECT permno, date, ret, prc, shrout, shrcd, exchcd
        FROM crsp.msf
        WHERE date BETWEEN '{start}' AND '{end}'
    """)


# -------------------------------------------------
# Filtering
# -------------------------------------------------

def apply_universe_filters(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply standard broad-universe filters:
      - Ordinary common shares (shrcd 10/11)
      - Major US exchanges only (exchcd 1/2/3)
      - Price > $1
    """
    mask = (
        df["shrcd"].isin([10, 11]) &
        df["exchcd"].isin([1, 2, 3]) &
        (df["prc"].abs() > 1)
    )

    return df[mask].copy()


# -------------------------------------------------
# Utilities (same as load_data.py)
# -------------------------------------------------

def add_tickers(df: pd.DataFrame, conn) -> pd.DataFrame:
    names = conn.raw_sql("""
        SELECT permno, ticker, namedt, nameendt
        FROM crsp.msenames
    """)

    names["nameendt"] = names["nameendt"].fillna(pd.Timestamp.today())

    df = df.merge(names, on="permno", how="left")

    df = df[
        (df["date"] >= df["namedt"]) &
        (df["date"] <= df["nameendt"])
    ]

    return df.drop(columns=["namedt", "nameendt"])


def make_pivot(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    return (
        df
        .pivot_table(
            index="date",
            columns="ticker",
            values=value_col,
            aggfunc="last"
        )
        .sort_index()
    )


# -------------------------------------------------
# Public API
# -------------------------------------------------

def load_complete_data(
    start: str = "1960-01-01",
    end: str = "2025-12-31",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load the full CRSP US equity universe (ticker-based), survivorship-free,
    monthly frequency, with standard academic filters applied.

    Returns
    -------
    returns_panel : pd.DataFrame
        Pivot table of returns (date x ticker)
    market_caps_panel : pd.DataFrame
        Pivot table of market caps in $thousands (date x ticker)
    """

    conn = wrds.Connection()

    raw = load_crsp_full(conn, start, end)
    filtered = apply_universe_filters(raw)

    returns_df = filtered[["permno", "date", "ret"]].copy()
    market_caps_df = filtered[["permno", "date", "me"]].copy()

    returns_df = add_tickers(returns_df, conn)
    market_caps_df = add_tickers(market_caps_df, conn)

    returns_panel = make_pivot(returns_df, "ret")
    market_caps_panel = make_pivot(market_caps_df, "me")

    # Align on dates only (not columns)
    common_dates = returns_panel.index.intersection(market_caps_panel.index)
    returns_panel = returns_panel.loc[common_dates].sort_index()
    market_caps_panel = market_caps_panel.loc[common_dates].sort_index()

    returns_panel.index = pd.to_datetime(returns_panel.index).to_period("M")
    market_caps_panel.index = pd.to_datetime(market_caps_panel.index).to_period("M")

    return returns_panel, market_caps_panel


# -------------------------------------------------
# Function to save datasets to CSV
# -------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"


def build_complete_crsp_dataset(
    start: str = "1960-01-01",
    end: str = "2025-12-31",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build and save full CRSP universe datasets (monthly).

    Parameters
    ----------
    start : str
    end : str

    Returns
    -------
    returns_panel, market_caps_panel
    """

    DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)

    returns, market_caps = load_complete_data(start=start, end=end)

    print(f"[CRSP Full Universe] Returns last date:     {returns.index.max()}")
    print(f"[CRSP Full Universe] Market caps last date: {market_caps.index.max()}")
    print(f"[CRSP Full Universe] Number of stocks:      {returns.shape[1]}")

    returns.to_csv(DATA_RAW_DIR / "complete_returns_monthly_with_tickers.csv")
    market_caps.to_csv(DATA_RAW_DIR / "complete_market_caps_monthly.csv")

    return returns, market_caps
