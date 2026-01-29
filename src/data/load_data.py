import pandas as pd
import wrds
from typing import Tuple
from pathlib import Path

# -------------------------------------------------
# Core WRDS pulls
# -------------------------------------------------

def load_sp500_membership(conn: wrds.Connection) -> pd.DataFrame:
    return conn.raw_sql("""
        SELECT permno, start, ending
        FROM crsp.dsp500list
    """)


def load_crsp_returns(conn, start, end, frequency: str) -> pd.DataFrame:

    table = "crsp.msf" if frequency == "monthly" else "crsp.dsf"

    return conn.raw_sql(f"""
        SELECT permno, date, ret
        FROM {table}
        WHERE date BETWEEN '{start}' AND '{end}'
    """)


def load_crsp_market_caps(conn, start, end, frequency: str) -> pd.DataFrame:

    table = "crsp.msf" if frequency == "monthly" else "crsp.dsf"

    df = conn.raw_sql(f"""
        SELECT permno, date, prc, shrout
        FROM {table}
        WHERE date BETWEEN '{start}' AND '{end}'
    """)

    # Market equity in thousands of dollars (CRSP convention)
    df["me"] = df["prc"].abs() * df["shrout"]

    return df[["permno", "date", "me"]]


# -------------------------------------------------
# Utilities
# -------------------------------------------------

def filter_to_sp500(df, membership) -> pd.DataFrame:
    df = df.merge(membership, on="permno", how="left")

    return df[
        (df["date"] >= df["start"]) &
        (df["date"] <= df["ending"])
    ].drop(columns=["start", "ending"])


# crsp doesnt have tickers we want to replace permno with ticker names
def add_tickers(df, conn) -> pd.DataFrame:
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

def load_sp500_data(
    start: str = "1960-01-01",
    end: str = "2025-12-31",
    frequency: str = "monthly"
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load S&P 500 CRSP panels (ticker-based), survivorship-free.
    """

    if frequency not in {"monthly", "daily"}:
        raise ValueError("frequency must be 'monthly' or 'daily'")

    conn = wrds.Connection()

    membership = load_sp500_membership(conn)

    returns = load_crsp_returns(conn, start, end, frequency)
    market_caps = load_crsp_market_caps(conn, start, end, frequency)

    returns = filter_to_sp500(returns, membership)
    market_caps = filter_to_sp500(market_caps, membership)

    returns = add_tickers(returns, conn)
    market_caps = add_tickers(market_caps, conn)

    returns_panel = make_pivot(returns, "ret")
    market_caps_panel = make_pivot(market_caps, "me")

    # -------------------------------------------------
    # IMPORTANT: align on DATES ONLY (not columns)
    # -------------------------------------------------
    common_dates = returns_panel.index.intersection(market_caps_panel.index)

    returns_panel = returns_panel.loc[common_dates].sort_index()
    market_caps_panel = market_caps_panel.loc[common_dates].sort_index()

    if frequency == "monthly":
        returns_panel.index = pd.to_datetime(returns_panel.index).to_period("M")
        market_caps_panel.index = pd.to_datetime(market_caps_panel.index).to_period("M")



    return returns_panel, market_caps_panel

# -------------------------------------------------
# Function to save datasets in csv
# -------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"


def build_crsp_dataset( start: str = "1960-01-01", end: str = "2025-12-31", frequency: str = "monthly") -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build and save CRSP S&P 500 datasets.

    Parameters
    ----------
    start : str
    end : str
    frequency : {"daily", "monthly"}

    Returns
    -------
    returns_panel, market_caps_panel
    """

    DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)

    returns, market_caps = load_sp500_data(
        start=start,
        end=end,
        frequency=frequency
    )

    # -----------------------------
    # Diagnostic: CRSP availability
    # -----------------------------
    print(f"[CRSP] Returns last date: {returns.index.max()}")
    print(f"[CRSP] Market caps last date: {market_caps.index.max()}")


    returns.to_csv(DATA_RAW_DIR / f"sp500_returns_{frequency}_with_tickers.csv")
    market_caps.to_csv(DATA_RAW_DIR / f"sp500_market_caps_{frequency}.csv")

    return returns, market_caps