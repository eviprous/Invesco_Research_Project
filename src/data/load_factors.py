import pandas as pd
import zipfile
import io
import requests
from pathlib import Path

# define root so files are saved in the correct place
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"


# URLs

FF5_MONTHLY_URL = ( "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_5_Factors_2x3_CSV.zip")

MOM_MONTHLY_URL = ("https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Momentum_Factor_CSV.zip")

FF5_DAILY_URL = ("https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_5_Factors_2x3_daily_CSV.zip")

MOM_DAILY_URL = ("https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Momentum_Factor_daily_CSV.zip")


def _load_ff_csv_from_zip(url: str, skiprows: int) -> pd.DataFrame:
    """
    Load and clean a Fama-French CSV from a zipped URL.
    Keeps only YYYYMM or YYYYMMDD rows.
    """
    r = requests.get(url)
    z = zipfile.ZipFile(io.BytesIO(r.content))
    name = z.namelist()[0]

    df = pd.read_csv(z.open(name), skiprows=skiprows)

    df.iloc[:, 0] = df.iloc[:, 0].astype(str).str.strip()
    df = df[df.iloc[:, 0].str.fullmatch(r"\d{6}|\d{8}")]

    return df


# -------------------------------------------------
# Monthly factors
# -------------------------------------------------

def load_ff_factors_monthly() -> pd.DataFrame:
    """
    Load FF5 + Momentum monthly factors.
    Index is month-start date.
    """
    ff5 = _load_ff_csv_from_zip(FF5_MONTHLY_URL, skiprows=3)
    mom = _load_ff_csv_from_zip(MOM_MONTHLY_URL, skiprows=13)

    ff5.columns = ["date", "MKT_RF", "SMB", "HML", "RMW", "CMA", "RF"]
    mom.columns = ["date", "MOM"]

    ff5["date"] = pd.to_datetime(ff5["date"], format="%Y%m") + pd.offsets.MonthBegin(0)
    mom["date"] = pd.to_datetime(mom["date"], format="%Y%m") + pd.offsets.MonthBegin(0)

    df = ff5.merge(mom, on="date", how="inner")
    df.set_index("date", inplace=True)

    return df.astype(float) / 100.0


# -------------------------------------------------
# Daily factors
# -------------------------------------------------

def load_ff_factors_daily() -> pd.DataFrame:
    """
    Load FF5 + Momentum daily factors.
    Index is daily date.
    """
    ff5 = _load_ff_csv_from_zip(FF5_DAILY_URL, skiprows=3)
    mom = _load_ff_csv_from_zip(MOM_DAILY_URL, skiprows=13)

    ff5.columns = ["date", "MKT_RF", "SMB", "HML", "RMW", "CMA", "RF"]
    mom.columns = ["date", "MOM"]

    ff5["date"] = pd.to_datetime(ff5["date"], format="%Y%m%d")
    mom["date"] = pd.to_datetime(mom["date"], format="%Y%m%d")

    df = ff5.merge(mom, on="date", how="inner")
    df.set_index("date", inplace=True)

    return df.astype(float) / 100.0


# -------------------------------------------------
# Public builder (CSV artifacts)
# -------------------------------------------------

def build_factors_dataset(frequency: str)-> pd.DataFrame:
    """
    Build and save Fama-French factor dataset.

    Parameters
    ----------
    frequency : {"daily", "monthly"}
    Returns
    -------
    pd.DataFrame
    """

    DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)

    if frequency == "daily":
        ff = load_ff_factors_daily()
        ff.to_csv(DATA_RAW_DIR / "ff_factors_daily.csv")

    elif frequency == "monthly":
        ff = load_ff_factors_monthly()
        ff.to_csv(DATA_RAW_DIR / "ff_factors_monthly.csv")

    else:
        raise ValueError("frequency must be 'daily' or 'monthly'")

    return ff
