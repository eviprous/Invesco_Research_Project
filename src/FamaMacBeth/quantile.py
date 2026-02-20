import numpy as np
import pandas as pd
from tqdm import tqdm

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


def build_double_sorted_portfolios(
    beta1_df,
    beta2_df,
    sp500,
    n_q1=3,
    n_q2=3,
    min_assets_per_cell=3,
    f1_name="EBC",
    f2_name="Cap_EBC",
    formation_frequency = 'quarterly',
):
    """Main portfolio construction routine."""

    if isinstance(beta1_df.index, pd.PeriodIndex):
        beta1_df = beta1_df.copy()
        beta1_df.index = beta1_df.index.to_timestamp()
    if isinstance(beta2_df.index, pd.PeriodIndex):
        beta2_df = beta2_df.copy()
        beta2_df.index = beta2_df.index.to_timestamp()
    if isinstance(sp500.index, pd.PeriodIndex):
        sp500 = sp500.copy()
        sp500.index = sp500.index.to_timestamp()

    all_dates = (beta1_df.index.intersection(beta2_df.index).intersection(sp500.index))
    print(all_dates)

    if formation_frequency == "daily":
        formation_dates = all_dates
    elif formation_frequency == "monthly":
        formation_dates = (
            pd.Series(all_dates)
            .groupby(pd.Series(all_dates).dt.to_period("M"))
            .last())
    elif formation_frequency == "quarterly":
        formation_dates = (
            pd.Series(all_dates)
            .groupby(pd.Series(all_dates).dt.to_period("Q"))
            .last()
            .values)

    else:
        raise ValueError("formation_frequency must be 'monthly' or 'quarterly'")



    dates = pd.DatetimeIndex(formation_dates).sort_values()


    port_rows, members_rows, cell_rows = [], [], []

    for date in tqdm(sorted(dates), desc="forming portfolios"):
        try:
            beta1 = beta1_df.loc[date]
            beta2 = beta2_df.loc[date]
        except KeyError:
            continue

        df = pd.DataFrame({
            "beta1": beta1,
            "beta2": beta2,
            "ret": sp500.loc[date],
        }).dropna()

        if df.empty:
            continue

        df["q1"] = assign_quantiles(df["beta1"], n_q1)
        df["q2"] = pd.Series(index=df.index, dtype="Int64")

        for i in range(1, n_q1 + 1):
            mask = df["q1"] == i
            if mask.any():
                df.loc[mask, "q2"] = assign_quantiles(df.loc[mask, "beta2"], n_q2)

        # (date, ticker) membership
        tmp = df[["q1", "q2"]].copy()
        tmp["date"] = date
        tmp["ticker"] = tmp.index
        members_rows.append(tmp.reset_index(drop=True))

        # cell membership + returns
        for i in range(1, n_q1 + 1):
            for j in range(1, n_q2 + 1):
                cell = df[(df.q1 == i) & (df.q2 == j)]
                n_assets = len(cell)
                ret = cell["ret"].mean() if n_assets >= min_assets_per_cell else np.nan

                port_rows.append({
                    "date": date,
                    "q1": i,
                    "q2": j,
                    "ret_ew": ret,
                    "n_assets": n_assets,
                })

                cell_rows.append({
                    "date": date,
                    "q1": i,
                    "q2": j,
                    "stocks": cell.index.tolist(),
                })
    portrets = (
        pd.DataFrame(port_rows)
        .assign(date=lambda x: pd.to_datetime(x.date))
        .set_index(["date", "q1", "q2"])
        .sort_index()
    )
    members = (
        pd.concat(members_rows, ignore_index=True)
        .assign(date=lambda x: pd.to_datetime(x.date))
        .set_index(["date", "ticker"])
        .sort_index()
    )
    quantile_map = (
        pd.DataFrame(cell_rows)
        .assign(date=lambda x: pd.to_datetime(x.date))
        .set_index(["date", "q1", "q2"])
        .sort_index()
    )
    return portrets, members, quantile_map, f1_name, f2_name
