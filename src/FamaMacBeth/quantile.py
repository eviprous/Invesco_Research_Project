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
    dic,
    dic2,
    sp500,
    n_q1=3,
    n_q2=3,
    min_assets_per_cell=3,
):
    """Main portfolio construction routine."""

    f1_name = list(dic.keys())[0]
    f2_name = list(dic2.keys())[0]

    dates = (
        dic[f1_name].index
        .intersection(dic2[f2_name].index)
        .intersection(sp500.index)
    )

    port_rows, members_rows, cell_rows = [], [], []

    for date in tqdm(sorted(dates), desc="forming portfolios"):
        try:
            beta1 = dic[f1_name].loc[date]
            beta2 = dic2[f2_name].loc[date]
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
