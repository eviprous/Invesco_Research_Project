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
    """
    Main portfolio construction routine.
    
    Logic:
    1. Form portfolios on 'formation_dates' (e.g., end of Quarter).
    2. Hold those specific stocks for the 'holding_steps' (e.g., next 3 months).
    3. Calculate and record monthly returns for every month in that holding period.
    """

    # --- 1. Standardize Indices to Timestamp ---
    if isinstance(beta1_df.index, pd.PeriodIndex):
        beta1_df = beta1_df.copy(); beta1_df.index = beta1_df.index.to_timestamp()
    if isinstance(beta2_df.index, pd.PeriodIndex):
        beta2_df = beta2_df.copy(); beta2_df.index = beta2_df.index.to_timestamp()
    if isinstance(sp500.index, pd.PeriodIndex):
        sp500 = sp500.copy(); sp500.index = sp500.index.to_timestamp()

    all_dates = (beta1_df.index.intersection(beta2_df.index).intersection(sp500.index))

    # --- 2. Identify Rebalancing (Formation) Dates ---
    if formation_frequency == "daily":
        formation_dates = all_dates
    elif formation_frequency == "monthly":
        formation_dates = pd.Series(all_dates).groupby(pd.Series(all_dates).dt.to_period("M")).last()
    elif formation_frequency == "quarterly":
        formation_dates = pd.Series(all_dates).groupby(pd.Series(all_dates).dt.to_period("Q")).last().values
    else:
        raise ValueError("formation_frequency must be 'monthly' or 'quarterly'")

    dates = pd.DatetimeIndex(formation_dates).sort_values()
    ret_dates = sp500.index  # <-- use return dates, not all_dates
    port_rows, members_rows, cell_rows = [], [], []

    # --- 3. Set Holding Window Logic ---

    for i, date in enumerate(tqdm(dates, desc=f"forming portfolios ({formation_frequency})")):
        try:
            # Characteristics (Betas) are known at 'date' (T)
            beta1 = beta1_df.loc[date]
            beta2 = beta2_df.loc[date]
            
            # DYNAMIC HOLDING LOGIC:
            # We hold from the day after 'date' until the next formation date (inclusive)
            if i + 1 < len(dates):
                next_formation = dates[i+1]
                holding_dates = ret_dates[(ret_dates > date) & (ret_dates <= next_formation)]
            else:
                holding_dates = ret_dates[ret_dates > date]

            if len(holding_dates) == 0:
                continue

        except (KeyError, IndexError):
            # Skip if we are at the end of the dataset or date is missing
            continue
        
        # --- 4. Sort Stocks into Quantiles ---
        df_chars = pd.DataFrame({"beta1": beta1, "beta2": beta2}).dropna()
        if df_chars.empty:
            continue

        df_chars["q1"] = assign_quantiles(df_chars["beta1"], n_q1)
        df_chars["q2"] = pd.Series(index=df_chars.index, dtype="Int64")

        for i in range(1, n_q1 + 1):
            mask = df_chars["q1"] == i
            if mask.any():
                df_chars.loc[mask, "q2"] = assign_quantiles(df_chars.loc[mask, "beta2"], n_q2)

        # --- 5. Record Returns for the Period ---
        # The stocks identified at 'date' are held constant for all 'hold_date' in this window
        for hold_date in holding_dates:
            current_rets = sp500.loc[hold_date]

            for i_q in range(1, n_q1 + 1):
                for j_q in range(1, n_q2 + 1):
                    # Tickers assigned at 'date' are used for 'hold_date'
                    tickers = df_chars[(df_chars.q1 == i_q) & (df_chars.q2 == j_q)].index
                    cell_rets = current_rets.reindex(tickers).dropna()
                    
                    n_assets = len(cell_rets)
                    ret = cell_rets.mean() if n_assets >= min_assets_per_cell else np.nan

                    port_rows.append({
                        "date": hold_date, 
                        "q1": i_q, "q2": j_q,
                        "ret_ew": ret, "n_assets": n_assets,
                    })
                    
                    # Record specific stock membership for this cell/month
                    cell_rows.append({
                        "date": hold_date,
                        "q1": i_q, "q2": j_q,
                        "stocks": tickers.tolist(),
                    })

            # Record full membership frame for cross-sectional alignment
            tmp = df_chars[["q1", "q2"]].copy()
            tmp["date"] = hold_date
            tmp["ticker"] = tmp.index
            members_rows.append(tmp.reset_index(drop=True))

    # --- 6. Final DataFrame Construction ---
    portrets = pd.DataFrame(port_rows).set_index(["date", "q1", "q2"]).sort_index()
    members = pd.concat(members_rows, ignore_index=True).set_index(["date", "ticker"]).sort_index()
    quantile_map = pd.DataFrame(cell_rows).set_index(["date", "q1", "q2"]).sort_index()
    

    return portrets, members, quantile_map, f1_name, f2_name

