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

def build_double_sorted_members(
    beta_EBC_df,
    beta_CW_EBC_df,
    returns_df,
    n_q1,
    n_q2,
    formation_frequency,
):
    """
    Constructs ONLY portfolio membership.
    
    Output:
    members: MultiIndex DataFrame indexed by (date, ticker)
             columns = ['q1', 'q2']
    """

    # --- Standardize Indices ---
    if isinstance(beta_EBC_df.index, pd.PeriodIndex):
        beta_EBC_df = beta_EBC_df.copy()
        beta_EBC_df.index = beta_EBC_df.index.to_timestamp()

    if isinstance(beta_CW_EBC_df.index, pd.PeriodIndex):
        beta_CW_EBC_df = beta_CW_EBC_df.copy()
        beta_CW_EBC_df.index = beta_CW_EBC_df.index.to_timestamp()

    if isinstance(returns_df.index, pd.PeriodIndex):
        returns_df = returns_df.copy()
        returns_df.index = returns_df.index.to_timestamp()

    # --- Common Dates ---
    all_dates = (
        beta_EBC_df.index
        .intersection(beta_CW_EBC_df.index)
        .intersection(returns_df.index)
    )

    # --- Formation Dates ---
    if formation_frequency == "daily":
        formation_dates = all_dates

    elif formation_frequency == "monthly":
        formation_dates = (
            pd.Series(all_dates)
            .groupby(pd.Series(all_dates).dt.to_period("M"))
            .last()
            .values
        )

    elif formation_frequency == "quarterly":
        formation_dates = (
            pd.Series(all_dates)
            .groupby(pd.Series(all_dates).dt.to_period("Q"))
            .last()
            .values
        )

    else:
        raise ValueError("formation_frequency must be daily/monthly/quarterly")

    dates = pd.DatetimeIndex(formation_dates).sort_values()
    ret_dates = returns_df.index

    members_rows = []

    # --- Main Loop ---
    for i, date in enumerate(dates):

        try:
            beta1 = beta_EBC_df.loc[date]
            beta2 = beta_CW_EBC_df.loc[date]

            # Hold AFTER formation date up to and including next formation date
            if i + 1 < len(dates):
                next_formation = dates[i + 1]
                holding_dates = ret_dates[
                    (ret_dates > date) & (ret_dates <= next_formation)
                ]
            else:
                holding_dates = ret_dates[ret_dates > date]

            if len(holding_dates) == 0:
                continue

        except KeyError:
            continue

        # --- Cross-Section ---
        df_chars = pd.DataFrame({
            "beta1": beta1,
            "beta2": beta2
        }).dropna()

        if df_chars.empty:
            continue

        # First sort
        df_chars["q1"] = assign_quantiles(df_chars["beta1"], n_q1)

        # Second sort within q1
        df_chars["q2"] = pd.Series(index=df_chars.index, dtype="Int64")

        for q in range(1, n_q1 + 1):
            mask = df_chars["q1"] == q
            if mask.any():
                df_chars.loc[mask, "q2"] = assign_quantiles(
                    df_chars.loc[mask, "beta2"],
                    n_q2
                )

        # --- Expand membership across holding window ---
        for hold_date in holding_dates:
            tmp = df_chars[["q1", "q2"]].copy()
            tmp["date"] = hold_date
            tmp["ticker"] = tmp.index
            members_rows.append(tmp.reset_index(drop=True))

    # --- Final Output ---
    members = (
        pd.concat(members_rows, ignore_index=True)
        .set_index(["date", "ticker"])
        .sort_index()
    )

    return members

def build_double_sorted_portfolios(
    beta_EBC_df,
    beta_CW_EBC_df,
    returns_df,
    caps_df,
    cap_weights,
    n_q1,
    n_q2,
    min_assets_per_cell,
    f1_name,
    f2_name,
    formation_frequency,
):
    """
    Main portfolio construction routine.
    
    Logic:
    1. Form portfolios on 'formation_dates' (e.g., end of Quarter).
    2. Hold those specific stocks for the 'holding_steps' (e.g., next 3 months).
    3. Calculate and record monthly returns for every month in that holding period.
    """

    # Standardize Indices to Timestamp 
    if isinstance(beta_EBC_df.index, pd.PeriodIndex):
        beta_EBC_df = beta_EBC_df.copy(); beta_EBC_df.index = beta_EBC_df.index.to_timestamp()
    if isinstance(beta_CW_EBC_df.index, pd.PeriodIndex):
        beta_CW_EBC_df = beta_CW_EBC_df.copy(); beta_CW_EBC_df.index = beta_CW_EBC_df.index.to_timestamp()
    if isinstance(returns_df.index, pd.PeriodIndex):
        returns_df = returns_df.copy(); returns_df.index = returns_df.index.to_timestamp()

    all_dates = (beta_EBC_df.index.intersection(beta_CW_EBC_df.index).intersection(returns_df.index))

    # Identify Rebalancing (Formation) Dates
    if formation_frequency == "daily":
        formation_dates = all_dates
    elif formation_frequency == "monthly":
        formation_dates = pd.Series(all_dates).groupby(pd.Series(all_dates).dt.to_period("M")).last()
    elif formation_frequency == "quarterly":
        formation_dates = pd.Series(all_dates).groupby(pd.Series(all_dates).dt.to_period("Q")).last().values
    else:
        raise ValueError("formation_frequency must be 'daily', 'monthly' or 'quarterly'")

    dates = pd.DatetimeIndex(formation_dates).sort_values()
    ret_dates = returns_df.index  # <-- use return dates, not all_dates
    port_rows, members_rows, cell_rows = [], [], []

    portfolio_weighting = "CW" if cap_weights else "EW"

    # Set Holding Window Logic 
    for i, date in enumerate(tqdm(dates, desc=f"forming portfolios ({formation_frequency})")):
        try:
            # Characteristics (Betas) are known at 'date'
            beta1 = beta_EBC_df.loc[date]
            beta2 = beta_CW_EBC_df.loc[date]
            
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


        # create weights for cap weighted at formation
        weights_dict = {}

        for i_q in range(1, n_q1 + 1):
            for j_q in range(1, n_q2 + 1):

                tickers = df_chars[
                    (df_chars.q1 == i_q) &
                    (df_chars.q2 == j_q)
                ].index

                if len(tickers) < min_assets_per_cell:
                    weights_dict[(i_q, j_q)] = None
                    continue

                if cap_weights:
                    formation_caps = caps_df.loc[date, tickers]
                    #w = formation_caps / formation_caps.sum()
                    if formation_caps.sum() == 0:
                        weights_dict[(i_q, j_q)] = None
                    else:
                        w = formation_caps / formation_caps.sum()
                        weights_dict[(i_q, j_q)] = w
                else:
                    w = pd.Series(1.0 / len(tickers), index=tickers)
                    weights_dict[(i_q, j_q)] = w


        # Record Returns for the Period
        # The stocks identified at 'date' are held constant for all 'hold_date' in this window
        for hold_date in holding_dates:
            current_rets = returns_df.loc[hold_date]

            for i_q in range(1, n_q1 + 1):
                for j_q in range(1, n_q2 + 1):

                    w = weights_dict.get((i_q, j_q))

                    if w is None:
                        ret = np.nan
                        n_assets = 0
                    else:
                       ret_s = returns_df.loc[hold_date, w.index]
                       ret = (w * ret_s).sum()
                       n_assets = ret_s.notna().sum()

                    port_rows.append({
                        "date": hold_date, 
                        "q1": i_q, "q2": j_q,
                        f"ret_{portfolio_weighting}": ret, "n_assets": n_assets,
                    })
                    
                    # Record specific stock membership for this cell/month
                    cell_rows.append({
                        "date": hold_date,
                        "q1": i_q, "q2": j_q,
                        "stocks": w.index.tolist() if w is not None else []
                    })

            # Record full membership frame for cross-sectional alignment
            tmp = df_chars[["q1", "q2"]].copy()
            tmp["date"] = hold_date
            tmp["ticker"] = tmp.index
            members_rows.append(tmp.reset_index(drop=True))

    # Final DataFrame Construction
    portrets = pd.DataFrame(port_rows).set_index(["date", "q1", "q2"]).sort_index()
    members = pd.concat(members_rows, ignore_index=True).set_index(["date", "ticker"]).sort_index()
    quantile_map = pd.DataFrame(cell_rows).set_index(["date", "q1", "q2"]).sort_index()
    

    return portrets, members, quantile_map, f1_name, f2_name

