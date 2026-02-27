import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

from src.FamaMacBeth.quantile import build_double_sorted_portfolios
from src.FamaMacBeth.cross_sectional_regression import (
    run_time_series_regressions,
    run_fama_macbeth,
)
from src.FamaMacBeth.pricing import pricing_errors, build_grids
import pandas as pd

def run_full_factor_pipeline(
    beta1_df,
    beta2_df,
    sp500,
    EBC,
    Cap_EBC,
    n_q1=3,
    n_q2=3,
    min_assets_per_cell=3,
    f1_name="EBC",
    f2_name="Cap_EBC",
    frequency="monthly",
    formation_frequency="quarterly"
):
    portrets, members, qmap, f1, f2 = build_double_sorted_portfolios(
    beta1_df,
    beta2_df,
    sp500,
    n_q1,
    n_q2,
    min_assets_per_cell,
    f1_name=f1_name,
    f2_name=f2_name,
    formation_frequency=formation_frequency
    )


    portrets_wide = portrets["ret_ew"].unstack([1, 2])
    portrets_wide.columns = [f"Q{a}_Q{b}" for a, b in portrets_wide.columns]

    #if frequency == "monthly":
       # portrets_wide.index = portrets_wide.index.to_period("M").to_timestamp()

    factors = pd.concat([EBC, Cap_EBC], axis=1).dropna()
    

    if frequency == "monthly":
        if isinstance(factors.index, pd.PeriodIndex):
            factors.index = factors.index.to_timestamp()

    if factors.shape[1] != 2:
        raise ValueError("Expected two single-column Series/DataFrames for factors.")
    
    factors.columns = [f1_name, f2_name]
    common_index = portrets_wide.index.intersection(factors.index)

    portrets_wide = portrets_wide.loc[common_index]
    factors = factors.loc[common_index]


    ts_summary = run_time_series_regressions(portrets_wide, factors, f1, f2, frequency=frequency)
    beta_table = ts_summary[["betaEBC", "betaCap_EBC"]].dropna()

    fm_table = run_fama_macbeth(portrets_wide[beta_table.index], beta_table, frequency=frequency)
    pricing_df = pricing_errors(portrets_wide, beta_table, ts_summary, fm_table)
    mean_grid, alpha_grid = build_grids(pricing_df, n_q1, n_q2)

    return {
        "portrets": portrets,
        "portrets_wide": portrets_wide,
        "members": members,
        "quantile_map": qmap,
        "ts_summary": ts_summary,
        "fm_table": fm_table,
        "pricing": pricing_df,
        "mean_grid": mean_grid,
        "alpha_grid": alpha_grid,
    }
