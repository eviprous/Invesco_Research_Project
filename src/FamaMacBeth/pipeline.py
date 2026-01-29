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
    dic,
    dic2,
    sp500,
    EBC,
    Cap_EBC,
    n_q1=3,
    n_q2=3,
    min_assets_per_cell=3,
):
    portrets, members, qmap, f1, f2 = build_double_sorted_portfolios(
        dic, dic2, sp500, n_q1, n_q2, min_assets_per_cell
    )

    portrets_wide = portrets["ret_ew"].unstack([1, 2])
    portrets_wide.columns = [f"Q{a}_Q{b}" for a, b in portrets_wide.columns]

    factors = pd.concat([EBC, Cap_EBC], axis=1).dropna()
    factors = factors.loc[portrets_wide.index]

    ts_summary = run_time_series_regressions(portrets_wide, factors, f1, f2)
    beta_table = ts_summary[["betaEBC", "betaCap_EBC"]].dropna()

    fm_table = run_fama_macbeth(portrets_wide[beta_table.index], beta_table, factors)
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
