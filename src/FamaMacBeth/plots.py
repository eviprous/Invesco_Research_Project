import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def plot_cum_log_returns(portfolio_returns, title=None):
    log_returns = np.log1p(portfolio_returns)
    cum_log_returns = log_returns.cumsum()

    plt.figure(figsize=(14, 8))
    palette = plt.cm.tab20.colors
    colors = (palette * ((len(cum_log_returns.columns) // len(palette)) + 1))[
        : len(cum_log_returns.columns)
    ]

    for col, color in zip(cum_log_returns.columns, colors):
        series = cum_log_returns[col].dropna()
        if series.empty:
            continue
        plt.plot(series.index, series.values, color=color, linewidth=2, label=col)

    if title is None:
        title = "Cumulative Log Excess Returns"
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Cumulative Log Return")
    plt.legend(
        ncol=2 if len(cum_log_returns.columns) > 6 else 1,
        fontsize=9,
        bbox_to_anchor=(1.05, 1),
        loc="upper left",
    )
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.show()


def plot_mean_grid(mean_grid, annualize=True, frequency = 'monthly'):
    n_q1, n_q2 = mean_grid.shape
    if annualize:
        if frequency == "monthly":
            ann_factor = 12
        elif frequency == "daily":
            ann_factor = 252
        else:
            raise ValueError("freq must be 'monthly' or 'daily'")

        display_grid = mean_grid * ann_factor * 100
        title = "Annualized Mean Portfolio Returns (%)"
    else:
        display_grid = mean_grid
        title = "Mean Portfolio Returns"

    plt.figure(figsize=(6, 5))
    im = plt.imshow(display_grid.values, cmap="YlGnBu")
    plt.title(title)
    plt.xlabel("Q2 (within Q1)")
    plt.ylabel("Q1")
    plt.xticks(range(n_q2), range(1, n_q2 + 1))
    plt.yticks(range(n_q1), range(1, n_q1 + 1))

    for i in range(n_q1):
        for j in range(n_q2):
            val = display_grid.iloc[i, j]
            if np.isfinite(val):
                label = f"{val:.2f}%" if annualize else f"{val:.4f}"
                plt.text(j, i, label, ha="center", va="center", color="white")

    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.show()

    return display_grid


def plot_alpha_grid(alpha_grid):
    n_q1, n_q2 = alpha_grid.shape
    vmax = np.nanmax(np.abs(alpha_grid.values))
    vmin = -vmax

    plt.figure(figsize=(6, 5))
    im = plt.imshow(alpha_grid.values, cmap="coolwarm", vmin=vmin, vmax=vmax)
    plt.title("Fama-MacBeth Alpha (by Portfolio)")
    plt.xlabel("Q2 (within Q1)")
    plt.ylabel("Q1")
    plt.xticks(range(n_q2), range(1, n_q2 + 1))
    plt.yticks(range(n_q1), range(1, n_q1 + 1))
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.show()


def plot_expected_return_grid(beta_table, fm_table, n_q1, n_q2, frequency = 'monthly'):
    def split_label(label):
        q1, q2 = label.replace("Q", "").split("_")
        return int(q1), int(q2)

    betaEBC_grid = pd.DataFrame(
        np.nan, index=range(1, n_q1 + 1), columns=range(1, n_q2 + 1)
    )
    betaCap_grid = betaEBC_grid.copy()

    for p in beta_table.index:
        q1, q2 = split_label(p)
        betaEBC_grid.loc[q1, q2] = beta_table.loc[p, "betaEBC"]
        betaCap_grid.loc[q1, q2] = beta_table.loc[p, "betaCap_EBC"]

    gamma0_bar = fm_table.loc["alpha", "lambda_mean"]
    gamma1_bar = fm_table.loc["lambdaEBC", "lambda_mean"]
    gamma2_bar = fm_table.loc["lambda_Cap_EBC", "lambda_mean"]

    fm_implied_grid = (
        gamma0_bar + gamma1_bar * betaEBC_grid + gamma2_bar * betaCap_grid
    )

    if frequency == "monthly":
        ann_factor = 12
    elif frequency == "daily":
        ann_factor = 252
    else:
        raise ValueError("freq must be 'monthly' or 'daily'")

    #fm_implied_grid_annual = (1 + fm_implied_grid.astype(float)) ** ann_factor - 1
    fm_implied_grid_annual = fm_implied_grid.astype(float) * ann_factor

    #fm_implied_grid_annual = (1 + fm_implied_grid.astype(float)) ** 12 - 1

    import seaborn as sns

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        fm_implied_grid_annual,
        cmap="YlGnBu",
        annot=True,
        fmt=".2%",
        cbar_kws={"label": "Annualized FM-Implied Return"},
    )
    plt.title("Fama–MacBeth Implied Returns by (EBC, Preference) Quantiles")
    plt.xlabel("Preference quantile")
    plt.ylabel("EBC quantile")
    plt.tight_layout()
    plt.show()

    return fm_implied_grid_annual


