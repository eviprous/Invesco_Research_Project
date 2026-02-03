import numpy as np
import matplotlib.pyplot as plt


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


def plot_mean_grid(mean_grid, annualize=True):
    n_q1, n_q2 = mean_grid.shape
    if annualize:
        display_grid = mean_grid * 12 * 100
        title = "Annualized Mean Portfolio Returns (%)"
    else:
        display_grid = mean_grid
        title = "Mean Portfolio Returns"

    plt.figure(figsize=(6, 5))
    im = plt.imshow(display_grid.values, cmap="viridis")
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


def plot_expected_return_grid(pricing_df, n_q1, n_q2, annualize=True):
    expected = pricing_df["mean_return"] - pricing_df["alpha_fm"]
    expected_grid = np.full((n_q1, n_q2), np.nan, dtype=float)

    for p, val in expected.items():
        q1, q2 = map(int, p.replace("Q", "").split("_"))
        expected_grid[q1 - 1, q2 - 1] = val

    if annualize:
        display_grid = expected_grid * 12 * 100
        title = "Annualized Expected Returns (FM, %)"
    else:
        display_grid = expected_grid
        title = "Expected Returns (FM)"

    plt.figure(figsize=(6, 5))
    im = plt.imshow(display_grid, cmap="viridis")
    plt.title(title)
    plt.xlabel("Q2 (within Q1)")
    plt.ylabel("Q1")
    plt.xticks(range(n_q2), range(1, n_q2 + 1))
    plt.yticks(range(n_q1), range(1, n_q1 + 1))

    for i in range(n_q1):
        for j in range(n_q2):
            val = display_grid[i, j]
            if np.isfinite(val):
                label = f"{val:.2f}%" if annualize else f"{val:.4f}"
                plt.text(j, i, label, ha="center", va="center", color="white")

    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.show()
