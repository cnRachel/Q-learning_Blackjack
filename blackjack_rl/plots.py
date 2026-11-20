"""Figures for the checkpoint. Colors: fixed categorical order, validated
for color-vision deficiency; every series is also direct-labeled."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap

SERIES = {  # fixed order; color follows the policy everywhere
    "Q-learning (trained, ε=0)": "#2a78d6",
    "Optimal (exact DP reference)": "#eb6834",
    "Stand on 17+": "#1baf7a",
    "Random": "#eda100",
}
INK, INK_2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, SURFACE = "#e1e0d9", "#c3c2b7", "#fcfcfb"
HIT, STAND = "#2a78d6", "#eb6834"


def _style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=INK_2, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def learning_curve(curve: pd.DataFrame, references: pd.DataFrame, train_rewards=None,
                   window=10_000):
    """`references` must be scored on the same evaluation deals as `curve`."""
    fig, ax = plt.subplots(figsize=(10, 5.2), facecolor=SURFACE)
    _style(ax)
    x = curve["episode"]
    end = x.iloc[-1]
    name = "Q-learning (trained, ε=0)"

    if train_rewards is not None:
        rolling = pd.Series(train_rewards).rolling(window).mean()
        ax.plot(np.arange(1, len(rolling) + 1), rolling, color=MUTED, linewidth=1.5,
                label=f"Training hands, with exploration (rolling {window:,})")
        i = int(len(rolling) * 0.30)
        ax.annotate("Training hands (ε-greedy, still exploring)", (i, rolling.iloc[i]),
                    xytext=(12, -6), textcoords="offset points", fontsize=9, color=INK_2)

    for ref in ["Optimal (exact DP reference)", "Stand on 17+", "Random"]:
        y = references.loc[ref, "avg_reward"]
        ax.hlines(y, 0, end, color=SERIES[ref], linewidth=1.5, linestyle=(0, (4, 3)), label=ref)
        dy = -7 if ref.startswith("Optimal") else 0
        ax.annotate(f"{ref.split(' (')[0]}  {y:+.3f}", (end, y), xytext=(8, dy),
                    textcoords="offset points", fontsize=9, color=INK, va="center")

    ax.fill_between(x, curve["ci95_low"], curve["ci95_high"], color=SERIES[name],
                    alpha=0.15, linewidth=0)
    ax.plot(x, curve["avg_reward"], color=SERIES[name], linewidth=2,
            label="Q-learning, greedy (ε = 0), separate eval games")
    ax.annotate(f"Q-learning, ε = 0  {curve['avg_reward'].iloc[-1]:+.3f}",
                (end, curve["avg_reward"].iloc[-1]), xytext=(8, 9),
                textcoords="offset points", fontsize=9, color=INK, va="center")

    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{int(v):,}"))
    ax.set_xlabel("Training episodes", color=INK_2)
    ax.set_ylabel("Average reward per hand", color=INK_2)
    ax.set_title(f"Learning curve: frozen greedy agent scored on the same "
                 f"{int(curve['games'].iloc[0]):,} unseen deals at every checkpoint",
                 loc="left", color=INK, fontsize=11)
    ax.set_xlim(0, end * 1.25)
    ax.legend(loc="lower right", bbox_to_anchor=(1.0, 0.15), fontsize=8, frameon=False)
    fig.tight_layout()
    return fig


def final_bars(results: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(9, 3.4), facecolor=SURFACE)
    _style(ax)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    names = list(results.index)[::-1]
    y = np.arange(len(names))
    vals = results.loc[names, "avg_reward"]
    err = [vals - results.loc[names, "ci95_low"], results.loc[names, "ci95_high"] - vals]
    ax.barh(y, vals, height=0.6, color=[SERIES[n] for n in names],
            xerr=err, error_kw={"ecolor": INK, "elinewidth": 1, "capsize": 3})
    lows = results.loc[names, "ci95_low"]
    for yi, v, lo in zip(y, vals, lows):
        ax.annotate(f"{v:+.4f}", (lo, yi), xytext=(-8, 0), textcoords="offset points",
                    ha="right", va="center", fontsize=9, color=INK)
    ax.set_yticks(y, names, color=INK)
    ax.axvline(0, color=AXIS, linewidth=1)
    ax.set_xlabel("Average reward per hand (95% CI), 100,000 identical unseen deals", color=INK_2)
    ax.set_title("Final evaluation, no exploration", loc="left", color=INK, fontsize=12)
    ax.set_xlim(vals.min() * 1.3, 0.02)
    fig.tight_layout()
    return fig


def policy_maps(policy, optimal, title="Learned policy"):
    """Hit/Stand grid for hard and soft hands; ✕ marks disagreement with optimal."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 6.2), facecolor=SURFACE,
                             gridspec_kw={"width_ratios": [1, 1]})
    ups = list(range(2, 11)) + [1]
    cmap = ListedColormap([STAND, HIT])
    for ax, usable, totals in [(axes[0], False, range(4, 22)), (axes[1], True, range(12, 22))]:
        totals = list(totals)[::-1]
        grid = np.array([[1 if policy((t, u, usable)) == "Hit" else 0 for u in ups] for t in totals])
        ax.imshow(grid, cmap=cmap, vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(range(len(ups)), ["A" if u == 1 else str(u) for u in ups])
        ax.set_yticks(range(len(totals)), [str(t) for t in totals])
        ax.set_xticks(np.arange(-.5, len(ups)), minor=True)
        ax.set_yticks(np.arange(-.5, len(totals)), minor=True)
        ax.grid(which="minor", color=SURFACE, linewidth=2)
        ax.tick_params(which="minor", length=0)
        ax.tick_params(colors=INK_2, labelsize=9)
        for s in ax.spines.values():
            s.set_visible(False)
        for i, t in enumerate(totals):
            for j, u in enumerate(ups):
                learned = policy((t, u, usable))
                mark = "✕" if learned != optimal((t, u, usable)) else ("H" if learned == "Hit" else "S")
                ax.text(j, i, mark, ha="center", va="center", fontsize=8 if mark != "✕" else 12,
                        color="#ffffff", fontweight="bold" if mark == "✕" else "normal")
        ax.set_xlabel("Dealer up-card", color=INK_2)
        ax.set_ylabel("Player total", color=INK_2)
        ax.set_title("Soft hands (usable Ace)" if usable else "Hard hands (no usable Ace)",
                     loc="left", color=INK, fontsize=11)
    fig.suptitle(f"{title}   H = Hit (blue), S = Stand (orange), ✕ = differs from optimal",
                 x=0.01, ha="left", color=INK, fontsize=12)
    fig.tight_layout()
    return fig


def dealer_check(table: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(9, 4), facecolor=SURFACE)
    _style(ax)
    ups = ["A" if u == 1 else str(u) for u in table.index]
    x = np.arange(len(ups))
    w = 0.38
    ax.bar(x - w / 2 - 0.01, table["dataset_bust_rate"], w, color="#2a78d6", label="Kaggle dataset (as recorded)")
    ax.bar(x + w / 2 + 0.01, table["exact_bust_rate"], w, color="#eb6834", label="Simulator rule (exact)")
    ax.set_xticks(x, ups)
    ax.set_xlabel("Dealer up-card", color=INK_2)
    ax.set_ylabel("Dealer bust rate", color=INK_2)
    ax.set_title(f"Simulator dealer rule vs. {int(table['dealer_hands'].sum()):,} recorded dealer hands (naturals excluded)", loc="left", color=INK, fontsize=12)
    ax.legend(frameon=False, fontsize=9, loc="upper right")
    fig.tight_layout()
    return fig
