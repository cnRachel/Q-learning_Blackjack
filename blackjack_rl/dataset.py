"""Kaggle dataset: "900,000 Hands of BlackJack Results" (mojocolors).

https://www.kaggle.com/datasets/mojocolors/900000-hands-of-blackjack-results

Layout: 150,000 rounds x 6 players. Each row is one player's hand; the six
rows of a round share the same dealer cards. Columns used here:
    card1..card5       player cards in order (0 = no card; Ace recorded as 1 or 11)
    dealcard1..5       dealer cards (dealcard1 is treated as the up-card)
    sumofcards         player's final total
    winloss            "Win" / "Loss" / "Push"

How the dataset is used in this project:
  1. Validate the simulator: dealer bust rate by up-card in the real data
     vs. the exact rate implied by our dealer rule.
  2. Offline Q-learning: each recorded hand becomes a sequence of
     (s, a, r, s') experiences, fed through the SAME Q-learning update.
"""

from __future__ import annotations

import random
from collections import defaultdict
from pathlib import Path

import pandas as pd

from .agent import ACTIONS, apply_transition
from .env import hand_value

KAGGLE_HANDLE = "mojocolors/900000-hands-of-blackjack-results"
CSV_NAME = "blkjckhands.csv"
REPO_ROOT = Path(__file__).resolve().parents[1]
FULL_CSV = REPO_ROOT / "data" / CSV_NAME
SAMPLE_CSV = REPO_ROOT / "data" / "sample_hands.csv"

PLAYER_COLS = [f"card{i}" for i in range(1, 6)]
DEALER_COLS = [f"dealcard{i}" for i in range(1, 6)]
REWARD = {"Win": 1, "Loss": -1, "Push": 0}


def download_full_dataset():
    """Download via kagglehub (needs Kaggle credentials; see README)."""
    import kagglehub  # imported lazily so the rest of the repo works without it

    path = Path(kagglehub.dataset_download(KAGGLE_HANDLE))
    return path / CSV_NAME


def load_hands(source="auto"):
    """Load the dataset.

    source="auto":   data/blkjckhands.csv if present, else download with
                     kagglehub, else fall back to data/sample_hands.csv
    source="sample": the 6,000-row sample committed to the repo
    source=<path>:   any CSV with the same columns
    """
    if source == "sample":
        path = SAMPLE_CSV
    elif source == "auto":
        if FULL_CSV.exists():
            path = FULL_CSV
        else:
            try:
                path = download_full_dataset()
            except Exception as exc:  # no kagglehub / no credentials / offline
                print(f"Full dataset unavailable ({exc.__class__.__name__}); using sample.")
                path = SAMPLE_CSV
    else:
        path = Path(source)
    df = pd.read_csv(path, index_col=0)
    df = df.reset_index(drop=True)
    df["round"] = df.index // 6
    print(f"Loaded {len(df):,} hands from {path.name}")
    return df


def _cards(row, cols):
    # Drop empty slots (the data sometimes leaves card3 = 0 and fills card4),
    # and record every Ace as 1; hand_value() decides whether it counts as 11.
    return [1 if c == 11 else int(c) for c in (row[c] for c in cols) if c != 0]


def hand_to_episode(row):
    """Convert one recorded hand into a list of (s, a, r, s', done).

    Returns None for hands with no decision (a natural on either side) or
    hands whose recorded total doesn't match the cards (data error).
    """
    player = _cards(row, PLAYER_COLS)
    dealer = _cards(row, DEALER_COLS)
    up = dealer[0]
    if hand_value(player[:2])[0] == 21 or hand_value(dealer[:2])[0] == 21:
        return None
    if hand_value(player)[0] != row["sumofcards"]:
        return None
    if hand_value(dealer)[0] != row["sumofdeal"]:   # dealer mis-scored -> reward unreliable
        return None

    def state(n):
        total, usable = hand_value(player[:n])
        return (total, up, usable)

    final_reward = REWARD[row["winloss"]]
    episode = []
    for n in range(2, len(player) + 1):
        s = state(n)
        if n < len(player):                      # another card followed -> Hit
            nxt = state(n + 1)
            busted = nxt[0] > 21
            episode.append((s, "Hit", -1 if busted else 0, None if busted else nxt, busted))
            if busted:
                break
        else:                                    # last card -> Stand
            episode.append((s, "Stand", final_reward, None, True))
    return episode


def build_episodes(df):
    episodes, skipped = [], 0
    for row in df.to_dict("records"):
        ep = hand_to_episode(row)
        if ep is None:
            skipped += 1
        else:
            episodes.append(ep)
    return episodes, skipped


def dealer_bust_by_upcard(df):
    """Observed dealer bust rate per up-card, one dealer hand per round,
    excluding dealer naturals (to match the simulator's conditioning).

    Also counts 'mis-scored' hands: the recorded total disagrees with correct
    Blackjack scoring of the recorded cards (e.g. A,A,4,6 recorded as 22
    instead of 12 because an Ace was never reduced from 11 to 1)."""
    rounds = df.groupby("round").first()
    rows = []
    for r in rounds[DEALER_COLS + ["sumofdeal"]].to_dict("records"):
        cards = _cards(r, DEALER_COLS)
        if hand_value(cards[:2])[0] == 21:
            continue
        mis_scored = hand_value(cards)[0] != r["sumofdeal"]
        rows.append((cards[0], r["sumofdeal"] > 21, mis_scored))
    out = pd.DataFrame(rows, columns=["up", "bust", "mis_scored"])
    return out.groupby("up").agg(
        dataset_bust_rate=("bust", "mean"),
        mis_scored_hands=("mis_scored", "sum"),
        dealer_hands=("bust", "count"),
    )


def train_offline_q(episodes, passes=20, alpha=0.01, gamma=1.0, seed=471):
    """Offline (batch) Q-learning: replay recorded experiences through the same
    update used online. No new games are played."""
    rng = random.Random(seed)
    Q = defaultdict(float)
    seen = defaultdict(int)
    order = list(range(len(episodes)))
    for _ in range(passes):
        rng.shuffle(order)
        for i in order:
            for s, a, r, s2, done in episodes[i]:
                apply_transition(Q, s, a, r, s2, done, alpha, gamma)
    for ep in episodes:
        for s, a, *_ in ep:
            seen[(s, a)] += 1
    return Q, seen


def offline_policy(Q, seen):
    """Greedy policy restricted to actions that were actually observed in the
    data. A Q-value for an action the recorded players never took is just the
    initial 0, not a learned estimate."""

    def policy(state, rng=None):
        observed = [a for a in ACTIONS if seen.get((state, a), 0) > 0]
        if not observed:
            return "Hit" if state[0] < 12 else "Stand"
        return max(observed, key=lambda a: Q[(state, a)])

    return policy
