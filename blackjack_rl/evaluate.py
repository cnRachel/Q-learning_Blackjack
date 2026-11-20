"""Exploration-free evaluation on separate games (instructor Required Revision #2).

* Evaluation never updates a Q-table; trained agents are frozen greedy
  policies (epsilon = 0).
* Evaluation uses its own seed (EVAL_SEED), separate from training seeds.
* Every policy plays the SAME pre-generated deals (common random numbers), so
  differences come from decisions, not luck of the cards.
"""

from __future__ import annotations

import math
import random

from .env import CARD_VALUES, BlackjackEnv

EVAL_SEED = 2026
CARDS_PER_DEAL = 24  # more than any hand can use; extra draws fall back to the RNG


def make_eval_deals(n_games, seed=EVAL_SEED):
    rng = random.Random(seed)
    return [[rng.choice(CARD_VALUES) for _ in range(CARDS_PER_DEAL)] for _ in range(n_games)]


def evaluate_policy(policy, deals, seed=EVAL_SEED):
    """Play every deal once with `policy`. Returns a summary dict."""
    env = BlackjackEnv(seed=seed + 1)        # only used if a deal runs out of cards
    policy_rng = random.Random(seed + 2)     # only used by the random baseline
    rewards = []
    for deal in deals:
        state, reward, done = env.reset(preset_cards=deal)
        while not done:
            state, reward, done = env.step(policy(state, policy_rng))
        rewards.append(reward)
    return summarize(rewards)


def summarize(rewards):
    n = len(rewards)
    mean = sum(rewards) / n
    var = sum((r - mean) ** 2 for r in rewards) / (n - 1)
    half_width = 1.96 * math.sqrt(var / n)
    return {
        "games": n,
        "avg_reward": mean,
        "ci95_low": mean - half_width,
        "ci95_high": mean + half_width,
        "win_rate": rewards.count(1) / n,
        "loss_rate": rewards.count(-1) / n,
        "push_rate": rewards.count(0) / n,
    }
