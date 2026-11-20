"""Experiment pipeline shared by the notebook and scripts/run_experiment.py."""

from __future__ import annotations

import pandas as pd

from .agent import policy_from_q, train_q_learning
from .baselines import random_policy, stand_on_17_policy
from .env import all_states
from .evaluate import evaluate_policy, make_eval_deals
from .optimal import optimal_policy, optimal_q_table

# Default hyperparameters (record these with any reported result).
CONFIG = {
    "episodes": 500_000,
    "alpha": 0.01,
    "gamma": 1.0,
    "eps_start": 1.0,
    "eps_end": 0.05,          # reached halfway through training, then held
    "train_seed": 471,
    "checkpoint_every": 10_000,
    "checkpoint_eval_games": 20_000,
    "checkpoint_eval_seed": 1000,
    "final_eval_games": 100_000,
    "final_eval_seed": 2026,
}


def train_with_learning_curve(cfg=CONFIG, trace=None):
    """Train, pausing every checkpoint to evaluate a frozen greedy copy of the
    Q-table on a fixed set of evaluation deals (never used for training)."""
    curve_deals = make_eval_deals(cfg["checkpoint_eval_games"], seed=cfg["checkpoint_eval_seed"])
    # Episode 0: untrained Q-table (all zeros; greedy ties go to "Stand").
    rows = [{"episode": 0, **evaluate_policy(policy_from_q({}), curve_deals,
                                               seed=cfg["checkpoint_eval_seed"])}]

    def on_checkpoint(episode, Q):
        result = evaluate_policy(policy_from_q(Q), curve_deals, seed=cfg["checkpoint_eval_seed"])
        rows.append({"episode": episode, **result})

    Q, train_rewards = train_q_learning(
        episodes=cfg["episodes"], alpha=cfg["alpha"], gamma=cfg["gamma"],
        eps_start=cfg["eps_start"], eps_end=cfg["eps_end"], seed=cfg["train_seed"],
        checkpoint_every=cfg["checkpoint_every"], on_checkpoint=on_checkpoint, trace=trace,
    )
    # Reference policies scored on the SAME checkpoint deals, so the curve and
    # the reference lines are directly comparable.
    refs = {name: evaluate_policy(p, curve_deals, seed=cfg["checkpoint_eval_seed"])
            for name, p in standard_policies(Q).items() if not name.startswith("Q-learning")}
    curve_refs = pd.DataFrame(refs).T
    return Q, pd.DataFrame(rows), train_rewards, curve_refs


def final_comparison(policies: dict, cfg=CONFIG):
    """Evaluate every policy on the same unseen deals. Returns a DataFrame."""
    deals = make_eval_deals(cfg["final_eval_games"], seed=cfg["final_eval_seed"])
    rows = [{"policy": name, **evaluate_policy(p, deals, seed=cfg["final_eval_seed"])}
            for name, p in policies.items()]
    return pd.DataFrame(rows).set_index("policy")


def standard_policies(Q):
    return {
        "Q-learning (trained, ε=0)": policy_from_q(Q),
        "Optimal (exact DP reference)": optimal_policy(),
        "Stand on 17+": stand_on_17_policy,
        "Random": random_policy,
    }


def policy_disagreements(policy):
    """States where `policy` differs from the exact optimal action, with the
    size of the optimal Q-value gap (how much the choice actually matters)."""
    q = optimal_q_table()
    opt = optimal_policy()
    rows = []
    for s in all_states():
        if policy(s) != opt(s):
            rows.append({
                "player_total": s[0], "dealer_up": "A" if s[1] == 1 else s[1],
                "usable_ace": s[2], "learned": policy(s), "optimal": opt(s),
                "Q*_stand": round(q[(s, "Stand")], 4), "Q*_hit": round(q[(s, "Hit")], 4),
                "cost_of_mistake": round(abs(q[(s, "Stand")] - q[(s, "Hit")]), 4),
            })
    return pd.DataFrame(rows)
