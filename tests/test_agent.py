"""Correctness checks for the Q-learning agent and the evaluation protocol."""

from collections import defaultdict

import pytest

from blackjack_rl.agent import (apply_transition, policy_from_q, q_learning_update,
                                train_q_learning)
from blackjack_rl.baselines import random_policy, stand_on_17_policy
from blackjack_rl.env import all_states
from blackjack_rl.evaluate import evaluate_policy, make_eval_deals
from blackjack_rl.optimal import optimal_policy


def test_q_update_matches_hw4_part_vi_hand_calculation():
    """HW4 6.2: Q=4, r=2, max Q'=8, alpha=0.5, gamma=0.75.
    By hand: target = 2 + 0.75*8 = 8, TD error = 8 - 4 = 4, new Q = 4 + 0.5*4 = 6."""
    assert q_learning_update(4, 2, 8, alpha=0.5, gamma=0.75) == (8, 4, 6)


def test_terminal_transition_ignores_next_state():
    """At a terminal step the target is just r (no future value).
    Q=0, r=+1, alpha=0.5 -> new Q = 0.5, even if other Q-values are large."""
    Q = defaultdict(float)
    Q[((20, 10, False), "Stand")] = 0.0
    Q[((21, 10, False), "Stand")] = 99.0      # must NOT leak into the update
    apply_transition(Q, (20, 10, False), "Stand", 1, (21, 10, False), True, 0.5, 1.0)
    assert Q[((20, 10, False), "Stand")] == pytest.approx(0.5)


def test_training_is_reproducible():
    """Same seed -> identical Q-table."""
    q1, _ = train_q_learning(episodes=5_000, seed=3)
    q2, _ = train_q_learning(episodes=5_000, seed=3)
    assert dict(q1) == dict(q2)


def test_evaluation_does_not_change_q_table():
    """Evaluation must be exploration-free and must not learn."""
    Q, _ = train_q_learning(episodes=5_000, seed=3)
    before = dict(Q)
    evaluate_policy(policy_from_q(Q), make_eval_deals(2_000))
    assert dict(Q) == before


@pytest.fixture(scope="module")
def trained():
    Q, _ = train_q_learning(episodes=200_000, seed=471)
    return policy_from_q(Q)


def test_learned_policy_on_obvious_states(trained):
    """Expected: always stand on hard 20-21, always hit on hard 8-11
    (hitting 11 or less can never bust).

    Hard 4-7 are left out on purpose: they only arise from deals like 2+2 or
    2+3 (hard 4 is 1 deal in 169), so with a fixed alpha = 0.01 their Q-values
    are still close to the initial 0 after training. See README "Known limitations"."""
    for total, up, usable in all_states():
        if not usable and total >= 20:
            assert trained((total, up, usable)) == "Stand"
        if not usable and 8 <= total <= 11:
            assert trained((total, up, usable)) == "Hit"


def test_trained_agent_beats_baselines_on_separate_games(trained):
    """Expected ordering on 50,000 unseen deals, no exploration:
    random < stand-on-17 < Q-learning, and Q-learning within 0.01 of optimal."""
    deals = make_eval_deals(50_000)
    rand = evaluate_policy(random_policy, deals)
    s17 = evaluate_policy(stand_on_17_policy, deals)
    agent = evaluate_policy(trained, deals)
    best = evaluate_policy(optimal_policy(), deals)
    assert agent["ci95_low"] > rand["ci95_high"]
    assert agent["avg_reward"] > s17["avg_reward"] + 0.01
    assert abs(agent["avg_reward"] - best["avg_reward"]) < 0.01
