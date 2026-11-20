r"""Tabular Q-learning, written to mirror CS471 HW4 Parts VI-VII.

    Q(s,a) <- Q(s,a) + alpha * [ r + gamma * max_a' Q(s',a') - Q(s,a) ]
                                 \_______ target _______/
                                 \__________ TD error ___________/
"""

from __future__ import annotations

import random
from collections import defaultdict

from .env import ACTIONS, BlackjackEnv


def q_learning_update(q_current, reward, max_next_q, alpha, gamma):
    """One Q-learning update. Returns (target, td_error, updated_q)."""
    target = reward + gamma * max_next_q
    td_error = target - q_current
    updated_q = q_current + alpha * td_error
    return target, td_error, updated_q


def epsilon_greedy_action(state, Q, epsilon, rng: random.Random):
    """Explore with probability epsilon, otherwise pick the best action
    (ties broken randomly, as in the class notebook)."""
    if rng.random() < epsilon:
        return rng.choice(ACTIONS)
    values = [Q[(state, a)] for a in ACTIONS]
    best = max(values)
    return rng.choice([a for a, v in zip(ACTIONS, values) if v == best])


def greedy_action(state, Q):
    """Exploitation only (epsilon = 0). Ties go to 'Stand' so evaluation is
    deterministic."""
    return max(ACTIONS, key=lambda a: Q[(state, a)])


def apply_transition(Q, state, action, reward, next_state, done, alpha, gamma):
    """Apply the update for one observed experience (s, a, r, s')."""
    max_next_q = 0.0 if done else max(Q[(next_state, a)] for a in ACTIONS)
    target, td_error, new_q = q_learning_update(
        Q[(state, action)], reward, max_next_q, alpha, gamma
    )
    Q[(state, action)] = new_q
    return td_error


def linear_epsilon(episode, episodes, eps_start=1.0, eps_end=0.05, decay_fraction=0.5):
    """Linear decay from eps_start to eps_end over the first decay_fraction of training."""
    decay_episodes = max(1, int(episodes * decay_fraction))
    frac = min(1.0, episode / decay_episodes)
    return eps_start + frac * (eps_end - eps_start)


def train_q_learning(episodes=500_000, alpha=0.01, gamma=1.0,
                     eps_start=1.0, eps_end=0.05, seed=471,
                     checkpoint_every=None, on_checkpoint=None, trace=None):
    """Train a Q-table by playing simulated hands.

    on_checkpoint(episode, Q) is called every `checkpoint_every` episodes so the
    caller can run a separate, exploration-free evaluation.
    trace: optional list; the first transitions are appended to it for display.
    """
    rng = random.Random(seed)
    env = BlackjackEnv(seed=seed + 1)
    Q = defaultdict(float)
    episode_rewards = []

    for ep in range(episodes):
        epsilon = linear_epsilon(ep, episodes, eps_start, eps_end)
        state, reward, done = env.reset()
        while not done:
            action = epsilon_greedy_action(state, Q, epsilon, rng)
            next_state, reward, done = env.step(action)
            old_q = Q[(state, action)]
            td = apply_transition(Q, state, action, reward, next_state, done, alpha, gamma)
            if trace is not None and len(trace) < 25:
                trace.append({
                    "episode": ep, "state": state, "action": action,
                    "reward": reward, "next_state": None if done else next_state,
                    "Q_before": round(old_q, 4), "TD_error": round(td, 4),
                    "Q_after": round(Q[(state, action)], 4),
                })
            state = next_state
        episode_rewards.append(reward)

        if checkpoint_every and on_checkpoint and (ep + 1) % checkpoint_every == 0:
            on_checkpoint(ep + 1, Q)

    return Q, episode_rewards


def policy_from_q(Q):
    """Freeze a Q-table into a greedy policy function (no exploration)."""
    frozen = dict(Q)

    def policy(state, rng=None):
        return max(ACTIONS, key=lambda a: frozen.get((state, a), 0.0))

    return policy
