"""Fixed comparison strategies (instructor Required Revision #1).

Every policy has the signature  policy(state, rng) -> "Hit" | "Stand".
"""

from __future__ import annotations


def random_policy(state, rng):
    """Hit or stand with 50% probability each. Lower bound: no learning."""
    return rng.choice(["Hit", "Stand"])


def stand_on_17_policy(state, rng=None):
    """Hit below 17, stand on 17 or more (mimics the dealer). A common
    human rule of thumb; ignores the dealer's up-card."""
    total, _, _ = state
    return "Stand" if total >= 17 else "Hit"
