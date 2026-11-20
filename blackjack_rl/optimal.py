"""Exact optimal Hit/Stand policy for the simulator's rules, by dynamic
programming (model-based, like the value iteration in HW4 Parts I-II).

Because the deck is infinite, the transition probabilities are known exactly,
so Q*(s, Stand) and Q*(s, Hit) can be computed directly. This serves as the
"best possible" reference that the model-free Q-learning agent is measured
against. Q-learning never sees these numbers.
"""

from __future__ import annotations

from functools import lru_cache

from .env import ACTIONS, all_states

# Probability of each card value from an infinite deck.
P_CARD = {c: (4 / 13 if c == 10 else 1 / 13) for c in range(1, 11)}


def _value(hard_sum, has_ace):
    if has_ace and hard_sum + 10 <= 21:
        return hard_sum + 10
    return hard_sum


@lru_cache(maxsize=None)
def _dealer_from(hard_sum, has_ace):
    """Distribution of the dealer's final total ('bust' or 17..21)."""
    v = _value(hard_sum, has_ace)
    if v > 21:
        return {"bust": 1.0}
    if v >= 17:  # stands on all 17s, including soft 17
        return {v: 1.0}
    out = {}
    for c, p in P_CARD.items():
        for k, q in _dealer_from(hard_sum + c, has_ace or c == 1).items():
            out[k] = out.get(k, 0.0) + p * q
    return out


@lru_cache(maxsize=None)
def dealer_final_distribution(up):
    """Dealer final-total distribution given the up-card, conditioned on the
    dealer NOT having a natural (the player only acts in that case)."""
    hole_probs = dict(P_CARD)
    if up == 1:
        hole_probs.pop(10)
    elif up == 10:
        hole_probs.pop(1)
    z = sum(hole_probs.values())
    out = {}
    for hole, p in hole_probs.items():
        for k, q in _dealer_from(up + hole, up == 1 or hole == 1).items():
            out[k] = out.get(k, 0.0) + (p / z) * q
    return out


def _stand_ev(total, up):
    ev = 0.0
    for k, q in dealer_final_distribution(up).items():
        if k == "bust" or total > k:
            ev += q
        elif total < k:
            ev -= q
    return ev


@lru_cache(maxsize=None)
def _q_values(hard_sum, has_ace, up):
    """Return (Q*(Stand), Q*(Hit)) for a player hand."""
    stand = _stand_ev(_value(hard_sum, has_ace), up)
    hit = 0.0
    for c, p in P_CARD.items():
        new_sum, new_ace = hard_sum + c, has_ace or c == 1
        if new_sum > 21:
            hit -= p
        else:
            hit += p * max(_q_values(new_sum, new_ace, up))
    return stand, hit


def optimal_q_table():
    """Q*[(state, action)] for every decision state."""
    table = {}
    for total, up, usable in all_states():
        hard_sum, has_ace = (total - 10, True) if usable else (total, False)
        stand, hit = _q_values(hard_sum, has_ace, up)
        table[((total, up, usable), "Stand")] = stand
        table[((total, up, usable), "Hit")] = hit
    return table


def optimal_policy():
    q = optimal_q_table()

    def policy(state, rng=None):
        return max(ACTIONS, key=lambda a: q[(state, a)])

    return policy
