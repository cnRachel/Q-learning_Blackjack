"""Correctness checks for the simulator rules (README "Simulator rules")."""

import random

import pytest

from blackjack_rl.env import BlackjackEnv, dealer_play, draw_card, hand_value
from blackjack_rl.optimal import dealer_final_distribution


def test_hand_value_counts_aces_correctly():
    """Expected: an Ace counts 11 unless that busts the hand, then 1."""
    assert hand_value([1, 6]) == (17, True)        # soft 17
    assert hand_value([1, 6, 10]) == (17, False)   # Ace forced down to 1
    assert hand_value([1, 1]) == (12, True)        # only one Ace can be 11
    assert hand_value([10, 6, 9]) == (25, False)   # bust


def test_dealer_stands_on_soft_17():
    """Expected: dealer holding A+6 takes no card (draw() must never be called)."""
    def no_draw():
        raise AssertionError("dealer drew on soft 17")
    assert dealer_play([1, 6], no_draw) == [1, 6]


def test_dealer_hits_16():
    """Expected: dealer holding 10+6 draws until reaching 17 or more."""
    cards = iter([5])
    assert dealer_play([10, 6], lambda: next(cards)) == [10, 6, 5]


@pytest.mark.parametrize("preset, expected", [
    ([1, 10, 9, 7], 1),    # player natural only
    ([9, 7, 1, 10], -1),   # dealer natural only
    ([1, 10, 10, 1], 0),   # both naturals
])
def test_naturals_resolved_before_player_acts(preset, expected):
    env = BlackjackEnv(seed=0)
    _, reward, done = env.reset(preset_cards=preset)
    assert done and reward == expected


def test_player_bust_is_immediate_loss():
    """Player 10+6 hits a 10 -> 26: reward -1, dealer never plays."""
    env = BlackjackEnv(seed=0)
    env.reset(preset_cards=[10, 6, 9, 7, 10])
    _, reward, done = env.step("Hit")
    assert done and reward == -1
    assert env.dealer == [9, 7]


def test_simulated_dealer_matches_exact_probabilities():
    """Expected: Monte Carlo dealer bust rate (200k hands per up-card) agrees with
    the exact DP probability to within 0.5 percentage points. This checks the
    simulator and the DP model against each other."""
    rng = random.Random(7)
    draw = lambda: draw_card(rng)  # noqa: E731
    for up in (1, 6, 10):
        busts = n = 0
        while n < 200_000:
            hand = [up, draw()]
            if hand_value(hand)[0] == 21:            # dealer natural: player never acts
                continue
            busts += hand_value(dealer_play(hand, draw))[0] > 21
            n += 1
        assert abs(busts / n - dealer_final_distribution(up)["bust"]) < 0.005
