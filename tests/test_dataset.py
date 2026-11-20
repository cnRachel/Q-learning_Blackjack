"""Checks for converting Kaggle hands into Q-learning experiences.
Uses only the 6,000-row sample committed in data/, so no Kaggle login is needed."""

from blackjack_rl.dataset import build_episodes, hand_to_episode, load_hands


def _row(player, dealer, total, dealer_total, result):
    row = {f"card{i+1}": c for i, c in enumerate(player + [0] * (5 - len(player)))}
    row.update({f"dealcard{i+1}": c for i, c in enumerate(dealer + [0] * (5 - len(dealer)))})
    row.update(sumofcards=total, sumofdeal=dealer_total, winloss=result)
    return row


def test_recorded_hand_becomes_hit_then_stand():
    """Dataset row: player 10,4,(0),5 = 19 vs dealer 10,8 -> Win.
    Expected: (14 vs 10) Hit -> (19 vs 10) Stand with reward +1.
    The empty card3 slot must be skipped."""
    row = _row([10, 4, 0, 5], [10, 8], 19, 18, "Win")
    assert hand_to_episode(row) == [
        ((14, 10, False), "Hit", 0, (19, 10, False), False),
        ((19, 10, False), "Stand", 1, None, True),
    ]


def test_recorded_bust_ends_with_minus_one():
    """Player 6,A,8,10: soft 17 -> hard 15 -> 25 bust. Ace recorded as 11 then 1."""
    row = _row([6, 1, 8, 10], [10, 8], 25, 18, "Loss")
    ep = hand_to_episode(row)
    assert [step[1] for step in ep] == ["Hit", "Hit"]
    assert ep[0][0] == (17, 10, True)
    assert ep[-1][2] == -1 and ep[-1][4] is True


def test_mis_scored_dealer_hand_is_skipped():
    """Dealer A,A,4,6 is really 12, but the dataset records 22. Expected: skipped."""
    row = _row([10, 9], [11, 1, 4, 6], 19, 22, "Win")
    assert hand_to_episode(row) is None


def test_sample_parses_into_valid_episodes():
    df = load_hands("sample")
    episodes, skipped = build_episodes(df)
    assert len(episodes) + skipped == len(df) == 6_000
    for ep in episodes:
        assert ep[-1][4] is True                         # every episode terminates
        assert all(step[2] in (-1, 0, 1) for step in ep)
        assert all(not step[4] for step in ep[:-1])      # only the last step is terminal
