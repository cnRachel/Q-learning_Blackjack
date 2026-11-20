"""Simplified Blackjack simulator.

Rules (fixed so every result in this repo is reproducible):

1. Infinite deck: every card is drawn independently with replacement.
   Ranks 2-9 count face value; 10/J/Q/K count 10 (so P(10-value) = 4/13);
   an Ace counts 11 unless that would bust the hand, then 1.
2. Player and dealer each get two cards; one dealer card (the up-card) is visible.
3. Naturals are resolved before the player acts (dealer "peeks"):
     player natural only  -> +1 (paid like a normal win, NOT 3:2)
     dealer natural only  -> -1
     both naturals        ->  0
4. Player actions: "Hit" or "Stand" only (no double, split, insurance, surrender).
5. Player total > 21 -> immediate loss (-1); the dealer does not play.
6. Dealer hits until reaching 17 or more and stands on ALL 17s (including soft 17).
7. Otherwise the higher total wins (+1 / -1); equal totals push (0).

State seen by the agent: (player_total, dealer_up_card, usable_ace)
  player_total   4..21
  dealer_up_card 1..10   (Ace = 1)
  usable_ace     True if the hand holds an Ace currently counted as 11
"""

from __future__ import annotations

import random

ACTIONS = ["Stand", "Hit"]

# One draw from an infinite deck: 1 = Ace, 10 = any ten-value card.
CARD_VALUES = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10]


def draw_card(rng: random.Random) -> int:
    return rng.choice(CARD_VALUES)


def hand_value(cards: list[int]) -> tuple[int, bool]:
    """Return (best total, usable_ace) for a list of card values (Ace = 1)."""
    total = sum(cards)
    if 1 in cards and total + 10 <= 21:
        return total + 10, True
    return total, False


def is_natural(cards: list[int]) -> bool:
    return len(cards) == 2 and hand_value(cards)[0] == 21


def dealer_play(cards: list[int], draw) -> list[int]:
    """Dealer hits until total >= 17; stands on all 17s (soft 17 included)."""
    cards = list(cards)
    while hand_value(cards)[0] < 17:
        cards.append(draw())
    return cards


def compare(player_total: int, dealer_total: int) -> int:
    if player_total > 21:
        return -1
    if dealer_total > 21 or player_total > dealer_total:
        return 1
    if player_total < dealer_total:
        return -1
    return 0


class CardSource:
    """Supplies cards. Optionally starts from a preset list (used so that
    different policies are evaluated on exactly the same deals)."""

    def __init__(self, rng: random.Random, preset: list[int] | None = None):
        self.rng = rng
        self.preset = list(preset) if preset else []
        self.i = 0

    def __call__(self) -> int:
        if self.i < len(self.preset):
            card = self.preset[self.i]
            self.i += 1
            return card
        return draw_card(self.rng)


class BlackjackEnv:
    """Episode interface:

        state, reward, done = env.reset()
        while not done:
            state, reward, done = env.step(action)

    `reset` can already return done=True when a natural ends the hand.
    """

    def __init__(self, seed: int | None = None):
        self.rng = random.Random(seed)
        self.player: list[int] = []
        self.dealer: list[int] = []
        self.done = True

    def _state(self):
        total, usable = hand_value(self.player)
        return (total, self.dealer[0], usable)

    def reset(self, preset_cards: list[int] | None = None):
        """Deal a new hand. preset_cards order: player, player, dealer-up,
        dealer-hole, then every later draw in sequence."""
        self.draw = CardSource(self.rng, preset_cards)
        self.player = [self.draw(), self.draw()]
        self.dealer = [self.draw(), self.draw()]
        self.done = False

        p_nat, d_nat = is_natural(self.player), is_natural(self.dealer)
        if p_nat or d_nat:
            self.done = True
            reward = 0 if (p_nat and d_nat) else (1 if p_nat else -1)
            return self._state(), reward, True
        return self._state(), 0, False

    def step(self, action: str):
        if self.done:
            raise RuntimeError("Episode is over; call reset().")
        if action == "Hit":
            self.player.append(self.draw())
            total, _ = hand_value(self.player)
            if total > 21:
                self.done = True
                return self._state(), -1, True
            return self._state(), 0, False
        if action == "Stand":
            self.dealer = dealer_play(self.dealer, self.draw)
            self.done = True
            reward = compare(hand_value(self.player)[0], hand_value(self.dealer)[0])
            return self._state(), reward, True
        raise ValueError(f"Unknown action {action!r}")


def all_states():
    """Every decision state the agent can face: hard 4-21 and soft 12-21,
    each against 10 dealer up-cards -> 180 + 100 = 280 states."""
    states = []
    for usable in (False, True):
        totals = range(12, 22) if usable else range(4, 22)
        for total in totals:
            for up in range(1, 11):
                states.append((total, up, usable))
    return states
