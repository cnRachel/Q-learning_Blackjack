"""Reproduce every result in one command (no notebook needed):

    python scripts/run_experiment.py            # full run, ~1 minute
    python scripts/run_experiment.py --quick    # 50k episodes, for a fast check

Writes CSV tables and PNG figures to results/ (or results/quick/ with --quick).
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")

from blackjack_rl import plots  # noqa: E402
from blackjack_rl.experiment import (CONFIG, final_comparison, policy_disagreements,  # noqa: E402
                                     standard_policies, train_with_learning_curve)
from blackjack_rl.agent import policy_from_q  # noqa: E402
from blackjack_rl.optimal import optimal_policy  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="50k episodes instead of 500k")
    args = parser.parse_args()

    cfg = dict(CONFIG)
    if args.quick:
        cfg.update(episodes=50_000, checkpoint_every=5_000)

    out = ROOT / "results" / "quick" if args.quick else ROOT / "results"
    out.mkdir(parents=True, exist_ok=True)
    print("Config:", json.dumps(cfg, indent=2))

    Q, curve, train_rewards, curve_refs = train_with_learning_curve(cfg)
    results = final_comparison(standard_policies(Q), cfg)
    disagreements = policy_disagreements(policy_from_q(Q))

    curve.to_csv(out / "learning_curve.csv", index=False)
    results.to_csv(out / "final_comparison.csv")
    disagreements.to_csv(out / "policy_disagreements.csv", index=False)
    (out / "config.json").write_text(json.dumps(cfg, indent=2))

    plots.learning_curve(curve, curve_refs, train_rewards).savefig(out / "learning_curve.png", dpi=150)
    plots.final_bars(results).savefig(out / "final_comparison.png", dpi=150)
    plots.policy_maps(policy_from_q(Q), optimal_policy(), "Q-learning policy").savefig(
        out / "policy_map.png", dpi=150)

    print(results[["avg_reward", "ci95_low", "ci95_high", "win_rate", "loss_rate", "push_rate"]]
          .round(4).to_string())
    print(f"\nStates differing from optimal: {len(disagreements)} of 280")
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()
