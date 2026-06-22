#!/usr/bin/env python
import argparse
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from lag_extensions.reward_shaping.vlm_label_dataset import generate_aircombat_semantic_dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-name", type=str, default="SingleCombat")
    parser.add_argument("--scenario-name", type=str, default="1v1/NoWeapon/Selfplay")
    parser.add_argument("--num-samples", type=int, default=100)
    parser.add_argument("--sample-interval", type=int, default=10)
    parser.add_argument("--output-dir", type=str, default="scripts/results/aircombat_semantic_dataset")
    parser.add_argument("--config", type=str, default="lag_extensions/reward_shaping/configs/qwen2_5_vl_1v1.yaml")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--agent-side", type=str, default="ego", choices=["ego", "all"])
    args = parser.parse_args()

    if args.env_name != "SingleCombat":
        raise NotImplementedError("Only SingleCombat 1v1 dataset generation is implemented.")

    summary = generate_aircombat_semantic_dataset(
        scenario_name=args.scenario_name,
        output_dir=args.output_dir,
        num_samples=args.num_samples,
        sample_interval=args.sample_interval,
        config_path=args.config,
        seed=args.seed,
        agent_side=args.agent_side,
    )
    print(summary)


if __name__ == "__main__":
    main()
