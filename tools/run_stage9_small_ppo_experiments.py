#!/usr/bin/env python
import argparse
import json
import os
import subprocess
import sys
import time


GROUPS = {
    "PPO_original": {
        "use_reward_shaping": False,
        "config": None,
    },
    "PPO_physical": {
        "use_reward_shaping": True,
        "config": "lag_extensions/reward_shaping/configs/physical_1v1.yaml",
    },
    "PPO_surrogate_v1_direct": {
        "use_reward_shaping": True,
        "config": "lag_extensions/reward_shaping/configs/surrogate_v1_hybrid_1v1.yaml",
    },
    "PPO_surrogate_v1_then_fusion": {
        "use_reward_shaping": True,
        "config": "lag_extensions/reward_shaping/configs/surrogate_v1_then_fusion_1v1.yaml",
    },
    "PPO_surrogate_v1_no_potential": {
        "use_reward_shaping": True,
        "config": "lag_extensions/reward_shaping/configs/surrogate_v1_no_potential_1v1.yaml",
    },
    "PPO_cached_vlm_fallback_diagnostic": {
        "use_reward_shaping": True,
        "config": "lag_extensions/reward_shaping/configs/qwen2_5_vl_hybrid_1v1.yaml",
    },
    "PPO_surrogate_v2_direct": {
        "use_reward_shaping": True,
        "config": "lag_extensions/reward_shaping/configs/surrogate_v2_hybrid_1v1.yaml",
    },
    "PPO_surrogate_v2_then_fusion": {
        "use_reward_shaping": True,
        "config": "lag_extensions/reward_shaping/configs/surrogate_v2_then_fusion_1v1.yaml",
    },
}


def main():
    parser = argparse.ArgumentParser(description="Run stage9 small PPO diagnostic experiments.")
    parser.add_argument("--groups", nargs="+", default=list(GROUPS.keys()), choices=list(GROUPS.keys()))
    parser.add_argument("--num-env-steps", type=int, default=50000)
    parser.add_argument("--buffer-size", type=int, default=128)
    parser.add_argument("--n-rollout-threads", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", default="scripts/results/stage9_small_ppo")
    parser.add_argument("--dry-run", action="store_true", default=False)
    args = parser.parse_args()
    summary = run(args)
    print(json.dumps(summary, indent=2, sort_keys=True))


def run(args):
    os.makedirs(args.output_dir, exist_ok=True)
    results = []
    for group in args.groups:
        spec = GROUPS[group]
        experiment_name = "stage9_{}_seed{}".format(group, args.seed)
        reward_log_dir = os.path.join(args.output_dir, group, "reward_shaping")
        stdout_path = os.path.join(args.output_dir, group, "stdout.log")
        stderr_path = os.path.join(args.output_dir, group, "stderr.log")
        os.makedirs(os.path.dirname(stdout_path), exist_ok=True)
        command = [
            sys.executable,
            "scripts/train/train_jsbsim.py",
            "--env-name", "SingleCombat",
            "--algorithm-name", "ppo",
            "--scenario-name", "1v1/NoWeapon/Selfplay",
            "--experiment-name", experiment_name,
            "--seed", str(args.seed),
            "--n-rollout-threads", str(args.n_rollout_threads),
            "--n-training-threads", "1",
            "--buffer-size", str(args.buffer_size),
            "--num-env-steps", str(args.num_env_steps),
            "--log-interval", "1",
            "--save-interval", "1",
        ]
        if spec["use_reward_shaping"]:
            command.extend([
                "--use-reward-shaping",
                "--reward-shaping-config", spec["config"],
                "--reward-shaping-log-dir", reward_log_dir,
            ])
        start = time.time()
        record = {
            "group": group,
            "command": command,
            "stdout": os.path.abspath(stdout_path),
            "stderr": os.path.abspath(stderr_path),
            "reward_shaping_log_dir": os.path.abspath(reward_log_dir) if spec["use_reward_shaping"] else None,
        }
        if args.dry_run:
            record["returncode"] = None
            record["duration_sec"] = 0.0
        else:
            with open(stdout_path, "w", encoding="utf-8") as out, open(stderr_path, "w", encoding="utf-8") as err:
                proc = subprocess.run(command, cwd=os.getcwd(), stdout=out, stderr=err, text=True)
            record["returncode"] = proc.returncode
            record["duration_sec"] = time.time() - start
        results.append(record)
    summary = {
        "output_dir": os.path.abspath(args.output_dir),
        "num_env_steps": args.num_env_steps,
        "buffer_size": args.buffer_size,
        "seed": args.seed,
        "results": results,
    }
    with open(os.path.join(args.output_dir, "run_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    return summary


if __name__ == "__main__":
    main()
