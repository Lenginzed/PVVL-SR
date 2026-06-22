#!/usr/bin/env python
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from lag_extensions.reward_shaping.config import load_reward_shaping_config, snapshot_config


GROUPS = {
    "PPO_original": {"use_reward_shaping": False, "config": None},
    "PPO_physical": {"use_reward_shaping": True, "config": "lag_extensions/reward_shaping/configs/physical_1v1.yaml"},
    "PPO_surrogate_v1_then_fusion": {
        "use_reward_shaping": True,
        "config": "lag_extensions/reward_shaping/configs/surrogate_v1_then_fusion_1v1.yaml",
    },
    "PPO_surrogate_v2_direct": {
        "use_reward_shaping": True,
        "config": "lag_extensions/reward_shaping/configs/surrogate_v2_direct_1v1.yaml",
    },
    "PPO_surrogate_v2_then_fusion": {
        "use_reward_shaping": True,
        "config": "lag_extensions/reward_shaping/configs/surrogate_v2_then_fusion_1v1.yaml",
    },
    "PPO_surrogate_v2_no_potential": {
        "use_reward_shaping": True,
        "config": "lag_extensions/reward_shaping/configs/surrogate_v2_no_potential_1v1.yaml",
    },
    "PPO_surrogate_v2_direct_neutral_stronger": {
        "use_reward_shaping": True,
        "config": "lag_extensions/reward_shaping/configs/surrogate_v2_direct_neutral_stronger_1v1.yaml",
    },
    "PPO_surrogate_v2_direct_beta_low": {
        "use_reward_shaping": True,
        "config": "lag_extensions/reward_shaping/configs/surrogate_v2_direct_beta_low_1v1.yaml",
    },
}


def main():
    parser = argparse.ArgumentParser(description="Run stage10 multiseed PPO pre-experiments.")
    parser.add_argument("--groups", nargs="+", default=list(GROUPS.keys()), choices=list(GROUPS.keys()))
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--num-env-steps", type=int, default=100000)
    parser.add_argument("--buffer-size", type=int, default=128)
    parser.add_argument("--n-rollout-threads", type=int, default=1)
    parser.add_argument("--n-training-threads", type=int, default=1)
    parser.add_argument("--checkpoint-every-steps", type=int, default=50000)
    parser.add_argument("--checkpoint-eval-episodes", type=int, default=50)
    parser.add_argument("--final-eval-episodes", type=int, default=100)
    parser.add_argument("--eval-seed", type=int, default=1101)
    parser.add_argument("--output-dir", default="scripts/results/stage10_multiseed_ppo")
    parser.add_argument("--env-name", default="SingleCombat")
    parser.add_argument("--scenario-name", default="1v1/NoWeapon/Selfplay")
    parser.add_argument("--dry-run", action="store_true", default=False)
    parser.add_argument("--resume", action="store_true", default=False)
    parser.add_argument("--skip-checkpoint-eval", action="store_true", default=False)
    args = parser.parse_args()
    summary = run(args)
    print(json.dumps(summary, indent=2, sort_keys=True))


def run(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for group in args.groups:
        for seed in args.seeds:
            trial = output_dir / group / "seed{}".format(seed)
            trial.mkdir(parents=True, exist_ok=True)
            record_path = trial / "run_record.json"
            final_eval_path = trial / "final_eval" / "summary.json"
            if args.resume and record_path.exists() and final_eval_path.exists():
                record = json.loads(record_path.read_text(encoding="utf-8"))
                record["skipped_existing"] = True
                results.append(record)
                continue
            record = run_one(args, group, seed, trial)
            results.append(record)
            record_path.write_text(json.dumps(record, indent=2, sort_keys=True), encoding="utf-8")
            write_summary(output_dir, args, results)
    summary = write_summary(output_dir, args, results)
    return summary


def run_one(args, group, seed, trial):
    spec = GROUPS[group]
    experiment_name = "stage10_{}_seed{}".format(group, seed)
    reward_log_dir = trial / "reward_shaping"
    stdout_path = trial / "stdout.txt"
    stderr_path = trial / "stderr.txt"
    command_path = trial / "command.txt"
    save_interval = max(1, int(args.checkpoint_every_steps) // max(1, int(args.buffer_size) * int(args.n_rollout_threads)))
    command = [
        sys.executable,
        "scripts/train/train_jsbsim.py",
        "--env-name", args.env_name,
        "--algorithm-name", "ppo",
        "--scenario-name", args.scenario_name,
        "--experiment-name", experiment_name,
        "--seed", str(seed),
        "--n-rollout-threads", str(args.n_rollout_threads),
        "--n-training-threads", str(args.n_training_threads),
        "--buffer-size", str(args.buffer_size),
        "--num-env-steps", str(args.num_env_steps),
        "--log-interval", "1",
        "--save-interval", str(save_interval),
    ]
    if spec["use_reward_shaping"]:
        command.extend([
            "--use-reward-shaping",
            "--reward-shaping-config", spec["config"],
            "--reward-shaping-log-dir", str(reward_log_dir),
        ])
        shutil.copyfile(spec["config"], trial / "reward_shaping_config.yaml")
        snapshot_config(load_reward_shaping_config(spec["config"]), str(trial / "reward_shaping_config_resolved.json"))
    (trial / "group_spec.json").write_text(json.dumps(spec, indent=2, sort_keys=True), encoding="utf-8")
    command_path.write_text(" ".join(command), encoding="utf-8")
    start = time.time()
    returncode = None
    if args.dry_run:
        duration = 0.0
    else:
        with stdout_path.open("w", encoding="utf-8") as out, stderr_path.open("w", encoding="utf-8") as err:
            proc = subprocess.run(command, cwd=os.getcwd(), stdout=out, stderr=err, text=True)
        returncode = proc.returncode
        duration = time.time() - start

    run_dir = find_latest_run_dir(args.env_name, args.scenario_name, "ppo", experiment_name)
    actor_latest = run_dir / "actor_latest.pt" if run_dir else None
    success = bool(returncode == 0 and actor_latest and actor_latest.exists())
    record = {
        "group": group,
        "seed": int(seed),
        "experiment_name": experiment_name,
        "command": command,
        "command_txt": str(command_path.resolve()),
        "stdout": str(stdout_path.resolve()),
        "stderr": str(stderr_path.resolve()),
        "reward_shaping_log_dir": str(reward_log_dir.resolve()) if spec["use_reward_shaping"] else None,
        "run_dir": str(run_dir.resolve()) if run_dir else None,
        "returncode": returncode,
        "success": success,
        "duration_sec": duration,
        "num_env_steps": int(args.num_env_steps),
        "buffer_size": int(args.buffer_size),
        "save_interval_updates": int(save_interval),
    }
    if success and not args.dry_run:
        record["checkpoint_eval"] = evaluate_checkpoints(args, group, seed, trial, run_dir)
        record["final_eval"] = evaluate_final(args, group, seed, trial, run_dir)
    return record


def find_latest_run_dir(env_name, scenario_name, algorithm_name, experiment_name):
    root = Path("scripts/results") / env_name / scenario_name / algorithm_name / experiment_name
    if not root.exists():
        return None
    runs = [path for path in root.iterdir() if path.is_dir() and path.name.startswith("run")]
    if not runs:
        return None
    return max(runs, key=lambda path: path.stat().st_mtime)


def evaluate_checkpoints(args, group, seed, trial, run_dir):
    rows = []
    out_jsonl = trial / "eval_by_checkpoint.jsonl"
    actor_files = sorted(run_dir.glob("actor_*.pt"), key=_checkpoint_sort_key)
    actor_files = [path for path in actor_files if path.name != "actor_latest.pt" and _checkpoint_sort_key(path) > 0]
    if args.skip_checkpoint_eval:
        return {"skipped": True, "path": str(out_jsonl.resolve())}
    with out_jsonl.open("w", encoding="utf-8") as f:
        for actor in actor_files:
            update = _checkpoint_sort_key(actor)
            critic = run_dir / "critic_{}.pt".format(update)
            eval_dir = trial / "checkpoint_eval" / "update_{}".format(update)
            command = [
                sys.executable,
                "tools/evaluate_stage10_policy.py",
                "--model-dir", str(run_dir),
                "--actor-file", str(actor),
                "--critic-file", str(critic),
                "--episodes", str(args.checkpoint_eval_episodes),
                "--seed", str(args.eval_seed),
                "--output-dir", str(eval_dir),
            ]
            eval_dir.mkdir(parents=True, exist_ok=True)
            with (eval_dir / "stdout.txt").open("w", encoding="utf-8") as out, (eval_dir / "stderr.txt").open("w", encoding="utf-8") as err:
                subprocess.run(command, cwd=os.getcwd(), check=True, stdout=out, stderr=err, text=True)
            summary_path = eval_dir / "summary.json"
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            row = {
                "group": group,
                "seed": int(seed),
                "update": int(update),
                "approx_env_steps": int((update + 1) * args.buffer_size * args.n_rollout_threads),
                "summary": summary,
            }
            f.write(json.dumps(row, sort_keys=True) + "\n")
            rows.append(row)
    return {"count": len(rows), "path": str(out_jsonl.resolve())}


def evaluate_final(args, group, seed, trial, run_dir):
    eval_dir = trial / "final_eval"
    command = [
        sys.executable,
        "tools/evaluate_stage10_policy.py",
        "--model-dir", str(run_dir),
        "--episodes", str(args.final_eval_episodes),
        "--seed", str(args.eval_seed),
        "--output-dir", str(eval_dir),
    ]
    eval_dir.mkdir(parents=True, exist_ok=True)
    with (eval_dir / "stdout.txt").open("w", encoding="utf-8") as out, (eval_dir / "stderr.txt").open("w", encoding="utf-8") as err:
        subprocess.run(command, cwd=os.getcwd(), check=True, stdout=out, stderr=err, text=True)
    return {
        "command": command,
        "summary": str((eval_dir / "summary.json").resolve()),
        "episodes": str((eval_dir / "episodes.jsonl").resolve()),
    }


def _checkpoint_sort_key(path):
    try:
        return int(path.stem.split("_")[-1])
    except Exception:
        return -1


def write_summary(output_dir, args, results):
    summary = {
        "output_dir": str(output_dir.resolve()),
        "groups": args.groups,
        "seeds": args.seeds,
        "num_env_steps": int(args.num_env_steps),
        "checkpoint_every_steps": int(args.checkpoint_every_steps),
        "checkpoint_eval_episodes": int(args.checkpoint_eval_episodes),
        "final_eval_episodes": int(args.final_eval_episodes),
        "results": results,
    }
    (output_dir / "run_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary


if __name__ == "__main__":
    main()
