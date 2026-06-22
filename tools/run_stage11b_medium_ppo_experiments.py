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
    "PPO_surrogate_v2_then_fusion": {
        "use_reward_shaping": True,
        "config": "lag_extensions/reward_shaping/configs/surrogate_v2_then_fusion_1v1.yaml",
    },
    "PPO_surrogate_v2_direct_beta_low": {
        "use_reward_shaping": True,
        "config": "lag_extensions/reward_shaping/configs/surrogate_v2_direct_beta_low_1v1.yaml",
    },
    "PPO_original_with_situation_curriculum": {
        "use_reward_shaping": False,
        "config": "lag_extensions/reward_shaping/configs/original_mixed_curriculum_1v1.yaml",
    },
    "PPO_surrogate_v2_then_fusion_with_situation_curriculum": {
        "use_reward_shaping": True,
        "config": "lag_extensions/reward_shaping/configs/surrogate_v2_then_fusion_mixed_curriculum_1v1.yaml",
    },
    "PPO_surrogate_v2_direct_beta_low_with_situation_curriculum": {
        "use_reward_shaping": True,
        "config": "lag_extensions/reward_shaping/configs/surrogate_v2_direct_beta_low_mixed_curriculum_1v1.yaml",
    },
    "PPO_surrogate_v2_then_fusion_curriculum_attack_weight_boost": {
        "use_reward_shaping": True,
        "config": "lag_extensions/reward_shaping/configs/surrogate_v2_then_fusion_mixed_curriculum_attack_weight_boost_1v1.yaml",
    },
    "PPO_surrogate_v2_then_fusion_curriculum_defense_penalty_relaxed": {
        "use_reward_shaping": True,
        "config": "lag_extensions/reward_shaping/configs/surrogate_v2_then_fusion_mixed_curriculum_defense_penalty_relaxed_1v1.yaml",
    },
}

EVAL_CONFIGS = {
    "default_eval": "lag_extensions/reward_shaping/configs/surrogate_v2_then_fusion_1v1.yaml",
    "offensive_eval": "lag_extensions/reward_shaping/configs/offensive_curriculum_original_1v1.yaml",
    "mixed_eval": "lag_extensions/reward_shaping/configs/mixed_curriculum_eval_1v1.yaml",
}


def main():
    parser = argparse.ArgumentParser(description="Run stage11B medium PPO comparisons.")
    parser.add_argument("--groups", nargs="+", default=[
        "PPO_original",
        "PPO_physical",
        "PPO_surrogate_v2_then_fusion",
        "PPO_surrogate_v2_direct_beta_low",
        "PPO_original_with_situation_curriculum",
        "PPO_surrogate_v2_then_fusion_with_situation_curriculum",
        "PPO_surrogate_v2_direct_beta_low_with_situation_curriculum",
    ], choices=list(GROUPS.keys()))
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--num-env-steps", type=int, default=200000)
    parser.add_argument("--buffer-size", type=int, default=128)
    parser.add_argument("--n-rollout-threads", type=int, default=1)
    parser.add_argument("--n-training-threads", type=int, default=1)
    parser.add_argument("--checkpoint-every-steps", type=int, default=50000)
    parser.add_argument("--checkpoint-eval-episodes", type=int, default=50)
    parser.add_argument("--final-eval-episodes", type=int, default=100)
    parser.add_argument("--eval-seed", type=int, default=2201)
    parser.add_argument("--output-dir", default="scripts/results/stage11b_medium_ppo")
    parser.add_argument("--env-name", default="SingleCombat")
    parser.add_argument("--scenario-name", default="1v1/NoWeapon/Selfplay")
    parser.add_argument("--resume", action="store_true", default=False)
    parser.add_argument("--dry-run", action="store_true", default=False)
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
            if args.resume and complete_record(record_path):
                record = json.loads(record_path.read_text(encoding="utf-8"))
                record["skipped_existing"] = True
                results.append(record)
                continue
            record = run_one(args, group, seed, trial)
            results.append(record)
            record_path.write_text(json.dumps(record, indent=2, sort_keys=True), encoding="utf-8")
            write_summary(output_dir, args, results)
    return write_summary(output_dir, args, results)


def complete_record(record_path):
    if not record_path.exists():
        return False
    try:
        record = json.loads(record_path.read_text(encoding="utf-8"))
    except Exception:
        return False
    if not record.get("success"):
        return False
    final = record.get("final_eval", {})
    return all(Path(final.get(name, {}).get("summary", "")).exists() for name in EVAL_CONFIGS)


def run_one(args, group, seed, trial):
    spec = GROUPS[group]
    experiment_name = "stage11b_{}_seed{}".format(group, seed)
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
    if spec.get("config"):
        command.extend(["--reward-shaping-config", spec["config"]])
        shutil.copyfile(spec["config"], trial / "reward_shaping_config.yaml")
        snapshot_config(load_reward_shaping_config(spec["config"]), str(trial / "reward_shaping_config_resolved.json"))
    if spec["use_reward_shaping"]:
        command.extend(["--use-reward-shaping", "--reward-shaping-log-dir", str(reward_log_dir)])
    (trial / "group_spec.json").write_text(json.dumps(spec, indent=2, sort_keys=True), encoding="utf-8")
    command_path.write_text(" ".join(command), encoding="utf-8")

    start = time.time()
    returncode = None
    existing_run_dir = find_latest_run_dir(args.env_name, args.scenario_name, "ppo", experiment_name)
    existing_actor = existing_run_dir / "actor_latest.pt" if existing_run_dir else None
    if args.dry_run:
        duration = 0.0
    elif args.resume and existing_actor and existing_actor.exists():
        returncode = 0
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
        record["final_eval"] = evaluate_all_final(args, group, seed, trial, run_dir)
    return record


def find_latest_run_dir(env_name, scenario_name, algorithm_name, experiment_name):
    root = Path("scripts/results") / env_name / scenario_name / algorithm_name / experiment_name
    if not root.exists():
        return None
    runs = [path for path in root.iterdir() if path.is_dir() and path.name.startswith("run")]
    return max(runs, key=lambda path: path.stat().st_mtime) if runs else None


def evaluate_checkpoints(args, group, seed, trial, run_dir):
    out_jsonl = trial / "eval_by_checkpoint.jsonl"
    if args.skip_checkpoint_eval:
        return {"skipped": True, "path": str(out_jsonl.resolve())}
    actor_files = sorted(run_dir.glob("actor_*.pt"), key=_checkpoint_sort_key)
    actor_files = [path for path in actor_files if path.name != "actor_latest.pt" and _checkpoint_sort_key(path) > 0]
    rows = []
    with out_jsonl.open("w", encoding="utf-8") as f:
        for actor in actor_files:
            update = _checkpoint_sort_key(actor)
            critic = run_dir / "critic_{}.pt".format(update)
            eval_dir = trial / "checkpoint_eval" / "default_update_{}".format(update)
            run_eval(run_dir, eval_dir, args.checkpoint_eval_episodes, args.eval_seed, EVAL_CONFIGS["default_eval"], actor, critic)
            summary = json.loads((eval_dir / "summary.json").read_text(encoding="utf-8"))
            row = {
                "group": group,
                "seed": int(seed),
                "eval_type": "default_eval",
                "update": int(update),
                "approx_env_steps": int((update + 1) * args.buffer_size * args.n_rollout_threads),
                "summary": summary,
            }
            f.write(json.dumps(row, sort_keys=True) + "\n")
            rows.append(row)
    return {"count": len(rows), "path": str(out_jsonl.resolve())}


def evaluate_all_final(args, group, seed, trial, run_dir):
    results = {}
    for eval_type, config in EVAL_CONFIGS.items():
        eval_dir = trial / eval_type
        run_eval(run_dir, eval_dir, args.final_eval_episodes, args.eval_seed, config)
        results[eval_type] = {
            "config": config,
            "summary": str((eval_dir / "summary.json").resolve()),
            "episodes": str((eval_dir / "episodes.jsonl").resolve()),
        }
    return results


def run_eval(run_dir, eval_dir, episodes, seed, config, actor=None, critic=None):
    summary_path = Path(eval_dir) / "summary.json"
    if summary_path.exists():
        return
    command = [
        sys.executable,
        "tools/evaluate_stage10_policy.py",
        "--model-dir", str(run_dir),
        "--episodes", str(episodes),
        "--seed", str(seed),
        "--config", config,
        "--output-dir", str(eval_dir),
    ]
    if actor is not None:
        command.extend(["--actor-file", str(actor)])
    if critic is not None and Path(critic).exists():
        command.extend(["--critic-file", str(critic)])
    eval_dir.mkdir(parents=True, exist_ok=True)
    with (eval_dir / "stdout.txt").open("w", encoding="utf-8") as out, (eval_dir / "stderr.txt").open("w", encoding="utf-8") as err:
        subprocess.run(command, cwd=os.getcwd(), check=True, stdout=out, stderr=err, text=True)
    (eval_dir / "command.txt").write_text(" ".join(command), encoding="utf-8")


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
        "eval_configs": EVAL_CONFIGS,
        "results": results,
    }
    (output_dir / "run_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary


if __name__ == "__main__":
    main()
