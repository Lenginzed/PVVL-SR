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
    "PPO_original_with_situation_curriculum": {
        "use_reward_shaping": False,
        "config": "lag_extensions/reward_shaping/configs/original_mixed_curriculum_1v1.yaml",
    },
    "PPO_physical": {
        "use_reward_shaping": True,
        "config": "lag_extensions/reward_shaping/configs/physical_1v1.yaml",
    },
    "PPO_surrogate_v2_then_fusion_with_situation_curriculum": {
        "use_reward_shaping": True,
        "config": "lag_extensions/reward_shaping/configs/surrogate_v2_then_fusion_mixed_curriculum_1v1.yaml",
    },
}


EVAL_CONFIGS = {
    "default_eval": "lag_extensions/reward_shaping/configs/surrogate_v2_then_fusion_1v1.yaml",
    "offensive_eval": "lag_extensions/reward_shaping/configs/offensive_curriculum_original_1v1.yaml",
    "mixed_eval": "lag_extensions/reward_shaping/configs/mixed_curriculum_eval_1v1.yaml",
    "unseen_eval": "lag_extensions/reward_shaping/configs/unseen_curriculum_eval_1v1.yaml",
}


def main():
    parser = argparse.ArgumentParser(description="Continue Stage12 300k checkpoints to Stage13 500k confirmation.")
    parser.add_argument("--groups", nargs="+", default=[
        "PPO_original_with_situation_curriculum",
        "PPO_physical",
        "PPO_surrogate_v2_then_fusion_with_situation_curriculum",
    ], choices=list(GROUPS.keys()))
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--stage12-dir", default="scripts/results/stage12_formal_ppo")
    parser.add_argument("--target-total-steps", type=int, default=500000)
    parser.add_argument("--continuation-steps", type=int, default=200000)
    parser.add_argument("--buffer-size", type=int, default=128)
    parser.add_argument("--checkpoint-every-steps", type=int, default=50000)
    parser.add_argument("--checkpoint-eval-episodes", type=int, default=30)
    parser.add_argument("--checkpoint-eval-types", nargs="+", default=["mixed_eval"])
    parser.add_argument("--final-eval-episodes", type=int, default=200)
    parser.add_argument("--eval-seed", type=int, default=4201)
    parser.add_argument("--output-dir", default="scripts/results/stage13_500k_confirmation")
    parser.add_argument("--env-name", default="SingleCombat")
    parser.add_argument("--scenario-name", default="1v1/NoWeapon/Selfplay")
    parser.add_argument("--experiment-prefix", default="stage13_500k")
    parser.add_argument("--n-rollout-threads", type=int, default=1)
    parser.add_argument("--n-training-threads", type=int, default=1)
    parser.add_argument("--resume", action="store_true", default=False)
    parser.add_argument("--dry-run", action="store_true", default=False)
    args = parser.parse_args()
    print(json.dumps(run(args), indent=2, sort_keys=True))


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
            record_path.write_text(json.dumps(record, indent=2, sort_keys=True), encoding="utf-8")
            results.append(record)
            write_summary(output_dir, args, results)
    return write_summary(output_dir, args, results)


def complete_record(path):
    if not path.exists():
        return False
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return False
    if not record.get("success"):
        return False
    return all(Path(record.get("final_eval", {}).get(name, {}).get("summary", "")).exists() for name in EVAL_CONFIGS)


def run_one(args, group, seed, trial):
    spec = GROUPS[group]
    source = load_stage12_record(Path(args.stage12_dir), group, seed)
    source_run_dir = Path(source["run_dir"])
    experiment_name = "{}_{}_seed{}".format(args.experiment_prefix, group, seed)
    reward_log_dir = trial / "reward_shaping"
    command_path = trial / "command.txt"
    stdout_path = trial / "stdout.txt"
    stderr_path = trial / "stderr.txt"
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
        "--num-env-steps", str(args.continuation_steps),
        "--model-dir", str(source_run_dir),
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
    (trial / "stage12_source_record.json").write_text(json.dumps(source, indent=2, sort_keys=True), encoding="utf-8")
    command_path.write_text(" ".join(command), encoding="utf-8")

    existing_run_dir = find_latest_run_dir(args.env_name, args.scenario_name, "ppo", experiment_name)
    existing_actor = existing_run_dir / "actor_latest.pt" if existing_run_dir else None
    start = time.time()
    if args.dry_run:
        returncode = 0
        duration = 0.0
    elif args.resume and existing_actor and existing_actor.exists() and training_complete(existing_run_dir, args.continuation_steps):
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
        "stage12_source_run_dir": str(source_run_dir.resolve()),
        "stage12_source_steps": int(source.get("num_env_steps", 300000)),
        "continuation_steps": int(args.continuation_steps),
        "target_total_steps": int(args.target_total_steps),
        "resume_mode": "actor_critic_weights_only_optimizer_reset",
        "buffer_size": int(args.buffer_size),
        "save_interval_updates": int(save_interval),
    }
    if success and not args.dry_run:
        record["checkpoint_eval"] = evaluate_checkpoints(args, group, seed, trial, run_dir)
        record["final_eval"] = evaluate_all_final(args, trial, run_dir)
    return record


def load_stage12_record(stage12_dir, group, seed):
    path = stage12_dir / group / "seed{}".format(seed) / "run_record.json"
    if not path.exists():
        raise FileNotFoundError("Missing Stage12 source record: {}".format(path))
    record = json.loads(path.read_text(encoding="utf-8"))
    if not record.get("success"):
        raise RuntimeError("Stage12 source record is not successful: {}".format(path))
    if not Path(record.get("run_dir", "")).exists():
        raise FileNotFoundError("Missing Stage12 run_dir: {}".format(record.get("run_dir")))
    return record


def find_latest_run_dir(env_name, scenario_name, algorithm_name, experiment_name):
    root = Path("scripts/results") / env_name / scenario_name / algorithm_name / experiment_name
    if not root.exists():
        return None
    runs = [path for path in root.iterdir() if path.is_dir() and path.name.startswith("run")]
    return max(runs, key=lambda path: path.stat().st_mtime) if runs else None


def training_complete(run_dir, required_steps):
    if not run_dir:
        return False
    path = Path(run_dir) / "train_metrics.jsonl"
    if not path.exists():
        return False
    last = None
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                last = line
    if not last:
        return False
    try:
        record = json.loads(last)
    except Exception:
        return False
    return int(record.get("total_num_steps", 0)) >= max(0, int(required_steps) - 256)


def evaluate_checkpoints(args, group, seed, trial, run_dir):
    out_jsonl = trial / "eval_by_checkpoint.jsonl"
    eval_types = [name for name in args.checkpoint_eval_types if name in EVAL_CONFIGS]
    actor_files = sorted(run_dir.glob("actor_*.pt"), key=_checkpoint_sort_key)
    actor_files = [path for path in actor_files if path.name != "actor_latest.pt" and _checkpoint_sort_key(path) > 0]
    rows = []
    with out_jsonl.open("w", encoding="utf-8") as f:
        for actor in actor_files:
            update = _checkpoint_sort_key(actor)
            critic = run_dir / "critic_{}.pt".format(update)
            for eval_type in eval_types:
                eval_dir = trial / "checkpoint_eval" / "{}_update_{}".format(eval_type, update)
                run_eval(run_dir, eval_dir, args.checkpoint_eval_episodes, args.eval_seed, EVAL_CONFIGS[eval_type], actor, critic)
                summary = json.loads((eval_dir / "summary.json").read_text(encoding="utf-8"))
                approx_total = 300000 + int((update + 1) * args.buffer_size * args.n_rollout_threads)
                row = {
                    "group": group,
                    "seed": int(seed),
                    "eval_type": eval_type,
                    "update": int(update),
                    "approx_env_steps": int(approx_total),
                    "summary": summary,
                }
                f.write(json.dumps(row, sort_keys=True) + "\n")
                rows.append(row)
    return {"count": len(rows), "path": str(out_jsonl.resolve()), "eval_types": eval_types}


def evaluate_all_final(args, trial, run_dir):
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
        "target_total_steps": int(args.target_total_steps),
        "continuation_steps": int(args.continuation_steps),
        "checkpoint_every_steps": int(args.checkpoint_every_steps),
        "checkpoint_eval_episodes": int(args.checkpoint_eval_episodes),
        "checkpoint_eval_types": args.checkpoint_eval_types,
        "final_eval_episodes": int(args.final_eval_episodes),
        "experiment_prefix": args.experiment_prefix,
        "eval_configs": EVAL_CONFIGS,
        "results": results,
    }
    (output_dir / "run_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary


if __name__ == "__main__":
    main()
