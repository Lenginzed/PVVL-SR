#!/usr/bin/env python
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from envs.JSBSim.envs.singlecombat_env import SingleCombatEnv
from lag_extensions.reward_shaping.config import load_reward_shaping_config, snapshot_config
from tools.summarize_stage10_results import summarize_reward_log, summarize_train_metrics


GROUPS = {
    "PPO_original": {"use_reward_shaping": False, "config": None},
    "PPO_physical": {"use_reward_shaping": True, "config": "lag_extensions/reward_shaping/configs/physical_1v1.yaml"},
    "PPO_surrogate_v2_then_fusion": {
        "use_reward_shaping": True,
        "config": "lag_extensions/reward_shaping/configs/surrogate_v2_then_fusion_1v1.yaml",
    },
}


def main():
    parser = argparse.ArgumentParser(description="Stage13 DodgeMissile preliminary validation.")
    parser.add_argument("--output-dir", default="scripts/results/stage13_dodgemissile_preliminary")
    parser.add_argument("--scenario", default=None)
    parser.add_argument("--smoke-steps", type=int, default=10000)
    parser.add_argument("--train-steps", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval-episodes", type=int, default=50)
    parser.add_argument("--resume", action="store_true", default=False)
    parser.add_argument("--skip-train", action="store_true", default=False)
    args = parser.parse_args()
    print(json.dumps(run(args), indent=2, sort_keys=True))


def run(args):
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    scenarios = list_dodge_scenarios()
    primary = args.scenario or choose_primary(scenarios)
    availability = [{"scenario": name, "smoke": env_smoke(name)} for name in scenarios]
    (out_dir / "dodgemissile_scenario_availability.json").write_text(json.dumps(availability, indent=2, sort_keys=True), encoding="utf-8")
    results = {
        "primary_scenario": primary,
        "availability": availability,
        "smoke_runs": [],
        "preliminary_runs": [],
        "shootmissile_todo": "ShootMissile action shape is not handled in Stage13; keep as TODO.",
    }
    for group in GROUPS:
        smoke = run_train_eval(args, group, primary, args.smoke_steps, out_dir / "smoke" / group, smoke=True)
        results["smoke_runs"].append(smoke)
    if all(item.get("success") for item in results["smoke_runs"]) and not args.skip_train:
        for group in GROUPS:
            results["preliminary_runs"].append(run_train_eval(args, group, primary, args.train_steps, out_dir / "preliminary_50k" / group, smoke=False))
    write_report(out_dir / "weapon_preliminary_report.md", results)
    (out_dir / "weapon_preliminary_summary.json").write_text(json.dumps(results, indent=2, sort_keys=True), encoding="utf-8")
    return results


def list_dodge_scenarios():
    root = Path("envs/JSBSim/configs/1v1/DodgeMissile")
    return ["1v1/DodgeMissile/{}".format(path.stem) for path in sorted(root.glob("*.yaml"))]


def choose_primary(scenarios):
    for preferred in ["1v1/DodgeMissile/Selfplay", "1v1/DodgeMissile/vsBaseline"]:
        if preferred in scenarios and env_smoke(preferred).get("step_ok"):
            return preferred
    for scenario in scenarios:
        if env_smoke(scenario).get("step_ok"):
            return scenario
    raise RuntimeError("No stable DodgeMissile scenario found.")


def env_smoke(scenario):
    out = {"scenario": scenario, "reset_ok": False, "step_ok": False, "info_keys": [], "error": None}
    try:
        env = SingleCombatEnv(scenario)
        env.seed(0)
        env.action_space.seed(0)
        env.reset()
        out["reset_ok"] = True
        keys = set()
        for _ in range(10):
            actions = np.array([env.action_space.sample() for _ in range(env.num_agents)])
            _, _, dones, info = env.step(actions)
            if isinstance(info, dict):
                keys.update(str(k) for k in info.keys())
            if bool(np.all(dones)):
                break
        env.close()
        out["step_ok"] = True
        out["info_keys"] = sorted(keys)
    except Exception as exc:
        out["error"] = "{}: {}".format(type(exc).__name__, exc)
    return out


def run_train_eval(args, group, scenario, steps, trial, smoke):
    spec = GROUPS[group]
    trial.mkdir(parents=True, exist_ok=True)
    record_path = trial / "run_record.json"
    if args.resume and record_path.exists():
        record = json.loads(record_path.read_text(encoding="utf-8"))
        if record.get("success"):
            return record
    exp = "stage13_dodgemissile_{}_{}_seed{}".format("smoke" if smoke else "prelim50k", group, args.seed)
    command = [
        sys.executable,
        "scripts/train/train_jsbsim.py",
        "--env-name", "SingleCombat",
        "--algorithm-name", "ppo",
        "--scenario-name", scenario,
        "--experiment-name", exp,
        "--seed", str(args.seed),
        "--n-rollout-threads", "1",
        "--n-training-threads", "1",
        "--buffer-size", "128",
        "--num-env-steps", str(steps),
        "--log-interval", "1",
        "--save-interval", "390",
    ]
    if spec.get("config"):
        command.extend(["--reward-shaping-config", spec["config"]])
        shutil.copyfile(spec["config"], trial / "reward_shaping_config.yaml")
        snapshot_config(load_reward_shaping_config(spec["config"]), str(trial / "reward_shaping_config_resolved.json"))
    if spec["use_reward_shaping"]:
        command.extend(["--use-reward-shaping", "--reward-shaping-log-dir", str(trial / "reward_shaping")])
    (trial / "command.txt").write_text(" ".join(command), encoding="utf-8")
    start = time.time()
    with (trial / "stdout.txt").open("w", encoding="utf-8") as out, (trial / "stderr.txt").open("w", encoding="utf-8") as err:
        proc = subprocess.run(command, cwd=os.getcwd(), stdout=out, stderr=err, text=True)
    run_dir = find_latest_run_dir("SingleCombat", scenario, "ppo", exp)
    success = bool(proc.returncode == 0 and run_dir and (run_dir / "actor_latest.pt").exists())
    record = {
        "group": group,
        "scenario": scenario,
        "steps": int(steps),
        "smoke": bool(smoke),
        "command": command,
        "run_dir": str(run_dir.resolve()) if run_dir else None,
        "returncode": proc.returncode,
        "success": success,
        "duration_sec": time.time() - start,
        "reward_summary": summarize_reward_log(trial / "reward_shaping" / "train_env0_steps.jsonl"),
        "train_summary": summarize_train_metrics((run_dir / "train_metrics.jsonl") if run_dir else trial / "missing.jsonl"),
        "metrics_available": {
            "missile_warning": "not available in info smoke keys",
            "lock": "not available in info smoke keys",
            "hit_or_avoided": "not available in info smoke keys",
        },
    }
    if success and not smoke:
        eval_dir = trial / "eval"
        config = spec.get("config") or "lag_extensions/reward_shaping/configs/surrogate_v2_then_fusion_1v1.yaml"
        run_eval(run_dir, eval_dir, args.eval_episodes, args.seed + 9000, scenario, config)
        record["eval_summary"] = str((eval_dir / "summary.json").resolve())
    record_path.write_text(json.dumps(record, indent=2, sort_keys=True), encoding="utf-8")
    return record


def find_latest_run_dir(env_name, scenario_name, algorithm_name, experiment_name):
    root = Path("scripts/results") / env_name / scenario_name / algorithm_name / experiment_name
    if not root.exists():
        return None
    runs = [path for path in root.iterdir() if path.is_dir() and path.name.startswith("run")]
    return max(runs, key=lambda path: path.stat().st_mtime) if runs else None


def run_eval(run_dir, eval_dir, episodes, seed, scenario, config):
    command = [
        sys.executable,
        "tools/evaluate_stage10_policy.py",
        "--scenario-name", scenario,
        "--model-dir", str(run_dir),
        "--episodes", str(episodes),
        "--seed", str(seed),
        "--config", config,
        "--output-dir", str(eval_dir),
    ]
    eval_dir.mkdir(parents=True, exist_ok=True)
    with (eval_dir / "stdout.txt").open("w", encoding="utf-8") as out, (eval_dir / "stderr.txt").open("w", encoding="utf-8") as err:
        subprocess.run(command, cwd=os.getcwd(), check=True, stdout=out, stderr=err, text=True)
    (eval_dir / "command.txt").write_text(" ".join(command), encoding="utf-8")


def write_report(path, results):
    lines = [
        "# DodgeMissile Preliminary Validation",
        "",
        "This is a preliminary validation, not a formal weapon experiment.",
        "",
        "- Goal: interface compatibility and rough transfer signal.",
        "- NoWeapon results must not be interpreted as missile-combat performance.",
        "- ShootMissile action-shape handling is left as TODO.",
        "",
        "Primary scenario: `{}`".format(results["primary_scenario"]),
        "",
        "## Smoke",
        "",
        "| group | success | reward log count | nan/inf | reward spikes |",
        "|---|---:|---:|---:|---:|",
    ]
    for item in results["smoke_runs"]:
        rs = item.get("reward_summary", {})
        lines.append("| {} | {} | {} | {} | {} |".format(
            item["group"],
            item.get("success"),
            rs.get("reward_log_count", 0),
            rs.get("reward_nan_inf_count", "n/a"),
            rs.get("reward_spike_abs_gt_10", "n/a"),
        ))
    lines.extend(["", "## 50k Preliminary", ""])
    lines.append("| group | success | win | loss | draw | return | episode length | survival | missile threat ratio | defensive escape | eval summary | missile/lock fields |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|")
    for item in results.get("preliminary_runs", []):
        eval_summary = load_eval_summary(item.get("eval_summary"))
        lines.append("| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
            item["group"],
            item.get("success"),
            fmt_metric(eval_summary.get("win_rate")),
            fmt_metric(eval_summary.get("loss_rate")),
            fmt_metric(eval_summary.get("draw_rate")),
            fmt_metric(eval_summary.get("return_ego_mean")),
            fmt_metric(eval_summary.get("episode_length_mean")),
            fmt_metric(eval_summary.get("survival_time_mean")),
            fmt_metric(eval_summary.get("enemy_missile_threat_zone_time_ratio_mean")),
            fmt_metric(eval_summary.get("defensive_escape_time_ratio_mean")),
            item.get("eval_summary", ""),
            "not available",
        ))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_eval_summary(path):
    if not path:
        return {}
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return {}


def fmt_metric(value):
    if value is None:
        return "n/a"
    try:
        return "{:.4f}".format(float(value))
    except Exception:
        return str(value)


if __name__ == "__main__":
    main()
