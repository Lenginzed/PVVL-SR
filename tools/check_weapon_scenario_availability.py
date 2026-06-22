#!/usr/bin/env python
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from envs.JSBSim.envs.singlecombat_env import SingleCombatEnv


def main():
    parser = argparse.ArgumentParser(description="Check weapon-enabled 1v1 JSBSim scenario availability.")
    parser.add_argument("--configs-root", default="envs/JSBSim/configs/1v1")
    parser.add_argument("--output-dir", default="scripts/results/stage12_weapon_scenario_check")
    parser.add_argument("--smoke", action="store_true", default=False)
    parser.add_argument("--steps", type=int, default=10)
    args = parser.parse_args()
    report = check(args)
    print(json.dumps(report, indent=2, sort_keys=True))


def check(args):
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    scenarios = []
    root = Path(args.configs_root)
    for yaml_path in sorted(root.glob("*/*.yaml")):
        scenario_name = str(yaml_path.with_suffix("").relative_to(root.parent)).replace("\\", "/")
        text = yaml_path.read_text(encoding="utf-8", errors="ignore")
        lower_name = scenario_name.lower()
        weapon_like = any(token in lower_name for token in ["missile", "shootmissile", "dodgemissile"])
        weapon_like = weapon_like or any(token in text.lower() for token in ["missile", "launch", "lock", "weapon"])
        if not weapon_like:
            continue
        item = {
            "scenario_name": scenario_name,
            "config_path": str(yaml_path.resolve()),
            "contains_missile_text": "missile" in text.lower(),
            "contains_launch_text": "launch" in text.lower(),
            "contains_lock_text": "lock" in text.lower(),
            "smoke": None,
        }
        if args.smoke:
            item["smoke"] = smoke_scenario(scenario_name, int(args.steps))
        scenarios.append(item)
    report = {
        "configs_root": str(root.resolve()),
        "scenario_count": len(scenarios),
        "scenarios": scenarios,
        "note": "Static availability check only unless --smoke is used. Do not treat this as Stage12 weapon training.",
    }
    (out_dir / "weapon_scenario_availability.json").write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    write_markdown(out_dir / "weapon_scenario_availability.md", report)
    return report


def smoke_scenario(scenario_name, steps):
    out = {
        "scenario_name": scenario_name,
        "env_constructed": False,
        "reset_ok": False,
        "step_ok": False,
        "info_keys": [],
        "error": None,
    }
    try:
        env = SingleCombatEnv(scenario_name)
        out["env_constructed"] = True
        env.seed(0)
        env.action_space.seed(0)
        env.reset()
        out["reset_ok"] = True
        info_keys = set()
        for _ in range(int(steps)):
            actions = np.array([env.action_space.sample() for _ in range(env.num_agents)])
            _, _, dones, info = env.step(actions)
            if isinstance(info, dict):
                info_keys.update(info.keys())
            if bool(np.all(dones)):
                break
        env.close()
        out["step_ok"] = True
        out["info_keys"] = sorted(str(key) for key in info_keys)
    except Exception as exc:
        out["error"] = "{}: {}".format(type(exc).__name__, exc)
    return out


def write_markdown(path, report):
    lines = [
        "# Weapon Scenario Availability",
        "",
        "| scenario | missile text | launch text | lock text | smoke reset | smoke step | error |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for item in report["scenarios"]:
        smoke = item.get("smoke") or {}
        lines.append("| {} | {} | {} | {} | {} | {} | {} |".format(
            item["scenario_name"],
            item["contains_missile_text"],
            item["contains_launch_text"],
            item["contains_lock_text"],
            smoke.get("reset_ok", ""),
            smoke.get("step_ok", ""),
            smoke.get("error", ""),
        ))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
