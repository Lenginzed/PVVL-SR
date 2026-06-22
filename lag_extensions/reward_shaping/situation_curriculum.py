import math
from typing import Dict, List, Optional

import numpy as np


class SituationCurriculumWrapper:
    """Optional reset adapter for offensive/neutral/defensive initial situations.

    The original LAG SingleCombatEnv shuffles the two configured aircraft init
    states on reset. This wrapper bypasses that shuffle only when
    situation_curriculum.enabled=true, and leaves the underlying environment
    unchanged otherwise. It is intended for curriculum experiments and smoke
    tests, not as a replacement for the original scenario definition.
    """

    def __init__(self, env, config: Dict, run_name: str = "env0"):
        self.env = env
        self.config = dict(config or {})
        self.enabled = bool(self.config.get("enabled", False))
        self.run_name = run_name
        self.reset_count = 0
        self.total_env_steps = 0
        self.current_situation_type = "default"
        self.last_stage_name = None

    def __getattr__(self, name):
        return getattr(self.env, name)

    def reset(self):
        if not self.enabled:
            self.current_situation_type = "default"
            return self.env.reset()
        situation = self._select_situation()
        self.reset_count += 1
        if situation == "random":
            self.current_situation_type = "random"
            return self.env.reset()
        states = build_init_states(situation, self.config, getattr(self.env, "np_random", None))
        self.current_situation_type = situation
        return self._reset_no_shuffle(states)

    def step(self, action):
        result = self.env.step(action)
        self.total_env_steps += 1
        return result

    def close(self):
        return self.env.close()

    def render(self, *args, **kwargs):
        return self.env.render(*args, **kwargs)

    def seed(self, seed=None):
        return self.env.seed(seed)

    def _select_situation(self) -> str:
        mode = self.config.get("mode", "fixed")
        if mode == "fixed":
            return self.config.get("stage", self.config.get("situation_level", "offensive_advantage"))
        stages = self.config.get("stages", [])
        if mode == "by_reset_count" and stages:
            current = stages[0]
            for stage in stages:
                if self.reset_count >= int(stage.get("start_reset", 0)):
                    current = stage
            return current.get("situation", current.get("name", "random"))
        if mode in ("mixed", "mixed_by_env_steps") and stages:
            current = stages[0]
            for stage in stages:
                if self.total_env_steps >= int(stage.get("start_step", 0)):
                    current = stage
            self.last_stage_name = current.get("name", "mixed")
            return sample_situation(current.get("probabilities", {}), getattr(self.env, "np_random", None))
        return self.config.get("stage", "random")

    def _reset_no_shuffle(self, states: List[Dict]):
        self.env.current_step = 0
        for sim, state in zip(self.env.agents.values(), states):
            sim.reload(state.copy())
        self.env._tempsims.clear()
        self.env.task.reset(self.env)
        return self.env._pack(self.env.get_obs())


def build_init_states(situation: str, config: Optional[Dict] = None, rng=None) -> List[Dict]:
    config = dict(config or {})
    rng = rng or np.random
    base_lon, base_lat, _ = config.get("battle_field_center", [120.0, 60.0, 0.0])
    altitude_ft = float(config.get("altitude_ft", 20000.0))
    distance_range = config.get("distance_m", [2800.0, 3600.0])
    lateral_range = config.get("lateral_m", [-300.0, 300.0])
    speed_ego_fps = float(config.get("ego_speed_fps", 880.0))
    speed_enemy_fps = float(config.get("enemy_speed_fps", 760.0))
    distance = _uniform(rng, distance_range)
    lateral = _uniform(rng, lateral_range)

    if situation == "offensive_advantage":
        ego_ne = (0.0, 0.0)
        enemy_ne = (distance, lateral)
        ego_heading = math.degrees(math.atan2(lateral, distance))
        enemy_heading = 0.0
        ego_speed, enemy_speed = speed_ego_fps, speed_enemy_fps
    elif situation == "neutral_merge":
        ego_ne = (0.0, 0.0)
        enemy_ne = (distance, lateral)
        ego_heading = math.degrees(math.atan2(lateral, distance))
        enemy_heading = (ego_heading + 180.0) % 360.0
        ego_speed = enemy_speed = 800.0
    elif situation == "defensive_disadvantage":
        ego_ne = (distance, lateral)
        enemy_ne = (0.0, 0.0)
        ego_heading = 0.0
        enemy_heading = math.degrees(math.atan2(lateral, distance))
        ego_speed, enemy_speed = speed_enemy_fps, speed_ego_fps
    elif situation == "random":
        raise ValueError("random should be handled by env.reset(), not build_init_states().")
    else:
        raise ValueError("Unknown situation curriculum stage: {}".format(situation))

    return [
        state_from_ne(base_lon, base_lat, ego_ne, altitude_ft, ego_heading, ego_speed),
        state_from_ne(base_lon, base_lat, enemy_ne, altitude_ft, enemy_heading, enemy_speed),
    ]


def sample_situation(probabilities: Dict, rng=None) -> str:
    if not probabilities:
        return "random"
    rng = rng or np.random
    names = list(probabilities.keys())
    weights = np.asarray([max(float(probabilities[name]), 0.0) for name in names], dtype=np.float64)
    if float(np.sum(weights)) <= 0.0:
        return "random"
    weights = weights / np.sum(weights)
    if hasattr(rng, "choice"):
        idx = int(rng.choice(len(names), p=weights))
    else:
        idx = int(np.random.choice(len(names), p=weights))
    return str(names[idx])


def state_from_ne(base_lon: float, base_lat: float, ne, altitude_ft: float, heading_deg: float, speed_fps: float) -> Dict:
    north, east = ne
    lat = float(base_lat) + float(north) / 111000.0
    lon = float(base_lon) + float(east) / (111000.0 * max(math.cos(math.radians(float(base_lat))), 1e-6))
    return {
        "ic_long_gc_deg": float(lon),
        "ic_lat_geod_deg": float(lat),
        "ic_h_sl_ft": float(altitude_ft),
        "ic_psi_true_deg": float(heading_deg) % 360.0,
        "ic_u_fps": float(speed_fps),
    }


def _uniform(rng, values):
    if isinstance(values, (int, float)):
        return float(values)
    lo, hi = values
    if hasattr(rng, "uniform"):
        return float(rng.uniform(float(lo), float(hi)))
    return float(np.random.uniform(float(lo), float(hi)))
