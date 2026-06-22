from typing import Dict, Optional

import numpy as np

from envs.JSBSim.core.catalog import Catalog as c


class StateExtractor:
    """Extract 1v1 air-combat state from LAG JSBSim environments.

    LAG's AircraftSimulator.get_position() returns local NEU coordinates
    (north, east, up) in meters. get_velocity() returns velocity in the same
    frame, with vertical velocity converted from JSBSim down to up.
    """

    def extract(self, env, agent_id: str) -> Dict:
        ego = env.agents[agent_id]
        if len(ego.enemies) == 0:
            raise ValueError("Reward shaping currently supports 1v1 tasks with one enemy.")
        enemy = ego.enemies[0]

        ego_state = self._aircraft_state(env, ego)
        enemy_state = self._aircraft_state(env, enemy)

        relative_position = enemy_state["position"] - ego_state["position"]
        distance = float(np.linalg.norm(relative_position))
        line_of_sight = relative_position / max(distance, 1e-8)
        relative_velocity = enemy_state["velocity"] - ego_state["velocity"]

        return {
            "agent_id": agent_id,
            "enemy_id": enemy.uid,
            "ego": ego_state,
            "enemy": enemy_state,
            "relative": {
                "relative_position": relative_position,
                "distance": distance,
                "line_of_sight_unit_vector": line_of_sight,
                "relative_velocity": relative_velocity,
                "distance_rate": float(np.dot(relative_velocity, line_of_sight)),
            },
        }

    def _aircraft_state(self, env, sim) -> Dict:
        position = np.asarray(sim.get_position(), dtype=np.float64)
        velocity = np.asarray(sim.get_velocity(), dtype=np.float64)
        roll, pitch, yaw = sim.get_rpy()
        speed = float(np.linalg.norm(velocity))
        heading_vector = self._heading_vector(float(pitch), float(yaw), velocity)
        missile_warning = sim.check_missile_warning() is not None

        return {
            "uid": sim.uid,
            "position": position,
            "velocity": velocity,
            "speed": speed,
            "altitude": float(sim.get_property_value(c.position_h_sl_m)),
            "roll": float(roll),
            "pitch": float(pitch),
            "yaw": float(yaw),
            "heading_vector": heading_vector,
            "missile_warning": bool(missile_warning),
            "num_under_missiles": int(sum(1 for missile in sim.under_missiles if missile.is_alive)),
            "num_launch_missiles": int(sum(1 for missile in sim.launch_missiles if missile.is_alive)),
            "lock": bool(self._has_attack_lock(env, sim.uid)),
            "is_alive": bool(sim.is_alive),
            "is_crash": bool(sim.is_crash),
            "is_shotdown": bool(sim.is_shotdown),
        }

    def _heading_vector(self, pitch: float, yaw: float, velocity: np.ndarray) -> np.ndarray:
        heading = np.array([
            np.cos(pitch) * np.cos(yaw),
            np.cos(pitch) * np.sin(yaw),
            np.sin(pitch),
        ], dtype=np.float64)
        norm = np.linalg.norm(heading)
        if norm > 1e-8:
            return heading / norm
        velocity_norm = np.linalg.norm(velocity)
        if velocity_norm > 1e-8:
            return velocity / velocity_norm
        return np.array([1.0, 0.0, 0.0], dtype=np.float64)

    def _has_attack_lock(self, env, uid: str) -> bool:
        lock_duration = getattr(getattr(env, "task", None), "lock_duration", None)
        if not isinstance(lock_duration, dict) or uid not in lock_duration:
            return False
        duration = lock_duration[uid]
        maxlen = getattr(duration, "maxlen", None)
        if not maxlen:
            return False
        return bool(np.sum(duration) >= maxlen)
