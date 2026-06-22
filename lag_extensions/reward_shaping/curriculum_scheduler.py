import copy
from typing import Dict, Optional


class CurriculumScheduler:
    """Reward curriculum scheduler.

    The current implementation switches stages by local environment steps.
    A win-rate based interface is left in place for later self-play/eval coupling.
    """

    def __init__(self, config: Dict, reward_config: Dict):
        self.config = config or {}
        self.reward_config = reward_config or {}
        self.enabled = bool(self.config.get("enabled", False))
        self.mode = self.config.get("mode", "by_training_steps")
        self.stages = sorted(self.config.get("stages", []), key=lambda x: int(x.get("start_step", 0)))
        self.current_stage = copy.deepcopy(self.stages[0]) if self.stages else self._base_stage()

    def update(self, training_step: int, win_rate: Optional[float] = None) -> Dict:
        if not self.enabled:
            self.current_stage = self._base_stage()
            return self.current_stage
        if self.mode == "by_training_steps":
            selected = self.stages[0] if self.stages else self._base_stage()
            for stage in self.stages:
                if training_step >= int(stage.get("start_step", 0)):
                    selected = stage
                else:
                    break
            self.current_stage = copy.deepcopy(selected)
            return self.current_stage
        if self.mode == "by_win_rate":
            # Placeholder: keep current stage until eval-driven win-rate logic is connected.
            return self.current_stage
        raise ValueError("Unknown curriculum mode: {}".format(self.mode))

    def _base_stage(self) -> Dict:
        return {
            "name": "disabled_or_base",
            "start_step": 0,
            "alpha": float(self.reward_config.get("alpha", 0.0)),
            "beta": float(self.reward_config.get("beta", 0.0)),
            "semantic_source": self.reward_config.get("semantic_source", "physical"),
            "semantic_call_interval": 1,
            "opponent_level": "unchanged",
            "situation_level": "unchanged",
        }
