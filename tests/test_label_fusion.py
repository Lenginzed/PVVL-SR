import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from lag_extensions.reward_shaping.config import LABEL_NAMES
from lag_extensions.reward_shaping.semantic_reward import LabelWiseSemanticFusion, PhysicalVerifier


def _scores(value):
    return {name: float(value) for name in LABEL_NAMES}


def test_vlm_high_physical_low_override():
    physical = _scores(0.0)
    semantic = _scores(0.0)
    semantic["ego_tail_advantage"] = 0.95
    verification = PhysicalVerifier().verify(physical, semantic)
    fused = LabelWiseSemanticFusion().fuse(verification)
    assert fused["override_info"]["ego_tail_advantage"]["override_triggered"]
    assert fused["fused_scores"]["ego_tail_advantage"] <= 0.1


def test_physical_high_vlm_low_override():
    physical = _scores(0.0)
    semantic = _scores(0.0)
    physical["enemy_tail_threat"] = 0.95
    verification = PhysicalVerifier().verify(physical, semantic)
    fused = LabelWiseSemanticFusion().fuse(verification)
    assert fused["override_info"]["enemy_tail_threat"]["override_triggered"]
    assert fused["fused_scores"]["enemy_tail_threat"] == physical["enemy_tail_threat"]


def test_energy_labels_are_physical():
    physical = _scores(0.0)
    semantic = _scores(1.0)
    physical["energy_advantage"] = 0.25
    physical["energy_disadvantage"] = 0.75
    verification = PhysicalVerifier().verify(physical, semantic)
    fused = LabelWiseSemanticFusion().fuse(verification)
    assert fused["fused_scores"]["energy_advantage"] == 0.25
    assert fused["fused_scores"]["energy_disadvantage"] == 0.75


def test_fused_scores_are_clipped():
    physical = _scores(0.2)
    semantic = _scores(0.8)
    verification = PhysicalVerifier().verify(physical, semantic)
    fused = LabelWiseSemanticFusion().fuse(verification)
    for value in fused["fused_scores"].values():
        assert 0.0 <= value <= 1.0


if __name__ == "__main__":
    test_vlm_high_physical_low_override()
    test_physical_high_vlm_low_override()
    test_energy_labels_are_physical()
    test_fused_scores_are_clipped()
    print("label fusion tests passed")
