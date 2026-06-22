from .config import LABEL_NAMES


PROMPT_VERSION = "compact_v1"
VERBOSE_PROMPT_VERSION = "verbose_v1"


def build_qwen_vl_prompt(prompt_version: str = PROMPT_VERSION) -> str:
    if prompt_version in ("v1", "verbose_v1"):
        return _build_verbose_prompt()
    if prompt_version == "compact_v1":
        return _build_compact_prompt()
    if prompt_version == "ultra_compact_v1":
        return _build_ultra_compact_prompt()
    if prompt_version == "metric_aware_v1":
        return _build_metric_aware_prompt()
    raise ValueError("Unsupported Qwen-VL prompt_version: {}".format(prompt_version))


def _build_ultra_compact_prompt() -> str:
    keys = ", ".join(LABEL_NAMES)
    return (
        "Return JSON only. Score this 1v1 air-combat diagram with floats in [0,1] for: "
        "{}. No action advice. No Markdown. No explanation."
    ).format(keys)


def _build_metric_aware_prompt() -> str:
    keys = ", ".join(LABEL_NAMES)
    return (
        "Score the current 1v1 air-combat diagram only. Return strict JSON only, no Markdown, no explanation. "
        "Use exactly these keys with float values in [0,1]: {}. "
        "For energy_advantage and energy_disadvantage, read the numeric panel: Ego V, Enemy V, Ego H, Enemy H, DeltaE. "
        "Positive DeltaE means ego has more specific energy; negative DeltaE means ego has less. "
        "For tail, attack-window, threat-zone, defensive, and neutral labels, use the aircraft positions, headings, cones, and distance rings. "
        "No action advice."
    ).format(keys)


def _build_compact_prompt() -> str:
    keys = ", ".join(LABEL_NAMES)
    return (
        "Score the current 1v1 air-combat diagram only. "
        "Do not suggest actions. Return one strict JSON object only, no Markdown, no code fence, no explanation. "
        "Use exactly these keys with float values in [0,1]: {}. "
        "0 means absent, 1 means strongly present, 0.5 means uncertain. "
        "Output format: "
        "{{\"ego_tail_advantage\":0.0,\"enemy_tail_threat\":0.0,"
        "\"effective_attack_window\":0.0,\"enemy_missile_threat_zone\":0.0,"
        "\"energy_advantage\":0.0,\"energy_disadvantage\":0.0,"
        "\"defensive_escape\":0.0,\"neutral_stalemate\":0.0}}"
    ).format(keys)


def _build_verbose_prompt() -> str:
    labels = "\n".join(["- {}".format(name) for name in LABEL_NAMES])
    return (
        "You are an air-combat situation semantic scorer. "
        "You only judge the current 1v1 situation shown in the image. "
        "Do not output actions, advice, tactics, explanations, Markdown, or code blocks.\n\n"
        "Score each label from 0.0 to 1.0. "
        "0.0 means the label is completely false, 1.0 means the label is completely true. "
        "If uncertain, output a value near 0.5 instead of inventing certainty.\n\n"
        "Label meanings:\n"
        "- ego_tail_advantage: ego is behind the enemy tail and points toward the enemy.\n"
        "- enemy_tail_threat: enemy is behind ego tail and points toward ego.\n"
        "- effective_attack_window: ego has suitable range, aspect, and aim for attack.\n"
        "- enemy_missile_threat_zone: ego is inside the enemy effective attack or missile threat zone.\n"
        "- energy_advantage: ego has clearly higher specific energy than enemy.\n"
        "- energy_disadvantage: ego has clearly lower specific energy than enemy.\n"
        "- defensive_escape: ego is threatened but separating, degrading enemy aim, or leaving the threat window.\n"
        "- neutral_stalemate: neither side has clear attack advantage and the situation is neutral or stalled.\n\n"
        "Return strict JSON with exactly these keys and numeric float values in [0, 1]:\n"
        "{}\n\n"
        "Output JSON only."
    ).format(labels)
