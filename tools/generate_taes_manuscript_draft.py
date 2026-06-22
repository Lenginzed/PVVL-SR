#!/usr/bin/env python
import argparse
import csv
import json
from pathlib import Path


TITLE_MAIN = "PVVL-SR: A Physics-Verified Vision-Language Semantic Reward Shaping Framework for UAV Air Combat Reinforcement Learning"
TITLE_CONSERVATIVE = "A Physics-Verified Vision-Language Reward Modeling Framework for Reinforcement Learning-Based UAV Air Combat Decision-Making"
TITLE_INTERPRETABLE = "Interpretable Vision-Language Semantic Reward Shaping for UAV Air Combat Reinforcement Learning"


def main():
    parser = argparse.ArgumentParser(description="Generate TAES manuscript draft package from the TAES experiment package.")
    parser.add_argument("--package-dir", default="scripts/results/taes_paper_package")
    parser.add_argument("--output-dir", default="scripts/results/taes_manuscript_draft")
    args = parser.parse_args()
    out = generate(Path(args.package_dir), Path(args.output_dir))
    print(json.dumps(out, indent=2, sort_keys=True))


def generate(pkg, out):
    out.mkdir(parents=True, exist_ok=True)
    data = load_package_data(pkg)
    writers = {
        "manuscript_blueprint.md": manuscript_blueprint,
        "title_and_abstract_candidates.md": title_and_abstract_candidates,
        "introduction_draft.md": introduction_draft,
        "related_work_plan.md": related_work_plan,
        "problem_formulation_draft.md": problem_formulation_draft,
        "method_section_draft.md": method_section_draft,
        "experiment_section_draft.md": experiment_section_draft,
        "results_section_draft.md": results_section_draft,
        "discussion_section_draft.md": discussion_section_draft,
        "conclusion_draft.md": conclusion_draft,
        "figure_table_placement_plan.md": figure_table_placement_plan,
        "notation_and_acronyms.md": notation_and_acronyms,
        "contribution_statement.md": contribution_statement,
        "reviewer_risk_and_response_plan.md": reviewer_risk_and_response_plan,
        "manuscript_claim_traceability_matrix.md": traceability_matrix,
        "taes_manuscript_outline.tex": manuscript_outline_tex,
        "taes_manuscript_outline.md": manuscript_outline_md,
        "manuscript_draft_package_report.md": package_report,
    }
    paths = {}
    for name, fn in writers.items():
        path = out / name
        path.write_text(fn(data), encoding="utf-8")
        paths[name] = str(path.resolve())
    return {"output_dir": str(out.resolve()), "files": paths}


def load_package_data(pkg):
    tables = pkg / "main_results_tables"
    return {
        "pkg": pkg,
        "mixed": keyed(read_csv(tables / "table1_mixed_eval_stage13.csv"), "Method"),
        "offensive": keyed(read_csv(tables / "table2_offensive_eval_stage13.csv"), "Method"),
        "unseen": keyed(read_csv(tables / "table3_unseen_eval_stage13.csv"), "Method"),
        "ablation": read_csv(tables / "table4_ablation_study.csv"),
        "dodge": keyed(read_csv(tables / "table5_dodgemissile_preliminary.csv"), "Method"),
        "claims": read_text(pkg / "paper_claims_and_limitations.md"),
        "protocol": read_text(pkg / "experiment_protocol_summary.md"),
        "consistency": read_text(pkg / "result_consistency_report.md"),
        "report": read_text(pkg / "taes_experiment_package_report.md"),
    }


def manuscript_blueprint(data):
    return f"""# Manuscript Blueprint

## Recommended Title

{TITLE_MAIN}

## Target Venue

IEEE Transactions on Aerospace and Electronic Systems (TAES), regular paper.

## Central Positioning

This manuscript should be framed as a physically grounded semantic reward modeling and reward shaping framework for reinforcement learning-based UAV air-combat decision-making. The central contribution is not a new PPO policy architecture and not an online VLM controller. Instead, the paper shows how an offline vision-language teacher can be made useful for air-combat reinforcement learning through physical verification, label-wise fusion, surrogate reward learning, potential-based shaping, and situation curriculum.

## Recommended Section Structure

1. Introduction
2. Related Work
3. Problem Formulation
4. Proposed Method: PVVL-SR
5. Experimental Setup
6. Results and Analysis
7. Discussion
8. Conclusion
9. Appendix, optional

## Primary Evidence Flow

- Fig. 1 introduces the pipeline.
- Fig. 2 and Table IV establish why raw VLM labels must be physically verified.
- Fig. 3 and Table IV justify the surrogate reward network.
- Table I and Fig. 4 provide the main Stage13 mixed-evaluation result.
- Table II and Table III separate offensive and unseen behavior.
- Fig. 5 explains the 300k-to-500k trade-off.
- Fig. 6 supports sample-efficiency discussion as an auxiliary result.
- Fig. 7 provides qualitative trajectory evidence.
- Table V and Fig. 8 show DodgeMissile compatibility only, not weapon-combat success.

## Recommended Main Claim

PPO-PVVL-SR provides a physically verified and deployable semantic reward shaping mechanism that improves mixed-evaluation win rate, draw reduction, attack-window occupancy, and stalemate reduction in the reported LAG 1v1 NoWeapon setting. PPO-Phys remains a strong threat-control baseline and must be presented as such.

## Writing Tone

Use restrained TAES language: "provides", "improves in the reported setting", "competitive with", "trade-off", "preliminary". Avoid "solves", "state-of-the-art", "fully outperforms", and any implication that Qwen controls the UAV online.
"""


def title_and_abstract_candidates(data):
    m = data["mixed"]["PPO-PVVL-SR"]
    p = data["mixed"]["PPO-Phys"]
    return f"""# Title and Abstract Candidates

## Title Candidates

1. **Recommended:** {TITLE_MAIN}
2. **More conservative:** {TITLE_CONSERVATIVE}
3. **More interpretable:** {TITLE_INTERPRETABLE}

## Recommended Final Title

**{TITLE_MAIN}**

This title is recommended because it gives the method a compact acronym, PVVL-SR, while explicitly stating the three technical anchors of the paper: physics verification, vision-language semantic reward modeling, and reward shaping for UAV air-combat reinforcement learning. It is specific enough to match the completed experiments and avoids overclaiming optimality, autonomy at weapon level, or state-of-the-art status.

## Abstract Version A: Performance-Oriented but Restrained

Reward design remains a central challenge for reinforcement learning-based UAV air-combat decision-making, where sparse outcomes and manually designed physical rewards may not fully capture tactical semantics. Vision-language models (VLMs) offer a way to evaluate spatial air-combat situations, but raw VLM scores are too slow for online training, unreliable for some physical and energy-related semantics, and difficult to cache in continuous simulator states. This paper proposes PVVL-SR, a physics-verified vision-language semantic reward shaping framework. The framework uses Qwen2.5-VL-7B-Instruct as an offline teacher on standardized air-combat situation diagrams, verifies semantic scores with geometry and energy metrics, applies label-wise fusion with physical override, trains a lightweight surrogate semantic reward network, and injects the resulting reward through potential-based shaping and mixed situation curriculum. Experiments in the LAG 1v1 NoWeapon/Selfplay environment with PPO show that PVVL-SR achieves the highest mixed-evaluation win rate ({m['win_rate']}) and lower draw rate ({m['draw_rate']}) among the compared Stage13 methods, while improving attack-window occupancy and reducing neutral-stalemate behavior. PPO-Phys remains a strong baseline for enemy-threat exposure ({p['enemy_threat_exposure']}), indicating an aggressiveness-safety trade-off. A small DodgeMissile study is included only as preliminary interface and transfer validation.

## Abstract Version B: Framework-Oriented and Conservative

This paper studies how vision-language semantic knowledge can be incorporated into reinforcement learning for UAV air combat without relying on online VLM inference or unconstrained learned rewards. We propose PVVL-SR, a physics-verified vision-language semantic reward shaping framework for PPO-based 1v1 air-combat decision-making. The framework first renders simulator states as standardized situation diagrams and uses Qwen2.5-VL-7B-Instruct as an offline semantic teacher over eight tactical labels. Because raw VLM scores are imperfect, particularly for energy and threat semantics, the scores are checked using physical air-combat geometry and energy metrics and fused label-wise with physical override rules. A lightweight surrogate network is then trained to predict the fused semantic scores from state features, avoiding online VLM calls and exact image-cache dependence. The surrogate reward is incorporated by potential-based shaping and trained with a mixed situation curriculum. Experiments in LAG 1v1 NoWeapon/Selfplay show that PVVL-SR improves mixed-evaluation win rate, draw reduction, and attack-window occupancy compared with curriculum PPO and physical shaping in the reported 500k setting, while physical shaping remains stronger for threat exposure and some unseen outcomes. DodgeMissile results are reported only as preliminary compatibility evidence. The results support physically constrained semantic reward modeling rather than direct VLM control.

## Recommendation

Use **Abstract Version A** for the first TAES draft because it states the main numerical evidence clearly while preserving the caveat that PPO-Phys remains strong in threat control. Version B is useful if the manuscript later needs a more conservative framing.
"""


def introduction_draft(data):
    return """# Introduction Draft

Autonomous UAV air-combat decision-making is a challenging problem for aerospace systems because the agent must reason over fast dynamics, adversarial interaction, partial tactical opportunities, and safety-critical constraints. Reinforcement learning (RL) is attractive in this setting because it can optimize closed-loop behavior directly from interaction with a simulator and can discover policies beyond simple scripted rules. However, the effectiveness of RL in air combat depends heavily on reward design. Sparse terminal rewards are often insufficient for learning useful maneuvering behavior, while dense rewards can easily bias the agent toward unintended strategies such as passive survival, excessive separation, or locally safe but tactically ineffective behavior.

A common way to address this problem is to design physically interpretable reward shaping terms. Air-combat geometry, relative distance, line-of-sight angles, energy difference, and threat-zone indicators are natural reward sources. These quantities are valuable because they are measurable, physically meaningful, and aligned with established tactical concepts. Nevertheless, physical reward shaping also has limitations. It usually requires manual threshold design and may only encode the specific geometric or energy quantities selected by the designer. Higher-level tactical semantics, such as whether the current situation resembles an effective attack opportunity, a defensive escape, or a prolonged stalemate, can be difficult to express with a single hand-crafted scalar reward.

Vision-language models (VLMs) provide a possible semantic layer for this problem. Given a standardized diagram of an air-combat situation, a VLM can evaluate spatial relationships and produce semantic descriptions or scores. This capability is appealing because air-combat decision-making is often described using tactical concepts rather than only raw coordinates. However, a VLM cannot be directly inserted into the online control loop. Large VLM inference is slow, raw model outputs may be unreliable for physical quantities such as energy advantage, and an unconstrained semantic score can conflict with safety-relevant geometry. Moreover, if VLM outputs are cached by rendered image hashes, the cache is unlikely to cover continuous PPO states encountered during training.

Our experiments confirm these issues. Qwen2.5-VL-7B-Instruct provides useful geometric semantic information, but raw VLM scores show large errors for energy- and threat-related labels. Exact cached-VLM lookup has zero hit rate in a random PPO rollout, motivating a learned surrogate rather than direct cache use. We also find that the default 1v1 NoWeapon initial-state distribution rarely produces simultaneous tail, aim, range, and closing-rate conditions required for an attack-window label. These observations motivate a framework that combines VLM semantic cues with physical verification, surrogate reward learning, and situation curriculum rather than treating the VLM as an online controller or a complete reward oracle.

This paper proposes PVVL-SR, a physics-verified vision-language semantic reward shaping framework for UAV air-combat RL. The framework renders simulator states into standardized situation diagrams, uses Qwen2.5-VL-7B-Instruct as an offline semantic teacher, checks VLM scores against air-combat geometry and energy metrics, fuses labels using label-wise policies, trains a lightweight surrogate semantic reward network, and injects the resulting scores through potential-based reward shaping. A mixed situation curriculum is used to expose the policy to offensive, neutral, defensive, and random initial conditions. During PPO training and deployment, the VLM is not called online; only the surrogate and physical metrics are used.

The main contributions are fourfold. First, we propose a physics-verified VLM semantic reward shaping framework for reinforcement learning-based UAV air combat. Second, we design a label-wise semantic fusion mechanism that uses VLM spatial reasoning where appropriate while relying on physical computation for energy and safety-critical labels. Third, we introduce a surrogate semantic reward network that avoids online VLM inference and supports continuous-state PPO training. Fourth, we conduct extensive LAG 1v1 experiments showing that the proposed method improves mixed-evaluation win rate, attack-window occupancy, and stalemate reduction in the reported setting, while also identifying that physical reward shaping remains a strong threat-control baseline.
"""


def related_work_plan(data):
    return """# Related Work Plan

This file intentionally uses reference placeholders. Do not convert placeholders into fabricated citations. A bibliography pass should be done separately.

## Reinforcement Learning for UAV Air-Combat Decision-Making

Needed references:

- [REF: UAV air combat RL]
- [REF: PPO]
- [REF: multi-agent air combat RL]
- [REF: self-play air combat]
- [REF: LAG environment]

What to cover:

Summarize the use of deep RL, PPO, self-play, and simulator-based training for air-combat decision-making. Emphasize that prior work often relies on sparse outcomes, engineered dense rewards, or scenario-specific curriculum. Explain that the present work does not propose a new policy optimizer, but focuses on semantic reward modeling compatible with PPO.

How this paper differs:

PVVL-SR targets reward construction rather than policy architecture. The VLM is not used as a controller and does not output actions. Instead, offline semantic labels are physically verified and distilled into a surrogate reward model.

## Reward Shaping and Curriculum Learning in Autonomous Decision-Making

Needed references:

- [REF: potential-based reward shaping]
- [REF: curriculum learning]
- [REF: reward shaping in robotics]
- [REF: safe reward shaping]

What to cover:

Discuss potential-based shaping as a way to add dense learning signals while preserving the conceptual structure of reward differences. Discuss curriculum learning as a method for exposing agents to progressively diverse or reachable situations. Mention that in air combat, attack-window conditions may be rare under default initialization.

How this paper differs:

PVVL-SR uses semantic potentials derived from physically verified VLM labels and uses situation curriculum to make tactical semantics reachable during training. The curriculum is not presented as a hidden performance trick; it is motivated by explicit reachability diagnostics.

## Vision-Language Models and VLM-Guided Reward Learning

Needed references:

- [REF: VLM reward model]
- [REF: CLIP/VLM feedback for robotics]
- [REF: VLM-as-judge]
- [REF: Qwen2.5-VL technical report/model card]

What to cover:

Summarize how VLMs have been used for scene understanding, reward feedback, task evaluation, or language-conditioned robotics. Highlight known issues: latency, hallucination, calibration, and physical inconsistency.

How this paper differs:

PVVL-SR does not trust raw VLM outputs directly. It uses Qwen2.5-VL only offline, checks semantic scores with air-combat physics, and trains a small surrogate network for online reward shaping.

## Trustworthy and Physics-Informed Learning for Aerospace Systems

Needed references:

- [REF: physics-informed RL / safe RL]
- [REF: trustworthy AI for aerospace]
- [REF: interpretable reward models]
- [REF: safety constraints in autonomous systems]

What to cover:

Discuss the importance of physical consistency, interpretability, and safety in aerospace learning systems. Explain that purely learned semantic rewards can be unsafe if they conflict with kinematics or energy constraints.

How this paper differs:

PVVL-SR explicitly separates geometry, energy, threat, defensive, and stalemate labels and applies label-wise fusion policies. This makes the reward model more interpretable than a black-box scalar reward while remaining more semantically expressive than hand-crafted physical shaping alone.

## Positioning Statement

This paper is positioned between direct VLM control and pure physical reward shaping. Unlike direct VLM-control methods, PVVL-SR never asks the VLM to output actions and never calls it in the PPO step loop. Unlike pure physical shaping, it uses visual-language semantic labels to represent tactical concepts. The central idea is physics-verified semantic reward modeling for RL training.
"""


def problem_formulation_draft(data):
    return """# Problem Formulation Draft

We consider a two-aircraft 1v1 UAV air-combat decision-making problem modeled as a Markov decision process (MDP). Let the MDP be denoted by

\\[
\\mathcal{M} = (\\mathcal{S}, \\mathcal{A}, P, R, \\gamma),
\\]

where \\(\\mathcal{S}\\) is the simulator state space, \\(\\mathcal{A}\\) is the action space, \\(P(s_{t+1}|s_t,a_t)\\) is the transition function induced by the flight dynamics simulator, \\(R\\) is the environment reward, and \\(\\gamma\\) is the discount factor. A policy \\(\\pi_\\theta(a_t|o_t)\\) maps the observation \\(o_t\\) of the ego aircraft to an action. In the experiments, the policy is trained with PPO in the LAG-master `1v1/NoWeapon/Selfplay` scenario.

The simulator state contains ego and enemy aircraft information, including position, velocity, speed, altitude, and attitude-derived heading information. From these quantities, relative geometry and energy metrics are computed, including distance, line-of-sight direction, closing rate, aim angle, tail angle, specific energy, and energy difference. These metrics are used both to define physical semantic labels and to support reward shaping.

We define an eight-dimensional semantic label space

\\[
\\mathcal{Y}=\\{y_1,\\ldots,y_8\\},
\\]

where the labels are ego_tail_advantage, enemy_tail_threat, effective_attack_window, enemy_missile_threat_zone, energy_advantage, energy_disadvantage, defensive_escape, and neutral_stalemate. Each label is represented by a score in \\([0,1]\\), where higher values indicate stronger presence of the corresponding tactical semantic condition.

The objective is to learn a PPO policy that benefits from semantic reward shaping without requiring online VLM inference. Let \\(\\hat{q}_i(s)\\) denote the final fused or surrogate-predicted semantic score for label \\(i\\). The semantic potential is defined as

\\[
\\Phi(s) = \\sum_i w_i \\hat{q}_i(s),
\\]

where \\(w_i\\) is the configured label weight. The potential-based semantic shaping term is

\\[
F(s_t,s_{t+1}) = \\gamma \\Phi(s_{t+1}) - \\Phi(s_t).
\\]

The training reward combines the original simulator reward, an optional physical reward term, and the semantic shaping term. The key constraint is that the VLM is used only offline to construct training targets for the surrogate reward network; PPO training uses state features, physical verification, and the surrogate network only.
"""


def method_section_draft(data):
    return """# Method Section Draft

## IV-A. Air-Combat Semantic Label Space

PVVL-SR represents tactical air-combat situations using eight semantic labels. The labels are designed to cover offensive geometry, defensive threat, energy state, escape behavior, and stalemate behavior. The offensive labels are ego_tail_advantage and effective_attack_window. The first indicates whether the ego aircraft is behind the opponent and oriented toward it, while the second further requires suitable range and launch geometry. The defensive and safety labels are enemy_tail_threat and enemy_missile_threat_zone, which indicate whether the opponent occupies a threatening rear-aspect or launch-like region. The energy labels are energy_advantage and energy_disadvantage, derived from speed and altitude. The final two labels, defensive_escape and neutral_stalemate, describe whether the aircraft is escaping a threat or remaining in a prolonged non-decisive situation.

Each label has both a hard interpretation and a soft score in \\([0,1]\\). Hard labels are useful for reporting tactical time ratios, while soft scores are used in reward shaping. This label space is intentionally compact. It does not attempt to describe every possible air-combat tactic, but it provides a structured vocabulary for semantic reward modeling in a 1v1 simulator.

## IV-B. Situation Rendering and Offline VLM Teacher

The VLM teacher is applied offline to standardized two-dimensional situation diagrams generated from LAG simulator states. The rendering includes the ego and enemy positions, heading directions, velocity directions, range rings, attack or threat sectors, and a compact numeric panel for state quantities such as speed, altitude, and energy difference. Qwen2.5-VL-7B-Instruct is prompted to output strict JSON scores for the eight semantic labels. The prompt asks only for situation evaluation and explicitly does not ask for action advice.

The VLM is not used online during PPO training and is not part of the deployed policy. This design avoids the latency and deployment burden of large VLM inference. The raw VLM labels are treated as semantic observations that require verification rather than as trusted reward values.

## IV-C. Physics Verification

Physical verification is based on air-combat geometry and energy metrics. Let \\(p_e\\) and \\(p_b\\) denote ego and enemy positions, \\(v_e\\) and \\(v_b\\) their velocities, and \\(h_e\\) and \\(h_b\\) their heading unit vectors. The relative vector is \\(r=p_b-p_e\\), the distance is \\(d=\\|r\\|\\), and the line-of-sight unit vector is \\(u=r/d\\). The ego aim angle, enemy aim angle, ego tail angle, enemy tail angle, closing rate, and specific energy difference are computed from these quantities. All arccos inputs are clipped to avoid numerical errors.

The physical score \\(p_i\\) for each label is compared with the semantic score \\(q_i\\) produced by the VLM or surrogate. A consistency confidence can be computed as

\\[
c_i = \\exp(-|q_i-p_i|/\\tau),
\\]

or by a hard threshold gate. The verified VLM score is \\(c_i q_i\\). Physical verification is especially important for energy-related labels, where a VLM may not reliably infer kinetic and potential energy from an image.

## IV-D. Label-Wise Semantic Fusion

PVVL-SR uses label-wise fusion because not all semantic labels should trust the VLM equally. Geometry labels can benefit from VLM spatial reasoning, while energy labels should be dominated by physical computation. For a hybrid label, the fused score is

\\[
\\tilde{q}_i = \\lambda_i c_i q_i + (1-\\lambda_i)p_i.
\\]

For physical labels, \\(\\lambda_i=0\\). For verified-VLM labels, the confidence-weighted VLM score is used. Threat labels use hybrid fusion with physical override. If the physical score strongly contradicts the VLM score, or if a high physical threat is assigned a low VLM score, the fused score is replaced or bounded by the physical score. This prevents semantic reward shaping from suppressing safety-critical physical information.

## IV-E. Surrogate Semantic Reward Network

A finite VLM cache is not sufficient for PPO training in a continuous state space. The Stage7 cache diagnosis shows an exact cached-VLM hit rate of zero under random rollout, which means that an image-hash cache would mostly fall back to physical scores during training. PVVL-SR therefore trains a lightweight MLP surrogate from compact state features to fused semantic label scores.

The surrogate input includes distance, aim and tail angles, closing rate, speed, altitude, and energy-difference features. The output is the eight fused semantic scores. Surrogate v2 is trained on the formal100 dataset, targeted geometry data, and boundary-augmented data. During PPO training, semantic scores are generated by this surrogate network in milliseconds or less, while Qwen remains offline.

## IV-F. Potential-Based Semantic Reward Shaping

The fused or surrogate semantic scores are converted into a semantic potential

\\[
\\Phi(s)=\\sum_i w_i \\tilde{q}_i(s).
\\]

The shaping term is

\\[
F(s_t,s_{t+1})=\\gamma\\Phi(s_{t+1})-\\Phi(s_t).
\\]

The total training reward is

\\[
R_t = R_{env,t}+\\alpha R_{phys,t}+\\beta F(s_t,s_{t+1}).
\\]

Potential-based shaping is used to reduce the risk that dense semantic rewards dominate the original task reward. The ablation results show that removing potential shaping is not uniformly worse on every tactical metric, but it weakens outcome and threat-control behavior.

## IV-G. Mixed Situation Curriculum

Default 1v1 NoWeapon rollouts rarely enter states that simultaneously satisfy tail, aim, range, and closing-rate requirements for an attack-window label. PVVL-SR therefore uses a mixed situation curriculum containing offensive_advantage, neutral_merge, defensive_disadvantage, and random initial situations. The curriculum is not intended to inflate results; it is a response to a reachability issue diagnosed by metric sanity checks, threshold sweeps, rollout geometry analysis, and offensive-initialization smoke tests. Evaluation is still reported separately under default, offensive, mixed, and unseen protocols.
"""


def experiment_section_draft(data):
    return """# Experimental Setup Section Draft

## V-A. Simulation Environment and Baselines

The main experiments are conducted in LAG-master using the `1v1/NoWeapon/Selfplay` scenario. This setting isolates maneuvering, geometry, energy, and tactical-positioning behavior without claiming full missile-combat validation. PPO is used as the base reinforcement learning algorithm, and the policy architecture is not modified by PVVL-SR.

The compared methods are named as follows. PPO is the original PPO baseline without reward shaping or curriculum. PPO-Phys adds physical reward shaping. PPO-Cur uses the mixed situation curriculum without semantic reward shaping. PPO-PVVL-SR is the proposed method, combining surrogate semantic scores, label-wise physical verification, potential-based shaping, and mixed situation curriculum. PPO-PVVL-SR-noPot is used as a potential-shaping ablation, and PPO-PVVL-SR-direct is treated as a diagnostic variant rather than a main method.

## V-B. Implementation Details

The VLM teacher is Qwen2.5-VL-7B-Instruct, deployed locally and used only offline. Situation diagrams are rendered from simulator states and labeled with eight semantic scores. Raw VLM scores are physically verified and fused label-wise. The fused labels are then used to train surrogate v2, a lightweight MLP that maps state features to semantic label scores. PPO training uses this surrogate and physical metrics only; it does not call Qwen online.

Stage13 uses three seeds, 0, 1, and 2, and evaluates 200 episodes for each evaluation type. The Stage13 500k confirmation is implemented as Stage12 300k actor/critic weights plus 200k continuation steps. The PPO optimizer state is not restored because the original runner does not save optimizer checkpoints. This caveat should be explicitly stated in the experimental protocol.

## V-C. Evaluation Protocol

Four evaluation settings are used. The default evaluation uses the simulator's default initial distribution. The offensive evaluation starts from offensive-advantage situations to test whether a policy can retain attack geometry. The mixed evaluation samples offensive, neutral, defensive, and random situations to match the curriculum distribution. The unseen evaluation uses a different mixture or perturbation pattern to check whether the policy is over-specialized to the training curriculum.

A DodgeMissile preliminary validation is also reported using `1v1/DodgeMissile/Selfplay`. This is a small-scale compatibility and weak-transfer check only. It is not treated as a full weapon-enabled benchmark, and missile warning, lock, and hit fields are not assumed when not exposed by the simulator info dictionary.

## V-D. Metrics

The primary outcome metrics are win rate, draw rate, average return, episode length, and survival time. Tactical metrics include attack-window ratio, tail-advantage ratio, enemy-threat exposure, enemy missile-threat-zone ratio, defensive-escape ratio, neutral-stalemate ratio, minimum ego aim angle, and minimum ego tail angle. Reward-shaping diagnostics include semantic potential, semantic shaping reward, physical reward, and total reward. Semantic modeling metrics include raw VLM MAE, fused-label MAE, surrogate MAE, high-difference cases, cache hit rate, and surrogate inference time.
"""


def results_section_draft(data):
    return f"""# Results and Analysis Section Draft

## VI-A. Semantic Label Reliability

Fig. 2 and Table IV show that raw VLM scores should not be used directly as rewards. On the formal100 semantic dataset, raw VLM overall MAE is 0.1421, while the physics-verified fused scores reduce the overall MAE to 0.0108. The number of high-difference cases decreases from 100/100 to 0/100. The improvement is most important for energy- and threat-related labels, where raw VLM scores are less physically reliable. This observation supports the label-wise fusion design: VLM spatial semantics are useful, but they must be checked by air-combat geometry and energy constraints.

## VI-B. Surrogate Reward Network

Fig. 3 and Table IV summarize the surrogate reward model results. Exact cached-VLM lookup is not feasible in continuous PPO states: the Stage7 diagnosis reports zero exact cache hits and full fallback under random rollout. Surrogate v1 achieves overall MAE 0.0377 with a high-difference ratio of 17.8%. After targeted geometry and boundary augmentation, surrogate v2 reduces MAE to 0.0256 and the high-difference ratio to 3.0%. The surrogate also provides fast CPU inference, approximately 7.36e-05 seconds per query, which makes it practical for online reward shaping without online VLM calls.

## VI-C. Main NoWeapon Results

Table I and Fig. 4 provide the main Stage13 500k mixed-evaluation comparison. PPO-PVVL-SR obtains the highest mixed win rate, {data['mixed']['PPO-PVVL-SR']['win_rate']}, compared with {data['mixed']['PPO-Cur']['win_rate']} for PPO-Cur and {data['mixed']['PPO-Phys']['win_rate']} for PPO-Phys. It also has the lowest draw rate, {data['mixed']['PPO-PVVL-SR']['draw_rate']}, and the highest attack-window ratio, {data['mixed']['PPO-PVVL-SR']['effective_attack_window_time_ratio_mean']}. Its neutral-stalemate ratio is also lower than the two baselines. These results indicate that semantic reward shaping can encourage more decisive and tactically active behavior in the mixed curriculum evaluation.

The result is not a universal advantage. PPO-Phys has lower enemy-threat exposure, {data['mixed']['PPO-Phys']['enemy_threat_exposure']}, than PPO-PVVL-SR, {data['mixed']['PPO-PVVL-SR']['enemy_threat_exposure']}. This is an important safety-related strength of physical shaping. The correct interpretation is that PVVL-SR improves mixed outcome and offensive-geometry indicators, while physical shaping remains a strong threat-control baseline.

## VI-D. Offensive and Unseen Evaluation

Table II shows that PPO-PVVL-SR performs strongly in the offensive evaluation. It reaches win rate {data['offensive']['PPO-PVVL-SR']['win_rate']} and attack-window ratio {data['offensive']['PPO-PVVL-SR']['effective_attack_window_time_ratio_mean']}, both higher than PPO-Phys in this evaluation. This supports the role of semantic reward shaping and curriculum in retaining offensive geometry when attack situations are reachable.

Table III presents a more mixed unseen-evaluation picture. PPO-Phys obtains a higher unseen win rate, {data['unseen']['PPO-Phys']['win_rate']}, than PPO-PVVL-SR, {data['unseen']['PPO-PVVL-SR']['win_rate']}. However, PPO-PVVL-SR retains a higher attack-window ratio and lower neutral-stalemate ratio. This suggests that PVVL-SR provides useful tactical incentives, but does not dominate the physical baseline under distribution shift.

## VI-E. Training Trend from 300k to 500k

Fig. 5 compares the main method from 300k to 500k in mixed evaluation. The trend indicates that PPO-PVVL-SR improves attack-window and tail-advantage occupancy and reduces neutral-stalemate behavior as training continues. At the same time, enemy-threat exposure increases relative to the 300k checkpoint. This trend suggests an aggressiveness-safety trade-off: longer training improves offensive activity but may expose the ego aircraft to more threat. This observation should motivate cautious interpretation rather than a claim of uniform improvement.

## VI-F. Ablation Studies

Table IV summarizes the major ablations and diagnostics. Label-wise fusion greatly reduces raw VLM label error. The exact cache diagnosis motivates the surrogate reward network. Surrogate v2 improves over surrogate v1 after targeted and boundary augmentation. The potential-based shaping ablation shows that removing potential shaping is not simply catastrophic on all tactical metrics, but it weakens outcome and threat-control behavior. Finally, curriculum is necessary because default rollouts rarely expose the policy to attack-window conditions, although curriculum alone does not solve every tactical metric.

## VI-G. Preliminary DodgeMissile Validation

Table V and Fig. 8 report the DodgeMissile preliminary validation. The wrapper and reward modules run successfully in `1v1/DodgeMissile/Selfplay`, and PPO-PVVL-SR shows slightly higher return, longer episode length, and longer survival time than the two preliminary baselines. However, all methods still lose in the 50k single-seed evaluation. Missile warning, lock, and hit metrics are not exposed in the available info fields. These results should be treated only as preliminary compatibility and weak-transfer evidence, not as full missile-combat validation.
"""


def discussion_section_draft(data):
    return """# Discussion Section Draft

The experiments suggest that VLM semantics are useful in UAV air-combat RL when they are treated as a structured source of tactical reward information rather than as direct control commands. The strongest role of the VLM appears in spatial and geometric reasoning. Situation diagrams make tail relations, heading, relative position, and attack sectors visually explicit, and these concepts are naturally aligned with tactical language. This helps explain why semantic reward shaping improves attack-window occupancy and reduces draw or stalemate behavior in the mixed evaluation.

At the same time, the results show why physics verification is necessary. Raw VLM scores are not uniformly reliable, especially for energy- and threat-related semantics that depend on speed, altitude, closing rate, or safety-critical geometry. A VLM may read a diagram correctly in a qualitative sense while still producing scores that are inconsistent with air-combat dynamics. Label-wise fusion addresses this by allowing geometry labels to preserve VLM contributions while energy labels and high-threat cases remain physically grounded.

The surrogate reward network is also essential. Large VLM inference is too slow for online PPO reward shaping, and exact rendered-image cache lookup has zero hit rate in continuous rollout states. The surrogate converts offline VLM-derived and physically verified labels into a fast state-feature reward model. This is the mechanism that makes the framework deployable in training and avoids online VLM dependence.

The Stage13 results reveal an aggressiveness-safety trade-off. PPO-PVVL-SR achieves the best mixed-evaluation win rate and attack-window ratio among the compared Stage13 methods, and it reduces draw and neutral-stalemate behavior. However, PPO-Phys has lower enemy-threat exposure, particularly in mixed and unseen evaluations. The proposed method should therefore be interpreted as a semantic reward shaping framework that improves offensive and mixed-situation behavior, not as a replacement for physical safety shaping.

The role of curriculum is also central. Stage11 diagnostics show that the default initial distribution rarely enters clear attack-window conditions. Without situation curriculum, an agent may receive little meaningful signal about offensive semantics. The mixed curriculum is therefore not an artificial post-hoc adjustment but a training mechanism that makes tactical situations reachable. Nevertheless, default evaluation still shows weak attack-window occupancy, indicating that broader initial-state and opponent diversity remain necessary for stronger generalization.

Several limitations remain. The main benchmark is NoWeapon and does not validate full missile-combat performance. DodgeMissile experiments are preliminary, single-seed, and short, and all methods still lose in the reported evaluation. The physical baseline remains strong and should be retained as a central comparator. Stage13 resumes actor and critic weights but resets PPO optimizer state. The experiments use three seeds and one main VLM teacher, Qwen2.5-VL-7B-Instruct. Future work should include weapon-enabled scenarios, stronger opponent pools, additional VLM teachers, explicit safety constraints, and multi-agent settings such as 2v2 air combat.
"""


def conclusion_draft(data):
    return """# Conclusion Draft

This paper proposed PVVL-SR, a physics-verified vision-language semantic reward shaping framework for reinforcement learning-based UAV air-combat decision-making. The framework uses Qwen2.5-VL-7B-Instruct as an offline semantic teacher, verifies semantic labels using air-combat geometry and energy metrics, applies label-wise fusion with physical override, trains a lightweight surrogate semantic reward network, and injects the resulting semantic potential through potential-based reward shaping. A mixed situation curriculum is used to expose PPO to offensive, neutral, defensive, and random initial conditions.

The experiments show that raw VLM scores are useful but not sufficiently reliable for direct reward use. Physics-verified fusion substantially reduces semantic label error, and surrogate v2 provides fast inference suitable for online reward shaping without online VLM calls. In the LAG 1v1 NoWeapon/Selfplay setting, PPO-PVVL-SR achieves the highest Stage13 mixed-evaluation win rate, the lowest draw rate, higher attack-window occupancy, and lower neutral-stalemate ratio among the compared Stage13 methods. However, PPO-Phys remains a strong baseline, especially in enemy-threat exposure and some unseen outcomes. These results support the value of physically grounded semantic reward modeling, while also revealing an aggressiveness-safety trade-off.

The DodgeMissile experiment provides preliminary compatibility evidence only. It shows that the wrapper, metrics, and surrogate reward can run in a weapon-enabled scenario, but it does not establish full missile-combat success. Future work should extend PVVL-SR to weapon-enabled benchmarks, stronger opponent pools, multi-agent 2v2 engagements, broader VLM comparisons, and explicit safety constraints. The broader direction is to combine semantic reasoning with physical consistency so that learned air-combat policies can receive richer training signals without relying on unverifiable or non-deployable online VLM inference.
"""


def figure_table_placement_plan(data):
    rows = [
        ("Fig. 1", "Overview of the PVVL-SR pipeline.", "End of Introduction or start of Method IV.", "Shows offline VLM teacher, physics verification, label-wise fusion, surrogate reward, potential shaping, and PPO update.", "scripts/results/taes_paper_package/figures/fig1_method_architecture.pdf", "Do not imply online VLM control."),
        ("Fig. 2", "Raw VLM versus physics-verified fused semantic label MAE.", "Results VI-A.", "Fusion reduces label error, especially for energy and threat semantics.", "fig2_label_quality_improvement.pdf", "Fused labels partly rely on physics; do not claim raw VLM is accurate."),
        ("Fig. 3", "Surrogate v1 versus surrogate v2 per-label MAE.", "Results VI-B.", "Boundary and geometry augmentation improves surrogate accuracy.", "fig3_surrogate_model_accuracy.pdf", "Surrogate is trained on fused labels, not raw VLM truth."),
        ("Fig. 4", "Stage13 mixed-evaluation performance.", "Results VI-C.", "PVVL-SR has highest mixed win and attack-window ratio and lower draw/stalemate.", "fig4_mixed_evaluation_performance.pdf", "PPO-Phys has lower enemy-threat exposure."),
        ("Fig. 5", "300k to 500k trend for PPO-PVVL-SR.", "Results VI-E.", "Attack and tail improve, neutral decreases, enemy threat increases.", "fig5_300k_to_500k_trend.pdf", "This is a trade-off, not uniform improvement."),
        ("Fig. 6", "Sample efficiency for mixed win >= 0.30.", "Results or Appendix.", "Auxiliary sample-efficiency comparison.", "fig6_sample_efficiency.pdf", "Checkpoint variance and 3 seeds limit strength."),
        ("Fig. 7", "Representative trajectories.", "Results VI-C or Discussion.", "Qualitative examples of offensive, safe/defensive, and stalemate behavior.", "fig7_representative_trajectories.pdf", "Representative, not statistical proof."),
        ("Fig. 8", "DodgeMissile preliminary validation.", "Results VI-G or Appendix.", "Compatibility and weak transfer signal.", "fig8_dodgemissile_preliminary.pdf", "All methods still lose; not weapon validation."),
        ("Table I", "Mixed evaluation.", "Results VI-C.", "Main quantitative comparison.", "table1_mixed_eval_stage13.tex", "Mention physical threat advantage."),
        ("Table II", "Offensive evaluation.", "Results VI-D.", "PVVL-SR maintains offensive geometry better.", "table2_offensive_eval_stage13.tex", "High variance due to 3 seeds."),
        ("Table III", "Unseen evaluation.", "Results VI-D.", "Physical has stronger unseen win/threat; PVVL-SR retains attack and neutral advantages.", "table3_unseen_eval_stage13.tex", "Do not claim unseen dominance."),
        ("Table IV", "Ablation study.", "Results VI-F.", "Supports fusion, surrogate, potential shaping, and curriculum rationale.", "table4_ablation_study.tex", "No-potential is nuanced."),
        ("Table V", "DodgeMissile preliminary validation.", "Results VI-G or Appendix.", "Wrapper compatibility and weak transfer signal.", "table5_dodgemissile_preliminary.tex", "Preliminary only."),
    ]
    lines = ["# Figure and Table Placement Plan", "", "| ID | Caption draft | Where to cite | Main message | Source file | Risk/caveat |", "|---|---|---|---|---|---|"]
    lines.extend(["| " + " | ".join(row) + " |" for row in rows])
    return "\n".join(lines) + "\n"


def notation_and_acronyms(data):
    return """# Notation and Acronyms

## Acronyms

| Acronym | Meaning |
|---|---|
| UAV | Unmanned aerial vehicle |
| RL | Reinforcement learning |
| PPO | Proximal policy optimization |
| VLM | Vision-language model |
| PVVL-SR | Physics-Verified Vision-Language Semantic Reward Shaping |
| MDP | Markov decision process |
| MAE | Mean absolute error |
| V1/V2 | Surrogate reward network versions |
| NoWeapon | Main maneuvering-only LAG scenario |

## Notation

| Symbol | Meaning |
|---|---|
| \\(s_t\\) | Simulator state at time step \\(t\\) |
| \\(o_t\\) | Ego aircraft observation |
| \\(a_t\\) | Ego action |
| \\(\\pi_\\theta\\) | PPO policy |
| \\(p_e, p_b\\) | Ego and enemy positions |
| \\(v_e, v_b\\) | Ego and enemy velocities |
| \\(h_e, h_b\\) | Ego and enemy heading unit vectors |
| \\(r=p_b-p_e\\) | Relative position |
| \\(d=\\|r\\|\\) | Relative distance |
| \\(u=r/d\\) | Line-of-sight unit vector |
| \\(p_i\\) | Physical score for semantic label \\(i\\) |
| \\(q_i\\) | VLM or surrogate semantic score |
| \\(c_i\\) | Physical consistency confidence |
| \\(\\tilde{q}_i\\) | Fused semantic score |
| \\(w_i\\) | Semantic label reward weight |
| \\(\\Phi(s)\\) | Semantic potential |
| \\(F(s_t,s_{t+1})\\) | Potential-based semantic shaping term |
| \\(\\alpha, \\beta\\) | Physical and semantic reward coefficients |
"""


def contribution_statement(data):
    return """# Contribution Statement

The proposed manuscript should state four contributions.

1. We propose PVVL-SR, a physics-verified vision-language semantic reward shaping framework for PPO-based UAV air-combat reinforcement learning. The framework uses VLM semantics as offline reward-modeling information rather than as online control actions.

2. We design a label-wise fusion mechanism that combines VLM spatial reasoning with physics-based geometry, energy, and threat verification. The mechanism preserves VLM contributions for suitable geometric labels while allowing physical scores and override rules to dominate energy and safety-critical semantics.

3. We introduce a surrogate semantic reward network that maps compact air-combat state features to fused semantic scores. This avoids online Qwen inference and solves the zero exact-cache-hit problem observed in continuous PPO rollouts.

4. We evaluate the framework in LAG 1v1 NoWeapon/Selfplay with multiple stages of diagnostics, ablations, and Stage13 500k confirmation. The results show improved mixed-evaluation win rate, draw reduction, attack-window occupancy, and neutral-stalemate reduction, while also identifying PPO-Phys as a strong threat-control baseline and DodgeMissile as preliminary only.
"""


def reviewer_risk_and_response_plan(data):
    risks = [
        ("Physical baseline is strong.", "It may appear that physical shaping is sufficient.", "Explicitly show that PPO-Phys has better threat exposure but lower mixed win and attack-window metrics than PPO-PVVL-SR in Stage13.", "Do not claim PVVL-SR fully replaces physical shaping.", "Run broader physical baselines or tune physical threat control in future work."),
        ("Threat exposure of PVVL-SR is higher than PPO-Phys.", "This is a safety-relevant weakness.", "Frame it as an aggressiveness-safety trade-off and discuss potential safety constraints.", "Do not hide the enemy-threat metric.", "Add safety-constrained shaping or threat-aware curriculum."),
        ("NoWeapon does not validate missile combat.", "TAES reviewers may expect weapon relevance.", "State that NoWeapon isolates maneuvering semantics and that DodgeMissile is preliminary.", "Do not equate NoWeapon with missile performance.", "Run full weapon-enabled experiments after action-shape compatibility is solved."),
        ("DodgeMissile is only preliminary.", "The table could be overread.", "Label it preliminary in captions and results; note all methods lose under 50k single-seed evaluation.", "Do not call it weapon success.", "Run multi-seed, longer DodgeMissile validation."),
        ("VLM teacher is offline and surrogate-based.", "Reviewers may ask whether the VLM is really used at deployment.", "Emphasize offline teacher and deployable surrogate as a design advantage.", "Do not imply online VLM reasoning.", "Compare more VLM teachers offline."),
        ("Only Qwen2.5-VL is used as main VLM.", "Generalization across VLMs is untested.", "Present Qwen as a fixed baseline teacher.", "Do not claim VLM-model agnosticism as proven.", "Add InternVL, LLaVA, or smaller VLM teachers."),
        ("Only 1v1, not 2v2 multi-agent.", "Scalability is not proven.", "State that this paper studies 1v1 semantic reward modeling first.", "Do not claim multi-agent scalability.", "Extend to 2v2 and cooperative tactics."),
        ("500k continuation optimizer state reset.", "Training continuation is not a perfect optimizer resume.", "Disclose it in protocol and limitations.", "Do not describe it as exact checkpoint resume.", "Save and restore optimizer state in future formal runs."),
        ("Only 3 seeds.", "Statistical strength is limited.", "Report mean +/- std and avoid claims of statistical significance.", "Do not say robustly superior.", "Run 5 or more seeds if resources allow."),
        ("Curriculum may dominate performance.", "Reviewers may question whether reward shaping adds beyond curriculum.", "Include PPO-Cur as a key baseline and discuss mixed/default/offensive differences.", "Do not omit PPO-Cur.", "Add more curriculum-only variants or matched curriculum ablations."),
    ]
    lines = ["# Reviewer Risk and Response Plan", "", "| Risk | Why it matters | Manuscript response | What not to claim | Possible future experiment |", "|---|---|---|---|---|"]
    lines.extend(["| " + " | ".join(row) + " |" for row in risks])
    return "\n".join(lines) + "\n"


def traceability_matrix(data):
    rows = [
        ("PVVL-SR improves mixed-evaluation win rate and reduces draw rate.", "Results VI-C", "table1_mixed_eval_stage13.md", "Table I, Fig. 4", "moderate", "achieves the highest mixed-evaluation win rate among the compared Stage13 methods", "consistently outperforms all baselines"),
        ("PVVL-SR improves attack-window occupancy in mixed and offensive evaluations.", "Results VI-C/VI-D", "table1, table2", "Table I, Table II", "moderate", "improves attack-window occupancy in the reported setting", "solves offensive air combat"),
        ("PPO-Phys remains stronger in enemy threat exposure.", "Results VI-C/Discussion", "table1_mixed_eval_stage13.md", "Table I, Fig. 4", "strong", "PPO-Phys retains lower enemy-threat exposure", "PVVL-SR is safer on all metrics"),
        ("Label-wise fusion improves raw VLM labels.", "Results VI-A", "table4_ablation_study.md; fig2 source", "Table IV, Fig. 2", "strong", "substantially reduces label MAE", "raw VLM is fully reliable"),
        ("Surrogate is necessary due to zero exact cache hits.", "Results VI-B", "table4_ablation_study.md", "Table IV, Fig. 3", "strong", "motivates surrogate reward learning", "exact cache failure means VLM is useless"),
        ("Potential shaping improves outcome/threat control.", "Results VI-F", "stage12_ablation_table; table4", "Table IV", "moderate", "helps outcome/threat control and stability", "no-potential always fails"),
        ("DodgeMissile shows compatibility only.", "Results VI-G", "table5_dodgemissile_preliminary.md", "Table V, Fig. 8", "preliminary", "preliminary compatibility and weak transfer signal", "weapon combat performance is validated"),
        ("VLM is offline teacher, not controller.", "Method/Discussion", "experiment_protocol_summary.md", "Fig. 1", "strong", "offline teacher distilled into surrogate reward", "VLM controls the UAV"),
    ]
    lines = ["# Manuscript Claim Traceability Matrix", "", "| Claim | Section | Evidence file | Figure/Table | Strength | Wording recommendation | Wording to avoid |", "|---|---|---|---|---|---|---|"]
    lines.extend(["| " + " | ".join(row) + " |" for row in rows])
    return "\n".join(lines) + "\n"


def manuscript_outline_tex(data):
    return r"""\documentclass[journal]{IEEEtran}
\usepackage{amsmath,amssymb}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{cite}

\title{PVVL-SR: A Physics-Verified Vision-Language Semantic Reward Shaping Framework for UAV Air Combat Reinforcement Learning}

\author{Author~Names~Withheld}

\begin{document}
\maketitle

\begin{abstract}
% Use Abstract Version A or B from title_and_abstract_candidates.md.
\end{abstract}

\begin{IEEEkeywords}
UAV air combat, reinforcement learning, reward shaping, vision-language model, physics verification, curriculum learning.
\end{IEEEkeywords}

\section{Introduction}
% See introduction_draft.md.

\section{Related Work}
\subsection{Reinforcement Learning for UAV Air Combat}
\subsection{Reward Shaping and Curriculum Learning}
\subsection{Vision-Language Models for Reward Learning}
\subsection{Trustworthy and Physics-Informed Aerospace Learning}

\section{Problem Formulation}
% See problem_formulation_draft.md.

\section{Proposed Method: PVVL-SR}
\subsection{Air-Combat Semantic Label Space}
\subsection{Situation Rendering and Offline VLM Teacher}
\subsection{Physics Verification}
\subsection{Label-Wise Semantic Fusion}
\subsection{Surrogate Semantic Reward Network}
\subsection{Potential-Based Semantic Reward Shaping}
\subsection{Mixed Situation Curriculum}

\section{Experimental Setup}
\subsection{Simulation Environment and Baselines}
\subsection{Implementation Details}
\subsection{Evaluation Protocol}
\subsection{Metrics}

\section{Results and Analysis}
\subsection{Semantic Label Reliability}
\subsection{Surrogate Reward Network}
\subsection{Main NoWeapon Results}
\subsection{Offensive and Unseen Evaluation}
\subsection{Training Trend from 300k to 500k}
\subsection{Ablation Studies}
\subsection{Preliminary DodgeMissile Validation}

\section{Discussion}

\section{Conclusion}

\appendices
\section{Additional Diagnostics}

\bibliographystyle{IEEEtran}
\bibliography{references}

\end{document}
"""


def manuscript_outline_md(data):
    return """# TAES Manuscript Outline

1. Title
2. Abstract
3. Index Terms
4. I. Introduction
5. II. Related Work
6. III. Problem Formulation
7. IV. Proposed Method: PVVL-SR
8. V. Experimental Setup
9. VI. Results and Analysis
10. VII. Discussion
11. VIII. Conclusion
12. Appendix: Additional diagnostics and preliminary weapon scenario details
"""


def package_report(data):
    return f"""# Manuscript Draft Package Report

## Generated Files

- manuscript_blueprint.md
- title_and_abstract_candidates.md
- introduction_draft.md
- related_work_plan.md
- problem_formulation_draft.md
- method_section_draft.md
- experiment_section_draft.md
- results_section_draft.md
- discussion_section_draft.md
- conclusion_draft.md
- figure_table_placement_plan.md
- notation_and_acronyms.md
- contribution_statement.md
- reviewer_risk_and_response_plan.md
- manuscript_claim_traceability_matrix.md
- taes_manuscript_outline.tex
- taes_manuscript_outline.md

## Recommended Title

{TITLE_MAIN}

## Recommended Abstract Version

Use Abstract Version A for the first TAES draft. It is performance-oriented but still restrained and includes the physical-baseline caveat. Abstract Version B is available if the manuscript needs a more conservative framing.

## Proposed Section Structure

The draft follows a regular TAES paper structure: Introduction, Related Work, Problem Formulation, Proposed Method, Experimental Setup, Results and Analysis, Discussion, and Conclusion.

## Main Claims

- PVVL-SR improves Stage13 mixed-evaluation win rate, draw reduction, attack-window occupancy, and neutral-stalemate reduction in the reported setting.
- Label-wise physics-verified fusion improves raw VLM label reliability.
- The surrogate reward network is necessary because exact cached VLM hits are zero in continuous rollout states.
- PPO-Phys remains a strong threat-control baseline.

## Main Limitations

- NoWeapon is not full missile combat.
- DodgeMissile validation is preliminary and single-seed.
- PPO-Phys has lower enemy-threat exposure than PPO-PVVL-SR.
- Stage13 continuation resets optimizer state.
- The study uses one main VLM teacher and three seeds.

## Figures/Tables Placement

See figure_table_placement_plan.md. The main result should cite Table I and Fig. 4. The DodgeMissile material should likely go near the end of Results or in an appendix.

## Remaining Manual Writing Tasks

- Add real references and replace [REF: ...] placeholders.
- Integrate the draft paragraphs into a single IEEEtran manuscript.
- Manually inspect all figures inside the TAES template.
- Decide whether DodgeMissile remains in the main text or moves to appendix.
- Add exact software versions and final author/affiliation information.

## Readiness

The package is ready for full manuscript drafting. It is not yet a final paper because references, final LaTeX integration, and human-level narrative polishing are still required.
"""


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def read_text(path):
    return Path(path).read_text(encoding="utf-8") if Path(path).exists() else ""


def keyed(rows, key):
    return {row[key]: row for row in rows}


if __name__ == "__main__":
    main()
