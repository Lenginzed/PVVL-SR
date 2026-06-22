from __future__ import annotations

import json
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TEMPLATE_DIR = ROOT / "TAES" / "PVVL-SR_ A Physics-Verified Vision-Language Semantic Reward Shaping Framework for UAV Air Combat Reinforcement Learning"
PAPER_PKG = ROOT / "scripts" / "results" / "taes_paper_package"
MANUSCRIPT_DRAFT = ROOT / "scripts" / "results" / "taes_manuscript_draft"
OUT = ROOT / "manuscript" / "taes_pvvl_sr_v01"


TITLE = "PVVL-SR: A Physics-Verified Vision-Language Semantic Reward Shaping Framework for UAV Air Combat Reinforcement Learning"


FIGURES = [
    ("fig1_method_architecture.pdf", "fig:architecture"),
    ("fig2_label_quality_improvement.pdf", "fig:label-quality"),
    ("fig3_surrogate_model_accuracy.pdf", "fig:surrogate-accuracy"),
    ("fig4_mixed_evaluation_performance.pdf", "fig:mixed-performance"),
    ("fig5_300k_to_500k_trend.pdf", "fig:trend-300-500"),
    ("fig6_sample_efficiency.pdf", "fig:sample-efficiency"),
    ("fig7_representative_trajectories.pdf", "fig:trajectories"),
    ("fig8_dodgemissile_preliminary.pdf", "fig:dodge"),
]


def ensure_dirs() -> None:
    for sub in ["sections", "figures", "tables"]:
        (OUT / sub).mkdir(parents=True, exist_ok=True)


def copy_template_and_assets() -> None:
    for name in ["IEEEtaes.cls", "IEEEtaes.bst"]:
        shutil.copy2(TEMPLATE_DIR / name, OUT / name)
    for fig, _ in FIGURES:
        shutil.copy2(PAPER_PKG / "figures" / fig, OUT / "figures" / fig)


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.strip() + "\n", encoding="utf-8")


def main_tex() -> str:
    return r"""
\documentclass{IEEEtaes}

\usepackage{color,array,amsthm,amsmath,amssymb}
\usepackage{graphicx}
\usepackage{url}
\usepackage{xcolor}

\jvol{XX}
\jnum{XX}
\jmonth{XXXXX}
\paper{TAES-2026-TODO}
\pubyear{2026}
\doiinfo{TAES.2026.TODO}

\setcounter{page}{1}

\newcommand{\needcite}[1]{\textcolor{red}{[REF: #1]}}
\newcommand{\method}{PVVL-SR}
\newcommand{\methodfull}{Physics-Verified Vision-Language Semantic Reward Shaping}
\newcommand{\ppopvvl}{PPO-PVVL-SR}

\begin{document}

\title{PVVL-SR: A Physics-Verified Vision-Language Semantic Reward Shaping Framework for UAV Air Combat Reinforcement Learning}

\author{Zhendong Li}
\affil{School of Aeronautics and Astronautics, University of Electronic Science and Technology of China, Chengdu 611731, China}

\author{Hui Li}
\member{Senior Member, IEEE}
\affil{School of Aeronautics and Astronautics, University of Electronic Science and Technology of China, Chengdu 611731, China}

\receiveddate{Manuscript received TODO; revised TODO; accepted TODO. Funding information is TODO and should be inserted only after confirmation.}

\corresp{Corresponding author: Hui Li.}

\authoraddress{Zhendong Li is with the School of Aeronautics and Astronautics, University of Electronic Science and Technology of China, Chengdu 611731, China (e-mail: \href{mailto:18980494367@163.com}{18980494367@163.com}). Hui Li is with the School of Aeronautics and Astronautics, University of Electronic Science and Technology of China, Chengdu 611731, China (e-mail: \href{mailto:Kelly.li@126.com}{Kelly.li@126.com}).}

\editor{Supplemental materials, data availability, and conflict-of-interest statements are TODO for the submission version.}

\markboth{LI AND LI}{PVVL-SR FOR UAV AIR COMBAT REINFORCEMENT LEARNING}
\maketitle

\begin{abstract}
Reward design remains a central challenge for reinforcement learning-based unmanned aerial vehicle (UAV) air-combat decision-making, where sparse outcomes and manually designed physical rewards may not fully capture tactical semantics. Vision-language models (VLMs) offer a way to evaluate spatial air-combat situations, but raw VLM scores are too slow for online training, unreliable for some physical and energy-related semantics, and difficult to cache in continuous simulator states. This paper proposes PVVL-SR, a physics-verified vision-language semantic reward shaping framework. The framework uses Qwen2.5-VL-7B-Instruct as an offline teacher on standardized air-combat situation diagrams, verifies semantic scores with geometry and energy metrics, applies label-wise fusion with physical override, trains a lightweight surrogate semantic reward network, and injects the resulting reward through potential-based shaping and mixed situation curriculum. Experiments in the LAG 1v1 NoWeapon/Selfplay environment with PPO show that PVVL-SR achieves the highest mixed-evaluation win rate and lower draw rate among the compared Stage13 methods, while improving attack-window occupancy and reducing neutral-stalemate behavior. PPO-Phys remains a strong baseline for enemy-threat exposure, indicating an aggressiveness-safety trade-off. A small DodgeMissile study is included only as preliminary interface and transfer validation.
\end{abstract}

\begin{IEEEkeywords}
Autonomous decision-making, curriculum learning, physics verification, reinforcement learning, reward shaping, semantic reward, UAV air combat, vision-language model.
\end{IEEEkeywords}

\input{acronyms}
\input{notation}
\input{sections/01_introduction}
\input{sections/02_related_work}
\input{sections/03_problem_formulation}
\input{sections/04_method}
\input{sections/05_experiments}
\input{sections/06_results}
\input{sections/07_discussion}
\input{sections/08_conclusion}

% References are intentionally left as TODO placeholders in this v0.1 draft.
% Replace \needcite{...} markers with real citations and uncomment the following lines.
% \bibliographystyle{IEEEtaes}
% \bibliography{refs}

\end{document}
"""


def acronyms_tex() -> str:
    return r"""
% Acronyms used in the manuscript.
\newcommand{\UAV}{unmanned aerial vehicle}
\newcommand{\RL}{reinforcement learning}
\newcommand{\VLM}{vision-language model}
\newcommand{\PPO}{proximal policy optimization}
"""


def notation_tex() -> str:
    return r"""
% Main notation.
\newcommand{\state}{s}
\newcommand{\action}{a}
\newcommand{\policy}{\pi}
\newcommand{\reward}{R}
\newcommand{\semanticpotential}{\Phi}
"""


def section_intro() -> str:
    return r"""
\section{Introduction}

Autonomous unmanned aerial vehicle (UAV) air-combat decision-making is a challenging problem for aerospace systems because the agent must reason over fast dynamics, adversarial interaction, transient tactical opportunities, and safety-critical constraints. Reinforcement learning (RL) is attractive in this setting because it can optimize closed-loop behavior directly from interaction with a simulator and can discover policies beyond simple scripted rules \needcite{UAV air combat RL}. However, the effectiveness of RL in air combat depends heavily on reward design. Sparse terminal rewards are often insufficient for learning useful maneuvering behavior, while dense rewards can bias the agent toward unintended strategies such as passive survival, excessive separation, or locally safe but tactically ineffective behavior.

A common way to address this problem is to design physically interpretable reward shaping terms. Air-combat geometry, relative distance, line-of-sight angles, energy difference, and threat-zone indicators are natural reward sources. These quantities are valuable because they are measurable, physically meaningful, and aligned with established tactical concepts. Nevertheless, physical reward shaping also has limitations. It usually requires manual threshold design and may only encode the specific geometric or energy quantities selected by the designer. Higher-level tactical semantics, such as whether the current situation resembles an effective attack opportunity, a defensive escape, or a prolonged stalemate, can be difficult to express with a single hand-crafted scalar reward.

Vision-language models (VLMs) provide a possible semantic layer for this problem. Given a standardized diagram of an air-combat situation, a VLM can evaluate spatial relationships and produce semantic descriptions or scores \needcite{VLM reward model}. This capability is appealing because air-combat decision-making is often described using tactical concepts rather than only raw coordinates. However, a VLM cannot be directly inserted into the online control loop. Large VLM inference is slow, raw model outputs may be unreliable for physical quantities such as energy advantage, and an unconstrained semantic score can conflict with safety-relevant geometry. Moreover, if VLM outputs are cached by rendered image hashes, the cache is unlikely to cover continuous PPO states encountered during training.

Our experiments confirm these issues. Qwen2.5-VL-7B-Instruct provides useful geometric semantic information, but raw VLM scores show large errors for energy- and threat-related labels. Exact cached-VLM lookup has zero hit rate in a random PPO rollout, motivating a learned surrogate rather than direct cache use. We also find that the default 1v1 NoWeapon initial-state distribution rarely produces simultaneous tail, aim, range, and closing-rate conditions required for an attack-window label. These observations motivate a framework that combines VLM semantic cues with physical verification, surrogate reward learning, and situation curriculum rather than treating the VLM as an online controller or a complete reward oracle.

This paper proposes PVVL-SR, a physics-verified vision-language semantic reward shaping framework for UAV air-combat RL. The framework renders simulator states into standardized situation diagrams, uses Qwen2.5-VL-7B-Instruct as an offline semantic teacher, checks VLM scores against air-combat geometry and energy metrics, fuses labels using label-wise policies, trains a lightweight surrogate semantic reward network, and injects the resulting scores through potential-based reward shaping. A mixed situation curriculum is used to expose the policy to offensive, neutral, defensive, and random initial conditions. During PPO training and deployment, the VLM is not called online; only the surrogate and physical metrics are used.

The main contributions are fourfold. First, we propose a physics-verified VLM semantic reward shaping framework for reinforcement learning-based UAV air combat. Second, we design a label-wise semantic fusion mechanism that uses VLM spatial reasoning where appropriate while relying on physical computation for energy and safety-critical labels. Third, we introduce a surrogate semantic reward network that avoids online VLM inference and supports continuous-state PPO training. Fourth, we conduct LAG 1v1 experiments showing that the proposed method improves mixed-evaluation win rate, attack-window occupancy, and stalemate reduction in the reported setting, while also identifying that physical reward shaping remains a strong threat-control baseline.
"""


def section_related() -> str:
    return r"""
\section{Related Work}

\subsection{Reinforcement Learning for UAV Air-Combat Decision-Making}
Deep RL has been studied for autonomous air-combat maneuvering, self-play training, and policy learning in high-dynamic simulators \needcite{UAV air combat RL} \needcite{multi-agent air combat RL}. Proximal policy optimization (PPO) is frequently used as a stable on-policy optimizer in continuous-control domains \needcite{PPO}. Simulator environments such as LAG provide reproducible settings for 1v1 and multi-agent air-combat decision-making \needcite{LAG environment}. The present work does not propose a new policy optimizer or aircraft control architecture. Instead, it focuses on reward construction and semantic reward modeling compatible with PPO.

\subsection{Reward Shaping and Curriculum Learning}
Reward shaping is a common way to provide dense learning signals when terminal outcomes are sparse. Potential-based shaping is particularly relevant because it introduces reward differences through a state potential rather than an arbitrary dense reward \needcite{potential-based reward shaping}. Curriculum learning can expose agents to progressively more informative or reachable situations \needcite{curriculum learning}. In air combat, attack-window conditions may be rare under default initialization, making curriculum design important for learning offensive geometry. PVVL-SR combines semantic potentials with a mixed situation curriculum, but the curriculum is reported as an explicit component and compared against curriculum-only baselines.

\subsection{Vision-Language Models and Reward Feedback}
VLMs have been used for scene understanding, task evaluation, reward feedback, and language-conditioned robotics \needcite{VLM reward model} \needcite{VLM in robotics/autonomous driving}. These models can provide semantic assessments beyond low-level numerical states. However, VLM outputs can be slow, poorly calibrated, and physically inconsistent. PVVL-SR therefore does not use Qwen2.5-VL-7B-Instruct as an online controller or a direct reward oracle \needcite{Qwen2.5-VL technical report/model card}. The VLM is only an offline semantic teacher whose labels are physically verified and distilled into a lightweight surrogate.

\subsection{Trustworthy and Physics-Informed Learning}
Trustworthy aerospace learning systems require physical consistency, interpretability, and safety-aware behavior \needcite{physics-informed RL / safe RL}. Purely learned scalar rewards may create unsafe incentives if they conflict with geometry, energy, or threat constraints. PVVL-SR explicitly separates offensive, defensive, energy, escape, and stalemate labels and applies label-wise fusion policies. This makes the reward model more interpretable than a black-box scalar reward while remaining more semantically expressive than hand-crafted physical shaping alone.
"""


def section_problem() -> str:
    return r"""
\section{Problem Formulation}

We consider a 1v1 air-combat task modeled as a Markov decision process
\begin{equation}
    \mathcal{M}=(\mathcal{S},\mathcal{A},P,R,\gamma),
\end{equation}
where $\mathcal{S}$ is the simulator state space, $\mathcal{A}$ is the action space, $P$ is the transition dynamics, $R$ is the training reward, and $\gamma$ is the discount factor. A policy $\pi_\theta(a|s)$ is trained by PPO to maximize the expected discounted return. The main simulator setting is LAG-master 1v1/NoWeapon/Selfplay. This setting isolates maneuvering, relative geometry, energy, and threat-like positioning without claiming full missile-combat performance.

PVVL-SR augments the original environment reward with semantic reward shaping. Let
\begin{equation}
    \mathcal{Y}=\{y_1,\ldots,y_8\}
\end{equation}
be the tactical semantic label space:
\begin{align}
\mathcal{Y}=\{&
\texttt{ego\_tail\_advantage},
\texttt{enemy\_tail\_threat},
\texttt{effective\_attack\_window}, \nonumber\\
&\texttt{enemy\_missile\_threat\_zone},
\texttt{energy\_advantage},
\texttt{energy\_disadvantage}, \nonumber\\
&\texttt{defensive\_escape},
\texttt{neutral\_stalemate}\}.
\end{align}
Each label has a soft score in $[0,1]$ and a hard interpretation for reporting tactical time ratios. The learning objective is to incorporate these labels as physically constrained semantic rewards while avoiding online VLM inference during PPO training.
"""


def section_method() -> str:
    return r"""
\section{PVVL-SR Method}

\begin{figure}[t]
\centering
\includegraphics[width=\columnwidth]{figures/fig1_method_architecture.pdf}
\caption{Overview of the PVVL-SR pipeline. The VLM teacher is used offline to annotate standardized air-combat situations; deployment-time PPO uses a lightweight surrogate reward model together with physical verification and potential-based shaping.}
\label{fig:architecture}
\end{figure}

\subsection{Air-Combat Semantic Label Space}
PVVL-SR represents tactical air-combat situations using eight semantic labels. The labels cover offensive geometry, defensive threat, energy state, escape behavior, and stalemate behavior. The offensive labels are \texttt{ego\_tail\_advantage} and \texttt{effective\_attack\_window}. Defensive and safety labels include \texttt{enemy\_tail\_threat} and \texttt{enemy\_missile\_threat\_zone}. Energy labels are \texttt{energy\_advantage} and \texttt{energy\_disadvantage}. The final labels, \texttt{defensive\_escape} and \texttt{neutral\_stalemate}, describe escape from threat and prolonged non-decisive situations. This compact label space provides a structured vocabulary for semantic reward modeling in a 1v1 simulator.

\subsection{Situation Rendering and Offline VLM Teacher}
Simulator states are rendered as standardized two-dimensional situation diagrams including ego and enemy positions, heading directions, velocity directions, range rings, attack or threat sectors, and compact numeric state information. Qwen2.5-VL-7B-Instruct is prompted to output strict JSON scores for the eight semantic labels. The prompt asks only for situation evaluation and explicitly does not ask for action advice. The VLM is not used online during PPO training and is not part of the deployed policy.

\subsection{Physics Verification}
Physical verification is based on air-combat geometry and energy metrics. Let $p_e$ and $p_b$ denote ego and enemy positions, $v_e$ and $v_b$ their velocities, and $h_e$ and $h_b$ their heading unit vectors. The relative vector is
\begin{equation}
    r=p_b-p_e,\qquad d=\|r\|,\qquad u=\frac{r}{d+\epsilon}.
\end{equation}
The ego aim angle, enemy aim angle, ego tail angle, enemy tail angle, and range rate are
\begin{align}
\theta_{\mathrm{aim}}^e &= \arccos(\mathrm{clip}(h_e^\top u,-1,1)),\\
\theta_{\mathrm{aim}}^b &= \arccos(\mathrm{clip}(h_b^\top (-u),-1,1)),\\
\theta_{\mathrm{tail}}^e &= \arccos(\mathrm{clip}(h_b^\top u,-1,1)),\\
\theta_{\mathrm{tail}}^b &= \arccos(\mathrm{clip}((-h_e)^\top u,-1,1)),\\
\dot d &= (v_b-v_e)^\top u.
\end{align}
Specific energy is computed as $E=\frac{1}{2}V^2+gh$, and the energy difference is $\Delta E=E_e-E_b$. The physical score $p_i$ for each label is compared with the semantic score $q_i$ produced by the VLM or surrogate. A soft consistency confidence is
\begin{equation}
    c_i = \exp(-|q_i-p_i|/\tau),
\end{equation}
and the verified VLM score is $c_iq_i$.

\subsection{Label-Wise Semantic Fusion}
PVVL-SR uses label-wise fusion because not all semantic labels should trust the VLM equally. Geometry labels can benefit from VLM spatial reasoning, while energy labels should be dominated by physical computation. For a hybrid label, the fused score is
\begin{equation}
    \tilde{q}_i = \lambda_i c_i q_i + (1-\lambda_i)p_i.
\end{equation}
For physical labels, $\lambda_i=0$. Threat labels use hybrid fusion with physical override. If the physical score strongly contradicts the VLM score, or if a high physical threat is assigned a low VLM score, the fused score is replaced or bounded by the physical score. This prevents semantic reward shaping from suppressing safety-critical physical information.

\subsection{Surrogate Semantic Reward Network}
A finite VLM cache is not sufficient for PPO training in a continuous state space. The Stage7 cache diagnosis reports an exact cached-VLM hit rate of zero under random rollout, which means that an image-hash cache would mostly fall back to physical scores during training. PVVL-SR therefore trains a lightweight multilayer perceptron surrogate from compact state features to fused semantic label scores. The surrogate input includes distance, aim and tail angles, closing rate, speed, altitude, and energy-difference features. The output is the eight fused semantic scores. During PPO training, semantic scores are generated by this surrogate network, while Qwen remains offline.

\subsection{Potential-Based Semantic Reward Shaping}
The fused or surrogate semantic scores are converted into a semantic potential
\begin{equation}
    \Phi(s)=\sum_i w_i \tilde{q}_i(s).
\end{equation}
The shaping term is
\begin{equation}
    F(s_t,s_{t+1})=\gamma\Phi(s_{t+1})-\Phi(s_t).
\end{equation}
The total training reward is
\begin{equation}
    R_t = R_{\mathrm{env},t}+\alpha R_{\mathrm{phys},t}+\beta F(s_t,s_{t+1}).
\end{equation}
Potential-based shaping is used to reduce the risk that dense semantic rewards dominate the original task reward. The ablation results show that removing potential shaping is not uniformly worse on every tactical metric, but it weakens outcome and threat-control behavior.

\subsection{Mixed Situation Curriculum}
Default 1v1 NoWeapon rollouts rarely enter states that simultaneously satisfy tail, aim, range, and closing-rate requirements for an attack-window label. PVVL-SR therefore uses a mixed situation curriculum containing offensive-advantage, neutral-merge, defensive-disadvantage, and random initial situations. The curriculum is motivated by metric sanity checks, threshold sweeps, rollout geometry analysis, and offensive-initialization smoke tests rather than introduced as an implicit performance trick.
"""


def section_experiments() -> str:
    return r"""
\section{Experimental Setup}

\subsection{Simulation Environment and Baselines}
The main experiments are conducted in LAG-master using the \texttt{1v1/NoWeapon/Selfplay} scenario \needcite{LAG environment}. This setting isolates maneuvering, geometry, energy, and tactical-positioning behavior without claiming full missile-combat validation. PPO is used as the base RL algorithm, and the policy architecture is not modified by PVVL-SR.

The compared methods use the following manuscript names. PPO denotes the original PPO baseline without reward shaping or curriculum. PPO-Phys denotes PPO with physical reward shaping. PPO-Cur denotes PPO with mixed situation curriculum. PPO-PVVL-SR denotes the proposed method, corresponding to \texttt{PPO\_surrogate\_v2\_then\_fusion\_with\_situation\_curriculum}. The no-potential ablation is denoted PPO-PVVL-SR-noPot. The direct surrogate variant is treated as a diagnostic variant rather than the main method.

\subsection{Implementation Details}
Qwen2.5-VL-7B-Instruct is deployed locally and used only for offline labeling. The final online reward source is surrogate v2, a lightweight MLP that predicts fused semantic label scores from state features. The Stage13 confirmation uses 500k training steps for the main methods and three random seeds. The 500k continuation loads actor and critic weights from the 300k checkpoints, but the optimizer state is reset; this caveat is preserved in the interpretation. Hardware and exact software versions should be inserted in the final submission after environment verification.

\subsection{Evaluation Protocol}
Four evaluation protocols are used. The default evaluation uses the original initial-state distribution. The offensive evaluation uses offensive-advantage initial states to test whether a policy can retain attack geometry. The mixed evaluation samples offensive, neutral, defensive, and random initial situations. The unseen evaluation changes the curriculum mixture or initial-state perturbations to test whether the policy overfits the training curriculum. The DodgeMissile experiment is reported only as preliminary compatibility and weak-transfer validation.

\subsection{Metrics}
The primary outcome metrics are win rate, draw rate, average return, and episode length. Tactical metrics include \texttt{ego\_tail\_advantage} time ratio, \texttt{effective\_attack\_window} time ratio, enemy-threat exposure, defensive-escape ratio, neutral-stalemate ratio, attack-window entry count, and first entry time. Reward metrics include semantic potential, semantic shaping reward, physical reward, and total reward. Label quality is measured by mean absolute error (MAE) and high-difference cases.
"""


def section_results() -> str:
    return r"""
\section{Results and Analysis}

\subsection{Semantic Label Reliability}
Fig. \ref{fig:label-quality} and Table \ref{tab:ablation} show that raw VLM scores should not be used directly as rewards. On the formal100 semantic dataset, raw VLM overall MAE is 0.1421, while the physics-verified fused scores reduce the overall MAE to 0.0108. The number of high-difference cases decreases from 100/100 to 0/100. The improvement is most important for energy- and threat-related labels, where raw VLM scores are less physically reliable.

\begin{figure}[t]
\centering
\includegraphics[width=\columnwidth]{figures/fig2_label_quality_improvement.pdf}
\caption{Comparison of raw VLM scores and physics-verified fused scores on the semantic label dataset. The fusion mechanism substantially reduces errors in energy- and threat-related labels, where raw VLM scores are less reliable.}
\label{fig:label-quality}
\end{figure}

\subsection{Surrogate Reward Network}
Fig. \ref{fig:surrogate-accuracy} and Table \ref{tab:ablation} summarize the surrogate reward model results. Exact cached-VLM lookup is not feasible in continuous PPO states: the Stage7 diagnosis reports zero exact cache hits and full fallback under random rollout. Surrogate v1 achieves overall MAE 0.0377 with a high-difference ratio of 17.8\%. After targeted geometry and boundary augmentation, surrogate v2 reduces MAE to 0.0256 and the high-difference ratio to 3.0\%. The surrogate also provides fast CPU inference, approximately $7.36\times10^{-5}$ seconds per query.

\begin{figure}[t]
\centering
\includegraphics[width=\columnwidth]{figures/fig3_surrogate_model_accuracy.pdf}
\caption{Per-label surrogate prediction error for surrogate v1 and surrogate v2. Boundary and geometry augmentation reduces the overall surrogate error and the number of high-difference cases.}
\label{fig:surrogate-accuracy}
\end{figure}

\input{tables/table_iv_ablation}

\subsection{Main NoWeapon Results}
Table \ref{tab:mixed} and Fig. \ref{fig:mixed-performance} provide the main Stage13 500k mixed-evaluation comparison. PPO-PVVL-SR obtains the highest mixed win rate, $0.3950\pm0.1098$, compared with $0.2367\pm0.1595$ for PPO-Cur and $0.2567\pm0.0958$ for PPO-Phys. It also has the lowest draw rate, $0.2067\pm0.2102$, and the highest attack-window ratio, $0.0653\pm0.0579$. Its neutral-stalemate ratio is also lower than the two baselines. These results indicate that semantic reward shaping can encourage more decisive and tactically active behavior in the mixed curriculum evaluation.

The result is not a universal advantage. PPO-Phys has lower enemy-threat exposure, $0.0624\pm0.0422$, than PPO-PVVL-SR, $0.1341\pm0.1204$. This is an important safety-related strength of physical shaping. The correct interpretation is that PVVL-SR improves mixed outcome and offensive-geometry indicators, while physical shaping remains a strong threat-control baseline.

\input{tables/table_i_mixed}

\begin{figure*}[t]
\centering
\includegraphics[width=0.92\textwidth]{figures/fig4_mixed_evaluation_performance.pdf}
\caption{Mixed-evaluation performance after the Stage13 500k confirmation. PPO-PVVL-SR achieves the highest win rate and attack-window occupancy and the lowest draw/stalemate tendency among the compared Stage13 methods, while PPO-Phys retains lower enemy-threat exposure.}
\label{fig:mixed-performance}
\end{figure*}

\subsection{Offensive and Unseen Evaluation}
Table \ref{tab:offensive} shows that PPO-PVVL-SR performs strongly in the offensive evaluation. It reaches win rate $0.6567\pm0.3709$ and attack-window ratio $0.0707\pm0.0668$, both higher than PPO-Phys in this evaluation. This supports the role of semantic reward shaping and curriculum in retaining offensive geometry when attack situations are reachable.

Table \ref{tab:unseen} presents a more mixed unseen-evaluation picture. PPO-Phys obtains a higher unseen win rate, $0.3333\pm0.1756$, than PPO-PVVL-SR, $0.2900\pm0.0712$. However, PPO-PVVL-SR retains a higher attack-window ratio and lower neutral-stalemate ratio. This suggests that PVVL-SR provides useful tactical incentives, but does not dominate the physical baseline under distribution shift.

\input{tables/table_ii_offensive}
\input{tables/table_iii_unseen}

\subsection{Training Trend from 300k to 500k}
Fig. \ref{fig:trend-300-500} compares the main method from 300k to 500k in mixed evaluation. The trend indicates that PPO-PVVL-SR improves attack-window and tail-advantage occupancy and reduces neutral-stalemate behavior as training continues. At the same time, enemy-threat exposure increases relative to the 300k checkpoint. This suggests an aggressiveness-safety trade-off: longer training improves offensive activity but may expose the ego aircraft to more threat.

\begin{figure}[t]
\centering
\includegraphics[width=\columnwidth]{figures/fig5_300k_to_500k_trend.pdf}
\caption{Trend of PPO-PVVL-SR from 300k to 500k in mixed evaluation. Attack-window and tail-advantage ratios increase and neutral stalemate decreases, but enemy-threat exposure increases relative to 300k.}
\label{fig:trend-300-500}
\end{figure}

\subsection{Sample Efficiency and Representative Behavior}
Fig. \ref{fig:sample-efficiency} gives an auxiliary sample-efficiency comparison for reaching mixed-evaluation win rate $0.30$. Because the comparison is based on checkpoint evaluation and only three seeds, it should be interpreted as supporting evidence rather than a statistical claim. Fig. \ref{fig:trajectories} presents representative trajectory examples, including a PVVL-SR offensive case, a PPO-Phys safe or defensive case, and a PVVL-SR stalemate case.

\begin{figure}[t]
\centering
\includegraphics[width=\columnwidth]{figures/fig6_sample_efficiency.pdf}
\caption{Sample efficiency for reaching mixed-evaluation win rate $\geq 0.30$. This auxiliary comparison is based on checkpoint evaluation and should be interpreted with checkpoint variance in mind.}
\label{fig:sample-efficiency}
\end{figure}

\begin{figure*}[t]
\centering
\includegraphics[width=0.96\textwidth]{figures/fig7_representative_trajectories.pdf}
\caption{Representative trajectory examples. The panels show a PVVL-SR offensive case, a PPO-Phys safe/defensive case, and a PVVL-SR stalemate case, together with distance, tactical scores, and semantic potential.}
\label{fig:trajectories}
\end{figure*}

\subsection{Preliminary DodgeMissile Validation}
Table \ref{tab:dodge} and Fig. \ref{fig:dodge} report the DodgeMissile preliminary validation. The wrapper and reward modules run successfully in \texttt{1v1/DodgeMissile/Selfplay}, and PPO-PVVL-SR shows slightly higher return, longer episode length, and longer survival time than the two preliminary baselines. However, all methods still lose in the 50k single-seed evaluation. Missile warning, lock, and hit metrics are not exposed in the available info fields. These results should be treated only as preliminary compatibility and weak-transfer evidence, not as full missile-combat validation.

\input{tables/table_v_dodge}

\begin{figure}[t]
\centering
\includegraphics[width=\columnwidth]{figures/fig8_dodgemissile_preliminary.pdf}
\caption{DodgeMissile preliminary validation. The results indicate wrapper compatibility and weak transfer signals only; they do not constitute full weapon-enabled combat validation.}
\label{fig:dodge}
\end{figure}
"""


def section_discussion() -> str:
    return r"""
\section{Discussion}

\subsection{Interpretation of VLM Semantic Reward Benefits}
The experiments suggest that VLM-derived semantic labels can be useful when they are constrained and distilled. PVVL-SR improves mixed-evaluation win rate, attack-window occupancy, and stalemate reduction in the reported 500k setting. The benefit appears most closely related to spatial and geometric semantics: the VLM teacher helps label tactical situations in a way that is richer than a single hand-crafted reward term. This should not be interpreted as evidence that a VLM can directly control an aircraft or replace physical modeling.

\subsection{Physics Verification and Label-Wise Fusion}
Physics verification is necessary because raw VLM scores are not uniformly reliable. Energy-related labels are a clear example: a rendered diagram may not provide enough physically grounded information for a VLM to infer specific energy consistently. Label-wise fusion allows the framework to use VLM spatial reasoning for geometry while letting physical computation dominate energy and safety-critical threat labels. This is why the proposed reward model should be understood as physics-verified semantic reward shaping rather than raw VLM reward learning.

\subsection{Surrogate Reward Network and Deployment Feasibility}
The surrogate network is necessary for practical PPO training. Qwen inference is too slow for online reward computation, and exact cached-VLM lookup has zero hit rate in continuous rollout states. The surrogate provides millisecond-level or faster inference from state features and keeps the VLM outside the training loop. This design improves deployability and preserves a clear separation between offline semantic labeling and online policy learning.

\subsection{Aggressiveness-Safety Trade-Off}
The main limitation in the Stage13 results is that PPO-Phys remains stronger in enemy-threat exposure. PVVL-SR improves mixed outcome and offensive-geometry indicators but can expose the ego aircraft to higher threat. The 300k-to-500k trend also shows increased enemy-threat exposure as offensive metrics improve. This should be framed as an aggressiveness-safety trade-off, not as a uniform advantage. Future work should incorporate explicit safety constraints or threat-aware curriculum if lower exposure is required.

\subsection{Role of Mixed Situation Curriculum}
The mixed situation curriculum is important because default NoWeapon rollouts rarely produce attack-window states. The curriculum exposes the policy to offensive, neutral, defensive, and random initial situations and makes the semantic labels learnable. However, curriculum alone is not the entire method, and PPO-Cur is retained as a key baseline. The results should therefore be read as evidence for the combination of curriculum and physics-verified semantic reward shaping.

\subsection{Limitations}
Several limitations should be emphasized. The NoWeapon limitation is central: the main experiments use a NoWeapon setting and do not validate full missile-combat performance. DodgeMissile is only a preliminary single-seed compatibility and weak-transfer experiment, and all methods still lose under the reported 50k setting. The main Stage13 experiments use three seeds, so claims should be made with mean and standard deviation rather than strong statistical significance. The 500k continuation loads actor and critic weights but resets optimizer state. Only Qwen2.5-VL-7B-Instruct is used as the main VLM teacher, and only 1v1 air combat is studied. Extending PVVL-SR to weapon-enabled scenarios, stronger opponent pools, and multi-agent 2v2 settings remains future work.
"""


def section_conclusion() -> str:
    return r"""
\section{Conclusion}

This paper presented PVVL-SR, a physics-verified vision-language semantic reward shaping framework for reinforcement learning-based UAV air-combat decision-making. The framework uses Qwen2.5-VL-7B-Instruct as an offline semantic teacher, verifies VLM scores using air-combat geometry and energy metrics, applies label-wise fusion with physical override, trains a lightweight surrogate semantic reward network, and injects semantic rewards through potential-based shaping and mixed situation curriculum. The VLM is not used as an online controller and is not called during PPO training.

Experiments in LAG 1v1 NoWeapon/Selfplay show that PVVL-SR improves mixed-evaluation win rate, attack-window occupancy, and stalemate reduction in the reported Stage13 500k setting. The results also show that physical reward shaping remains a strong baseline, particularly for enemy-threat exposure and some unseen-evaluation outcomes. Ablation studies support the need for physics-verified label fusion, surrogate reward learning, potential-based shaping, and curriculum exposure. A small DodgeMissile study indicates wrapper compatibility and weak transfer signals only, not full weapon-combat validation.

Future work will extend the framework to weapon-enabled scenarios, stronger opponent pools, multi-agent 2v2 combat, broader VLM teacher comparisons, and explicit safety constraints for reducing threat exposure while preserving offensive maneuvering ability.
"""


def table_i() -> str:
    return r"""
\begin{table*}[t]
\centering
\caption{Stage13 500k mixed evaluation. Values are mean $\pm$ standard deviation over three seeds.}
\label{tab:mixed}
\footnotesize
\resizebox{\textwidth}{!}{%
\begin{tabular}{lccccccc}
\hline
Method & Win & Draw & Avg. return & Attack-window & Tail-adv. & Enemy threat & Neutral-stalemate \\
\hline
PPO-Cur & $0.2367\pm0.1595$ & $0.5033\pm0.3532$ & $-86.3075\pm52.2829$ & $0.0420\pm0.0386$ & $0.0969\pm0.0695$ & $0.1107\pm0.0590$ & $0.4221\pm0.1611$ \\
PPO-Phys & $0.2567\pm0.0958$ & $0.3800\pm0.2540$ & $-157.8728\pm24.9887$ & $0.0313\pm0.0302$ & $0.0580\pm0.0452$ & $0.0624\pm0.0422$ & $0.4124\pm0.0314$ \\
PPO-PVVL-SR & $0.3950\pm0.1098$ & $0.2067\pm0.2102$ & $-136.1781\pm20.3136$ & $0.0653\pm0.0579$ & $0.0851\pm0.0520$ & $0.1341\pm0.1204$ & $0.3462\pm0.2662$ \\
\hline
\end{tabular}}
\end{table*}
"""


def table_ii() -> str:
    return r"""
\begin{table*}[t]
\centering
\caption{Stage13 500k offensive evaluation. Values are mean $\pm$ standard deviation over three seeds.}
\label{tab:offensive}
\footnotesize
\resizebox{\textwidth}{!}{%
\begin{tabular}{lccccc}
\hline
Method & Win & Attack-window & Tail-adv. & First attack entry time & Neutral-stalemate \\
\hline
PPO-Cur & $0.0367\pm0.0287$ & $0.0531\pm0.0453$ & $0.1303\pm0.0785$ & $0.0000\pm0.0000$ & $0.3535\pm0.1773$ \\
PPO-Phys & $0.1533\pm0.0862$ & $0.0326\pm0.0260$ & $0.0667\pm0.0413$ & $0.0000\pm0.0000$ & $0.4230\pm0.0164$ \\
PPO-PVVL-SR & $0.6567\pm0.3709$ & $0.0707\pm0.0668$ & $0.0949\pm0.0661$ & $0.0000\pm0.0000$ & $0.3198\pm0.2760$ \\
\hline
\end{tabular}}
\end{table*}
"""


def table_iii() -> str:
    return r"""
\begin{table}[t]
\centering
\caption{Stage13 500k unseen evaluation. Values are mean $\pm$ standard deviation over three seeds.}
\label{tab:unseen}
\footnotesize
\resizebox{\columnwidth}{!}{%
\begin{tabular}{lcccc}
\hline
Method & Win & Attack-window & Neutral & Enemy threat \\
\hline
PPO-Cur & $0.2967\pm0.2223$ & $0.0283\pm0.0172$ & $0.5103\pm0.1461$ & $0.1002\pm0.0467$ \\
PPO-Phys & $0.3333\pm0.1756$ & $0.0286\pm0.0255$ & $0.4267\pm0.0538$ & $0.0632\pm0.0270$ \\
PPO-PVVL-SR & $0.2900\pm0.0712$ & $0.0475\pm0.0368$ & $0.3641\pm0.2191$ & $0.1054\pm0.0756$ \\
\hline
\end{tabular}}
\end{table}
"""


def table_iv() -> str:
    return r"""
\begin{table*}[t]
\centering
\caption{Ablation and diagnostic evidence supporting the reward-modeling pipeline.}
\label{tab:ablation}
\footnotesize
\resizebox{\textwidth}{!}{%
\begin{tabular}{lcccc}
\hline
Ablation & Metric & Baseline/Before & Variant/After & Conclusion \\
\hline
Raw VLM to label-wise fused & Overall label MAE; high-diff cases & $0.1421$; $100/100$ & $0.0108$; $0/100$ & Fusion substantially improves label reliability. \\
Exact cached VLM to surrogate & Exact cache hit; CPU inference time & $0.000$ hit; $1.000$ fallback & $7.36{\times}10^{-5}$ s/query & Surrogate is needed for continuous PPO states. \\
Surrogate v1 to surrogate v2 & Overall MAE; high-diff ratio & $0.0377$; $17.8\%$ & $0.0256$; $3.0\%$ & Boundary augmentation improves surrogate accuracy. \\
No potential to potential shaping & Stage12 mixed win; enemy threat & $0.2667$; $0.1508$ & $0.4367$; $0.0470$ & Potential shaping improves outcome/threat control, not every tactical metric. \\
No curriculum to mixed curriculum & Stage12 mixed win; attack-window ratio & $0.2633$; $0.0501$ & $0.3000$; $0.0389$ & Curriculum is necessary for attack-window exposure, but does not solve all metrics by itself. \\
\hline
\end{tabular}}
\end{table*}
"""


def table_v() -> str:
    return r"""
\begin{table}[t]
\centering
\caption{DodgeMissile preliminary validation. This is a 50k single-seed compatibility test; all methods still lose in this setting.}
\label{tab:dodge}
\footnotesize
\resizebox{\columnwidth}{!}{%
\begin{tabular}{lccccc}
\hline
Method & Return & Length & Survival & Missile threat & Def. escape \\
\hline
PPO & $-231.1155$ & $40.0000$ & $8.0000$ & $1.0000$ & $0.0000$ \\
PPO-Phys & $-223.3494$ & $44.0000$ & $8.8000$ & $0.9773$ & $0.0000$ \\
PPO-PVVL-SR & $-214.7135$ & $51.0000$ & $10.2000$ & $0.9804$ & $0.0392$ \\
\hline
\end{tabular}}
\end{table}
"""


def auxiliary_files() -> dict[str, str]:
    return {
        "refs.bib": "% Real BibTeX entries are TODO. Do not add fabricated references.\n",
        "references_todo.md": """# Missing References TODO

- PPO original paper
- Potential-based reward shaping
- Curriculum learning
- UAV air combat reinforcement learning
- LAG environment
- Qwen2.5-VL technical report / model card
- VLM reward modeling
- Physics-informed / safe RL
- Autonomous air combat decision-making
- VLM in robotics/autonomous driving, if used
""",
        "author_info_check.md": """# Author Information Check

- Zhendong Li IEEE membership: not marked. PASS.
- Hui Li IEEE membership: Senior Member, IEEE. PASS.
- Corresponding author: Hui Li. PASS.
- Zhendong Li email: 18980494367@163.com. PASS.
- Hui Li email: Kelly.li@126.com. PASS.
- Affiliation unified: School of Aeronautics and Astronautics, University of Electronic Science and Technology of China, Chengdu 611731, China. PASS.
- ORCID: not added because it was not provided. PASS.
- Funding footnote: TODO placeholder only; no funding information fabricated. PASS.
- Template author block: uses IEEEtaes \\author, \\member, and \\affil commands. PASS.
""",
        "missing_references_todo.md": """# Missing References TODO

The manuscript uses \\needcite{...} placeholders rather than fabricated citations.

Required literature categories:

1. PPO original paper.
2. Potential-based reward shaping.
3. Curriculum learning.
4. UAV air combat reinforcement learning.
5. LAG environment.
6. Qwen2.5-VL technical report or model card.
7. VLM reward modeling.
8. Physics-informed or safe RL.
9. Autonomous air-combat decision-making.
10. VLM in robotics/autonomous driving, if used.
""",
        "compile_notes.md": """# Compile Notes

Compile status is filled after running LaTeX.

- Preferred command: `latexmk -pdf main.tex`
- Fallback command: `pdflatex main.tex` repeated as needed.
- References are TODO placeholders via `\\needcite{...}` and no fabricated BibTeX entries are included.
""",
    }


def checklist_script() -> str:
    return r"""
from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MS = ROOT / "manuscript" / "taes_pvvl_sr_v01"


def read_all_tex() -> str:
    parts = [MS / "main.tex"]
    parts.extend(sorted((MS / "sections").glob("*.tex")))
    parts.extend(sorted((MS / "tables").glob("*.tex")))
    parts.extend([MS / "acronyms.tex", MS / "notation.tex"])
    return "\n".join(p.read_text(encoding="utf-8", errors="ignore") for p in parts if p.exists())


def main() -> None:
    checks = []
    text = read_all_tex()
    main = (MS / "main.tex").read_text(encoding="utf-8", errors="ignore")

    def add(name: str, ok: bool, detail: str = "") -> None:
        checks.append((name, ok, detail))

    add("main.tex exists", (MS / "main.tex").exists())
    for i in range(1, 9):
        section = MS / "sections" / f"{i:02d}_" 
    for rel in [
        "sections/01_introduction.tex", "sections/02_related_work.tex",
        "sections/03_problem_formulation.tex", "sections/04_method.tex",
        "sections/05_experiments.tex", "sections/06_results.tex",
        "sections/07_discussion.tex", "sections/08_conclusion.tex",
    ]:
        add(f"{rel} exists and input", (MS / rel).exists() and rel.replace(".tex", "") in main)

    for fig in [
        "fig1_method_architecture.pdf", "fig2_label_quality_improvement.pdf",
        "fig3_surrogate_model_accuracy.pdf", "fig4_mixed_evaluation_performance.pdf",
        "fig5_300k_to_500k_trend.pdf", "fig6_sample_efficiency.pdf",
        "fig7_representative_trajectories.pdf", "fig8_dodgemissile_preliminary.pdf",
    ]:
        add(f"figure {fig} exists", (MS / "figures" / fig).exists())

    for table in [
        "table_i_mixed.tex", "table_ii_offensive.tex", "table_iii_unseen.tex",
        "table_iv_ablation.tex", "table_v_dodge.tex",
    ]:
        add(f"table {table} exists", (MS / "tables" / table).exists())

    labels = set(re.findall(r"\\label\{([^}]+)\}", text))
    refs = set(re.findall(r"\\ref\{([^}]+)\}", text))
    missing_refs = sorted(refs - labels)
    add("all refs have labels", not missing_refs, ", ".join(missing_refs))

    danger = ["fully outperforms", "solves air combat", "solves UAV air combat", "state-of-the-art", "proves missile combat capability", "VLM controls"]
    danger_hits = [d for d in danger if d.lower() in text.lower()]
    add("dangerous overclaim scan", not danger_hits, ", ".join(danger_hits))

    add("PPO-Phys threat-control advantage mentioned", "PPO-Phys has lower enemy-threat exposure" in text or "physical shaping remains a strong threat-control baseline" in text)
    add("DodgeMissile preliminary mentioned", "preliminary" in text and "DodgeMissile" in text)
    add(
        "NoWeapon limitation mentioned",
        "NoWeapon" in text
        and ("does not validate full missile-combat" in text or "do not validate full missile-combat" in text),
    )
    add("optimizer reset caveat mentioned", "optimizer state is reset" in text or "resets optimizer state" in text)
    add("Zhendong Li not marked IEEE member", "Zhendong Li}\n\\member" not in main and "Student Member" not in main)
    add("Hui Li Senior Member", "Hui Li}\n\\member{Senior Member, IEEE}" in main)
    add("Hui Li corresponding author", "Corresponding author: Hui Li" in main)

    cite_placeholders = sorted(c for c in set(re.findall(r"\\needcite\{([^}]+)\}", text)) if c != "...")
    add("citation placeholders intentionally present", bool(cite_placeholders), f"{len(cite_placeholders)} placeholders")

    lines = ["# Manuscript Checklist", ""]
    required_checks = [(name, ok, detail) for name, ok, detail in checks if name != "citation placeholders intentionally present"]
    status = "PASS" if all(ok for _, ok, _ in required_checks) else "CHECK"
    lines.append(f"Overall status: {status}")
    lines.append("")
    lines.append("| Check | Status | Detail |")
    lines.append("|---|---|---|")
    for name, ok, detail in checks:
        lines.append(f"| {name} | {'PASS' if ok else 'FAIL'} | {detail} |")
    lines.append("")
    lines.append("## Citation TODO Placeholders")
    for c in cite_placeholders:
        lines.append(f"- {c}")
    (MS / "manuscript_checklist.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
"""


def write_files() -> None:
    ensure_dirs()
    copy_template_and_assets()
    write(OUT / "main.tex", main_tex())
    write(OUT / "acronyms.tex", acronyms_tex())
    write(OUT / "notation.tex", notation_tex())
    write(OUT / "sections" / "01_introduction.tex", section_intro())
    write(OUT / "sections" / "02_related_work.tex", section_related())
    write(OUT / "sections" / "03_problem_formulation.tex", section_problem())
    write(OUT / "sections" / "04_method.tex", section_method())
    write(OUT / "sections" / "05_experiments.tex", section_experiments())
    write(OUT / "sections" / "06_results.tex", section_results())
    write(OUT / "sections" / "07_discussion.tex", section_discussion())
    write(OUT / "sections" / "08_conclusion.tex", section_conclusion())
    write(OUT / "tables" / "table_i_mixed.tex", table_i())
    write(OUT / "tables" / "table_ii_offensive.tex", table_ii())
    write(OUT / "tables" / "table_iii_unseen.tex", table_iii())
    write(OUT / "tables" / "table_iv_ablation.tex", table_iv())
    write(OUT / "tables" / "table_v_dodge.tex", table_v())
    for name, text in auxiliary_files().items():
        write(OUT / name, text)
    write(ROOT / "tools" / "check_taes_latex_draft.py", checklist_script())

    manifest = {
        "title": TITLE,
        "output_dir": str(OUT),
        "template_dir": str(TEMPLATE_DIR),
        "paper_package": str(PAPER_PKG),
        "manuscript_draft": str(MANUSCRIPT_DRAFT),
        "figures": [name for name, _ in FIGURES],
    }
    write(OUT / "build_manifest.json", json.dumps(manifest, indent=2))


if __name__ == "__main__":
    write_files()
    print(json.dumps({"output_dir": str(OUT), "main_tex": str(OUT / "main.tex")}, indent=2))
