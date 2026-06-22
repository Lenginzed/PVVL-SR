下面是一版适合 GitHub 开源首页的 `README.md`，你可以直接复制覆盖原来的内容。整体语气我写得比较克制，避免把论文结果和大模型权重一起绑定到开源仓库里。

```markdown
# PVVL-SR

**Physics-Verified Vision-Language Semantic Reward Shaping for Maneuvering UAV Air Combat Reinforcement Learning**

PVVL-SR is a research framework built on top of LAG-master for studying physics-verified semantic reward shaping in 1v1 UAV air-combat maneuvering tasks. The framework uses an offline vision-language model (VLM) teacher, physics-based tactical verification, label-wise semantic fusion, a lightweight surrogate semantic reward network, and potential-based reward shaping for PPO training.

This repository focuses on the algorithmic framework and code implementation. Large experiment logs, trained checkpoints, Qwen model weights, generated datasets, and manuscript artifacts are not included.

## Overview

PVVL-SR addresses a practical issue in reinforcement learning for air-combat maneuvering: manually designed physical rewards are interpretable but often limited in tactical semantics, while raw VLM feedback can be slow, physically inconsistent, and unsuitable for online control.

The framework follows this pipeline:

1. Extract ego/enemy states from the LAG 1v1 air-combat environment.
2. Compute air-combat geometry, energy, threat, and maneuvering metrics.
3. Render standardized 2D air-combat situation diagrams for offline VLM labeling.
4. Use Qwen2.5-VL-7B-Instruct as an offline semantic teacher.
5. Verify VLM scores with physics-based tactical labels.
6. Apply label-wise semantic fusion with physical override.
7. Train a lightweight MLP surrogate to predict fused semantic scores from state features.
8. Use the surrogate online during PPO training.
9. Convert semantic scores into a potential-based reward shaping term.
10. Combine environment reward, physical reward, and semantic shaping reward.

Qwen2.5-VL is **not** used as an online controller and is **not** called inside the PPO step loop.

## Key Features

- Physics-based air-combat tactical metric extraction
- Eight tactical semantic labels for 1v1 maneuvering combat
- Offline VLM semantic labeling interface
- Physics verification and label-wise semantic fusion
- Physical override for unreliable VLM labels
- Lightweight surrogate semantic reward network
- Potential-based semantic reward shaping
- Mixed situation curriculum for offensive, neutral, defensive, and random initial states
- PPO integration with configurable reward shaping switch
- Logging and diagnostic tools for reward, label, and curriculum analysis

## Tactical Semantic Labels

PVVL-SR uses eight semantic labels:

1. `ego_tail_advantage`
2. `enemy_tail_threat`
3. `effective_attack_window`
4. `enemy_missile_threat_zone`
5. `energy_advantage`
6. `energy_disadvantage`
7. `defensive_escape`
8. `neutral_stalemate`

Different labels use different fusion policies:

- Geometry labels use VLM + physics hybrid fusion.
- Energy labels are physical-dominant.
- Threat labels use hybrid fusion with physical override.
- If VLM and physics strongly disagree, physical override is triggered.

## Repository Structure

```text
.
├── algorithms/                         # PPO/MAPPO baseline algorithms from LAG
├── envs/                               # LAG/JSBSim environments and scenarios
├── lag_extensions/
│   └── reward_shaping/                 # PVVL-SR core implementation
│       ├── state_extractor.py          # Ego/enemy state extraction
│       ├── aircombat_metrics.py        # Geometry, range-rate, and energy metrics
│       ├── tactical_labels.py          # Physical tactical label scoring
│       ├── semantic_reward.py          # Verification, fusion, and reward shaping
│       ├── reward_wrapper.py           # Environment reward wrapper
│       ├── curriculum_scheduler.py     # Reward curriculum scheduler
│       ├── situation_curriculum.py     # Mixed situation reset curriculum
│       ├── situation_renderer.py       # 2D air-combat situation rendering
│       ├── vlm_prompt.py               # VLM prompt templates
│       ├── qwen_vl_scorer.py           # Offline Qwen2.5-VL inference interface
│       ├── vlm_cache.py                # VLM cache management
│       ├── semantic_surrogate.py       # MLP surrogate reward model
│       └── configs/                    # Reward shaping and surrogate configs
├── scripts/
│   └── train/
│       └── train_jsbsim.py             # PPO training entry
├── tools/                              # Dataset, labeling, training, evaluation tools
├── tests/                              # Smoke and unit tests
├── config.py                           # LAG configuration
├── README.md
└── LICENSE
```

## What Is Not Included

The following files are intentionally excluded from the public repository:

- Qwen2.5-VL model weights
- Trained PPO checkpoints
- Trained surrogate model checkpoints
- Generated air-combat semantic datasets
- VLM cache files
- Experiment logs and TensorBoard files
- Paper manuscript files and PDFs
- Large result folders under `scripts/results/`

If you want to reproduce the full experiment pipeline, you need to generate datasets, VLM labels, surrogate models, and PPO training logs locally.

## Installation

The project is based on LAG-master and JSBSim.

A typical setup is:

```bash
conda create -n pvvl_sr python=3.8
conda activate pvvl_sr
```

Install the core dependencies:

```bash
pip install torch numpy scipy pandas matplotlib pillow pyyaml scikit-learn
pip install gym==0.20.0 pymap3d geographiclib jsbsim==1.1.6
pip install wandb icecream setproctitle
```

Initialize JSBSim data if needed:

```bash
git submodule update --init --recursive
```

Optional dependencies for offline Qwen2.5-VL labeling:

```bash
pip install transformers accelerate safetensors qwen-vl-utils
```

Depending on your CUDA environment, you may need to install a compatible PyTorch build manually.

## Basic Usage

### 1. Original PPO Training

Run original PPO without reward shaping:

```bash
python scripts/train/train_jsbsim.py ^
  --env-name SingleCombat ^
  --algorithm-name ppo ^
  --scenario-name 1v1/NoWeapon/Selfplay
```

### 2. PPO with Physical Reward Shaping

```bash
python scripts/train/train_jsbsim.py ^
  --env-name SingleCombat ^
  --algorithm-name ppo ^
  --scenario-name 1v1/NoWeapon/Selfplay ^
  --use-reward-shaping ^
  --reward-shaping-config lag_extensions/reward_shaping/configs/physical_1v1.yaml
```

### 3. PPO with Surrogate Semantic Reward Shaping

This requires a trained surrogate model. By default, configs may point to:

```text
scripts/results/surrogate_models/stage9_surrogate_v2/
```

If this directory does not exist, train or provide a surrogate model first.

```bash
python scripts/train/train_jsbsim.py ^
  --env-name SingleCombat ^
  --algorithm-name ppo ^
  --scenario-name 1v1/NoWeapon/Selfplay ^
  --use-reward-shaping ^
  --reward-shaping-config lag_extensions/reward_shaping/configs/surrogate_v2_then_fusion_mixed_curriculum_1v1.yaml
```

## Offline VLM Labeling Pipeline

The VLM is used offline only.

### 1. Generate Air-Combat Situation Dataset

```bash
python tools/generate_aircombat_semantic_dataset.py ^
  --env-name SingleCombat ^
  --scenario-name 1v1/NoWeapon/Selfplay ^
  --num-samples 100 ^
  --output-dir scripts/results/aircombat_semantic_dataset
```

### 2. Label with Qwen2.5-VL

```bash
python tools/label_with_qwen_vl.py ^
  --dataset-dir scripts/results/aircombat_semantic_dataset ^
  --config lag_extensions/reward_shaping/configs/qwen2_5_vl_hybrid_1v1.yaml ^
  --cache scripts/results/vlm_cache/qwen2_5_vl_cache.jsonl ^
  --resume
```

For testing without loading Qwen:

```bash
python tools/label_with_qwen_vl.py ^
  --dataset-dir scripts/results/aircombat_semantic_dataset ^
  --config lag_extensions/reward_shaping/configs/qwen2_5_vl_hybrid_1v1.yaml ^
  --mock-vlm ^
  --resume
```

### 3. Evaluate VLM Label Quality

```bash
python tools/evaluate_vlm_labels.py ^
  --dataset-dir scripts/results/aircombat_semantic_dataset ^
  --cache scripts/results/vlm_cache/qwen2_5_vl_cache.jsonl ^
  --config lag_extensions/reward_shaping/configs/qwen2_5_vl_hybrid_1v1.yaml ^
  --output-dir scripts/results/vlm_label_eval
```

## Surrogate Reward Network

The surrogate network is a lightweight MLP that maps compact air-combat state features to fused semantic scores.

### Export Surrogate Dataset

```bash
python tools/export_semantic_surrogate_dataset.py ^
  --dataset-dir scripts/results/aircombat_semantic_dataset ^
  --cache scripts/results/vlm_cache/qwen2_5_vl_cache.jsonl ^
  --config lag_extensions/reward_shaping/configs/qwen2_5_vl_hybrid_1v1.yaml ^
  --output scripts/results/semantic_surrogate_dataset/surrogate_dataset.jsonl
```

### Train Surrogate MLP

```bash
python tools/train_semantic_surrogate.py ^
  --dataset scripts/results/semantic_surrogate_dataset/surrogate_dataset.jsonl ^
  --output-dir scripts/results/surrogate_models/surrogate_v1 ^
  --target-type fused_scores ^
  --epochs 400 ^
  --batch-size 32 ^
  --hidden-dim 128 ^
  --num-layers 3 ^
  --lr 1e-3 ^
  --seed 0
```

### Evaluate Surrogate

```bash
python tools/evaluate_semantic_surrogate.py ^
  --dataset scripts/results/semantic_surrogate_dataset/surrogate_dataset.jsonl ^
  --model-dir scripts/results/surrogate_models/surrogate_v1 ^
  --output-dir scripts/results/surrogate_eval
```

## Reward Formulation

PVVL-SR constructs a semantic potential function:

```text
Phi(s) = sum_i w_i * fused_score_i(s)
```

The potential-based shaping reward is:

```text
F(s_t, s_{t+1}) = gamma * Phi(s_{t+1}) - Phi(s_t)
```

The total PPO reward is:

```text
R_total = R_env + alpha * R_physical + beta * F(s_t, s_{t+1})
```

All terms are configurable and logged separately.

## Mixed Situation Curriculum

The mixed situation curriculum increases tactical-state coverage during training. It samples initial situations from:

- offensive advantage
- neutral merge
- defensive disadvantage
- random initialization

The curriculum is optional and only enabled through configuration. It does not modify the original LAG environment unless explicitly enabled.

## Important Notes

- Qwen2.5-VL is used as an offline semantic teacher only.
- The PPO policy does not depend on Qwen during deployment or online training.
- The surrogate reward network is lightweight and suitable for online PPO reward computation.
- The framework is designed for maneuvering-oriented 1v1 air-combat experiments.
- NoWeapon results should not be interpreted as full missile-combat validation.
- DodgeMissile support is preliminary and intended for compatibility checking.

## Recommended `.gitignore`

Large artifacts should not be committed:

```gitignore
models/
scripts/results/
manuscript/
TAES/
wandb/
runs/
results/

*.pt
*.pth
*.ckpt
*.safetensors
*.bin
*.onnx

*.jsonl
*.npy
*.npz
*.csv
*.acmi

*.pdf
*.aux
*.bbl
*.blg
*.log
*.out
*.toc
*.synctex.gz
```

## Citation

A manuscript based on this framework is in preparation.

```bibtex
@misc{pvvl_sr,
  title = {PVVL-SR: A Physics-Verified Vision-Language Semantic Reward Shaping Framework for Maneuvering UAV Air Combat Reinforcement Learning},
  author = {Li, Zhendong and Li, Hui},
  year = {2026},
  note = {Research code},
}
```

Please also cite the original LAG environment if you use this repository.

## Acknowledgement

This project builds on LAG-master, a JSBSim-based air-combat reinforcement learning environment. PVVL-SR extends it with physics-verified semantic reward modeling, offline VLM labeling, surrogate reward inference, and curriculum-based PPO training.

## License

This repository follows the license terms of the included project files. Please check `LICENSE` before redistribution or commercial use.
```
