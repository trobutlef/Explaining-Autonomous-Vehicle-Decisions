# Explaining-Autonomous-Vehicle-Decisions

XAI for autonomous vehicle decisions using a lightweight simulator and standard explanation methods (Grad-CAM, SHAP/LIME).

## What we're building

- **Simulator-first (Option A)**: Train an agent in `highway-env` (Gymnasium) and generate explanations.
- **Perception explanations**: Grad-CAM on CNN features from simulator frames.
- **Action explanations**: SHAP/LIME on tabular state features feeding the policy/Q-network.

## Repo structure (who does what)

- **configs/**: YAML configs (env, agent, logging, XAI toggles).
- **scripts/run_train.sh**: Entrypoint to run training.
- **src/envs/highway_env.py**: Build and wrap the `highway-env` environment. [Owner: Env/Training]
- **src/agents/dqn_agent.py**: DQN agent (network, replay, act/learn, save/load). [Owner: Agent]
- **src/training/train_highway.py**: Training loop, eval, checkpoints, logging. [Owner: Env/Training]
- **src/utils/config.py**: Load YAML config. [Owner: Env/Training]
- **src/utils/logging.py**: Project logger. [Owner: Env/Training]
- **src/xai/grad_cam.py**: Grad-CAM for CNN perception. [Owner: XAI]
- **src/xai/shap_lime.py**: SHAP/LIME for state features. [Owner: XAI]
- **src/visualization/dashboard.py**: Overlay saliency on frames + action timeline. [Owner: XAI]
- **tests/**: Sanity tests and unit tests.

## Datasets

- **Primary (perception subset)**
  - BDD100K (use ~10k images subset): https://github.com/bdd100k/bdd100k | https://www.kaggle.com/datasets/solesensei/solesensei_bdd100k
  - KITTI (small detection subset): http://www.cvlibs.net/datasets/kitti/
  - nuScenes mini (≈7 GB): https://www.nuscenes.org/download
- **Simulator state (tabular)**
  - Logged directly from `highway-env` (e.g., speed, distances, lane index). Used for SHAP/LIME.

## How to run or setup

1. Create env and install deps

```
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Train (uses configs/config.yaml)

```
chmod +x scripts/run_train.sh
./scripts/run_train.sh
```

3. Generate Explanations (Hybrid XAI)

Runs the trained agent, computes SHAP values for actions, and generates a dashboard video.

```
python3 scripts/explain_agent.py
```
Output video will be saved to `outputs/explanation_video.mp4`.

## Tasks

- Env/Training
  - Implement `highway_env.make_env`
  - Implement `utils.config.load_config`, `utils.logging.get_logger`
  - Implement `DQNAgent` and `train_highway.py`
- XAI/Visualization
  - Implement `xai/grad_cam.py`, `xai/shap_lime.py`
  - Implement `visualization/dashboard.py`
  - Prepare a small BDD100K/KITTI/nuScenes-mini subset for Grad-CAM demos

## Simulator reference

- highway-env: https://github.com/Farama-Foundation/HighwayEnv

## Notes

- Keep explanations optional via `configs/config.yaml` (`xai.enable_*` flags).
- Save logs/checkpoints to `logs/` and `outputs/` (gitignored).
