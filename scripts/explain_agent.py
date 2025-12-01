import sys
import os
from pathlib import Path
import numpy as np
import torch
import gymnasium as gym

# Add src to path
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

from src.utils.config import load_config
from src.envs.highway_env import make_env
from src.agents.dqn_agent import DQNAgent
from src.xai.shap_lime import explain_with_shap, explain_with_lime
from src.visualization.dashboard import render_episode_overview

def main():
    # 1. Load Config & Env
    config_path = project_root / "configs/config.yaml"
    cfg = load_config(str(config_path))
    
    # Force render mode to rgb_array for video capture
    cfg.environment.render = False 
    env = make_env(cfg)
    
    # 2. Load Agent
    obs_space = env.observation_space
    action_space = env.action_space
    agent = DQNAgent(cfg, obs_space, action_space)
    
    model_path = project_root / "outputs/dqn_highway_final.pt"
    if model_path.exists():
        print(f"Loading model from {model_path}")
        agent.load(str(model_path))
    else:
        print("No trained model found! Running with random weights.")

    # 3. Run Episode & Collect Data
    print("Running episode...")
    frames = []
    actions = []
    states = []
    
    state, _ = env.reset()
    done = False
    truncated = False
    
    while not (done or truncated):
        # Render frame
        frame = env.render()
        frames.append(frame)
        
        # Act
        action = agent.act(state, exploit=True)
        actions.append(action)
        states.append(state)
        
        # Step
        next_state, reward, done, truncated, _ = env.step(action)
        state = next_state

    print(f"Episode finished. Steps: {len(frames)}")
    
    # 4. Compute Explanations (SHAP)
    print("Computing SHAP values...")
    
    # Use a subset of states as background for SHAP
    states_tensor = torch.as_tensor(np.stack(states), dtype=torch.float32).to(agent.device)
    # Flatten states if needed (DQNAgent expects flattened input for the network, but handles it internally)
    # However, explain_with_shap expects the input to the model.
    # The DQNAgent.q_net expects flattened input.
    
    # We need to flatten the states to match QNetwork input
    states_flat = states_tensor.view(states_tensor.shape[0], -1)
    
    # Background: Random 50 samples from the episode (or less if short)
    bg_size = min(50, len(states))
    background_indices = np.random.choice(len(states), bg_size, replace=False)
    background_data = states_flat[background_indices]
    
    # Compute SHAP for all states in the episode
    # Note: DeepExplainer might be memory intensive. If so, loop.
    shap_values_list = []
    
    # explain_with_shap returns (shap_values, expected_value)
    # shap_values is a list of arrays (one for each output class)
    # We want the shap values for the *chosen* action
    
    # To avoid OOM, process in batches or one by one if needed. 
    # For < 100 steps, all at once is likely fine.
    s_vals, _ = explain_with_shap(agent.q_net, states_flat, background_data)
    
    # s_vals is list of [N, Features] arrays, one per action.
    # We want to select the array corresponding to the action taken at each step.
    
    final_saliencies = []
    for i, action in enumerate(actions):
        # s_vals[action] is [N, Features]
        # We want the i-th row
        final_saliencies.append(s_vals[action][i])
        
    # 5. Feature Names
    # Highway-env Kinematics: [presence, x, y, vx, vy] for each vehicle
    # Usually 5 vehicles.
    feature_names = []
    num_vehicles = states_flat.shape[1] // 5
    for v in range(num_vehicles):
        feature_names.extend([f"V{v}_Pres", f"V{v}_x", f"V{v}_y", f"V{v}_vx", f"V{v}_vy"])
        
    # 6. Generate Video
    output_video = project_root / "outputs/explanation_video.mp4"
    render_episode_overview(frames, actions, final_saliencies, feature_names, str(output_video))

if __name__ == "__main__":
    main()
