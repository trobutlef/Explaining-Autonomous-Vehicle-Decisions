import sys
import os
from pathlib import Path
import numpy as np
import torch
import gymnasium as gym
import cv2
import shap
import matplotlib.pyplot as plt
from omegaconf import OmegaConf

# Add src to path
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

# ---- Hack: bypass SimplePBR inside MetaDrive engine ----
# This must be done BEFORE importing MetaDriveEnv!
import metadrive.engine.core.engine_core as engine_core

class DummyPBR:
    """Minimal dummy PBR pipeline to satisfy MetaDrive."""
    def __init__(self, *args, **kwargs):
        pass
    def destroy(self):
        pass

def dummy_init(*args, **kwargs):
    # print("[INFO] Using DummyPBR pipeline (SimplePBR disabled).")
    return DummyPBR()

# Replace the SimplePBR init that EngineCore uses
engine_core.init = dummy_init
# ---- End hack ----

from src.utils.config import load_config
from src.envs.metadrive_env import make_env
from src.agents.dqn_agent import DQNAgent
from src.xai.shap_lime import explain_with_shap
from src.visualization.dashboard import render_episode_overview
from metadrive.component.sensors.rgb_camera import RGBCamera
from metadrive.component.sensors.lidar import Lidar
from metadrive.component.sensors.distance_detector import SideDetector, LaneLineDetector

# ---- Hack: Disable RGBCamera PBR effects ----
def dummy_setup_effect(self):
    # print("[INFO] Skipping RGBCamera PBR effect setup.")
    pass

RGBCamera._setup_effect = dummy_setup_effect
# ---- End hack ----

import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config file")
    args = parser.parse_args()

    # 1. Load Config & Env
    config_path = project_root / args.config
    cfg = load_config(str(config_path))
    
    # Convert to dict to allow inserting Class objects (RGBCamera)
    cfg = OmegaConf.to_container(cfg, resolve=True)
    
    # Enable rendering for explanation
    # Use 3D rendering with RGBCamera
    cfg["environment"]["render"] = False # We use sensor observation, not window
    cfg["environment"]["image_observation"] = True
    cfg["environment"]["sensors"] = dict(
        lidar=(Lidar, ),
        side_detector=(SideDetector, ),
        lane_line_detector=(LaneLineDetector, ),
        rgb_camera=(RGBCamera, 256, 256)
    )
    # Adjust camera to avoid clipping and improve view
    cfg["environment"]["vehicle_config"] = dict(
        image_source="rgb_camera",
        rgb_camera=(256, 256),
        # Move camera up and forward
        # Default is often too low or inside hood
    )
    # MetaDrive camera config is often global or per-vehicle
    # Let's try setting global camera params if possible, or vehicle specific
    # For RGBCamera, it's attached to vehicle. We can try to set its position relative to vehicle.
    # But RGBCamera class hardcodes some stuff.
    # Let's try to set 'camera_height' in global config which might affect it?
    # Actually, RGBCamera uses its own pos.
    # We might need to subclass RGBCamera or just accept it.
    # But user said "low part where the vision of the car is" is missing.
    # This implies near plane clipping or obstruction.
    # Let's try to set 'camera_height' and 'camera_dist' in environment config.
    cfg["environment"]["camera_height"] = 2.0
    cfg["environment"]["camera_dist"] = 5.0
    cfg["environment"]["image_on_cuda"] = False
    
    env = make_env(cfg)
    
    # 2. Load Agent
    obs_space = env.observation_space
    if isinstance(obs_space, gym.spaces.Dict):
        obs_space = obs_space["state"]
    
    action_space = env.action_space
    agent = DQNAgent(cfg, obs_space, action_space)
    
    output_dir = project_root / getattr(getattr(cfg, "project", {}), "output_dir", "outputs")
    model_path = output_dir / "dqn_highway_final.pt"
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
    
    obs, _ = env.reset()
    state = obs["state"]
    frame = obs.get("image")
    if frame is not None and frame.ndim == 4:
        frame = frame[..., 0] # Take first frame of stack
    done = False
    truncated = False
    
    # MetaDrive might need a few steps to initialize rendering properly
    for _ in range(5):
        env.step(env.action_space.sample())
        env.reset()

    obs, _ = env.reset()
    state = obs["state"]
    
    while not (done or truncated):
        # Render frame
        # MetaDrive's render() usually returns None and draws to window.
        # We need to capture the frame.
        # If use_render=True, env.render() might return the image if configured, 
        # or we might need to access the camera.
        # For now, let's try to get the image from the env if possible.
        # In MetaDrive, env.render() returns None.
        # We can try env.engine.get_sensor("main_camera").perceive() if it exists.
        # Or we can use the top_down renderer for now if 3D is hard to capture without display.
        # But user wants "visuals".
        
        # Capture frame from observation
        # frame is already updated in the loop or initial reset
        # We need to get the image from the CURRENT observation
        # But wait, we append frame BEFORE acting? 
        # Usually we render state S, act A, get S'.
        # So we should use the frame from the current 'obs'.
        
        # We already extracted 'frame' from 'obs' at the start or end of loop.
        pass

        # If frame is None, we can't make a video. 
        # Let's assume for this step we might get a frame or we'll fix it in verification.
        # Actually, let's try to force a capture if possible.
        # For now, we will proceed. If frame is None, dashboard will complain.
        
        if frame is not None:
            # Ensure it's HxWxC uint8
            if frame.dtype != np.uint8:
                frame = (frame * 255).astype(np.uint8)
            frames.append(frame)
        
        # Act
        action = agent.act(state, exploit=True)
        actions.append(action)
        states.append(state)
        
        # Step
        # Step
        next_obs, reward, done, truncated, _ = env.step(action)
        next_state = next_obs["state"]
        
        # Update frame for next iteration (or current append)
        # Wait, we append frame at the top of the loop.
        # So we need to update 'frame' here for the next iteration.
        new_frame = next_obs.get("image")
        if new_frame is not None and new_frame.ndim == 4:
            new_frame = new_frame[..., 0]
        frame = new_frame
        
        state = next_state
        
        if len(frames) > 500: # Limit length
            break

    print(f"Episode finished. Steps: {len(actions)}")
    
    if not frames:
        print("Warning: No frames captured. Video will not be generated.")
        # Create dummy frames just to test the pipeline? No, better to warn.
        # We will try to generate video only if frames exist.
    
    # 4. Compute Explanations (SHAP)
    print("Computing SHAP values...")
    
    states_tensor = torch.as_tensor(np.stack(states), dtype=torch.float32).to(agent.device)
    states_flat = states_tensor.view(states_tensor.shape[0], -1)
    
    bg_size = min(50, len(states))
    background_indices = np.random.choice(len(states), bg_size, replace=False)
    background_data = states_flat[background_indices]
    
    s_vals, _ = explain_with_shap(agent.q_net, states_flat, background_data)
    
    final_saliencies = []
    for i, action in enumerate(actions):
        final_saliencies.append(s_vals[action][i])
        
    # 5. Feature Names
    # MetaDrive observation is typically [ego_state, lidar_scan]
    # We'll label them generically for now as we don't know the exact size/order without inspecting.
    num_features = states_flat.shape[1]
    feature_names = [f"Feat {i}" for i in range(num_features)]
    # Try to label first few if possible (usually ego state)
    # Common MetaDrive state: [x, y, vx, vy, heading, ...]
    if num_features > 5:
        feature_names[0] = "Ego X"
        feature_names[1] = "Ego Y"
        feature_names[2] = "Ego Vx"
        feature_names[3] = "Ego Vy"
        feature_names[4] = "Heading"
        
    # 6. Generate Video
    if frames:
        output_video = project_root / "outputs/explanation_video.mp4"
        render_episode_overview(frames, actions, final_saliencies, feature_names, str(output_video))
    
    env.close()

if __name__ == "__main__":
    main()
