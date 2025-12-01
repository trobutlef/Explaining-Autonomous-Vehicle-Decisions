import gymnasium as gym
import highway_env
import numpy as np
import torch
import shap
import cv2
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

# Monkeypatch HerReplayBuffer BEFORE importing ANY SB3 related modules
import sys
import stable_baselines3.common.buffers
from stable_baselines3 import HerReplayBuffer

class PatchedHerReplayBuffer(HerReplayBuffer):
    def __init__(self, *args, **kwargs):
        print(f"PatchedHerReplayBuffer init called with keys: {list(kwargs.keys())}")
        if 'online_sampling' in kwargs:
            print("Removing online_sampling argument")
            del kwargs['online_sampling']
        if 'max_episode_length' in kwargs:
            print("Removing max_episode_length argument")
            del kwargs['max_episode_length']
        super().__init__(*args, **kwargs)

print(f"Original HerReplayBuffer: {HerReplayBuffer}")
stable_baselines3.HerReplayBuffer = PatchedHerReplayBuffer
stable_baselines3.common.buffers.HerReplayBuffer = PatchedHerReplayBuffer
print(f"Patched HerReplayBuffer: {stable_baselines3.common.buffers.HerReplayBuffer}")

# Try to patch in 'her' module if it exists (old SB3 structure)
try:
    import stable_baselines3.her.her_replay_buffer
    stable_baselines3.her.her_replay_buffer.HerReplayBuffer = PatchedHerReplayBuffer
    print("Patched stable_baselines3.her.her_replay_buffer.HerReplayBuffer")
except ImportError:
    print("stable_baselines3.her.her_replay_buffer not found")

# Now import SB3/HuggingFace modules
from huggingface_sb3 import load_from_hub
from sb3_contrib import TQC

def render_continuous_explanation(frames, actions, saliencies, feature_names, output_path="outputs/parking_explanation.mp4", fps=15):
    """
    Render video for continuous action explanations (e.g. Throttle, Steering).
    saliencies: List of length T. Each element is a list of arrays [shap_throttle, shap_steering].
    """
    if not frames:
        print("No frames to render.")
        return

    height, width, _ = frames[0].shape
    
    # Layout: Frame on top, 2 plots below (Throttle, Steering)
    out_height = height + 400
    out_width = max(width, 800)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output_path, fourcc, fps, (out_width, out_height))
    
    fig, axes = plt.subplots(1, 2, figsize=(out_width/100, 4), dpi=100)
    
    for i, (frame, action, saliency_list) in enumerate(zip(frames, actions, saliencies)):
        # 1. Prepare Frame
        img_canvas = np.zeros((out_height, out_width, 3), dtype=np.uint8)
        h_offset = 0
        w_offset = (out_width - width) // 2
        img_canvas[h_offset:h_offset+height, w_offset:w_offset+width] = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        # 2. Plots
        action_names = ["Acceleration", "Steering"]
        
        for ax, val, act_val, name in zip(axes, saliency_list, action, action_names):
            ax.clear()
            # val is array of shape (features,)
            colors = ['red' if v > 0 else 'blue' for v in val]
            y_pos = np.arange(len(feature_names))
            
            ax.barh(y_pos, val, align='center', color=colors)
            ax.set_yticks(y_pos)
            ax.set_yticklabels(feature_names)
            ax.invert_yaxis()
            ax.set_title(f"{name}: {act_val:.2f}")
            ax.set_xlabel("Impact")
            
        # Draw
        canvas = FigureCanvas(fig)
        canvas.draw()
        plot_img = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')
        plot_img = plot_img.reshape(canvas.get_width_height()[::-1] + (3,))
        plot_img = cv2.cvtColor(plot_img, cv2.COLOR_RGB2BGR)
        
        # Resize and place
        plot_h, plot_w, _ = plot_img.shape
        scale = out_width / plot_w
        new_plot_h = int(plot_h * scale)
        plot_img_resized = cv2.resize(plot_img, (out_width, new_plot_h))
        
        remaining_h = out_height - height
        if new_plot_h > remaining_h:
            plot_img_resized = plot_img_resized[:remaining_h, :]
            
        img_canvas[height:height+plot_img_resized.shape[0], :] = plot_img_resized
        
        video.write(img_canvas)
        
    video.release()
    plt.close(fig)
    print(f"Saved video to {output_path}")

class PaddingWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        # Extend observation space by 1
        old_space = env.observation_space
        low = np.concatenate([old_space['observation'].low, [0.0]])
        high = np.concatenate([old_space['observation'].high, [1.0]])
        self.observation_space = gym.spaces.Dict({
            'observation': gym.spaces.Box(low=low, high=high, dtype=np.float32),
            'achieved_goal': gym.spaces.Box(low=old_space['achieved_goal'].low, high=old_space['achieved_goal'].high, dtype=np.float32),
            'desired_goal': gym.spaces.Box(low=old_space['desired_goal'].low, high=old_space['desired_goal'].high, dtype=np.float32),
        })
        
    def observation(self, obs):
        # Pad 'observation' with 0.0 and cast all to float32
        new_obs = obs.copy()
        new_obs['observation'] = np.concatenate([obs['observation'], [0.0]]).astype(np.float32)
        new_obs['achieved_goal'] = obs['achieved_goal'].astype(np.float32)
        new_obs['desired_goal'] = obs['desired_goal'].astype(np.float32)
        return new_obs

def main():
    # 2. Setup Env
    env = gym.make("parking-v0", render_mode='rgb_array')
    env = PaddingWrapper(env)

    # 1. Load Model
    print("Loading pretrained model...")
    checkpoint = load_from_hub("sb3/tqc-Parking-v0", "tqc-parking-v0.zip")
    model = TQC.load(checkpoint, env=env)
    
    # 3. Run Episode
    print("Running episode...")
    obs, info = env.reset()
    done = False
    truncated = False
    
    frames = []
    actions = []
    observations = []
    
    while not (done or truncated):
        frames.append(env.render())
        observations.append(obs)
        
        # Predict
        action, _ = model.predict(obs, deterministic=True)
        actions.append(action)
        
        obs, reward, done, truncated, info = env.step(action)
        
    print(f"Episode finished. Steps: {len(frames)}")
    
    # 4. SHAP Explanation
    print("Computing SHAP values...")
    
    def flatten_obs(obs_dict):
        return np.concatenate([
            obs_dict['observation'],
            obs_dict['achieved_goal'],
            obs_dict['desired_goal']
        ])
        
    flat_obs = np.array([flatten_obs(o) for o in observations])
    print(f"Flat obs shape: {flat_obs.shape}")
    
    def predict_wrapper(flat_x):
        n_samples = flat_x.shape[0]
        obs_dim = 7
        goal_dim = 6
        
        batch_obs = {
            'observation': flat_x[:, :obs_dim].astype(np.float32),
            'achieved_goal': flat_x[:, obs_dim:obs_dim+goal_dim].astype(np.float32),
            'desired_goal': flat_x[:, obs_dim+goal_dim:].astype(np.float32)
        }
        
        actions, _ = model.predict(batch_obs, deterministic=True)
        return actions

    bg = flat_obs[np.random.choice(flat_obs.shape[0], min(50, len(flat_obs)), replace=False)]
    
    explainer = shap.KernelExplainer(predict_wrapper, bg)
    
    # Subsample for explanation to speed up demo
    step = 5
    indices = np.arange(0, len(flat_obs), step)
    flat_obs_subset = flat_obs[indices]
    
    print(f"Explaining {len(flat_obs_subset)} samples (subsampled)...")
    shap_vals = explainer.shap_values(flat_obs_subset)
    
    # Re-organize
    per_frame_saliencies = []
    # Fill in missing frames with previous saliency or None
    # Actually, let's just render the subsampled frames or interpolate.
    # For simplicity, let's just render the subsampled frames in the video.
    
    # We need to filter frames and actions too
    frames_subset = [frames[i] for i in indices]
    actions_subset = [actions[i] for i in indices]
    
    for i in range(len(flat_obs_subset)):
        sals = [shap_vals[0][i], shap_vals[1][i]]
        per_frame_saliencies.append(sals)
        
    feature_names = [
        "Ego_x", "Ego_y", "Ego_vx", "Ego_vy", "Ego_cos", "Ego_sin", "Pad",
        "Ach_x", "Ach_y", "Ach_vx", "Ach_vy", "Ach_cos", "Ach_sin",
        "Goal_x", "Goal_y", "Goal_vx", "Goal_vy", "Goal_cos", "Goal_sin"
    ]
            
    # 5. Render
    render_continuous_explanation(frames_subset, actions_subset, per_frame_saliencies, feature_names, fps=5)

if __name__ == "__main__":
    main()
