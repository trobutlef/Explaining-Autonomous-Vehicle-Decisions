import sys
from pathlib import Path
from copy import deepcopy
import torch
import numpy as np

# Add project root to path
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

from src.utils.config import load_config
from src.envs.metadrive_env import make_env
from src.agents.dqn_agent import DQNAgent

def test_load():
    print("Project root:", project_root)
    
    cfg = load_config("configs/config.yaml")
    
    # Make a copy of cfg with render=True (mimic notebook)
    eval_cfg = deepcopy(cfg)
    eval_cfg.environment.render = True
    
    print("Creating environment...")
    try:
        env = make_env(eval_cfg)
    except Exception as e:
        print(f"Failed to create env: {e}")
        return

    obs_space = env.observation_space
    action_space = env.action_space
    
    print(f"Observation space: {obs_space}")
    print(f"Action space: {action_space}")
    
    if hasattr(obs_space, "shape"):
        print(f"Obs dim: {np.prod(obs_space.shape)}")
    
    print("Initializing agent...")
    try:
        agent = DQNAgent(cfg, obs_space, action_space)
    except Exception as e:
        print(f"Failed to init agent: {e}")
        return

    model_path = project_root / "outputs/dqn_highway_final.pt"
    print(f"Loading model from {model_path}...")
    
    if not model_path.exists():
        print("Model file not found!")
        return

    try:
        agent.load(str(model_path))
        print("Agent loaded successfully!")
    except Exception as e:
        print(f"Failed to load agent: {e}")
        # Print model structure to debug
        print("Current model state_dict keys and shapes:")
        for k, v in agent.q_net.state_dict().items():
            print(f"  {k}: {v.shape}")
            
        # Try to load checkpoint and inspect
        ckpt = torch.load(model_path, map_location="cpu")
        print("Checkpoint q_net keys and shapes:")
        if "q_net" in ckpt:
            for k, v in ckpt["q_net"].items():
                print(f"  {k}: {v.shape}")
        else:
            print("Checkpoint structure:", ckpt.keys())

if __name__ == "__main__":
    test_load()
