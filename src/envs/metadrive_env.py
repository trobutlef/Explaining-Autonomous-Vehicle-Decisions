import gymnasium as gym
import numpy as np
from metadrive.envs.metadrive_env import MetaDriveEnv
from gymnasium import spaces

class DiscreteMetaDriveWrapper(gym.ActionWrapper):
    def __init__(self, env):
        super().__init__(env)
        # Define discrete actions: 
        # 0: Idle, 1: Accel, 2: Decel, 3: Left, 4: Right
        self.action_space = spaces.Discrete(5)
        self.action_map = {
            0: [0.0, 0.0],  # Idle
            1: [0.0, 1.0],  # Accel
            2: [0.0, -1.0], # Decel
            3: [-1.0, 0.0], # Left
            4: [1.0, 0.0],  # Right
        }

    def action(self, action):
        return np.array(self.action_map[action])

    def reset(self, *, seed=None, options=None):
        return self.env.reset(seed=seed)

    def render(self, *args, **kwargs):
        return self.env.render(*args, **kwargs)

def make_env(cfg):
    """
    Create and wrap the MetaDrive environment.
    """
    # Helper to get value from config (dict or object)
    def get_cfg_val(obj, key, default=None):
        if isinstance(obj, dict):
            return obj.get(key, default)
        return getattr(obj, key, default)

    env_cfg = get_cfg_val(cfg, "environment", {})
    
    use_render = get_cfg_val(env_cfg, "render", False)
    traffic_density = get_cfg_val(env_cfg, "traffic_density", 0.1)
    map_type = get_cfg_val(env_cfg, "map", "S")
    
    # Project config for seed
    proj_cfg = get_cfg_val(cfg, "project", {})
    seed = get_cfg_val(proj_cfg, "seed", 0)

    env_config = {
        "use_render": use_render,
        "traffic_density": traffic_density,
        "map": map_type,
        "start_seed": seed,
        "random_traffic": True,
    }
    
    # Forward sensor configuration
    for key in ["sensors", "image_observation", "vehicle_config", "image_on_cuda"]:
        val = get_cfg_val(env_cfg, key)
        if val is not None:
            env_config[key] = val
        
    env = MetaDriveEnv(env_config)
    env = DiscreteMetaDriveWrapper(env)
    
    return env
