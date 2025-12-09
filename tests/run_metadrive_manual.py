from metadrive.envs.metadrive_env import MetaDriveEnv

env = MetaDriveEnv(
    {
        "use_render": True,
        "manual_control": True,   # enable keyboard
        "start_seed": 0,
        "horizon": 2000,          # longer episode
    }
)

obs, info = env.reset()
while True:
    # For manual_control=True, MetaDrive reads keyboard events internally,
    # but you still need to call step() every frame to advance the sim.
    obs, reward, terminated, truncated, info = env.step([0.0, 0.0])

    if terminated or truncated:
        obs, info = env.reset()
