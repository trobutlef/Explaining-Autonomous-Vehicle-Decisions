from metadrive.envs.metadrive_env import MetaDriveEnv

env = MetaDriveEnv({
    "use_render": True,
    "manual_control": False,
})

obs, info = env.reset()

for _ in range(200):
    action = env.action_space.sample()
    obs, reward, done, truncated, info = env.step(action)
    if done:
        obs, info = env.reset()

env.close()
