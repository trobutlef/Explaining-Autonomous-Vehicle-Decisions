from typing import Any

def make_env(config: Any):
    # return configured highway-env environment
    try:
        import gymnasium as gym
        import highway_env  # noqa: F401 - needed to register envs
    except Exception as e:
        raise RuntimeError("gymnasium/highway-env not installed. Run `pip install -r requirements.txt`.") from e

    env_name = getattr(getattr(config, "environment", {}), "name", "highway-v0")

    render_flag = getattr(getattr(config, "environment", {}), "render", False)
    # Use rgb_array by default for notebooks; switch to human when explicitly requested
    render_mode = "human" if render_flag else "rgb_array"

    # Some envs require explicit render_mode kwarg; others accept none.
    try:
        env = gym.make(env_name, render_mode=render_mode)
    except TypeError:
        env = gym.make(env_name)

    # Apply optional wrappers listed in config.environment.wrappers
    wrappers = getattr(getattr(config, "environment", {}), "wrappers", []) or []
    for w in wrappers:
        # Placeholder: resolve wrapper by name/path if needed in future
        # Example: env = SomeWrapper(env, **w.get("kwargs", {}))
        pass

    return env
