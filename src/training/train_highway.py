import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch

from src.utils.config import load_config
from src.envs.highway_env import make_env
from src.agents.dqn_agent import DQNAgent


def _set_seed(seed: int) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate(env, agent: DQNAgent, n_episodes: int = 5) -> Dict[str, float]:
    returns: List[float] = []
    lengths: List[int] = []

    for _ in range(n_episodes):
        state, _ = env.reset()
        done = False
        total_r = 0.0
        t = 0
        while not done:
            action = agent.act(state, exploit=True)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_r += float(reward)
            state = next_state
            t += 1
        returns.append(total_r)
        lengths.append(t)

    return {
        "avg_return": float(np.mean(returns)),
        "std_return": float(np.std(returns)),
        "avg_length": float(np.mean(lengths)),
    }


def main(config_path: str = "configs/config.yaml") -> None:
    cfg = load_config(config_path)

    seed = int(getattr(getattr(cfg, "project", {}), "seed", 42))
    _set_seed(seed)

    project_root = Path(__file__).resolve().parents[2]
    output_dir = project_root / getattr(getattr(cfg, "project", {}), "output_dir", "outputs")
    output_dir.mkdir(parents=True, exist_ok=True)

    log_dir = project_root / getattr(getattr(cfg, "logging", {}), "log_dir", "logs")
    log_dir.mkdir(parents=True, exist_ok=True)

    env = make_env(cfg)
    obs_space = env.observation_space
    action_space = env.action_space

    agent = DQNAgent(cfg, obs_space, action_space)

    agent_cfg = getattr(cfg, "agent", {})
    total_timesteps = int(getattr(agent_cfg, "total_timesteps", 100000))
    eval_interval = int(getattr(agent_cfg, "eval_interval", 5000))
    save_interval = int(getattr(agent_cfg, "save_interval", 10000))

    # Simple training loop (episodic)
    state, _ = env.reset()
    episode_return = 0.0
    episode_len = 0
    episode = 0

    rewards_log: List[Dict[str, float]] = []

    start_time = time.time()
    for t in range(1, total_timesteps + 1):
        action = agent.act(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        agent.remember(state, action, reward, next_state, done)
        loss, mean_q = agent.learn()

        episode_return += float(reward)
        episode_len += 1
        state = next_state

        if done:
            rewards_log.append(
                {
                    "timestep": t,
                    "episode": episode,
                    "return": episode_return,
                    "length": episode_len,
                }
            )
            state, _ = env.reset()
            episode_return = 0.0
            episode_len = 0
            episode += 1

        # Periodic evaluation
        if t % eval_interval == 0:
            eval_stats = evaluate(env, agent, n_episodes=5)
            elapsed = time.time() - start_time
            print(
                f"Step {t}: eval_return={eval_stats['avg_return']:.2f} ",
                f"len={eval_stats['avg_length']:.1f} time={elapsed/60:.1f}min",
            )

        # Periodic checkpoint
        if t % save_interval == 0:
            ckpt_path = output_dir / f"dqn_highway_step{t}.pt"
            agent.save(str(ckpt_path))

    # Final save
    final_path = output_dir / "dqn_highway_final.pt"
    agent.save(str(final_path))

    # Save rewards log as npy for later plotting in notebooks
    rewards_path = output_dir / "highway_rewards.npy"
    np.save(rewards_path, rewards_log, allow_pickle=True)
    print(f"Training finished. Saved agent to {final_path} and rewards to {rewards_path}.")


if __name__ == "__main__":
    main()
