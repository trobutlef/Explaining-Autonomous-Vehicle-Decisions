import random
from collections import deque, namedtuple
from typing import Any, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


Transition = namedtuple(
    "Transition", ["state", "action", "reward", "next_state", "done"]
)


class QNetwork(nn.Module):
    """Simple MLP Q-network for low-dimensional highway-env observations."""

    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)

    def push(self, *args):
        self.buffer.append(Transition(*args))

    def sample(self, batch_size: int) -> Transition:
        batch = random.sample(self.buffer, batch_size)
        return Transition(*zip(*batch))

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.buffer)


class DQNAgent:
    def __init__(self, config: Any, obs_space, action_space):
        """DQN agent for highway-env.

        Args:
            config: global config (OmegaConf DictConfig or similar)
            obs_space: env.observation_space
            action_space: env.action_space (Discrete)
        """

        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        def get_cfg_val(obj, key, default=None):
            if isinstance(obj, dict):
                return obj.get(key, default)
            return getattr(obj, key, default)

        # DQN hyperparameters (with reasonable fallbacks)
        agent_cfg = get_cfg_val(config, "agent", {})
        self.gamma = float(get_cfg_val(agent_cfg, "gamma", 0.99))
        self.lr = float(get_cfg_val(agent_cfg, "lr", 1e-3))
        self.batch_size = int(get_cfg_val(agent_cfg, "batch_size", 64))
        self.buffer_size = int(get_cfg_val(agent_cfg, "buffer_size", 100000))
        self.eps_start = float(get_cfg_val(agent_cfg, "eps_start", 1.0))
        self.eps_end = float(get_cfg_val(agent_cfg, "eps_end", 0.05))
        self.eps_decay = float(get_cfg_val(agent_cfg, "eps_decay", 50000))
        self.target_update_interval = int(
            get_cfg_val(agent_cfg, "target_update_interval", 1000)
        )

        # Infer dimensions from spaces (assume Box + Discrete)
        if hasattr(obs_space, "shape") and obs_space.shape is not None:
            obs_dim = int(np.prod(obs_space.shape))
        else:
            raise ValueError("Unsupported observation space for DQNAgent")

        if not hasattr(action_space, "n"):
            raise ValueError("DQNAgent expects a discrete action space")
        action_dim = int(action_space.n)

        hidden_dim = int(get_cfg_val(agent_cfg, "hidden_dim", 256))
        self.obs_dim = obs_dim
        self.q_net = QNetwork(self.obs_dim, action_dim, hidden_dim).to(self.device)
        self.target_net = QNetwork(self.obs_dim, action_dim, hidden_dim).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=self.lr)

        self.replay_buffer = ReplayBuffer(self.buffer_size)
        self.total_steps = 0

    def _epsilon(self) -> float:
        # Linear decay from eps_start to eps_end over eps_decay steps
        frac = min(1.0, self.total_steps / max(1, self.eps_decay))
        return self.eps_start + frac * (self.eps_end - self.eps_start)

    def act(self, state: np.ndarray, exploit: bool = False) -> int:
        """Select an action using epsilon-greedy policy."""

        self.total_steps += 1
        eps = self.eps_end if exploit else self._epsilon()

        if not exploit and random.random() < eps:
            # explore
            # action_space is implicit: use q_net output dimension
            with torch.no_grad():
                dummy = self.q_net(torch.zeros(1, self.q_net.net[0].in_features))
            return int(random.randrange(dummy.shape[-1]))

        state_t = self._state_to_tensor(state)
        with torch.no_grad():
            q_values = self.q_net(state_t)
            action = q_values.argmax(dim=1).item()
        return int(action)

    def _state_to_tensor(self, state: np.ndarray) -> torch.Tensor:
        x = torch.as_tensor(state, dtype=torch.float32, device=self.device)
        if x.dim() == 1:
            x = x.unsqueeze(0)
        else:
            x = x.view(1, -1)
        return x

    def remember(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        self.replay_buffer.push(state, action, reward, next_state, done)

    def learn(self) -> Tuple[float, float]:
        """Sample from replay buffer and perform one gradient step.

        Returns (loss, mean_Q) for logging. If insufficient data, returns (0.0, 0.0).
        """

        if len(self.replay_buffer) < self.batch_size:
            return 0.0, 0.0

        transitions = self.replay_buffer.sample(self.batch_size)
        state_batch_np = np.stack(transitions.state)
        state_batch = torch.as_tensor(
            state_batch_np.reshape(self.batch_size, self.obs_dim),
            dtype=torch.float32,
            device=self.device,
        )
        action_batch = torch.as_tensor(
            transitions.action, dtype=torch.int64, device=self.device
        ).unsqueeze(1)
        reward_batch = torch.as_tensor(
            transitions.reward, dtype=torch.float32, device=self.device
        ).unsqueeze(1)
        next_state_batch_np = np.stack(transitions.next_state)
        next_state_batch = torch.as_tensor(
            next_state_batch_np.reshape(self.batch_size, self.obs_dim),
            dtype=torch.float32,
            device=self.device,
        )
        done_batch = torch.as_tensor(
            transitions.done, dtype=torch.float32, device=self.device
        ).unsqueeze(1)

        # Q(s, a)
        q_values = self.q_net(state_batch).gather(1, action_batch)

        # target: r + gamma * max_a' Q_target(s', a') * (1 - done)
        with torch.no_grad():
            next_q_values = self.target_net(next_state_batch).max(dim=1, keepdim=True)[0]
            target = reward_batch + self.gamma * next_q_values * (1.0 - done_batch)

        loss = F.mse_loss(q_values, target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        mean_q = q_values.detach().mean().item()

        # Periodically update target network
        if self.total_steps % self.target_update_interval == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

        return float(loss.item()), float(mean_q)

    def save(self, path: str) -> None:
        state = {
            "q_net": self.q_net.state_dict(),
            "target_net": self.target_net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "total_steps": self.total_steps,
        }
        torch.save(state, path)

    def load(self, path: str, map_location=None) -> None:
        loc = map_location or ("cuda" if torch.cuda.is_available() else "cpu")
        state = torch.load(path, map_location=loc)
        self.q_net.load_state_dict(state["q_net"])
        self.target_net.load_state_dict(state["target_net"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.total_steps = int(state.get("total_steps", 0))
