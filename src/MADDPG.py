from NetworkModels import Actor, Critic
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from ReplayBuffer import Buffer
from Noise import OUNoise
from torch.optim import Adam

class MADDPG:
    def __init__(self, n_agents, obs_dims, act_dims, buffer_size=100000, batch_size=128,
                 gamma=0.99, tau=0.01, actor_lr=1e-3, critic_lr=1e-3, device="cpu"):

        self.n_agents = n_agents
        self.obs_dims = obs_dims
        self.act_dims = act_dims
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.device = torch.device(device)

        # Networks
        self.actors = [Actor(obs_dims[i], act_dims[i]).to(self.device) for i in range(n_agents)]
        self.critics = [Critic(sum(obs_dims), sum(act_dims)).to(self.device) for _ in range(n_agents)]
        self.target_actors = [Actor(obs_dims[i], act_dims[i]).to(self.device) for i in range(n_agents)]
        self.target_critics = [Critic(sum(obs_dims), sum(act_dims)).to(self.device) for _ in range(n_agents)]

        # Optimizers
        self.actor_opts = [optim.Adam(self.actors[i].parameters(), lr=actor_lr) for i in range(n_agents)]
        self.critic_opts = [optim.Adam(self.critics[i].parameters(), lr=critic_lr) for i in range(n_agents)]

        # Replay buffer (shared across agents)
        self.buffer = Buffer(buffer_size, obs_dims, act_dims)

        # Exploration noise
        self.noises = [OUNoise(act_dims[i]) for i in range(n_agents)]

        self._hard_update_targets()

    def _hard_update_targets(self):
        for i in range(self.n_agents):
            self.target_actors[i].load_state_dict(self.actors[i].state_dict())
            self.target_critics[i].load_state_dict(self.critics[i].state_dict())

    def soft_update(self, net, target_net):
        for target_param, param in zip(target_net.parameters(), net.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

    def select_action(self, obs_list, noise=True):
        actions = []
        for i, obs in enumerate(obs_list):
            obs_tensor = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            action = self.actors[i](obs_tensor).detach().cpu().numpy()[0]
            if noise:
                action += self.noises[i].sample()
            actions.append(np.clip(action, -1.0, 1.0))  # assuming actions in [-1,1]
        return actions

    def update(self):
        if len(self.buffer) < self.batch_size:
            return  # not enough samples yet

        obs_n, act_n, rew_n, next_obs_n, done_n = self.buffer.sample(self.batch_size)

        # Convert to tensors
        obs_n = [torch.tensor(o, dtype=torch.float32, device=self.device) for o in obs_n]
        act_n = [torch.tensor(a, dtype=torch.float32, device=self.device) for a in act_n]
        next_obs_n = [torch.tensor(no, dtype=torch.float32, device=self.device) for no in next_obs_n]
        rew_n = [torch.tensor(r, dtype=torch.float32, device=self.device).unsqueeze(-1) for r in rew_n]
        done_n = [torch.tensor(d, dtype=torch.float32, device=self.device).unsqueeze(-1) for d in done_n]

        # Concatenate for critics
        obs_cat = torch.cat(obs_n, dim=-1)
        act_cat = torch.cat(act_n, dim=-1)
        next_obs_cat = torch.cat(next_obs_n, dim=-1)
        next_act_cat = torch.cat([self.target_actors[i](next_obs_n[i]) for i in range(self.n_agents)], dim=-1)

        for i in range(self.n_agents):
            # --- Critic update ---
            q_val = self.critics[i](obs_cat, act_cat)
            with torch.no_grad():
                target_q = self.target_critics[i](next_obs_cat, next_act_cat)
                y = rew_n[i] + self.gamma * (1 - done_n[i]) * target_q
            critic_loss = F.mse_loss(q_val, y)

            self.critic_opts[i].zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critics[i].parameters(), 0.5)
            self.critic_opts[i].step()

            # --- Actor update ---
            curr_act_cat = torch.cat([
                self.actors[j](obs_n[j]) if j == i else act_n[j]
                for j in range(self.n_agents)
            ], dim=-1)
            actor_loss = -self.critics[i](obs_cat, curr_act_cat).mean()

            self.actor_opts[i].zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actors[i].parameters(), 0.5)
            self.actor_opts[i].step()

            # --- Soft update ---
            self.soft_update(self.actors[i], self.target_actors[i])
            self.soft_update(self.critics[i], self.target_critics[i])
