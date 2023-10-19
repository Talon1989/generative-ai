import copy
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import gym
from utilities import ReplayBuffer, display_graph


env_ = gym.make('CartPole-v1')


class Agent(nn.Module):
    def __init__(self, in_dim:int, hidden_shape:np.array, out_dim:int):
        super().__init__()
        self.hidden_layers = nn.ModuleList()
        current_n_feat = in_dim
        for h in hidden_shape:
            self.hidden_layers.append(nn.Linear(current_n_feat, h))
            current_n_feat = h
        self.output_layer = nn.Linear(current_n_feat, out_dim)
        self.relu = nn.ReLU()
        # dim=1: operates on column, this requires unsqueezing single states
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        for layer in self.hidden_layers:
            x = self.relu(layer(x))
        x = self.softmax(self.output_layer(x))
        return x


class PolicyGradientMethod:
    def __init__(self, env, hidden_shape:np.array, alpha:float=1/1_000, gamma:float=99/100):
        self.env = env
        self.n_s = self.env.observation_space.shape[0]
        self.n_a = self.env.action_space.n
        self.gamma = gamma
        self.buffer = ReplayBuffer(max_size=99_999)
        self.nn = Agent(self.n_s, hidden_shape, self.n_a)
        self.optimizer = torch.optim.Adam(params=self.nn.parameters(), lr=alpha)
        self.criterion = nn.MSELoss()

    def _custom_loss(self, states, actions, norm_returns):
        states = torch.tensor(states)
        actions = torch.tensor(actions)
        norm_returns = torch.tensor(norm_returns)
        probabilities = self.nn(states)
        distribution = torch.distributions.Categorical(probabilities)
        log_action_probas = distribution.log_prob(value=actions)
        losses = torch.sum(- log_action_probas * norm_returns)
        return losses

    def _choose_action(self, s):
        self.nn.eval()
        s = torch.tensor([s])
        with torch.no_grad():
            probabilities = self.nn(s)
            distribution = torch.distributions.Categorical(probabilities)
        return int(distribution.sample())

    def _store_transition(self, s, a, r):
        self.buffer.remember(s, a, r, None, None)

    def _fit(self):
        states, actions, rewards, _, _ = self.buffer.get_buffer(
            self.buffer.get_buffer_size(), randomized=False, cleared=True)
        returns = []
        cum_reward = 0
        for r in reversed(rewards):
            cum_reward = r + self.gamma * cum_reward
            returns.append(cum_reward)
        returns = np.array(returns[::-1])
        norm_returns = (returns - np.mean(returns)) / np.std(returns)
        self.nn.train()
        self.optimizer.zero_grad()
        losses = self._custom_loss(states, actions, norm_returns)
        losses.backward()
        self.optimizer.step()

    def train(self, n_episodes=5_000, graph=False):
        scores, avg_scores = [], []
        for ep in range(1, n_episodes + 1):
            score = 0
            s = self.env.reset()[0]
            while True:
                a = self._choose_action(s)
                s_, r, d, t, _ = self.env.step(a)
                score += r
                self._store_transition(s, a, r)
                if d or t:
                    break
                s = s_
            scores.append(score)
            avg_scores.append(np.sum(scores[-50:]) / len(scores[-50:]))
            self._fit()
            if ep % 10 == 0:
                print('Episode %d | avg score: %.3f' % (ep, avg_scores[-1]))
            if ep % 100 == 0 and graph:
                pass


# agent = Agent(4, np.array([16, 16, 32]), 2)
# s = env_.reset()[0]
# # ss = torch.tensor([s])
# ss = np.array(s)
# ss = torch.from_numpy(ss)
# states = []
# for _ in range(7):
#     states.append(env_.reset()[0])
# states = torch.tensor(states)


pgm = PolicyGradientMethod(env_, np.array([16, 16, 32, 32, 64]))
pgm.train(500, graph=False)











































































































































