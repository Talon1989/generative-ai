import copy
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import gym
from utilities import ReplayBuffer


env_ = gym.make('CartPole-v1')
# env_ = gym.make('CartPole-v1', render_mode='human')


def random_steps(n_episodes=5, visualize=True):
    if visualize:
        env = gym.make('CartPole-v1', render_mode='human')
    else:
        env = gym.make('CartPole-v1')
    for ep in range(n_episodes):
        score = 0
        env.reset()
        while True:
            # env.render()  # no need to call it anymore with render_mode='human'
            a = env.action_space.sample()
            s_, r, d, t, _ = env.step(a)
            score += r
            if d or t:
                break
        print('Episode %d | score: %.3f' % ((ep+1), score))
    if visualize:
        env.close()


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

    def forward(self, x):
        for layer in self.hidden_layers:
            x = self.relu(layer(x))
        x = self.output_layer(x)  # linear activation
        return x


# hardcoded to work with discrete action spaces
class DQL:
    def __init__(self, env, hidden_shape:np.array, epsilon_decay=999/1_000,
                 alpha:float=1/1000, gamma:float=99/100, batch_size:int=64):
        self.env = env
        self.n_s, self.n_a = self.env.observation_space.shape[0], self.env.action_space.n
        self.gamma = gamma
        self.batch_size = batch_size
        self.buffer = ReplayBuffer(max_size=1_000)
        self.epsilon = 1.
        self.epsilon_decay = epsilon_decay
        self.main_nn = Agent(self.n_s, hidden_shape, self.n_a)
        self.target_nn = copy.deepcopy(self.main_nn)  # = would be in-place
        self.optimizer = torch.optim.Adam(params=self.main_nn.parameters(), lr=alpha)
        self.criterion = nn.MSELoss()

    def hard_update(self):
        self.target_nn.load_state_dict(self.main_nn.state_dict())

    def choose_action(self, s):
        if np.random.uniform(0, 1) < self.epsilon:
            return self.env.action_space.sample()
        else:
            s = torch.tensor(s)
            values = self.main_nn(torch.tensor(s))
            if torch.sum(torch.abs(values)) == 0:
                return self.env.action_space.sample()
            a = torch.argmax(values)
            return int(a)

    def store_transition(self, s, a, r, s_, d):
        self.buffer.remember(s, a, r, s_, d)

    def fit(self):
        if self.buffer.get_buffer_size() < 256:
            return
        states, actions, rewards, states_, dones = self.buffer.get_buffer(
            self.batch_size, randomized=True, cleared=False
        )
        pass

    def train(self, n_episodes=200, graph=True):
        scores, avg_scores = [], []
        for ep in range(1, n_episodes+1):
            s = self.env.reset()[0]
            score = 0
            while True:
                a = self.choose_action(s)
                s_, r, d, t, _ = self.env.step(a)
                score += r
                if d or t:
                    break
                self.fit()
                s = s_
            if self.epsilon > 1/10:
                self.epsilon = self.epsilon * self.epsilon_decay
            scores.append(score)
            avg_scores.append(np.sum(scores[-50:]) / len(scores[-50:]))
            if ep % 10 == 0:
                print('Episode %d | Avg score: %.3f | Epsilon: %.3f'
                      % (ep, avg_scores[-1], self.epsilon))
            if graph and ep % 100 == 0:
                plt.scatter(np.arange(len(scores)), scores, c='g', s=1, label='scores')
                plt.plot(avg_scores, c='b', linewidth=1, label='avg scores')
                plt.xlabel('episode')
                plt.ylabel('score')
                plt.legend(loc='best')
                plt.title('Episode %d DQL' % ep)
                plt.show()
                plt.clf()





































































































































































































































































































