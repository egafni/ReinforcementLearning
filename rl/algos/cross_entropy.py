from dataclasses import dataclass

import gym
import numpy
import torch
import torch.nn as nn
import torch.optim as optim
from pytorch_lightning.utilities.seed import seed_everything


class Net(nn.Module):
    def __init__(self, obs_size, hidden_size, n_actions):
        super(Net, self).__init__()
        self.net = nn.Sequential(nn.Linear(obs_size, hidden_size), nn.ReLU(), nn.Linear(hidden_size, n_actions))

    def forward(self, x):
        return self.net(x)


@dataclass
class Episode:
    observations: torch.FloatTensor
    actions: torch.LongTensor
    rewards: torch.FloatTensor


@dataclass
class Batch:
    episodes: numpy.ndarray  # array of Episodes

    def filter(self, percentile):
        """Only keep the top percentile of episodes in batch based on total reward"""
        total_rewards = torch.stack([episode.rewards.sum() for episode in self.episodes])
        bound = numpy.percentile(total_rewards, percentile)
        batch = Batch(self.episodes[total_rewards >= bound])
        assert len(batch.episodes)
        return batch, bound

    @property
    def mean_reward(self):
        return numpy.mean([ep.rewards.sum() for ep in self.episodes])


def play(env: gym.Env, net: torch.nn.Module, n_actions: int):
    """Play one episode of the environment using net"""
    observations = []
    rewards = []
    actions = []
    is_done = False

    obs = env.reset()

    while not is_done:
        observations.append(obs)
        action_logits = net(torch.FloatTensor(obs))

        action_probs = torch.softmax(action_logits, dim=0).detach().numpy()
        action = numpy.random.choice(range(n_actions), 1, p=action_probs)[0]
        actions.append(action)

        obs, reward, is_done, _ = env.step(action)

        rewards.append(reward)

    return Episode(
        torch.FloatTensor(observations),
        torch.LongTensor(actions),
        torch.FloatTensor(rewards),
    )


def main(hidden_size=128, batch_size=16, percentile=70, max_epochs=1000, seed=None):
    seed_everything(1)
    env = gym.make("CartPole-v0")
    env.seed(seed)
    obs_size = env.observation_space.shape[0]
    n_actions = env.action_space.n

    net = Net(obs_size, hidden_size, n_actions)
    objective = nn.CrossEntropyLoss()
    optimizer = optim.Adam(params=net.parameters(), lr=0.01)

    for i in range(max_epochs):
        optimizer.zero_grad()
        batch = Batch(numpy.array([play(env, net, n_actions) for _ in range(batch_size)]))
        batch, bound = batch.filter(percentile)
        observations = torch.cat([e.observations for e in batch.episodes])
        action_logits = net(observations)
        actions = torch.cat([e.actions for e in batch.episodes])
        loss = objective(action_logits, actions)
        loss.backward()
        optimizer.step()

        print(f"{i} loss: {loss:.2}, bound: {bound}, mean_reward: {batch.mean_reward:.4}")

        if batch.mean_reward >= 200:
            # environment is solved
            break
    return loss


if __name__ == "__main__":
    main()
