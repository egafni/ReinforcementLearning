#!/usr/bin/env python3
import argparse
import os
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import ptan
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from oracle.common import make_env_oracle
from oracle.rl.ppo.lib import calc_logprob, model, test_net
from oracle.utils.misc_utils import DataClassMixin


@dataclass
class PPOParams(DataClassMixin):
    env_name: str = "HalfCheetahBulletEnv-v0"
    env_kwargs: dict = field(default_factory=dict)
    name: str = "ppo_test"  # name for log files
    gamma: float = 0.99  # discount rate
    gae_lambda: float = 0.95  # specifies the lambda factor in the advantage estimator

    trajectory_size: int = 2049
    lr_actor: float = 1e-5
    lr_critic: float = 1e-4

    """
    For every batch of TRAJECTORY_SIZE samples, we perform PPO_EPOCHES iterations of
    the PPO objective, with mini-batches of 64 samples
    """
    ppo_eps: float = 0.2  # clipping value for the ratio of the new and the old policy
    ppo_epochs: int = 10  # during training, we do several epochs over the sampled training batch
    ppo_batch_size: int = 64  # mini-batch size, where each observation is a trajectory of TRAJECTORY_SIZE

    test_iters: int = 100000
    cpu: bool = False  # train on the gpu
    seed: Optional[int] = None


def calc_advantage_reference(trajectory, net_critic, states_v, gamma, gae_lambda, device):
    """
    By trajectory calculate advantage and 1-step ref value

    Takes the trajectory with steps and calculates advantages for
    the actor and reference values for the critic training.
    Our trajectory is not a single episode,
    but can be several episodes concatenated together.

    :param trajectory: trajectory list
    :param net_critic: critic network
    :param states_v: states tensor
    :return: tuple with advantage numpy array and reference values
    """
    values_v = net_critic(states_v)
    values = values_v.squeeze().data.cpu().numpy()
    # generalized advantage estimator: smoothed version of the advantage
    # As the first step, we ask the critic to convert states into values
    last_gae = 0.0
    result_adv = []
    result_ref = []
    for val, next_val, (exp,) in zip(reversed(values[:-1]), reversed(values[1:]), reversed(trajectory[:-1])):
        """
        This loop joins the values obtained and experience points. For every trajectory step,
        we need the current value (obtained from the current state) and the value for the
        next subsequent step (to perform the estimation using the Bellman equation). We
        also traverse the trajectory in reverse order, to be able to calculate more recent values
        of the advantage in one step.
        """
        if exp.done:
            delta = exp.reward - val
            last_gae = delta
        else:
            delta = exp.reward + gamma * next_val - val
            last_gae = delta + gamma * gae_lambda * last_gae
        result_adv.append(last_gae)
        result_ref.append(last_gae + val)

    adv_v = torch.FloatTensor(list(reversed(result_adv)))
    ref_v = torch.FloatTensor(list(reversed(result_ref)))
    return adv_v.to(device), ref_v.to(device)


def train(params: PPOParams):
    import pybullet_envs  # noqa  required to register Cheetah env

    device = torch.device("cpu" if params.cpu else "cuda")

    save_path = os.path.join("saves", "ppo-" + params.name)
    os.makedirs(save_path, exist_ok=True)

    env = make_env_oracle(params.env_name, params.env_kwargs)
    test_env = make_env_oracle(params.env_name, params.env_kwargs)

    net_actor = model.ModelActor(env.observation_space.shape[0], env.action_space.shape[0]).to(device)
    net_critic = model.ModelCritic(env.observation_space.shape[0]).to(device)
    print(net_actor)
    print(net_critic)

    writer = SummaryWriter(comment="-ppo_" + params.name)
    agent = model.AgentA2C(net_actor, device=device)
    exp_source = ptan.experience.ExperienceSource(env, agent, steps_count=1)

    opt_actor = optim.Adam(net_actor.parameters(), lr=params.lr_actor)
    opt_critic = optim.Adam(net_critic.parameters(), lr=params.lr_critic)

    trajectory = []
    best_reward = None
    with ptan.common.utils.RewardTracker(writer) as reward_tracker:
        for step_idx, exp in enumerate(exp_source):
            # exp: ptan.experience.Experience  # state, action, reward, done
            rewards_steps = exp_source.pop_rewards_steps()
            if rewards_steps:
                # episode ended
                total_rewards, total_steps = zip(*rewards_steps)
                writer.add_scalar("episode_steps", np.mean(total_steps), step_idx)
                reward_tracker.reward(np.mean(total_rewards), step_idx)

            if step_idx % params.test_iters == 0:
                ts = time.time()
                total_rewards, total_steps = test_net(net_actor, test_env, device=device)
                print("Test done in %.2f sec, reward %.3f, steps %d" % (time.time() - ts, total_rewards, total_steps))
                writer.add_scalar("test_reward", total_rewards, step_idx)
                writer.add_scalar("test_steps", total_steps, step_idx)
                if best_reward is None or best_reward < total_rewards:
                    if best_reward is not None:
                        print("Best reward updated: %.3f -> %.3f" % (best_reward, total_rewards))
                        name = "best_%+.3f_%d.dat" % (total_rewards, step_idx)
                        fname = os.path.join(save_path, name)
                        torch.save(net_actor.state_dict(), fname)
                    best_reward = total_rewards

            trajectory.append(exp)
            if len(trajectory) < params.trajectory_size:
                continue

            traj_states = [t[0].state for t in trajectory]
            traj_actions = [t[0].action for t in trajectory]
            traj_states_v = torch.tensor(traj_states)
            traj_states_v = traj_states_v.to(device)
            traj_actions_v = torch.tensor(traj_actions)
            traj_actions_v = traj_actions_v.to(device)
            traj_adv_v, traj_ref_v = calc_advantage_reference(
                trajectory, net_critic, traj_states_v, params.gamma, params.gae_lambda, device=device
            )
            mu_v = net_actor(traj_states_v)
            old_logprob_v = calc_logprob(mu_v, net_actor.logstd, traj_actions_v)

            # normalize advantages
            traj_adv_v = traj_adv_v - torch.mean(traj_adv_v)
            traj_adv_v /= torch.std(traj_adv_v)

            # drop last entry from the trajectory, an our adv and ref value calculated without it
            trajectory = trajectory[:-1]
            old_logprob_v = old_logprob_v[:-1].detach()

            sum_loss_value = 0.0
            sum_loss_policy = 0.0
            count_steps = 0

            for epoch in range(params.ppo_epochs):
                for batch_ofs in range(0, len(trajectory), params.ppo_batch_size):
                    batch_l = batch_ofs + params.ppo_batch_size
                    states_v = traj_states_v[batch_ofs:batch_l]
                    actions_v = traj_actions_v[batch_ofs:batch_l]
                    batch_adv_v = traj_adv_v[batch_ofs:batch_l]
                    batch_adv_v = batch_adv_v.unsqueeze(-1)
                    batch_ref_v = traj_ref_v[batch_ofs:batch_l]
                    batch_old_logprob_v = old_logprob_v[batch_ofs:batch_l]

                    # critic training
                    opt_critic.zero_grad()
                    value_v = net_critic(states_v)
                    loss_value_v = F.mse_loss(value_v.squeeze(-1), batch_ref_v)
                    loss_value_v.backward()
                    opt_critic.step()

                    # actor training
                    opt_actor.zero_grad()
                    mu_v = net_actor(states_v)
                    logprob_pi_v = calc_logprob(mu_v, net_actor.logstd, actions_v)
                    ratio_v = torch.exp(logprob_pi_v - batch_old_logprob_v)
                    surr_obj_v = batch_adv_v * ratio_v
                    c_ratio_v = torch.clamp(ratio_v, 1.0 - params.ppo_eps, 1.0 + params.ppo_eps)
                    clipped_surr_v = batch_adv_v * c_ratio_v
                    loss_policy_v = -torch.min(surr_obj_v, clipped_surr_v).mean()
                    loss_policy_v.backward()
                    opt_actor.step()

                    sum_loss_value += loss_value_v.item()
                    sum_loss_policy += loss_policy_v.item()
                    count_steps += 1

            trajectory.clear()
            writer.add_scalar("advantage", traj_adv_v.mean().item(), step_idx)
            writer.add_scalar("values", traj_ref_v.mean().item(), step_idx)
            writer.add_scalar("loss_policy", sum_loss_policy / count_steps, step_idx)
            writer.add_scalar("loss_value", sum_loss_value / count_steps, step_idx)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    PPOParams.add_to_argparser(p)
    args = p.parse_args()
    train(PPOParams.from_parsed_args(args))
