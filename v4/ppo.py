import argparse
import os
import random
import time

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from stable_baselines3.common.atari_wrappers import (
    ClipRewardEnv,
    EpisodicLifeEnv,
    FireResetEnv,
    MaxAndSkipEnv,
    NoopResetEnv
)
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, default=os.path.basename(__file__).rstrip('.py'))
    parser.add_argument('--gym_id', type=str, default='BreakoutNoFrameskip-v4')
    parser.add_argument('--learning_rate', type=float, default=2.5e-4)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--total_steps', type=int, default=int(1e7))
    parser.add_argument('--use_cuda', type=bool, default=True)
    parser.add_argument('--num_envs', type=int, default=8)
    parser.add_argument('--num_steps', type=int, default=128)
    parser.add_argument('--lr_decay', type=bool, default=True)
    parser.add_argument('--use_gae', type=bool, default=True)
    parser.add_argument('--gae_lambda', type=float, default=0.95)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--num_mini_batches', type=int, default=4)
    parser.add_argument('--update_epochs', type=int, default=4)
    parser.add_argument('--norm_adv', type=bool, default=True)
    parser.add_argument('--clip_value_loss', type=bool, default=True)
    parser.add_argument('--c_1', type=float, default=1.0)
    parser.add_argument('--c_2', type=float, default=0.01)
    parser.add_argument('--max_grad_norm', type=float, default=0.5)
    parser.add_argument('--clip_epsilon', type=float, default=0.2)
    a = parser.parse_args()
    a.batch_size = int(a.num_envs * a.num_steps)
    a.minibatch_size = int(a.batch_size // a.num_mini_batches)
    return a


def make_env(gym_id, seed):
    def thunk():
        env = gym.make(gym_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = NoopResetEnv(env, noop_max=30)
        env = MaxAndSkipEnv(env, skip=4)
        env = EpisodicLifeEnv(env)
        if 'FIRE' in env.unwrapped.get_action_meanings():
            env = FireResetEnv(env)
        env = ClipRewardEnv(env)
        env = gym.wrappers.ResizeObservation(env, (84, 84))
        env = gym.wrappers.GrayScaleObservation(env)
        env = gym.wrappers.FrameStack(env, 4)
        env.seed(seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env
    return thunk


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, e):
        super(Agent, self).__init__()
        self.network = nn.Sequential(
            layer_init(nn.Conv2d(4, 32, 8, stride=4)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, 4, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, 3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(64 * 7 * 7, 512)),
            nn.ReLU(),
        )
        self.actor = layer_init(nn.Linear(512, e.single_action_space.n), std=0.01)
        self.critic = layer_init(nn.Linear(512, 1), std=1)

    def get_value(self, x):
        return self.critic(self.network(x / 255.0))

    def get_action_and_value(self, x, a=None, show_all=False):
        hidden = self.network(x / 255.0)
        log = self.actor(hidden)
        p = Categorical(logits=log)
        if a is None:
            a = p.sample()
        if show_all:
            return a, p.log_prob(a), p.entropy(), self.critic(hidden), p.probs
        return a, p.log_prob(a), p.entropy(), self.critic(hidden)


def main(env_id, seed):
    args = get_args()
    args.gym_id = env_id
    args.seed = seed
    run_name = (
            'ppo' +
            '_epoch_' + str(args.update_epochs) +
            '_seed_' + str(args.seed)
    )

    # Save training logs
    path_string = str(args.gym_id).split('NoFrameskip')[0] + '/' + run_name
    writer = SummaryWriter(path_string)
    writer.add_text(
        'Hyperparameter',
        '|param|value|\n|-|-|\n%s' % ('\n'.join([f'|{key}|{value}|' for key, value in vars(args).items()])),
    )

    # Random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Initialize environments
    device = torch.device('cuda' if torch.cuda.is_available() and args.use_cuda else 'cpu')
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.gym_id, args.seed + i) for i in range(args.num_envs)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), 'only discrete action space is supported'
    agent = Agent(envs).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # Initialize buffer
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    probs = torch.zeros((args.num_steps, args.num_envs, envs.single_action_space.n)).to(device)
    log_probs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # Data collection
    global_step = 0
    start_time = time.time()
    next_obs = torch.Tensor(envs.reset()).to(device)
    next_done = torch.zeros(args.num_envs).to(device)
    num_updates = int(args.total_steps // args.batch_size)

    for update in tqdm(range(1, num_updates + 1)):

        # Linear decay of learning rate
        if args.lr_decay:
            frac = 1.0 - (update - 1.0) / num_updates
            lr_now = frac * args.learning_rate
            optimizer.param_groups[0]['lr'] = lr_now

        for step in range(0, args.num_steps):
            global_step += 1 * args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            # Compute the logarithm of the action probability output by the old policy network
            with torch.no_grad():
                action, log_prob, _, value, prob = agent.get_action_and_value(next_obs, show_all=True)
                values[step] = value.flatten()
            actions[step] = action
            probs[step] = prob
            log_probs[step] = log_prob

            # Update the environments
            next_obs, reward, done, info = envs.step(action.cpu().numpy())
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(done).to(device)

            for item in info:
                if 'episode' in item.keys():
                    writer.add_scalar('charts/episodic_return', item['episode']['r'], global_step)
                    break

        # Use GAE (Generalized Advantage Estimation) technique to estimate the advantage function
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            if args.use_gae:
                advantages = torch.zeros_like(rewards).to(device)
                last_gae_lam = 0
                for t in reversed(range(args.num_steps)):
                    if t == args.num_steps - 1:
                        next_non_terminal = 1.0 - next_done
                        next_values = next_value
                    else:
                        next_non_terminal = 1.0 - dones[t + 1]
                        next_values = values[t + 1]
                    delta = rewards[t] + args.gamma * next_values * next_non_terminal - values[t]
                    advantages[t] = last_gae_lam = (
                            delta + args.gamma * args.gae_lambda * next_non_terminal * last_gae_lam
                    )
                returns = advantages + values
            else:
                returns = torch.zeros_like(rewards).to(device)
                for t in reversed(range(args.num_steps)):
                    if t == args.num_steps - 1:
                        next_non_terminal = 1.0 - next_done
                        next_return = next_value
                    else:
                        next_non_terminal = 1.0 - dones[t + 1]
                        next_return = returns[t + 1]
                    returns[t] = rewards[t] + args.gamma * next_non_terminal * next_return
                advantages = returns - values

        # ---------------------- We have collected enough data, now let's start training ---------------------- #
        # Flatten each batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_probs = probs.reshape((-1, envs.single_action_space.n))
        b_log_probs = log_probs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Update the policy network and value network
        b_index = np.arange(args.batch_size)
        for epoch in range(1, args.update_epochs + 1):
            np.random.shuffle(b_index)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_index = b_index[start:end]

                # The latest outputs of the policy network and value network
                _, new_log_prob, entropy, new_value, new_probs = (
                    agent.get_action_and_value(b_obs[mb_index], b_actions.long()[mb_index], show_all=True)
                )

                # Compute KL divergence
                d = torch.sum(
                    b_probs[mb_index] * torch.log((b_probs[mb_index] + 1e-12) / (new_probs + 1e-12)), 1
                )
                writer.add_scalar('charts/average_kld', d.mean(), global_step)
                writer.add_scalar('others/min_kld', d.min(), global_step)
                writer.add_scalar('others/max_kld', d.max(), global_step)
                log_ratio = new_log_prob - b_log_probs[mb_index]
                ratio = log_ratio.exp()

                # Advantage normalization
                mb_advantages = b_advantages[mb_index]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-12)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_epsilon, 1 + args.clip_epsilon)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                new_value = new_value.view(-1)
                if args.clip_value_loss:
                    v_loss_un_clipped = (new_value - b_returns[mb_index]) ** 2
                    v_clipped = b_values[mb_index] + torch.clamp(
                        new_value - b_values[mb_index],
                        -args.clip_epsilon,
                        args.clip_epsilon,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_index]) ** 2
                    v_loss_max = torch.max(v_loss_un_clipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((new_value - b_returns[mb_index]) ** 2).mean()

                # Policy entropy
                entropy_loss = entropy.mean()

                # Total loss
                loss = pg_loss + v_loss * args.c_1 - entropy_loss * args.c_2

                # Save the data during the training process
                writer.add_scalar('losses/value_loss', v_loss.item(), global_step)
                writer.add_scalar('losses/policy_loss', pg_loss.item(), global_step)
                writer.add_scalar('losses/entropy', entropy_loss.item(), global_step)
                writer.add_scalar('losses/delta', torch.abs(ratio - 1).mean().item(), global_step)

                # Update network parameters
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

        y_pre, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pre) / var_y

        # Save the data during the training process
        writer.add_scalar('charts/learning_rate', optimizer.param_groups[0]['lr'], global_step)
        writer.add_scalar('others/explained_variance', explained_var, global_step)
        writer.add_scalar('charts/SPS', int(global_step / (time.time() - start_time)), global_step)

    envs.close()
    writer.close()


def run():
    for env_id in ['Breakout']:
        for seed in [1, 2, 3]:
            print(env_id + 'NoFrameskip-v4', 'seed:', seed)
            main(env_id + 'NoFrameskip-v4', seed)


if __name__ == '__main__':
    run()
