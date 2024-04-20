import argparse
import os
import random
import time

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, default=os.path.basename(__file__).rstrip('.py'))
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--torch_deterministic', type=bool, default=True)
    parser.add_argument('--cuda', type=bool, default=True)
    parser.add_argument('--env_id', type=str, default='Humanoid-v4')
    parser.add_argument('--total_time_steps', type=int, default=int(1e7))
    parser.add_argument('--learning_rate', type=float, default=3e-4)
    parser.add_argument('--num_envs', type=int, default=8)
    parser.add_argument('--num_steps', type=int, default=256)
    parser.add_argument('--anneal_lr', type=bool, default=True)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--gae_lambda', type=float, default=0.95)
    parser.add_argument('--num_mini_batches', type=int, default=4)
    parser.add_argument('--update_epochs', type=int, default=10)
    parser.add_argument('--norm_adv', type=bool, default=True)
    parser.add_argument('--clip_value_loss', type=bool, default=True)
    parser.add_argument('--c_1', type=float, default=0.5)
    parser.add_argument('--c_2', type=float, default=0.0)
    parser.add_argument('--max_grad_norm', type=float, default=0.5)
    parser.add_argument('--kld_max', type=float, default=0.02)
    a = parser.parse_args()
    a.batch_size = int(a.num_envs * a.num_steps)
    a.minibatch_size = int(a.batch_size // a.num_mini_batches)
    return a


def make_env(env_id, gamma):
    def thunk():
        env = gym.make(env_id)
        env = gym.wrappers.FlattenObservation(env)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = gym.wrappers.ClipAction(env)
        env = gym.wrappers.NormalizeObservation(env)
        env = gym.wrappers.TransformObservation(env, lambda o: np.clip(o, -10, 10))
        env = gym.wrappers.NormalizeReward(env, gamma=gamma)
        env = gym.wrappers.TransformReward(env, lambda r: float(np.clip(r, -10, 10)))
        return env
    return thunk


def layer_init(layer, s=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, s)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, e):
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(e.single_observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), s=1.0),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(np.array(e.single_observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, np.array(e.single_action_space.shape).prod()), s=0.01),
        )
        self.actor_log_std = nn.Parameter(torch.zeros(1, np.array(e.single_action_space.shape).prod()))

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, a=None, show_all=False):
        action_mean = self.actor_mean(x)
        action_log_std = self.actor_log_std.expand_as(action_mean)
        action_std = torch.exp(action_log_std)
        probs = Normal(action_mean, action_std)
        if a is None:
            a = probs.sample()
        if show_all:
            return a, probs.log_prob(a).sum(1), probs.entropy().sum(1), self.critic(x), probs
        return a, probs.log_prob(a).sum(1), probs.entropy().sum(1), self.critic(x)


def compute_kld(mu_1, sigma_1, mu_2, sigma_2):
    return torch.log(sigma_2 / sigma_1) + ((mu_1 - mu_2) ** 2 + (sigma_1 ** 2 - sigma_2 ** 2)) / (2 * sigma_2 ** 2)


def main(env_id, seed):
    args = get_args()
    args.env_id = env_id
    args.seed = seed
    run_name = (
            'spo' +
            '_epoch_' + str(args.update_epochs) +
            '_seed_' + str(args.seed)
    )

    # Save training logs
    path_string = str(args.env_id) + '/' + run_name
    writer = SummaryWriter(path_string)
    writer.add_text(
        'Hyperparameter',
        '|param|value|\n|-|-|\n%s' % ('\n'.join([f'|{key}|{value}|' for key, value in vars(args).items()])),
    )

    # Random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device('cuda' if torch.cuda.is_available() and args.cuda else 'cpu')

    # Initialize environments
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, args.gamma) for _ in range(args.num_envs)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Box), 'only continuous action space is supported'

    agent = Agent(envs).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # Initialize buffer
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    log_probs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)
    mean = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    std = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)

    # Data collection
    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)
    num_updates = args.total_time_steps // args.batch_size

    for update in tqdm(range(1, num_updates + 1)):

        # Linear decay of learning rate
        if args.anneal_lr:
            frac = 1.0 - (update - 1.0) / num_updates
            lr_now = frac * args.learning_rate
            optimizer.param_groups[0]['lr'] = lr_now

        for step in range(0, args.num_steps):
            global_step += 1 * args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            # Compute the logarithm of the action probability output by the old policy network
            with torch.no_grad():
                action, log_prob, _, value, mean_std = agent.get_action_and_value(next_obs, show_all=True)
                values[step] = value.flatten()
            actions[step] = action
            log_probs[step] = log_prob

            # Mean and variance (mini_batch_size, num_envs, action_dim)
            mean[step] = mean_std.loc
            std[step] = mean_std.scale

            # Update the environments
            next_obs, reward, terminations, truncations, info = envs.step(action.cpu().numpy())
            done = np.logical_or(terminations, truncations)
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(done).to(device)

            if 'final_info' not in info:
                continue

            for item in info['final_info']:
                if item is None:
                    continue
                writer.add_scalar('charts/episodic_return', item['episode']['r'][0], global_step)

        # Use GAE (Generalized Advantage Estimation) technique to estimate the advantage function
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
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
                advantages[t] = last_gae_lam = delta + args.gamma * args.gae_lambda * next_non_terminal * last_gae_lam
            returns = advantages + values

        # ---------------------- We have collected enough data, now let's start training ---------------------- #
        # Flatten each batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_log_probs = log_probs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Obtain the mean and variance of a batch
        b_mean = mean.reshape(args.batch_size, -1)
        b_std = std.reshape(args.batch_size, -1)

        # Update the policy network and value network
        b_index = np.arange(args.batch_size)
        for epoch in range(1, args.update_epochs + 1):
            np.random.shuffle(b_index)
            t = 0
            for start in range(0, args.batch_size, args.minibatch_size):
                t += 1
                end = start + args.minibatch_size
                mb_index = b_index[start:end]

                # The latest outputs of the policy network and value network
                _, new_log_prob, entropy, new_value, new_mean_std = agent.get_action_and_value(b_obs[mb_index],
                                                                                               b_actions[mb_index],
                                                                                               show_all=True)
                # Compute KL divergence
                new_mean = new_mean_std.loc.reshape(args.minibatch_size, -1)
                new_std = new_mean_std.scale.reshape(args.minibatch_size, -1)
                d = compute_kld(b_mean[mb_index], b_std[mb_index], new_mean, new_std).sum(1)

                writer.add_scalar('charts/average_kld', d.mean(), global_step)
                writer.add_scalar('others/min_kld', d.min(), global_step)
                writer.add_scalar('others/max_kld', d.max(), global_step)

                log_ratio = new_log_prob - b_log_probs[mb_index]
                ratios = log_ratio.exp()
                mb_advantages = b_advantages[mb_index]

                # Advantage normalization
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-12)

                # Policy loss (main code of SPO)
                if epoch == 1 and t == 1:
                    pg_loss = (-mb_advantages * ratios).mean()
                else:
                    # d_clip
                    d_clip = torch.clamp(input=d, min=0, max=args.kld_max)
                    # d_clip / d
                    ratio = d_clip / (d + 1e-12)
                    # sign_a
                    sign_a = torch.sign(mb_advantages)
                    # (d_clip / d + sign_a - 1) * sign_a
                    result = (ratio + sign_a - 1) * sign_a
                    pg_loss = (-mb_advantages * ratios * result).mean()

                # Value loss
                new_value = new_value.view(-1)
                if args.clip_value_loss:
                    v_loss_un_clipped = (new_value - b_returns[mb_index]) ** 2
                    v_clipped = b_values[mb_index] + torch.clamp(
                        new_value - b_values[mb_index],
                        -0.2,
                        0.2,
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
                writer.add_scalar('losses/policy_loss', pg_loss.item(), global_step)
                writer.add_scalar('losses/value_loss', v_loss.item(), global_step)
                writer.add_scalar('losses/entropy', entropy_loss.item(), global_step)

                # Update network parameters
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

        y_pre, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pre) / var_y
        writer.add_scalar('others/explained_variance', explained_var, global_step)

        # Save the data during the training process
        writer.add_scalar('charts/learning_rate', optimizer.param_groups[0]['lr'], global_step)
        writer.add_scalar('charts/SPS', int(global_step / (time.time() - start_time)), global_step)

    envs.close()
    writer.close()


def run():
    for env_id in ['Humanoid-v4']:
        for seed in range(1, 6):
            print(env_id, 'seed:', seed)
            main(env_id, seed)


if __name__ == '__main__':
    run()
