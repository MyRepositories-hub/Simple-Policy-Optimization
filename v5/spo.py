import argparse
import os
import time

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3.common.atari_wrappers import FireResetEnv, EpisodicLifeEnv, ClipRewardEnv
from torch import nn, optim
from torch.distributions import Categorical
from torch.nn.utils.clip_grad import clip_grad_norm_
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm import tqdm


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, default=os.path.basename(__file__).rstrip('.py'))
    parser.add_argument('--env_id', type=str, default='ALE/Breakout-v5')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--use_cuda', type=bool, default=True)
    parser.add_argument('--learning_rate', type=float, default=2.5e-4)
    parser.add_argument('--lr_decay', type=bool, default=True)
    parser.add_argument('--total_steps', type=int, default=int(1e7))
    parser.add_argument('--num_envs', type=int, default=8)
    parser.add_argument('--num_steps', type=int, default=128)
    parser.add_argument('--update_epochs', type=int, default=8)
    parser.add_argument('--num_mini_batches', type=int, default=4)
    parser.add_argument('--gae_lambda', type=float, default=0.95)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--clip_value_loss', type=bool, default=True)
    parser.add_argument('--c_1', type=float, default=1.0)
    parser.add_argument('--c_2', type=float, default=0.01)
    parser.add_argument('--clip_grad_norm', type=float, default=0.5)
    parser.add_argument('--kld_max', type=float, default=0.02)
    args = parser.parse_args()
    args.device = torch.device('cuda' if torch.cuda.is_available() and args.use_cuda else 'cpu')
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_mini_batches)
    args.num_updates = int(args.total_steps // args.batch_size)
    return args


def make_env(env_id):
    def thunk():
        env = gym.make(env_id, frameskip=1, repeat_action_probability=0.0, full_action_space=False)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if 'FIRE' in env.unwrapped.get_action_meanings():
            env = FireResetEnv(env)
        env = EpisodicLifeEnv(env)
        env = ClipRewardEnv(env)
        env = gym.wrappers.AtariPreprocessing(env, scale_obs=True)
        env = gym.wrappers.FrameStack(env, 4)
        return env
    return thunk


def compute_advantages(rewards, flags, values, last_value, args):
    advantages = torch.zeros((args.num_steps, args.num_envs)).to(args.device)
    adv = torch.zeros(args.num_envs).to(args.device)
    for i in reversed(range(args.num_steps)):
        returns = rewards[i] + args.gamma * flags[i] * last_value
        delta = returns - values[i]
        adv = delta + args.gamma * args.gae_lambda * flags[i] * adv
        advantages[i] = adv
        last_value = values[i]
    return advantages


class Buffer:
    def __init__(self, num_steps, num_envs, observation_shape, action_dim, device):
        self.states = np.zeros((num_steps, num_envs, *observation_shape), dtype=np.float32)
        self.actions = np.zeros((num_steps, num_envs), dtype=np.int64)
        self.rewards = np.zeros((num_steps, num_envs), dtype=np.float32)
        self.flags = np.zeros((num_steps, num_envs), dtype=np.float32)
        self.log_probs = np.zeros((num_steps, num_envs), dtype=np.float32)
        self.probs = np.zeros((num_steps, num_envs, action_dim), dtype=np.float32)
        self.values = np.zeros((num_steps, num_envs), dtype=np.float32)
        self.step = 0
        self.num_steps = num_steps
        self.device = device

    def push(self, state, action, reward, flag, log_prob, prob, value):
        self.states[self.step] = state
        self.actions[self.step] = action
        self.rewards[self.step] = reward
        self.flags[self.step] = flag
        self.log_probs[self.step] = log_prob
        self.probs[self.step] = prob
        self.values[self.step] = value
        self.step = (self.step + 1) % self.num_steps

    def get(self):
        return (
            torch.from_numpy(self.states).to(self.device),
            torch.from_numpy(self.actions).to(self.device),
            torch.from_numpy(self.rewards).to(self.device),
            torch.from_numpy(self.flags).to(self.device),
            torch.from_numpy(self.log_probs).to(self.device),
            torch.from_numpy(self.values).to(self.device),
        )

    def get_probs(self):
        return torch.from_numpy(self.probs).to(self.device)


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, action_dim, device):
        super().__init__()
        self.encoder = nn.Sequential(
            layer_init(nn.Conv2d(4, 32, 8, stride=4)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, 4, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, 3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(64 * 7 * 7, 512)),
            nn.ReLU()
        )
        self.actor_net = layer_init(nn.Linear(512, action_dim), std=0.01)
        self.critic_net = layer_init(nn.Linear(512, 1), std=1)

        if device.type == 'cuda':
            self.cuda()

    def forward(self, state):
        hidden = self.encoder(state)
        actor_value = self.actor_net(hidden)
        distribution = Categorical(logits=actor_value)
        action = distribution.sample()
        log_prob = distribution.log_prob(action)
        value = self.critic_net(hidden).squeeze(-1)
        return action, log_prob, value, distribution.probs

    def evaluate(self, states, actions):
        hidden = self.encoder(states)
        actor_values = self.actor_net(hidden)
        distribution = Categorical(logits=actor_values)
        log_probs = distribution.log_prob(actions)
        entropy = distribution.entropy()
        values = self.critic_net(hidden).squeeze(-1)
        return log_probs, values, entropy, distribution.probs

    def critic(self, state):
        return self.critic_net(self.encoder(state)).squeeze(-1)


def train(env_id, seed):
    args = get_args()
    args.env_id = env_id
    args.seed = seed
    run_name = (
            'spo_' + str(args.kld_max) +
            '_epoch_' + str(args.update_epochs) +
            '_seed_' + str(args.seed)
    )

    # 保存训练日志
    path_string = str(args.env_id)[4:] + '/' + run_name
    writer = SummaryWriter(path_string)
    writer.add_text(
        'Hyperparameter',
        '|param|value|\n|-|-|\n%s' % ('\n'.join([f'|{key}|{value}|' for key, value in vars(args).items()])),
    )

    # 初始化并行环境
    envs = gym.vector.AsyncVectorEnv([make_env(args.env_id) for _ in range(args.num_envs)])

    # 状态空间和动作空间
    observation_shape = envs.single_observation_space.shape
    action_dim = envs.single_action_space.n

    # 随机数种子
    if args.seed:
        numpy_rng = np.random.default_rng(args.seed)
        torch.manual_seed(args.seed)
        state, _ = envs.reset(seed=args.seed)
    else:
        numpy_rng = np.random.default_rng()
        state, _ = envs.reset()

    # 价值网络和策略网络
    agent = Agent(action_dim, args.device)

    # 优化器
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate)

    # 存储数据的buffer
    rollout_buffer = Buffer(args.num_steps, args.num_envs, observation_shape, action_dim, args.device)
    global_step = 0
    start_time = time.time()

    # 开始收集数据
    for _ in tqdm(range(args.num_updates)):

        # 学习率线性递减
        if args.lr_decay:
            optimizer.param_groups[0]['lr'] -= (args.learning_rate - 1e-12) / args.num_updates

        for _ in range(args.num_steps):
            global_step += 1 * args.num_envs

            with torch.no_grad():
                action, log_prob, value, prob = agent(torch.from_numpy(state).to(args.device).float())

            action = action.cpu().numpy()
            next_state, reward, terminated, truncated, all_info = envs.step(action)

            # 存储数据
            flag = 1.0 - np.logical_or(terminated, truncated)
            log_prob = log_prob.cpu().numpy()
            prob = prob.cpu().numpy()
            value = value.cpu().numpy()
            rollout_buffer.push(state, action, reward, flag, log_prob, prob, value)
            state = next_state

            if 'final_info' not in all_info:
                continue

            # 写入训练过程的数据
            for info in all_info['final_info']:
                if info is None:
                    continue
                if 'episode' in info.keys():
                    writer.add_scalar('charts/episodic_return', info['episode']['r'], global_step)
                    # print(float(info['episode']['r']))
                    break

        # ------------------------------- 上面收集了足够的数据，下面开始更新 ------------------------------- #
        states, actions, rewards, flags, log_probs, values = rollout_buffer.get()
        probs = rollout_buffer.get_probs()

        with torch.no_grad():
            last_value = agent.critic(torch.from_numpy(next_state).to(args.device).float())

        # 计算优势值和TD目标
        advantages = compute_advantages(rewards, flags, values, last_value, args)
        td_target = advantages + values

        # 将数据展平
        states = states.reshape(-1, *observation_shape)
        actions = actions.reshape(-1)
        log_probs = log_probs.reshape(-1)
        probs = probs.reshape((-1, action_dim))
        td_target = td_target.reshape(-1)
        advantages = advantages.reshape(-1)
        values = values.reshape(-1)
        batch_indexes = np.arange(args.batch_size)

        # 更新策略网络和价值网络
        for e in range(1, args.update_epochs + 1):
            numpy_rng.shuffle(batch_indexes)
            t = 0
            for start in range(0, args.batch_size, args.minibatch_size):
                t += 1
                end = start + args.minibatch_size
                index = batch_indexes[start:end]

                # 得到最新的策略网络和价值网络输出
                new_log_probs, td_predict, entropy, new_probs = agent.evaluate(states[index], actions[index])
                log_ratio = new_log_probs - log_probs[index]
                ratios = log_ratio.exp()

                # 计算kl散度
                d = torch.sum(
                    probs[index] * torch.log((probs[index] + 1e-12) / (new_probs + 1e-12)), 1
                )
                writer.add_scalar('charts/average_kld', d.mean(), global_step)
                writer.add_scalar('others/min_kld', d.min(), global_step)
                writer.add_scalar('others/max_kld', d.max(), global_step)

                # 优势值标准化
                b_advantages = advantages[index]
                b_advantages = (b_advantages - b_advantages.mean()) / (b_advantages.std() + 1e-12)

                # 策略网络和价值网络损失
                if e == 1 and t == 1:
                    policy_loss = (-b_advantages * ratios).mean()
                else:
                    # d_clip
                    d_clip = torch.clamp(input=d, min=0, max=args.kld_max)
                    # d_clip / d
                    ratio = d_clip / (d + 1e-12)
                    # sign_a
                    sign_a = torch.sign(b_advantages)
                    # (d_clip / d + sign_a - 1) * sign_a
                    result = (ratio + sign_a - 1) * sign_a
                    # 策略网络损失
                    policy_loss = (-b_advantages * ratios * result).mean()

                # 价值网络损失
                if args.clip_value_loss:
                    v_loss_un_clipped = (td_predict - td_target[index]) ** 2
                    v_clipped = td_target[index] + torch.clamp(
                        td_predict - td_target[index],
                        -0.2,
                        0.2,
                    )
                    v_loss_clipped = (v_clipped - td_target[index]) ** 2
                    v_loss_max = torch.max(v_loss_un_clipped, v_loss_clipped)
                    value_loss = 0.5 * v_loss_max.mean()
                else:
                    value_loss = 0.5 * ((td_predict - td_target[index]) ** 2).mean()

                entropy_loss = entropy.mean()

                # 保存训练过程中的一些数据
                writer.add_scalar('losses/value_loss', value_loss.item(), global_step)
                writer.add_scalar('losses/policy_loss', policy_loss.item(), global_step)
                writer.add_scalar('losses/entropy', entropy_loss.item(), global_step)
                writer.add_scalar('losses/delta', torch.abs(ratios - 1).mean().item(), global_step)

                # 总的损失
                loss = policy_loss + value_loss * args.c_1 - entropy_loss * args.c_2

                # 更新网络参数
                optimizer.zero_grad()
                loss.backward()
                clip_grad_norm_(agent.parameters(), args.clip_grad_norm)
                optimizer.step()

        explained_var = (
            np.nan if torch.var(td_target) == 0 else 1 - torch.var(td_target - values) / torch.var(td_target)
        )
        writer.add_scalar('charts/learning_rate', optimizer.param_groups[0]['lr'], global_step)
        writer.add_scalar('charts/SPS', int(global_step / (time.time() - start_time)), global_step)
        writer.add_scalar('others/explained_var', explained_var, global_step)

    envs.close()
    writer.close()


def main():
    for env_id in ['Breakout']:
        for seed in [1, 2, 3]:
            print(env_id, seed)
            train('ALE/' + env_id + '-v5', seed)


if __name__ == '__main__':
    main()
