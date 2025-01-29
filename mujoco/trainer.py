import numpy as np
import torch
import torch.nn as nn


def compute_kld(mu_1, sigma_1, mu_2, sigma_2):
    return torch.log(sigma_2 / sigma_1) + ((mu_1 - mu_2) ** 2 + (sigma_1 ** 2 - sigma_2 ** 2)) / (2 * sigma_2 ** 2)


class Trainer:
    def __init__(self, args, agent, optimizer, writer):
        self.args = args
        self.agent = agent
        self.optimizer = optimizer
        self.writer = writer

    def train(self, global_step, b_obs, b_actions, b_log_probs, b_advantages, b_returns, b_values, b_mean, b_std):
        b_index = np.arange(self.args.batch_size)

        for epoch in range(self.args.update_epochs):
            np.random.shuffle(b_index)

            for start in range(0, self.args.batch_size, self.args.minibatch_size):
                end = start + self.args.minibatch_size
                mb_index = b_index[start:end]

                # The latest outputs of the policy network and value network
                _, new_log_prob, new_entropy, new_value, new_mean_std = self.agent.get_action_and_value(
                    b_obs[mb_index], b_actions[mb_index]
                )

                # Probability ratio
                log_ratio = new_log_prob - b_log_probs[mb_index]
                ratios = log_ratio.exp()

                # Adaptive learning rate for ppo
                new_mean = new_mean_std.loc.reshape(self.args.minibatch_size, -1)
                new_std = new_mean_std.scale.reshape(self.args.minibatch_size, -1)
                kl = compute_kld(b_mean[mb_index], b_std[mb_index], new_mean, new_std).sum(-1)
                if self.args.adaptive_learning_rate and self.args.algo == 'ppo':
                    if kl.mean() > self.args.desired_kl * 2.0:
                        self.args.learning_rate = max(1e-5, self.args.learning_rate / 1.5)
                        self.optimizer.param_groups[0]['lr'] = self.args.learning_rate
                    if kl.mean() < self.args.desired_kl / 2.0:
                        self.args.learning_rate = min(1e-2, self.args.learning_rate * 1.5)
                        self.optimizer.param_groups[0]['lr'] = self.args.learning_rate
                self.writer.add_scalar('This is for plotting/kl_divergence', kl.mean(), global_step)

                # Advantage normalization
                mb_advantages = b_advantages[mb_index]
                if self.args.advantage_normalization:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                policy_loss = self.compute_policy_loss(ratios, mb_advantages)

                # Value loss
                value_loss = self.compute_value_loss(new_value, b_returns[mb_index], b_values[mb_index])

                # Policy entropy
                entropy_loss = new_entropy.mean()

                # Total loss
                loss = policy_loss + value_loss * self.args.c_1 - entropy_loss * self.args.c_2

                # Save the data during the training process
                self.writer.add_scalar('charts/ratio_deviation', torch.abs(ratios - 1).mean(), global_step)
                self.writer.add_scalar('losses/policy_loss', policy_loss.item(), global_step)
                self.writer.add_scalar('losses/value_loss', value_loss.item(), global_step)
                self.writer.add_scalar('losses/entropy', entropy_loss.item(), global_step)

                # Update network parameters
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.agent.parameters(), self.args.max_grad_norm)
                self.optimizer.step()

    def compute_value_loss(self, new_value, mb_returns, mb_values):
        """
        Compute value loss
        """
        new_value = new_value.view(-1)
        if self.args.clip_value_loss:
            value_loss_un_clipped = (new_value - mb_returns) ** 2
            value_clipped = mb_values + torch.clamp(new_value - mb_values, -self.args.epsilon, self.args.epsilon)
            value_loss_clipped = (value_clipped - mb_returns) ** 2
            value_loss_max = torch.max(value_loss_un_clipped, value_loss_clipped)
            value_loss = 0.5 * value_loss_max.mean()
        else:
            value_loss = 0.5 * ((new_value - mb_returns) ** 2).mean()

        return value_loss

    def compute_policy_loss(self, ratios, mb_advantages):
        """
        Compute the policy loss
        """
        if self.args.algo == 'ppo':
            policy_loss_1 = mb_advantages * ratios
            policy_loss_2 = mb_advantages * torch.clamp(
                ratios, 1 - self.args.epsilon, 1 + self.args.epsilon
            )
            policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean()

        if self.args.algo == 'spo':
            policy_loss = -(
                    mb_advantages * ratios -
                    torch.abs(mb_advantages) * torch.pow(ratios - 1, 2) / (2 * self.args.epsilon)
            ).mean()

        return policy_loss
