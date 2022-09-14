import torch
import numpy as np
from torch.optim import Adam
from .policies import SquashedGaussianActor
from .buffer import ReplayBuffer, RegressionBuffer
from sklearn.linear_model import LinearRegression


class AttentionAgent(object):
    """
    General class for Attention agents (policy, target policy)
    """

    def __init__(self,
                 dim_in_actor,
                 dim_out_actor,
                 action_scaling_coef,
                 buffer_length,
                 reg_buffer_length,
                 start_regression,
                 regression_frequency,
                 action_spaces,
                 hidden_dim=(400, 300),
                 reward_scaling=5.,
                 lr=0.001):
        """
        Inputs:
        :param dim_in_actor: number of dimensions for policy input
        :param dim_out_actor: number of dimensions for policy output
        :param hidden_dim:
        :param lr:
        """
        self.norm_mean = 0
        self.norm_std = 0
        self.r_norm_mean = 0
        self.r_norm_std = 0
        self.norm_flag = 0
        self.regression_flag = 0
        self.regression_freq = regression_frequency
        self.buffer_length = buffer_length
        self.reg_buffer_length = reg_buffer_length
        self.replay_buffer = ReplayBuffer(self.buffer_length)
        self.start_regression = start_regression
        self.reg_buffer = RegressionBuffer(self.reg_buffer_length)
        self.action_spaces = action_spaces
        self.reward_scaling = reward_scaling
        self.policy = SquashedGaussianActor(dim_in_actor, dim_out_actor, action_spaces, action_scaling_coef, hidden_dim)
        self.target_policy = SquashedGaussianActor(dim_in_actor, dim_out_actor, action_spaces, action_scaling_coef, hidden_dim)
        # Freeze target networks with respect to optimizers (only update via polyak averaging)
        for p in self.target_policy.parameters():
            p.requires_grad = False
        self.policy_optimizer = Adam(self.policy.parameters(), lr=lr)
        self.expected_demand = 0
        self.expected_demand_next = 0
        self.elec_estimator = LinearRegression()

    def step(self,
             obs: torch.Tensor,
             action_spaces,
             encoder,
             encoder_reg,
             time_step,
             explore: bool = False,
             deterministic: bool = False
             ) -> torch.Tensor:
        """
        Take a step forward in environment for a minibatch of observations
        Inputs:
        :param time_step:
        :param encoder_reg:
        :param encoder:
        :param action_spaces:
        :param obs: Observations for this agent
        :param explore: Whether to sample or not
        :param deterministic:
        Outputs:
        :return: action: Action for this agent
        """
        return self.policy.choose_action(obs, action_spaces, encoder, encoder_reg, time_step, self.start_regression,
                                         self.elec_estimator, self.norm_mean, self.norm_std, explore, deterministic)

    def normalize_buffer(self):
        if self.norm_flag == 0:
            # normalizing the states in replay buffer
            S = np.array([j[0] for j in self.replay_buffer.buffer])
            self.norm_mean = np.mean(S, axis=0)
            self.norm_std = np.std(S, axis=0) + 1e-2

            # normalizing the rewards in replay buffer
            R = np.array([j[2] for j in self.replay_buffer.buffer])
            self.r_norm_mean = np.mean(R)
            self.r_norm_std = np.std(R) / self.reward_scaling + 1e-2

            new_buffer = []
            for s, a, r, s2, dones in self.replay_buffer.buffer:
                s_buffer = np.hstack(((s - self.norm_mean) / self.norm_std).reshape(1, -1)[0])
                s2_buffer = np.hstack(((s2 - self.norm_mean) / self.norm_std).reshape(1, -1)[0])
                new_buffer.append(
                    (s_buffer, a, (r - self.r_norm_mean) / self.r_norm_std, s2_buffer, dones))

            self.replay_buffer.buffer = new_buffer
            self.norm_flag = 1

    def add_to_buffer(self, encoder, encoder_reg, state, act, reward, next_state, done, time_step,
                      expected_demand, expected_demand_next):
        # Run once the regression model has been fitted. Normalize all the states using periodical normalization,
        # one-hot encoding, or -1, 1 scaling. It also removes states that are not necessary (solar radiation if
        # there are no solar PV panels).
        x_reg = np.array([j for j in np.hstack(encoder_reg * state) if j is not None][:-1])
        y_reg = [j for j in np.hstack(encoder_reg * next_state) if j is not None][-1]

        # Push inputs and targets to the regression buffer. The targets are the net electricity consumption.
        self.reg_buffer.push(x_reg, y_reg)

        # Run once the regression model has been fitted
        if self.regression_flag > 1:
            o = np.concatenate((np.array([j for j in np.hstack(encoder * state) if j is not None]), expected_demand))
            o2 = np.concatenate((
                np.array([j for j in np.hstack(encoder * next_state) if j is not None]), expected_demand_next))

            if self.norm_flag > 0:
                o = (o - self.norm_mean) / self.norm_std
                o2 = (o2 - self.norm_mean) / self.norm_std
                reward = (reward - self.r_norm_mean) / self.r_norm_std

            self.replay_buffer.push(o, act, reward, o2, done)

        if time_step >= self.start_regression and (
                self.regression_flag < 2 or time_step % self.regression_freq == 0):
            # Fit regression model for the first time.
            self.elec_estimator.fit(self.reg_buffer.x, self.reg_buffer.y)

            if self.regression_flag < 2:
                self.regression_flag += 1
