import torch.optim as optim
from torch.optim import Adam
import gym
from agents.DDPG import core
from common.preprocessing import *
from common.rl import *
import json


class DDPG:
    """
    Deep Deterministic Policy Gradient (DDPG)
    """
    def __init__(self, building_ids,
                 buildings_states_actions,
                 building_info,
                 observation_spaces=None,
                 action_spaces=None,
                 hidden_dim=[256, 256],
                 discount=0.99,
                 tau=5e-3,
                 polyak=0.995,
                 lr_actor=2e-4,
                 lr_critic=1e-3,
                 batch_size=256,
                 replay_buffer_capacity=5e5,
                 start_training=100,
                 exploration_period=8000,
                 action_scaling_coef=0.5,
                 reward_scaling=5.,
                 update_per_step=10,
                 act_noise=0.2,
                 seed=0):

        with open(buildings_states_actions) as json_file:
            self.buildings_states_actions = json.load(json_file)

        self.building_ids = building_ids
        self.start_training = start_training
        self.discount = discount
        self.batch_size = batch_size
        self.tau = tau
        self.action_scaling_coef = action_scaling_coef
        self.reward_scaling = reward_scaling
        torch.manual_seed(seed)
        np.random.seed(seed)
        # self.deterministic = False
        self.update_per_step = update_per_step
        self.exploration_period = exploration_period

        self.action_list_ = []

        self.time_step = 0
        self.norm_flag = {uid: 0 for uid in building_ids}
        self.action_spaces = {uid: a_space for uid, a_space in zip(building_ids, action_spaces)}
        self.action_limit = {uid: 0 for uid in building_ids}
        self.observation_spaces = {uid: o_space for uid, o_space in zip(building_ids, observation_spaces)}

        # device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print('Device:' + "cuda" if torch.cuda.is_available() else "cpu")

        self.critic_loss_, self.actor_loss_ = {}, {}

        self.replay_buffer, self.critic_net, self.target_critic_net, self.actor_net, self.target_actor_net, self.critic_optimizer, self.actor_optimizer, self.encoder, self.norm_mean, self.norm_std, self.r_norm_mean, self.r_norm_std, self.norm_mean, self.norm_std, self.r_norm_mean, self.r_norm_std = {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}

        for uid in building_ids:
            self.critic_loss_[uid], self.actor_loss_[uid] = [], []
            self.encoder[uid] = []
            state_n = 0
            for s_name, s in self.buildings_states_actions[uid]['states'].items():
                if not s:
                    self.encoder[uid].append(0)
                elif s_name in ["month", "hour"]:
                    self.encoder[uid].append(periodic_normalization(self.observation_spaces[uid].high[state_n]))
                    state_n += 1
                elif s_name == "day":
                    self.encoder[uid].append(onehot_encoding([1, 2, 3, 4, 5, 6, 7,8]))
                    state_n += 1
                elif s_name == "daylight_savings_status":
                    self.encoder[uid].append(onehot_encoding([0, 1]))
                    state_n += 1
                elif s_name == "net_electricity_consumption":
                    self.encoder[uid].append(remove_feature())
                    state_n += 1
                else:
                    self.encoder[uid].append(normalize(self.observation_spaces[uid].low[state_n],
                                                       self.observation_spaces[uid].high[state_n]))
                    state_n += 1

            self.encoder[uid] = np.array(self.encoder[uid])

            # If there is no solar PV installed, remove solar radiation variables
            if building_info[uid]['solar_power_capacity (kW)'] == 0:
                for k in range(12, 20):
                    if self.encoder[uid][k] != 0:
                        self.encoder[uid][k] = -1
                if self.encoder[uid][24] != 0:
                    self.encoder[uid][24] = -1
            if building_info[uid]['Annual_DHW_demand (kWh)'] == 0 and self.encoder[uid][26] != 0:
                self.encoder[uid][26] = -1
            if building_info[uid]['Annual_cooling_demand (kWh)'] == 0 and self.encoder[uid][25] != 0:
                self.encoder[uid][25] = -1
            if building_info[uid]['Annual_nonshiftable_electrical_demand (kWh)'] == 0 and self.encoder[uid][23] != 0:
                self.encoder[uid][23] = -1

            self.encoder[uid] = self.encoder[uid][self.encoder[uid] != 0]
            self.encoder[uid][self.encoder[uid] == -1] = remove_feature()

            state_dim = len(
                [j for j in np.hstack(self.encoder[uid] * np.ones(len(self.observation_spaces[uid].low))) if
                 j is not None])

            action_dim = self.action_spaces[uid].shape[0]
            self.action_limit[uid] = self.action_spaces[uid].high[0]

            self.replay_buffer[uid] = ReplayBuffer(int(replay_buffer_capacity))

            # init critic networks
            self.critic_net[uid] = core.Critic(state_dim, action_dim, hidden_dim).to(self.device)
            self.target_critic_net[uid] = core.Critic(state_dim, action_dim, hidden_dim).to(self.device)
            for target_param, param in zip(self.target_critic_net[uid].parameters(),
                                           self.critic_net[uid].parameters()):
                target_param.data.copy_(param.data)

            # actor network
            self.actor_net[uid] = core.Actor(state_dim, action_dim, hidden_dim).to(self.device)
            self.target_actor_net[uid] = core.Actor(state_dim, action_dim, hidden_dim).to(self.device)

            # optimizers
            self.critic_optimizer[uid] = optim.Adam(self.critic_net[uid].parameters(), lr=lr_critic)
            self.actor_optimizer[uid] = optim.Adam(self.actor_net[uid].parameters(), lr=lr_actor)

    def select_action(self, states, noise_scale):

        self.time_step += 1
        explore = self.time_step <= self.exploration_period
        actions = []
        k = 0
        # deterministic = (self.time_step > 3 * 8760)
        for uid, state in zip(self.building_ids, states):
            if explore:
                actions.append(self.action_scaling_coef * self.action_spaces[uid].sample())
            else:
                state_ = np.array([j for j in np.hstack(self.encoder[uid] * state) if j is not None])

                state_ = (state_ - self.norm_mean[uid]) / self.norm_std[uid]
                state_ = torch.FloatTensor(state_).unsqueeze(0).to(self.device)

                self.actor_net[uid].eval()
                with torch.no_grad():
                    act = self.actor_net(state_).cpu().data.numpy()
                self.actor_net[uid].train()
                act += noise_scale * np.random.randn(self.action_dim)
                act = np.clip(act, -self.action_limit[uid], self.action_limit[uid])
                actions.append(act.detach().cpu().numpy()[0])

        return np.array(actions)

    def add_to_buffer(self, states, actions, rewards, next_states, done):

        for (uid, o, a, r, o2,) in zip(self.building_ids, states, actions, rewards, next_states):
            # Run once the regression model has been fitted Normalize all the states using periodical normalization,
            # one-hot encoding, or -1, 1 scaling. It also removes states that are not necessary (solar radiation if
            # there are no solar PV panels).

            o = np.array([j for j in np.hstack(self.encoder[uid] * o) if j is not None])
            o2 = np.array([j for j in np.hstack(self.encoder[uid] * o2) if j is not None])

            if self.norm_flag[uid] > 0:
                o = (o - self.norm_mean[uid]) / self.norm_std[uid]
                o2 = (o2 - self.norm_mean[uid]) / self.norm_std[uid]
                r = (r - self.r_norm_mean[uid]) / self.r_norm_std[uid]

            self.replay_buffer[uid].push(o, a, r, o2, done)

        if self.time_step >= self.start_training and self.batch_size <= len(self.replay_buffer[self.building_ids[0]]):
            print("start training")
            for uid in self.building_ids:
                if self.norm_flag[uid] == 0:
                    X = np.array([j[0] for j in self.replay_buffer[uid].buffer])
                    self.norm_mean[uid] = np.mean(X, axis=0)
                    self.norm_std[uid] = np.std(X, axis=0) + 1e-5

                    R = np.array([j[2] for j in self.replay_buffer[uid].buffer])
                    self.r_norm_mean[uid] = np.mean(R)
                    self.r_norm_std[uid] = np.std(R) / self.reward_scaling + 1e-5

                    new_buffer = []
                    for s, a, r, s2, dones in self.replay_buffer[uid].buffer:
                        s_buffer = np.hstack(((s - self.norm_mean[uid]) / self.norm_std[uid]).reshape(1, -1)[0])
                        s2_buffer = np.hstack(((s2 - self.norm_mean[uid]) / self.norm_std[uid]).reshape(1, -1)[0])
                        new_buffer.append(
                            (s_buffer, a, (r - self.r_norm_mean[uid]) / self.r_norm_std[uid], s2_buffer, dones))

                    self.replay_buffer[uid].buffer = new_buffer
                    self.norm_flag[uid] = 1

            for _ in range(self.update_per_step):
                for uid in self.building_ids:
                    state, action, reward, next_state, done = self.replay_buffer[uid].sample(self.batch_size)

                    if self.device.type == "cuda":
                        state = torch.cuda.FloatTensor(state).to(self.device)
                        next_state = torch.cuda.FloatTensor(next_state).to(self.device)
                        action = torch.cuda.FloatTensor(action).to(self.device)
                        reward = torch.cuda.FloatTensor(reward).unsqueeze(1).to(self.device)
                        done = torch.cuda.FloatTensor(done).unsqueeze(1).to(self.device)
                    else:
                        state = torch.FloatTensor(state).to(self.device)
                        next_state = torch.FloatTensor(next_state).to(self.device)
                        action = torch.FloatTensor(action).to(self.device)
                        reward = torch.FloatTensor(reward).unsqueeze(1).to(self.device)
                        done = torch.FloatTensor(done).unsqueeze(1).to(self.device)

                    with torch.no_grad():
                        # Update Q-values
                        next_action = self.target_actor_net[uid](state)

                        # The updated Q-value is found by subtracting the logprob of the sampled action (proportional
                        # to the entropy) to the Q-values estimated by the target networks.
                        q_target_next = self.target_critic_net[uid](torch.as_tensor(next_state), torch.as_tensor(next_action))
                        q_target = reward + (self.discount * q_target_next * (1 - done))

                        q_expected = self.critic_net[uid](state, action)

                    # Update Q-Networks
                    q_expected = self.critic_net[uid](state, action)

                    critic_loss = F.mse_loss(q_expected, q_target)

                    self.critic_optimizer[uid].zero_grad()
                    critic_loss.backward()
                    self.critic_optimizer[uid].step()

                    # Update Policy
                    action_pred = self.actor_net[uid](state)
                    actor_loss = -self.critic_net[uid](state, action_pred).mean()

                    # Minimize the loss
                    self.actor_optimizer[uid].zero_grad()
                    actor_loss.backward()
                    self.actor_optimizer[uid].step()

                    # ----------------------- update target networks ----------------------- #
                    for target_param, param in zip(self.target_critic_net[uid].parameters(),
                                                   self.critic_net[uid].parameters()):
                        target_param.data.copy_(
                            target_param.data * (1.0 - self.tau) + param.data * self.tau
                        )

