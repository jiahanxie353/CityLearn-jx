import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from itertools import chain
from typing import List, Tuple


class AttentionCritic(nn.Module):
    """
    Attention network, used as critic for all agents. Each agent gets itw own
    observation and action, and can also attend over the other agent's encoded
    observations and actions.
    """

    def __init__(self,
                 sa_sizes: List[Tuple[int, int]],
                 hidden_dim: int,
                 norm_in: bool = True,
                 attend_heads: int = 1
                 ):
        """
        Inputs:
        :param sa_sizes: Size of (state,action) [= (state dim, action dim)] spaces per agent
        :param hidden_dim: Number of hidden dimensions
        :param norm_in: Whether to apply batch norm to input
        :param attend_heads: Number of attention heads to use (needs to be
                             divisible by hidden_dim)
        """
        super(AttentionCritic, self).__init__()
        assert (hidden_dim % attend_heads) == 0
        self.sa_size = sa_sizes
        self.num_agents = len(sa_sizes)
        self.attend_heads = attend_heads

        # extract/embed the features
        self.s_encoders = nn.ModuleList()
        self.sa_encoders = nn.ModuleList()
        self.critics = nn.ModuleList()
        self.values = nn.ModuleList()

        # iterate over agents
        for state_dim, action_dim in sa_sizes:
            input_dim = state_dim + action_dim
            output_dim = 1

            s_encoder = nn.Sequential()
            if norm_in:
                pass
            s_encoder.add_module('state encoder fc 1', nn.Linear(state_dim, hidden_dim))
            s_encoder.add_module('state encoder activation', nn.LeakyReLU())
            self.s_encoders.append(s_encoder)

            sa_encoder = nn.Sequential()
            if norm_in:
                pass
                # perform batch normalization
                # sa_encoder.add_module('critic encoder batch norm',
                #                      nn.BatchNorm1d(input_dim, affine=False))
            sa_encoder.add_module('critic encoder fc 1', nn.Linear(input_dim, hidden_dim))
            sa_encoder.add_module('critic encoder activation', nn.LeakyReLU())
            self.sa_encoders.append(sa_encoder)

            critic = nn.Sequential()
            critic.add_module('critic fc 1', nn.Linear(2 * hidden_dim, hidden_dim))
            critic.add_module('critic activation 1', nn.LeakyReLU())
            critic.add_module('critic fc 2', nn.Linear(hidden_dim, output_dim))
            self.critics.append(critic)

            value = nn.Sequential()
            value.add_module('value fc 1', nn.Linear(2 * hidden_dim, hidden_dim))
            value.add_module('value activation 1', nn.LeakyReLU())
            value.add_module('value fc 2', nn.Linear(hidden_dim, output_dim))
            self.values.append(value)

            attend_dim = hidden_dim // attend_heads
            self.key_extractors = nn.ModuleList()
            self.selector_extractors = nn.ModuleList()
            self.value_extractors = nn.ModuleList()

            for i in range(attend_heads):
                self.key_extractors.append(nn.Linear(hidden_dim, attend_dim, bias=False))
                self.selector_extractors.append(nn.Linear(hidden_dim, attend_dim, bias=False))
                self.value_extractors.append(nn.Sequential(nn.Linear(hidden_dim,
                                                                     attend_dim),
                                                           nn.LeakyReLU()))

            self.shared_modules = [self.key_extractors, self.selector_extractors,
                                   self.value_extractors, self.sa_encoders]

    def shared_parameters(self):
        """
        Parameters shared across agents and reward heads
        :return:
        """
        return chain(*[m.parameters() for m in self.shared_modules])

    def scale_shared_grads(self):
        """
        Scale gradients for parameters that are shared since they accumulate
        gradients from the critic loss function multiple times
        """
        for p in self.shared_parameters():
            p.grad.data.mul_(1. / self.num_agents)

    def forward(self, inps, agents=None, return_q=True, return_v=True, regularize=False):
        """
        Inputs:
        :param inps: (list of PyTorch matrices) Inputs to each agent's encoder (batch of obs + ac)
                for the target critic network, the actions are sampled from the *current* policy;
                for the critic network, the actions come from a batch of replay buffer.
        :param agents: indices of agents
        :param return_q: return Q-value
        :param regularize: return values to add to loss function for regularization
        :return:
        """
        if agents is None:
            agents = range(len(self.sa_encoders))
        states = [s for s, a in inps]
        s_encodings = [self.s_encoders[a_i](states[a_i]) for a_i in agents]
        inps = [torch.cat((s, a), dim=1) for s, a in inps]
        # extract state-action encoding for each agent
        sa_encodings = [sa_encoder(inp) for sa_encoder, inp in zip(self.sa_encoders, inps)]
        # extract keys for each head for each agent
        all_head_keys = [[k_ext(enc) for enc in sa_encodings] for k_ext in self.key_extractors]
        # extract sa values for each head for each agent
        all_head_values = [[v_ext(enc) for enc in sa_encodings] for v_ext in self.value_extractors]
        # extract queries for each head for each agent
        all_head_selectors = [[sel_ext(enc) for i, enc in enumerate(sa_encodings) if i in agents]
                              for sel_ext in self.selector_extractors]

        other_all_values = [[] for _ in range(len(agents))]
        all_attend_logits = [[] for _ in range(len(agents))]

        # calculate attention per head
        for curr_head_keys, curr_head_values, curr_head_selectors in zip(
                all_head_keys, all_head_values, all_head_selectors):
            # iterate over agents
            for i, a_i, selector in zip(range(len(agents)), agents, curr_head_selectors):
                keys = [k for j, k in enumerate(curr_head_keys)]
                values = [v for j, v in enumerate(curr_head_values)]
                # calculate attention across agents
                attend_logits = torch.matmul(selector.view(selector.shape[0], 1, -1),
                                             torch.stack(keys).permute(1, 2, 0))
                # scale dot-products by size of key (from Attention is All You Need)
                scaled_attend_logits = attend_logits / np.sqrt(keys[0].shape[1])
                attend_weights = F.softmax(scaled_attend_logits, dim=2)
                other_values = (torch.stack(values).permute(1, 2, 0) *
                                attend_weights).sum(dim=2)
                other_all_values[i].append(other_values)
                all_attend_logits[i].append(attend_logits)

        # calculate Q per agent
        all_rets = []
        for i, a_i in enumerate(agents):
            agent_rets = []
            critic_in = torch.cat((sa_encodings[i], *other_all_values[i]), dim=1)
            value_in = torch.cat((s_encodings[i], *other_all_values[i]), dim=1)
            one_q = self.critics[a_i](critic_in)
            one_v = self.values[a_i](value_in)
            if return_q:
                agent_rets.append(one_q)
            if return_v:
                agent_rets.append(one_v)
            if regularize:
                # regularize magnitude of attention logits
                attend_magn_reg = 1e-3 * sum((logit ** 2).mean() for logit in all_attend_logits[i])
                regs = (attend_magn_reg,)
                agent_rets.append(regs)
            if len(agent_rets) == 1:
                all_rets.append(agent_rets[0])
            else:
                all_rets.append(agent_rets)
        if len(all_rets) == 1:
            return all_rets[0]
        else:
            return all_rets
