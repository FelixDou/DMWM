from typing import List, Optional

import numpy as np
import torch
import torch.distributions
from torch import jit, nn
from torch.distributions.normal import Normal
from torch.distributions.transformed_distribution import TransformedDistribution
from torch.nn import functional as F
import utils
import random


def bottle(f, x_tuple):
    x_sizes = tuple(map(lambda x: x.size(), x_tuple))
    y = f(*map(lambda x: x[0].view(x[1][0] * x[1][1], *x[1][2:]), zip(x_tuple, x_sizes)))
    y_size = y.size()
    output = y.view(x_sizes[0][0], x_sizes[0][1], *y_size[1:])
    return output


class TransitionModel(jit.ScriptModule):
    __constants__ = ['min_std_dev']

    def __init__(
            self,
            belief_size,
            state_size,
            action_size,
            hidden_size,
            embedding_size,
            activation_function='relu',
            min_std_dev=0.1,
    ):
        super().__init__()
        self.act_fn = getattr(F, activation_function)
        self.min_std_dev = min_std_dev
        self.fc_embed_state_action = nn.Linear(state_size + action_size, belief_size)
        self.rnn = nn.GRUCell(belief_size, belief_size)
        self.fc_embed_belief_prior = nn.Linear(belief_size, hidden_size)
        self.fc_state_prior = nn.Linear(hidden_size, 2 * state_size)
        self.fc_embed_belief_posterior = nn.Linear(belief_size + embedding_size, hidden_size)
        self.fc_state_posterior = nn.Linear(hidden_size, 2 * state_size)
        self.modules = [
            self.fc_embed_state_action,
            self.fc_embed_belief_prior,
            self.fc_state_prior,
            self.fc_embed_belief_posterior,
            self.fc_state_posterior,
        ]

    @jit.script_method
    def forward(
            self,
            prev_state: torch.Tensor,
            actions: torch.Tensor,
            prev_belief: torch.Tensor,
            observations: Optional[torch.Tensor] = None,
            nonterminals: Optional[torch.Tensor] = None,
    ) -> List[torch.Tensor]:

        T = actions.size(0) + 1
        beliefs, prior_states, prior_means, prior_std_devs, posterior_states, posterior_means, posterior_std_devs = (
            [torch.empty(0)] * T,
            [torch.empty(0)] * T,
            [torch.empty(0)] * T,
            [torch.empty(0)] * T,
            [torch.empty(0)] * T,
            [torch.empty(0)] * T,
            [torch.empty(0)] * T,
        )
        beliefs[0], prior_states[0], posterior_states[0] = prev_belief, prev_state, prev_state
        # Loop over time sequence
        for t in range(T - 1):
            _state = (
                prior_states[t] if observations is None else posterior_states[t]
            )  # Select appropriate previous state
            _state = (
                _state if nonterminals is None else _state * nonterminals[t]
            )  # Mask if previous transition was terminal
            # Compute belief (deterministic hidden state)
            hidden = self.act_fn(self.fc_embed_state_action(torch.cat([_state, actions[t]], dim=1)))
            beliefs[t + 1] = self.rnn(hidden, beliefs[t])
            # Compute state prior by applying transition dynamics
            hidden = self.act_fn(self.fc_embed_belief_prior(beliefs[t + 1]))
            prior_means[t + 1], _prior_std_dev = torch.chunk(self.fc_state_prior(hidden), 2, dim=1)
            prior_std_devs[t + 1] = F.softplus(_prior_std_dev) + self.min_std_dev
            prior_states[t + 1] = prior_means[t + 1] + prior_std_devs[t + 1] * torch.randn_like(prior_means[t + 1])
            if observations is not None:
                # Compute state posterior by applying transition dynamics and using current observation
                t_ = t - 1  # Use t_ to deal with different time indexing for observations
                hidden = self.act_fn(
                    self.fc_embed_belief_posterior(torch.cat([beliefs[t + 1], observations[t_ + 1]], dim=1))
                )
                posterior_means[t + 1], _posterior_std_dev = torch.chunk(self.fc_state_posterior(hidden), 2, dim=1)
                posterior_std_devs[t + 1] = F.softplus(_posterior_std_dev) + self.min_std_dev
                posterior_states[t + 1] = posterior_means[t + 1] + posterior_std_devs[t + 1] * torch.randn_like(
                    posterior_means[t + 1]
                )
        # Return new hidden states
        hidden = [
            torch.stack(beliefs[1:], dim=0),
            torch.stack(prior_states[1:], dim=0),
            torch.stack(prior_means[1:], dim=0),
            torch.stack(prior_std_devs[1:], dim=0),
        ]
        if observations is not None:
            hidden += [
                torch.stack(posterior_states[1:], dim=0),
                torch.stack(posterior_means[1:], dim=0),
                torch.stack(posterior_std_devs[1:], dim=0),
            ]
        return hidden


class SymbolicObservationModel(jit.ScriptModule):
    def __init__(self, observation_size, belief_size, state_size, embedding_size, activation_function='relu'):
        super().__init__()
        self.act_fn = getattr(F, activation_function)
        self.fc1 = nn.Linear(belief_size + state_size, embedding_size)
        self.fc2 = nn.Linear(embedding_size, embedding_size)
        self.fc3 = nn.Linear(embedding_size, observation_size)
        self.modules = [self.fc1, self.fc2, self.fc3]

    @jit.script_method
    def forward(self, belief, state):
        hidden = self.act_fn(self.fc1(torch.cat([belief, state], dim=1)))
        hidden = self.act_fn(self.fc2(hidden))
        observation = self.fc3(hidden)
        return observation


class VisualObservationModel(jit.ScriptModule):
    __constants__ = ['embedding_size']

    def __init__(self, belief_size, state_size, embedding_size, activation_function='relu'):
        super().__init__()
        self.act_fn = getattr(F, activation_function)
        self.embedding_size = embedding_size
        self.fc1 = nn.Linear(belief_size + state_size, embedding_size)
        self.conv1 = nn.ConvTranspose2d(embedding_size, 128, 5, stride=2)
        self.conv2 = nn.ConvTranspose2d(128, 64, 5, stride=2)
        self.conv3 = nn.ConvTranspose2d(64, 32, 6, stride=2)
        self.conv4 = nn.ConvTranspose2d(32, 3, 6, stride=2)
        self.modules = [self.fc1, self.conv1, self.conv2, self.conv3, self.conv4]

    @jit.script_method
    def forward(self, belief, state):
        hidden = self.fc1(torch.cat([belief, state], dim=1))  # No nonlinearity here
        hidden = hidden.view(-1, self.embedding_size, 1, 1)
        hidden = self.act_fn(self.conv1(hidden))
        hidden = self.act_fn(self.conv2(hidden))
        hidden = self.act_fn(self.conv3(hidden))
        observation = self.conv4(hidden)
        return observation


def ObservationModel(symbolic, observation_size, belief_size, state_size, embedding_size, activation_function='relu'):
    if symbolic:
        return SymbolicObservationModel(observation_size, belief_size, state_size, embedding_size, activation_function)
    else:
        return VisualObservationModel(belief_size, state_size, embedding_size, activation_function)


class RewardModel(jit.ScriptModule):
    def __init__(self, belief_size, state_size, hidden_size, activation_function='relu'):
        # [--belief-size: 200, --hidden-size: 200, --state-size: 30]
        super().__init__()
        self.act_fn = getattr(F, activation_function)
        self.fc1 = nn.Linear(belief_size + state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)
        self.modules = [self.fc1, self.fc2, self.fc3]

    @jit.script_method
    def forward(self, belief, state):
        x = torch.cat([belief, state], dim=1)
        hidden = self.act_fn(self.fc1(x))
        hidden = self.act_fn(self.fc2(hidden))
        reward = self.fc3(hidden).squeeze(dim=1)
        return reward


class ValueModel(jit.ScriptModule):
    def __init__(self, belief_size, state_size, hidden_size, activation_function='relu'):
        super().__init__()
        self.act_fn = getattr(F, activation_function)
        self.fc1 = nn.Linear(belief_size + state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, 1)
        self.modules = [self.fc1, self.fc2, self.fc3, self.fc4]

    @jit.script_method
    def forward(self, belief, state):
        x = torch.cat([belief, state], dim=1)
        hidden = self.act_fn(self.fc1(x))
        hidden = self.act_fn(self.fc2(hidden))
        hidden = self.act_fn(self.fc3(hidden))
        reward = self.fc4(hidden).squeeze(dim=1)
        return reward


class ActorModel(jit.ScriptModule):
    def __init__(
            self,
            belief_size,
            state_size,
            hidden_size,
            action_size,
            dist='tanh_normal',
            activation_function='elu',
            min_std=1e-4,
            init_std=5,
            mean_scale=5,
    ):
        super().__init__()
        self.act_fn = getattr(F, activation_function)
        self.fc1 = nn.Linear(belief_size + state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, hidden_size)
        self.fc5 = nn.Linear(hidden_size, 2 * action_size)
        self.modules = [self.fc1, self.fc2, self.fc3, self.fc4, self.fc5]

        self._dist = dist
        self._min_std = min_std
        self._init_std = init_std
        self._mean_scale = mean_scale

    @jit.script_method
    def forward(self, belief, state):
        raw_init_std = torch.log(torch.exp(self._init_std) - 1)
        x = torch.cat([belief, state], dim=1)
        hidden = self.act_fn(self.fc1(x))
        hidden = self.act_fn(self.fc2(hidden))
        hidden = self.act_fn(self.fc3(hidden))
        hidden = self.act_fn(self.fc4(hidden))
        action = self.fc5(hidden).squeeze(dim=1)

        action_mean, action_std_dev = torch.chunk(action, 2, dim=1)
        action_mean = self._mean_scale * torch.tanh(action_mean / self._mean_scale)
        action_std = F.softplus(action_std_dev + raw_init_std) + self._min_std
        return action_mean, action_std

    def get_action(self, belief, state, det=False):
        action_mean, action_std = self.forward(belief, state)
        dist = Normal(action_mean, action_std)
        dist = TransformedDistribution(dist, TanhBijector())
        dist = torch.distributions.Independent(dist, 1)
        dist = SampleDist(dist)
        if det:
            return dist.mode()
        else:
            return dist.rsample()


class SymbolicEncoder(jit.ScriptModule):
    def __init__(self, observation_size, embedding_size, activation_function='relu'):
        super().__init__()
        self.act_fn = getattr(F, activation_function)
        self.fc1 = nn.Linear(observation_size, embedding_size)
        self.fc2 = nn.Linear(embedding_size, embedding_size)
        self.fc3 = nn.Linear(embedding_size, embedding_size)
        self.modules = [self.fc1, self.fc2, self.fc3]

    @jit.script_method
    def forward(self, observation):
        hidden = self.act_fn(self.fc1(observation))
        hidden = self.act_fn(self.fc2(hidden))
        hidden = self.fc3(hidden)
        return hidden


class VisualEncoder(jit.ScriptModule):
    __constants__ = ['embedding_size']

    def __init__(self, embedding_size, activation_function='relu'):
        super().__init__()
        self.act_fn = getattr(F, activation_function)
        self.embedding_size = embedding_size
        self.conv1 = nn.Conv2d(3, 32, 4, stride=2)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 128, 4, stride=2)
        self.conv4 = nn.Conv2d(128, 256, 4, stride=2)
        self.fc = nn.Identity() if embedding_size == 1024 else nn.Linear(1024, embedding_size)
        self.modules = [self.conv1, self.conv2, self.conv3, self.conv4]

    @jit.script_method
    def forward(self, observation):
        hidden = self.act_fn(self.conv1(observation))
        hidden = self.act_fn(self.conv2(hidden))
        hidden = self.act_fn(self.conv3(hidden))
        hidden = self.act_fn(self.conv4(hidden))
        hidden = hidden.view(-1, 1024)
        hidden = self.fc(hidden)  # Identity if embedding size is 1024 else linear projection
        return hidden


def Encoder(symbolic, observation_size, embedding_size, activation_function='relu'):
    if symbolic:
        return SymbolicEncoder(observation_size, embedding_size, activation_function)
    else:
        return VisualEncoder(embedding_size, activation_function)



def atanh(x):
    return 0.5 * torch.log((1 + x) / (1 - x))


class TanhBijector(torch.distributions.Transform):
    def __init__(self):
        super().__init__()
        self.bijective = True
        self.domain = torch.distributions.constraints.real
        self.codomain = torch.distributions.constraints.interval(-1.0, 1.0)

    @property
    def sign(self):
        return 1.0

    def _call(self, x):
        return torch.tanh(x)

    def _inverse(self, y: torch.Tensor):
        y = torch.where((torch.abs(y) <= 1.0), torch.clamp(y, -0.99999997, 0.99999997), y)
        y = atanh(y)
        return y

    def log_abs_det_jacobian(self, x, y):
        return 2.0 * (np.log(2) - x - F.softplus(-2.0 * x))


class SampleDist:
    def __init__(self, dist, samples=100):
        self._dist = dist
        self._samples = samples

    @property
    def name(self):
        return 'SampleDist'

    def __getattr__(self, name):
        return getattr(self._dist, name)

    def mean(self):
        sample = self._dist.rsample()
        return torch.mean(sample, 0)

    def mode(self):
        dist = self._dist.expand((self._samples, *self._dist.batch_shape))
        sample = dist.rsample()
        logprob = dist.log_prob(sample)
        batch_size = sample.size(1)
        feature_size = sample.size(2)
        indices = torch.argmax(logprob, dim=0).reshape(1, batch_size, 1).expand(1, batch_size, feature_size)
        return torch.gather(sample, 0, indices).squeeze(0)

    def entropy(self):
        dist = self._dist.expand((self._samples, *self._dist.batch_shape))
        sample = dist.rsample()
        logprob = dist.log_prob(sample)
        return -torch.mean(logprob, 0)

    def sample(self):
        return self._dist.sample()

# Logic Model for System 2
class NLR(nn.Module):
    def __init__(self, state_size, action_size, vector_size, layer_num, logic_distance, r_logic=0.1, r_length=0.001,
                 activation_function='relu'):
        super(NLR, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.vector_size = vector_size
        self.r_logic = r_logic
        self.r_length = r_length
        self.layer_num = layer_num
        self.act_fn = getattr(F, activation_function)
        self.logic_distance = logic_distance

    def _init_weights(self):
        self.l2_embeddings = []
        self.true = torch.nn.Parameter(utils.numpy_to_torch(
            np.random.uniform(0, 1, size=[1, self.vector_size]).astype(np.float32)), requires_grad=False)

        self.not_layer = torch.nn.Linear(self.vector_size, self.vector_size)
        for i in range(self.layer_num):
            setattr(self, 'not_layer_%d' % i, torch.nn.Linear(self.vector_size, self.vector_size))

        self.and_layer_cat = torch.nn.Linear(self.vector_size * 2, self.vector_size)
        for i in range(self.layer_num):
            setattr(self, 'and_layer_cat_%d' % i, torch.nn.Linear(self.vector_size * 2, self.vector_size * 2))
        self.and_layer_mul = torch.nn.Linear(self.vector_size, self.vector_size)
        for i in range(self.layer_num):
            setattr(self, 'and_layer_mul_%d' % i, torch.nn.Linear(self.vector_size, self.vector_size))

        self.or_layer_cat = torch.nn.Linear(self.vector_size * 2, self.vector_size)
        for i in range(self.layer_num):
            setattr(self, 'or_layer_cat_%d' % i, torch.nn.Linear(self.vector_size * 2, self.vector_size * 2))
        self.or_layer_mul = torch.nn.Linear(self.vector_size, self.vector_size)
        for i in range(self.layer_num):
            setattr(self, 'or_layer_mul_%d' % i, torch.nn.Linear(self.vector_size, self.vector_size))

        self.sim_layer = torch.nn.Linear(self.vector_size, 1)
        for i in range(self.layer_num):
            setattr(self, 'sim_layer_%d' % i, torch.nn.Linear(self.vector_size, self.vector_size))

        self.f1_state_logic = torch.nn.Linear(self.state_size, self.vector_size)
        self.f2_state_logic = torch.nn.Linear(self.vector_size, self.vector_size)
        self.f3_state_logic = torch.nn.Linear(self.vector_size, self.vector_size)

        self.f1_action_logic = torch.nn.Linear(self.action_size + self.state_size, self.vector_size)
        self.f2_action_logic = torch.nn.Linear(self.vector_size, self.vector_size)
        self.f3_action_logic = torch.nn.Linear(self.vector_size, self.vector_size)

    def logic_action_embedding(self, state, action):
        vector_cat = torch.cat((state, action), dim=-1)
        a1 = self.act_fn(self.f1_action_logic(vector_cat))
        a2 = self.act_fn(self.f2_action_logic(a1))
        a3 = self.f3_action_logic(a2)
        return a3

    def logic_state_embedding(self, state):
        s1 = self.act_fn(self.f1_state_logic(state))
        s2 = self.act_fn(self.f2_state_logic(s1))
        s3 = self.f3_state_logic(s2)
        return s3

    def logic_not(self, vector):
        for i in range(self.layer_num):
            vector = self.act_fn(getattr(self, 'not_layer_%d' % i)(vector))
        vector = self.not_layer(vector)
        return vector
    
    def logic_and(self, vector1, vector2):
        vector_cat = torch.cat((vector1, vector2), dim=-1)
        for i in range(self.layer_num):
            vector_cat = self.act_fn(getattr(self, 'and_layer_cat_%d' % i)(vector_cat))
        vector_mul = vector1 * vector2
        for i in range(self.layer_num):
            vector_mul = self.act_fn(getattr(self, 'and_layer_mul_%d' % i)(vector_mul))
        vector = self.and_layer_cat(vector_cat) + self.and_layer_mul(vector_mul)
        return vector

    def logic_or(self, vector1, vector2):
        vector_cat = torch.cat((vector1, vector2), dim=-1)
        for i in range(self.layer_num):
            vector_cat = self.act_fn(getattr(self, 'or_layer_cat_%d' % i)(vector_cat))
        vector_mul = vector1 * vector2
        for i in range(self.layer_num):
            vector_mul = self.act_fn(getattr(self, 'or_layer_mul_%d' % i)(vector_mul))
        vector = self.or_layer_cat(vector_cat) + self.or_layer_mul(vector_mul)
        return vector

    def mse(self, vector1, vector2):
        return (vector1 - vector2) ** 2

    def similarity(self, vector1, vector2, sigmoid=True):
        result = F.cosine_similarity(vector1, vector2, dim=-1)
        if sigmoid:
            return result.sigmoid()
        return result

    def predict(self, state, action, next_state):
        constraint = []
        input1 = [state, action]
        random.shuffle(input1)
        and_a_b = self.logic_and(input1[0], input1[1])
        constraint.append(and_a_b)

        vector = and_a_b.clone()
        length = and_a_b.size(0)
        for i in range(length):
            for j in range(1, min(self.logic_distance + 1, length - i)):
                input2 = [and_a_b[i], and_a_b[i + j]]
                random.shuffle(input2)
                vector[i + j] = self.logic_and(input2[0], input2[1])
        constraint.append(vector)

        not_and_a_b = self.logic_not(vector)
        constraint.append(not_and_a_b)
        input3 = [not_and_a_b, next_state]
        random.shuffle(input3)
        result_vector = self.logic_or(input3[0], input3[1])
        constraint.append(result_vector)

        prediction = self.similarity(result_vector, self.true).view([-1])

        return prediction, torch.cat(tuple(constraint), dim=1)

    def logic_regular(self, constraint):
        true_matrix = self.true.unsqueeze(0).expand(constraint.size())

        false = self.logic_not(self.true)
        false_matrix = self.true.unsqueeze(0).expand(constraint.size())

        # Regular
        r_length = constraint.norm(dim=2).mean()

        # Logic Rules are given as follows

        # Not
        r_not_true = self.similarity(false, self.true)
        r_not_true = r_not_true.mean()
        r_not_self = self.similarity(self.logic_not(constraint), constraint).mean()
        r_not_not_self = 1 - self.similarity(self.logic_not(self.logic_not(constraint)), constraint).mean()

        # And
        r_and_true = 1 - self.similarity(self.logic_and(constraint, true_matrix), constraint).mean()

        r_and_false = 1 - self.similarity(self.logic_and(constraint, false_matrix), false).mean()

        r_and_self = 1 - self.similarity(self.logic_and(constraint, constraint), constraint).mean()

        r_and_not_self = 1 - self.similarity(self.logic_and(constraint, self.logic_not(constraint)),
                                             false_matrix).mean()

        # Or
        r_or_true = 1 - self.similarity(self.logic_or(constraint, true_matrix), true_matrix).mean()

        r_or_false = 1 - self.similarity(self.logic_or(constraint, false_matrix), constraint).mean()

        r_or_self = 1 - self.similarity(self.logic_or(constraint, constraint), constraint).mean()

        r_or_not_self = 1 - self.similarity(self.logic_or(constraint, self.logic_not(constraint)),
                                            true_matrix).mean()

        # IMPLY
        r_imply_ture = 1 - self.similarity(self.logic_or(self.logic_not(constraint), true_matrix), true_matrix).mean()

        r_imply_false = 1 - self.similarity(self.logic_or(self.logic_not(constraint), false_matrix),
                                            self.logic_not(constraint)).mean()

        r_imply_ide = 1 - self.similarity(self.logic_or(self.logic_not(constraint), constraint), true_matrix).mean()

        r_imply_con = 1 - self.similarity(self.logic_or(constraint, self.logic_not(constraint)),
                                          self.logic_not(constraint)).mean()

        # Loss
        r_loss = 0
        r_loss += r_not_true + r_not_self + r_not_not_self + \
                  r_and_true + r_and_false + r_and_self + r_and_not_self + \
                  r_or_true + r_or_false + r_or_self + r_or_not_self + \
                  r_imply_ture + r_imply_false + r_imply_ide + r_imply_con
        r_loss = r_loss * self.r_logic
        r_loss += r_length * self.r_length

        return r_loss

    def l2(self):
        l2_loss = 0.0
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if param.requires_grad and name not in self.l2_embeddings:
                l2_loss += torch.sum(param ** 2)
        return l2_loss

    def forward(self, state, action, next_state):
        state_vector = self.logic_state_embedding(state)
        next_state_vector = self.logic_state_embedding(next_state)
        action_vector = self.logic_action_embedding(state, action)
        pre_sim, constraint = self.predict(state_vector, action_vector, next_state_vector)
        r_loss = self.logic_regularizer(constraint)
        l2_loss = self.l2()

        return pre_sim, r_loss, l2_loss

    def predict_for_test(self, state, action, next_state):
        predictions = []
        for i in range(state.size(0)):
            predictions_s = []
            for j in range(action.size(0)):
                and_a_b = self.logic_and(state[i], action[j])
                not_and_a_b = self.logic_not(and_a_b)
                print(not_and_a_b.size())
                print(next_state[-1].size())

                result_vector = self.logic_or(not_and_a_b, next_state[-1])
                prediction = self.similarity(result_vector, self.true).view([-1])
                print(prediction.size())
                predictions_s.append(prediction[0].item())
            predictions.append(predictions_s)

        return predictions

    def test(self, state, action, next_state):
        state_vector = self.logic_state_embedding(state)
        next_state_vector = self.logic_state_embedding(next_state)
        action_vector = self.logic_action_embedding(state, action)
        pre_sim = self.predict_for_test(state_vector, action_vector, next_state_vector)

        return pre_sim