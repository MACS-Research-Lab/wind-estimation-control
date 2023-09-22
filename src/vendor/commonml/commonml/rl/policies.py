from typing import Callable

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Distribution, Bernoulli, Categorical, MultivariateNormal



def init_orthogonal_(w: torch.Tensor, one_dim_init_= lambda w: nn.init.normal_(w, 0., 0.01)):
    if w.ndim >= 2:
        nn.init.orthogonal_(w)
    else:
        one_dim_init_(w)



class Policy(nn.Module):

    dist = Distribution
    dist_kwargs = {}


    def __init__(self, state_dim, action_dim, n_latent_var,
        init_fn: Callable=init_orthogonal_, device='cpu'):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.n_latent_var = n_latent_var
        self.init_fn = init_fn
        self.device = device

        self.base = None

        self.value_layer = nn.Sequential(
                    nn.Linear(state_dim, n_latent_var),
                    nn.Tanh(),
                    nn.Linear(n_latent_var, n_latent_var),
                    nn.Tanh(),
                    nn.Linear(n_latent_var, n_latent_var),
                    nn.Tanh(),
                    nn.Linear(n_latent_var, 1)
                    )
        
        self.action_layer = lambda x: x

        for w in self.value_layer.parameters():
            self.init_fn(w)


    def forward(self, state, action):
        return self.evaluate(state, action)


    def predict(self, state: np.ndarray):
        # pylint: disable=no-member
        state = torch.from_numpy(state).float().to(self.device)
        action_probs = self.action_layer(state)  # Discrete
        dist = self.dist(action_probs, **self.dist_kwargs)
        action = dist.sample()
        return action.cpu(), dist.log_prob(action).cpu()


    def evaluate(self, state: torch.Tensor, action: torch.Tensor):
        action_probs = self.action_layer(state)  # Discrete
        dist = self.dist(action_probs, **self.dist_kwargs)
        
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy() # TODO, sum entropy over variables
        
        state_value = self.value_layer(state)
        
        return action_logprobs, torch.squeeze(state_value), dist_entropy



class ActorCriticDiscrete(Policy):

    dist = Categorical
    dist_kwargs = {}

    def __init__(self, state_dim, action_dim, n_latent_var,
        init_fn: Callable=init_orthogonal_, device='cpu'):
        super().__init__(state_dim=state_dim, action_dim=action_dim,
            n_latent_var=n_latent_var, init_fn=init_fn, device=device)

        self.action_layer = nn.Sequential(
                nn.Linear(state_dim, n_latent_var),
                nn.Tanh(),
                nn.Linear(n_latent_var, n_latent_var),
                nn.Tanh(),
                nn.Linear(n_latent_var, n_latent_var),
                nn.Tanh(),
                nn.Linear(n_latent_var, action_dim),
                nn.Softmax(dim=-1)
                )
        for w in self.action_layer.parameters():
            self.init_fn(w)


    def predict(self, state):
        action, log_prob = super().predict(state)
        return action.item(), log_prob.item()


class ActorCriticMultiBinary(Policy):

    dist = Bernoulli
    dist_kwargs = {}


    def __init__(self, state_dim, action_dim, n_latent_var,
        init_fn: Callable=init_orthogonal_, device='cpu'):
        super().__init__(state_dim, action_dim, n_latent_var, init_fn=init_fn, device=device)

        self.action_layer = nn.Sequential(
                nn.Linear(state_dim, n_latent_var),
                nn.Tanh(),
                nn.Linear(n_latent_var, n_latent_var),
                nn.Tanh(),
                nn.Linear(n_latent_var, n_latent_var),
                nn.Tanh(),
                nn.Linear(n_latent_var, action_dim),
                nn.Sigmoid()
                )
        for w in self.action_layer.parameters():
            self.init_fn(w)


    def predict(self, state):
        action, log_prob = super().predict(state)
        return action.numpy(), log_prob.sum(-1).item()


    def evaluate(self, state, action):
        action_logprobs, state_value, dist_entropy = \
            super().evaluate(state, action)
        return action_logprobs.sum(-1), state_value, dist_entropy



class ActorCriticBox(Policy):

    dist = MultivariateNormal
    dist_kwargs = {}


    def __init__(self, state_dim, action_dim, n_latent_var,
        init_fn: Callable=init_orthogonal_, device='cpu',
        action_std=0.1, activation=nn.Tanh, trainable_cov=True):
        super().__init__(state_dim=state_dim, action_dim=action_dim,
            n_latent_var=n_latent_var, init_fn=init_fn, device=device)
        self.activation = activation
        self.action_layer = nn.Sequential(
                nn.Linear(state_dim, n_latent_var),
                nn.Tanh(),
                nn.Linear(n_latent_var, n_latent_var),
                nn.Tanh(),
                nn.Linear(n_latent_var, n_latent_var),
                nn.Tanh(),
                nn.Linear(n_latent_var, action_dim),
                nn.Identity() if self.activation is None else self.activation()
                )
        for w in self.action_layer.parameters():
            self.init_fn(w)

        cov_exp = (torch.ones(action_dim, action_dim)).to(self.device)
        if trainable_cov:
            cov_exp.fill_(-10)
            cov_exp.fill_diagonal_(np.log(action_std))
            self.cov_exp = nn.Parameter(cov_exp)
        else:
            cov_exp.fill_(-torch.inf)
            cov_exp.fill_diagonal_(np.log(action_std))
            self.cov_exp = cov_exp


    def predict(self, state: np.ndarray):
        # pylint: disable=no-member
        state = torch.from_numpy(state).float().to(self.device)
        action_probs = self.action_layer(state)  # Discrete
        cov = self.cov_exp.exp()
        dist = self.dist(action_probs, covariance_matrix=cov)
        action = dist.sample()
        return action.cpu().numpy(), dist.log_prob(action).cpu().item()


    def evaluate(self, state: torch.Tensor, action: torch.Tensor):
        action_probs = self.action_layer(state)  # Discrete
        cov = self.cov_exp.exp()
        dist = self.dist(action_probs, covariance_matrix=cov)
        
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy() # TODO, sum entropy over variables
        
        state_value = self.value_layer(state)
        
        return action_logprobs, torch.squeeze(state_value), dist_entropy
