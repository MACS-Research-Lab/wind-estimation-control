"""
Proximal Policy Optimization (PPO)
"""
# TODO: https://ppo-details.cleanrl.dev//2021/11/05/ppo-implementation-details/
from typing import Callable, List, Union, Tuple
from collections import deque
import warnings

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter
from higher import innerloop_ctx
from higher.optim import DifferentiableOptimizer
import gym
import numpy as np
from tqdm.auto import trange

from .policies import Policy

# pylint: disable=no-member
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Memory:


    def __init__(self, states=(), actions=(), rewards=(), logprobs=(), is_terminals=(),
                 maxlen=None):
        self.maxlen = maxlen
        self.states = deque(states, maxlen=maxlen)
        self.actions = deque(actions, maxlen=maxlen)
        self.rewards = deque(rewards, maxlen=maxlen)
        self.logprobs = deque(logprobs, maxlen=maxlen)
        self.is_terminals = deque(is_terminals, maxlen=maxlen)
        self._lists = (self.states, self.actions, self.rewards, self.logprobs,
                       self.is_terminals)


    def __len__(self):
        return len(self.states)


    def add(self, state, action, logprob, reward, done):
        self.states.append(state)
        self.actions.append(action)
        self.logprobs.append(logprob)
        self.rewards.append(reward)
        self.is_terminals.append(done)


    def clear(self):
        """
        Clear all entries.
        """
        for l in self._lists:
                    for _ in range(len(l)):
                        del l[0]


    def flush(self):
        """
        Delete all entries except those belonging to the last, unfinished episode.
        """
        size = len(self.states)
        for i, done in enumerate(reversed(self.is_terminals)):
            if done:
                truncate = size - i
                for l in self._lists:
                    for idx in range(truncate):
                        del l[0]
                break
    

    def as_array(self):
        return \
            np.asarray(self.states), \
            np.asarray(self.actions), \
            np.asarray(self.rewards), \
            np.asarray(self.logprobs), \
            np.asarray(self.is_terminals)


     
class PPO:


    def __init__(self, env, policy: Policy, state_dim, action_dim, n_latent_var=64, lr=0.02,
                 betas=(0.9, 0.999), gamma=0.99, epochs=5, batchsize=None, eps_clip=0.2,
                 truncate=False, update_interval=2000, seed=None, normalize_state: Callable=None,
                 device=DEVICE, summary: SummaryWriter=None, **policy_kwargs):
        self.random = np.random.RandomState(seed)
        self.seed = seed
        if seed is not None: torch.manual_seed(self.seed)
        self.env = env
        self.lr = lr
        self.betas = betas
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.truncate = truncate
        self.epochs = epochs
        self.batchsize = batchsize
        self.update_interval = update_interval
        self.device = device
        self.normalize_state = normalize_state
        
        policy_kwargs['device'] = device
        self.policy = policy(state_dim, action_dim, n_latent_var, **policy_kwargs).to(self.device)
        if isinstance(lr, tuple):
            self.optimizer = torch.optim.Adam([
                dict(params=self.policy.action_layer.parameters(), lr=lr[0], betas=betas),
                dict(params=self.policy.value_layer.parameters(), lr=lr[1], betas=betas)
            ])
        else:
            self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr, betas=betas)
        
        self.MseLoss = nn.MSELoss()

        self.summary = summary
        self.meta_policy = None

    

    def update(self, policy, memory: Memory, epochs: int=1, batchsize: int=None, optimizer=None,
            summary: SummaryWriter=None, update_number: int=None,
            normalize_state: Callable=None, grad_callback=None):

        
        states, actions, rewards, logprobs, is_terminals = memory.as_array()
        if normalize_state is not None:
            states = normalize_state(states)
        ret = returns(rewards, is_terminals, self.gamma, truncate=self.truncate)
        truncate = len(ret)
        # If the returns calculated are zero length, i.e. when memory does not
        # contain a single full episode, because returns() truncated incompleted
        # episodes, abort update:
        if truncate == 0:
            return np.nan
        # Casting to correct data type and DEVICE
        # pylint: disable=not-callable
        old_states = torch.tensor(states[:truncate]).float().to(self.device).detach()
        old_actions = torch.tensor(actions[:truncate]).float().to(self.device).detach()
        old_logprobs = torch.tensor(logprobs[:truncate]).float().to(self.device).detach()
        ret = torch.tensor(ret[:truncate]).float().to(self.device)
        # Normalizing the rewards:
        # ret_norm = (ret - ret.mean()) / (ret.std() + 1e-5)

        # If states/actions are 1D arrays of single number states/actions,
        # convert them to 2D matrix of 1 column where each row is one timestep.
        # This is to make sure the 0th dimension always indexes time, and the
        # last dimension indexes feature.
        if policy.state_dim == 1 and old_states.ndim == 1:
            old_states = old_states.unsqueeze(dim=-1)
        if policy.action_dim == 1 and old_actions.ndim == 1:
            old_actions = old_actions.unsqueeze(dim=-1)

        
        # Optimize policy for multiple epochs:
        mb_losses, mb_vlosses, mb_plosses = [], [], []
        batchsize = len(ret) if batchsize is None else batchsize
        for e in range(epochs):
            permutations = torch.randperm(ret.size()[0])
            # iterate over minibatches (mb) for stochastic updates
            for mb in range(0, ret.size()[0], batchsize):
                idx = permutations[mb: mb+batchsize]

                mb_ret, mb_old_states, mb_old_actions, mb_old_logprobs = \
                    ret[idx], old_states[idx], old_actions[idx], old_logprobs[idx]

                # Evaluating old actions and values:
                logprobs, state_values, dist_entropy = policy(mb_old_states, mb_old_actions)
                
                # Finding the ratio (pi_theta / pi_theta__old):
                ratios = torch.exp(logprobs - mb_old_logprobs.detach())
                    
                # Finding Surrogate Loss:
                advantages = mb_ret - state_values.detach()
                adv_norm = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
                surr1 = ratios * adv_norm
                surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * adv_norm
                policy_loss = -torch.min(surr1, surr2)
                value_loss = 0.5 * self.MseLoss(state_values, mb_ret)
                loss = value_loss + policy_loss
                loss -= 0.01 * dist_entropy  # randomness (exploration)
                loss = loss.mean()
                mb_losses.append(loss.item())
                mb_vlosses.append(value_loss.mean().item())
                mb_plosses.append(policy_loss.mean().item())
                
                # Take gradient step. If optimizer==None, then just backpropagate
                # gradients.
                if optimizer is not None:
                    # the 'higher' library wraps optimizers to make parameters
                    # differentiable w.r.t earlier versions of parameters. Those
                    # optimizers to not need `backward()` and `zero_grad()`
                    if not isinstance(optimizer, DifferentiableOptimizer):
                        optimizer.zero_grad()
                        loss.backward(retain_graph=True)
                        if grad_callback is not None:
                            apply_grad_callback(policy, grad_callback)
                        optimizer.step()
                    else:
                        optimizer.step(loss, grad_callback=grad_callback)
                else:
                    loss.backward(retain_graph=True)
                    if grad_callback is not None:
                        apply_grad_callback(policy, grad_callback)

        if summary is not None:
            summary.add_scalar('Loss/Total Loss', np.mean(mb_losses), update_number)
            summary.add_scalar('Loss/Value Loss', np.mean(mb_vlosses), update_number)
            summary.add_scalar('Loss/Policy Loss', np.mean(mb_plosses), update_number)
            summary.add_scalar('Perf/Returns', ret.mean().item(), update_number)
            idx = np.where(is_terminals)[0]
            lens = np.diff(idx)
            summary.add_scalar('Perf/Episode Length', np.mean(lens), update_number)
            # TODO: entropy logging
        return np.mean(mb_losses)


    def experience(self, memory, timesteps, env, policy, state0=None):
        state = env.reset() if state0 is None else state0
        for t in range(timesteps):
            # Running policy:
            memory.states.append(state)
            action, logprob = policy.predict(state)
            # print('a', action, 'log(p(a))', logprob)
            state, reward, done, _ = env.step(action)
            if done:
                state = env.reset()
            memory.actions.append(action)
            memory.logprobs.append(logprob)
            memory.rewards.append(reward)
            memory.is_terminals.append(done)


    def learn(self, timesteps, update_interval=None, track_higher_grads=False,
              lr_scheduler=None,
              step_callback=None, interval_callback=None,
              reward_aggregation='episodic',
              start_at_timestep: int=0) -> List[float]:
        """
        Run learning.

        Parameters
        ----------
        timesteps : int
            Number of steps to interact with environment.
        update_interval : int, optional
            Number of steps after which to update policy, by default None
        track_higher_grads : bool, optional
            Whether to track rate of change of parameters w.r.t themselves, by default False
        lr_scheduler : [type], optional
            A scheduler to adjust learning rate, by default None
        step_callback : Callable, optional
            A function to call after every step. It is passed a dictionary of
            local variables, by default None
        interval_callback : Callable, optional
            A function to call after every update interval. It is passed a 
            dictionary of local variables, by default None
        reward_aggregation : str, optional
            One of 'episodic', 'episodic.normalized', 'interval'.
            'episodic' returns total rewards per episode. If normalized
            returns reward divided by episode length. 'interval' returns
            rewards per update interval., by default 'episodic'
        start_at_timestep : int, optional
            The time value to start logging in summary writer

        Returns
        -------
        List[float]
            A list of aggregated rewards.
        """
        if update_interval is None:
            update_interval = self.update_interval
        state = self.env.reset()
        memory = Memory()
        episodic_rewards = [0.]
        interval_rewards = [0.]
        t_episode = 0
        # This context wraps the policy and optimizer to track parameter updates
        # over time such that d Params(time=t) / d Params(time=t-n) can be calculated.
        # If not tracking higher gradients, a dummy context is used which does
        # nothing.
        with innerloop_ctx(self.policy, self.optimizer, track_higher_grads=track_higher_grads,
                           copy_initial_weights=False) as (policy, optimizer):

            for t in trange(1 + start_at_timestep, int(timesteps) + 1 + start_at_timestep, leave=False):
                # Running policy:
                action, logprob = policy.predict(state)
                new_state, reward, done, info = self.env.step(action)
                episodic_rewards[-1] += reward
                interval_rewards[-1] += reward
                t_episode += 1
                if done:
                    new_state = self.env.reset()
                    if reward_aggregation.endswith('normalized'):
                        episodic_rewards[-1] /= t_episode
                    episodic_rewards.append(0.)
                    t_episode = 0
                memory.add(state, action, logprob, reward, done)

                if step_callback is not None:
                    step_callback(locals())
                
                # update if its time
                if t % update_interval == 0:
                    interval_rewards.append(0.)
                    loss = self.update(
                        policy=policy, memory=memory,
                        epochs=self.epochs, batchsize=self.batchsize,
                        optimizer=optimizer,
                        summary=self.summary, update_number=t // update_interval,
                        normalize_state=self.normalize_state)
                    if interval_callback is not None:
                        interval_callback(locals())
                    if lr_scheduler is not None:
                        with warnings.catch_warnings():
                            # warning for calling scheduler before optimizer step,
                            # which is not the case, so ignoring.
                            warnings.simplefilter('ignore', UserWarning)
                            lr_scheduler()
                    if np.isnan(loss):  # a full episode is not present, so no update was done
                        memory.flush()
                    else:
                        memory.clear()
                    
                state = new_state

        
            self.meta_policy = policy if track_higher_grads else None
        self.policy.load_state_dict(policy.state_dict())

        if reward_aggregation.startswith('episodic'):
            return episodic_rewards[:-1 if len(episodic_rewards) > 1 else None]
        else:
            return interval_rewards


    def predict(self, state) -> Tuple[Union[np.ndarray, int], float]:
        return self.policy.predict(state)



def returns(rewards, is_terminals, gamma, truncate=False):
    # Monte Carlo estimate of state rewards:
    # Discarding rewards for incomplete episodes because their returns
    # will be inaccurate. For e.g. if episode rewards = 1,1,1,1,1 but the current
    # batch only has rewards for first 3 steps =1,1,1, then the returns=3,2,1,
    # where they should be 5,4,3
    if truncate:
        if True in is_terminals:
            # TODO: optimize by using reversed() instead of copying array using slicing[::-1]
            idx_from_end = np.where(is_terminals[::-1]==True)[0][0]
            if idx_from_end > 0:
                rewards = rewards[:-idx_from_end]
                is_terminals = is_terminals[:-idx_from_end]
        else:
            is_terminals = []
    returns = []
    discounted_reward = 0
    for reward, is_terminal in zip(reversed(rewards), reversed(is_terminals)):
        if is_terminal:
            discounted_reward = 0
        discounted_reward = reward + (gamma * discounted_reward)
        returns.insert(0, discounted_reward)
    return returns



def apply_grad_callback(model: nn.Module, callback: Callable[[List[Tensor]], List[Tensor]]):
    new_grads = callback([p.grad for p in model.parameters()])
    for p, ng in zip(model.parameters(), new_grads):
        p.grad = ng
