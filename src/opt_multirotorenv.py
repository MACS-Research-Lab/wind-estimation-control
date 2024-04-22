import sys
import os

import torch
# from .setup import local_path
import multiprocessing as mp
from argparse import ArgumentParser, Namespace
import pickle
import optuna
import numpy as np
from multirotor.trajectories import Trajectory
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from environments import OctorotorEnvSelector
from trajectories import square_100, circle_100, nasa_wp
from parameters import pid_params

# from systems.long_blending import LongBlendingEnv
# from systems.blending import BlendingEnv

# from .scripts.opt_pidcontroller import (
#     get_controller as get_controller_base,
#     apply_params as apply_params_pid,
#     make_controller_from_trial,
#     make_env,
#     DEFAULTS as PID_DEFAULTS
# )


DEFAULTS = Namespace(
    ntrials = 100,
    nprocs = 5,
    safety_radius = 5,
    max_velocity = 15,
    max_acceleration = 3,
    max_tilt = np.deg2rad(22.5),
    scurve = False,
    leashing = False,
    sqrt_scaling = False,
    use_yaw = False,
    wind = True,
    fault = False,
    num_sims = 5,
    max_steps = 50_000,
    study_name = 'MultirotorTrajEnv',
    env_kind = "lstm",
    pid_params = '',
    use_trial = None
)

class Callback(BaseCallback):

    def __init__(self, trial: optuna.Trial, verbose: int = 0):
        super().__init__(verbose)
        self.trial = trial
        self.should_stop = False
        self.rollouts = 0

    def _on_rollout_end(self):
        self.rollouts += 1
        self.trial.report(
            value=np.nanmean([info['r'] for info in self.model.ep_info_buffer]),
            step=self.rollouts
        )
        self.should_stop = self.trial.should_prune()

    def _on_step(self) -> bool:
        return not self.should_stop



def get_study(study_name: str=DEFAULTS.study_name, seed:int=0, args: Namespace=DEFAULTS) -> optuna.Study:
    storage_name = "sqlite:///studies/{}.db".format(study_name)
    study = optuna.create_study(
        study_name=study_name, direction='maximize',
        storage=storage_name,
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=seed),
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=5,
            n_warmup_steps=20, # number of rollouts
            n_min_trials=1
        )
    )
    return study


def objective(trial: optuna.Trial):
    
    scaling_factor = trial.suggest_int('scaling_factor', 2, 5, step=1) # for now, use this to determine the action range
    wind_d = trial.suggest_int("window_distance", 10, 20)
    
    policy_layers = trial.suggest_categorical("policy_layers", [1,2,3])
    policy_size = trial.suggest_int("policy_size", 32, 256, step=32)

    learning_rate = trial.suggest_float('learning_rate', 1e-8, 1e-3, log=True)
    n_epochs = trial.suggest_int('n_epochs', 1, 5)
    n_steps = trial.suggest_int('n_steps', 16, 12000, step=16) # what if we allow this to be much higher?
    # n_steps = 4000 // env_kwargs['steps_u'] # expect around 4000 when interacts every timestep
    batch_size = trial.suggest_int('batch_size', 32, 256, step=32)
    total_timesteps = trial.suggest_categorical('total_timesteps', [50000, 100000, 200000])
    

    wp_options = [square_100]
    env_selector = OctorotorEnvSelector()
    
    pos_mult = 0.2
    vel_mult = 0.2
    pid_parameters = pid_params(
        pos_p=[pos_mult*0.3, pos_mult*0.3, 0.2],
        vel_p=[vel_mult*1, vel_mult*1, 100],
        vel_i=[vel_mult*0.1, vel_mult*0.1, 0]
    )
    
    best_params = {'steps_u':50, 'scaling_factor':scaling_factor, 'window_distance':wind_d, 'pid_parameters': pid_parameters}
    env = env_selector.get_env("lstm", best_params, [(5,12), (5,12), (0,0)], square_100, start_alt=30, has_turbulence=True)
    env.wp_options = wp_options
    env.base_env.fault_type=None
    
    ppo = PPO(policy="MlpPolicy", env=env, learning_rate=learning_rate, n_epochs=n_epochs, n_steps=n_steps, 
              policy_kwargs=dict(squash_output=False,net_arch=[dict(pi=[policy_size]*policy_layers, vf=[policy_size]*policy_layers)]),
              batch_size=batch_size, verbose=0)
    
    agent = ppo.learn(total_timesteps=total_timesteps, progress_bar=True)
    agent.save(f'./saved_models/lower_pid/{trial.number}')
    
    done = False
    env = env_selector.get_env("lstm", best_params, [(0,0), (12,12), (0,0)], square_100, start_alt=30, has_turbulence=True)
    state = np.array(env.reset(), dtype=np.float32)
    rewards = []
    while not done:
        action = agent.predict(state, deterministic=True)[0]
        state, reward, done, info = env.step(action)
        rewards.append(reward)
        state = np.array(state, dtype=np.float32)

    return np.mean(rewards)



def optimize(args: Namespace=DEFAULTS, seed: int=0, queue=[]):
    study = get_study(args.study_name, seed=seed, args=args)
    for params in queue:
        study.enqueue_trial(params, skip_if_exists=True)
    study.optimize(
        objective,
        n_trials=args.ntrials//args.nprocs,
    )
    return study


if __name__=='__main__':
    parser = ArgumentParser()
    parser.add_argument('study_name', help='Name of study', default=DEFAULTS.study_name, type=str, nargs='?')
    parser.add_argument('--nprocs', help='Number of processes.', default=DEFAULTS.nprocs, type=int)
    parser.add_argument('--ntrials', help='Number of trials.', default=DEFAULTS.ntrials, type=int)
    parser.add_argument('--max_velocity', default=DEFAULTS.max_velocity, type=float)
    parser.add_argument('--max_acceleration', default=DEFAULTS.max_acceleration, type=float)
    parser.add_argument('--max_tilt', default=DEFAULTS.max_tilt, type=float)
    parser.add_argument('--scurve', action='store_true', default=DEFAULTS.scurve)
    parser.add_argument('--leashing', action='store_true', default=DEFAULTS.leashing)
    parser.add_argument('--sqrt_scaling', action='store_true', default=DEFAULTS.sqrt_scaling)
    parser.add_argument('--use_yaw', action='store_true', default=DEFAULTS.use_yaw)
    parser.add_argument('--wind', help='wind force from heading "force@heading"', default=DEFAULTS.wind)
    parser.add_argument('--fault', help='motor loss of effectiveness "loss@motor"', default=DEFAULTS.fault)
    parser.add_argument('--safety_radius', help="size of safety corridor in meters", default=DEFAULTS.safety_radius, type=float)
    parser.add_argument('--cardinal', help="whether to experience wind from the cardinal directions", default=False)
    parser.add_argument('--num_sims', default=DEFAULTS.num_sims, type=int)
    parser.add_argument('--max_steps', default=DEFAULTS.max_steps, type=int)
    parser.add_argument('--append', action='store_true', default=False)
    parser.add_argument('--pid_params', help='File to load pid params from.', type=str, default=DEFAULTS.pid_params)
    parser.add_argument('--comment', help='Comments to attach to study.', type=str, default='')
    parser.add_argument('--env_kind', help='"[traj,alloc]"', default=DEFAULTS.env_kind)
    parser.add_argument('--use_study', help='Use top 10 parameters from this trial to start', default=None)
    args = parser.parse_args()

    # if not args.append:
    #     import shutil
    #     path = local_path / ('tensorboard/MultirotorTrajEnv/optstudy/' + args.study_name)
    #     shutil.rmtree(path=path, ignore_errors=True)
    #     try:
    #         os.remove(local_path / ('studies/' + args.study_name + '.db'))
    #     except OSError:
    #         pass
    
    # create study if it doesn't exist. The study will be reused with a new seed
    # by each process
    study = get_study(args.study_name, args=args)
    # reuse parameters from another study
    if args.use_study is not None:
        trials = sorted(get_study(args.use_study).trials, key= lambda t: t.value, reverse=True)[:10]
        params = [r.params for r in trials]
        trials_per_proc = len(params) // args.nprocs
        reuse = [params[i:i+trials_per_proc] for i in range(0, len(params), trials_per_proc)]
        remainder = len(params) % trials_per_proc
        if remainder > 0: # add remainder to last process's queue
            reuse[-1].extend(params[-remainder:])
        # for p in params:
        #     study.enqueue_trial(p)
        # reuse = [[] for _ in range(args.nprocs)]
    else:
        reuse = [[] for _ in range(args.nprocs)]

    for key in vars(args):
        study.set_user_attr(key, getattr(args, key))
        
    
    mp.set_start_method('spawn')
    with mp.Pool(args.nprocs) as pool:
        pool.starmap(optimize, [(args, i, reuse[i]) for i in range(args.nprocs)])
        # pool.starmap(optimize, [(args, i) for i in range(args.nprocs)])