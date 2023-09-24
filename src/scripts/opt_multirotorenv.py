import sys
import os

import torch
from .setup import local_path
import multiprocessing as mp
from argparse import ArgumentParser, Namespace
import pickle
import optuna
import numpy as np
from multirotor.trajectories import Trajectory
from rl import learn_rl, evaluate_rl, load_agent
from stable_baselines3.common.callbacks import BaseCallback
from systems.multirotor_naive import Multirotor, MultirotorTrajEnv, VP
from systems.long_multirotor_naive import LongTrajEnv
# from systems.long_blending import LongBlendingEnv
# from systems.blending import BlendingEnv

from .opt_pidcontroller import (
    get_controller as get_controller_base,
    apply_params as apply_params_pid,
    make_controller_from_trial,
    make_env,
    DEFAULTS as PID_DEFAULTS
)


DEFAULTS = Namespace(
    ntrials = 100,
    nprocs = 5,
    safety_radius = PID_DEFAULTS.safety_radius,
    max_velocity = PID_DEFAULTS.max_velocity,
    max_acceleration = PID_DEFAULTS.max_acceleration,
    max_tilt = PID_DEFAULTS.max_tilt,
    scurve = PID_DEFAULTS.scurve,
    leashing = False,
    sqrt_scaling = False,
    use_yaw = PID_DEFAULTS.use_yaw,
    wind = PID_DEFAULTS.wind,
    fault = PID_DEFAULTS.fault,
    num_sims = 5,
    max_steps = 50_000,
    study_name = 'MultirotorTrajEnv',
    env_kind = PID_DEFAULTS.env_kind,
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



def get_established_controller(m: Multirotor, leash=False, speed: int = 15):
    """
    This returns the manually-tuned PID controller that is fixed for all experiments. It was tuned by an expert to handle up to 15 m/s wind. 
    (Though, of course, it is less safe at higher winds.)
    """
    ctrl = get_controller_base(m, scurve=False, leashing=leash)

    with open('../src/params/manual_pid.pkl', 'rb') as f: #TODO: make relative
        optimal_params = pickle.load(f)

    optimal_params['ctrl_z']['k_p'] = np.array([0.4])
    optimal_params['ctrl_z']['k_i'] = np.array([0.0])
    optimal_params['ctrl_z']['k_d'] = np.array([0.0])
    optimal_params['ctrl_p']['max_velocity'] = speed # to modify the max speed of the UAV, modify this

    z_params = optimal_params['ctrl_z']
    optimal_params['ctrl_vz']['k_p'] = 25
    vz_params = optimal_params['ctrl_vz']

    ctrl.set_params(**optimal_params)
    ctrl.ctrl_z.set_params(**z_params)
    ctrl.ctrl_vz.set_params(**vz_params)
    return ctrl



def get_env(wind_ranges, scurve=False, **kwargs):  
    kw = dict(
        safety_radius=kwargs['safety_radius'],
        vp=VP,get_controller_fn=lambda m: get_established_controller(m),
        steps_u=kwargs['steps_u'],
        scaling_factor=kwargs['scaling_factor'],
        wind_ranges=wind_ranges,
        proximity=5,
        seed=kwargs['seed'])
    return MultirotorTrajEnv(**kw)



def make_objective(args: Namespace=DEFAULTS):
    def objective(trial: optuna.Trial):
        all_directions = args.cardinal == 'True'
        bounding_rect_length = 200 # can suggest this if we want to use it
        # bounding_rect_length = trial.suggest_int("bounding_rect_length", 5, 50, step=5)
        bounding_rect_width = args.safety_radius
        
        env_kwargs = dict(
            safety_corridor = bounding_rect_width, 
            seed=0,
            get_controller_fn=lambda m: get_established_controller(m, args),
            vp = VP,
            safety_radius=bounding_rect_width,
           
        )

        env_kwargs['steps_u'] = 50 # assume half a second
        env_kwargs['scaling_factor'] = trial.suggest_int('scaling_factor', 1, 7, step=1) # for now, use this to determine the action range
        # env_kwargs['scaling_factor'] = 5
        
        square_np = np.array([[100,0,30], [100,100,30], [0,100,30], [0,0,30]]) # set up your trajectory here
        square_traj = Trajectory(None, points=square_np, resolution=bounding_rect_length)
        square_wpts = square_traj.generate_trajectory(curr_pos=np.array([0,0,30]))

        wind_d = trial.suggest_int("window_distance", 10, 50)
        # wind_d = 10
        
        env = LongTrajEnv(
            waypoints = square_wpts,
            base_env = get_env(wind_ranges = [(0,10), (0,10), (0,0)], **env_kwargs), # what ranges of wind you want to experience in hyperparmeter optimization [(xmin, xmax), (ymin, ymax), (zmin, zmax)]
            initial_waypoints=square_np,
            randomize_direction=True, # whether to randomize the direction the trajectory is flown during HPO
            always_modify_wind=False, # whether to generate a different wind vector for each bounding box, note: if you include this, fix the length of the bounding box
            random_cardinal_wind=all_directions,
            window_distance = wind_d,
            has_turbulence=True
        )
        
        env.reset()
        
        env.base_env.vehicle.mass = np.random.normal(10.33, 0.75)
        
        env.start_alt = 30
        env.base_env.vehicle.position[2] = 30
        
        policy_layers = trial.suggest_categorical("policy_layers", [1,2,3])
        policy_size = trial.suggest_int("policy_size", 32, 256, step=32)

        learning_rate = trial.suggest_float('learning_rate', 1e-8, 1e-3, log=True)
        n_epochs = trial.suggest_int('n_epochs', 1, 5)
        n_steps = trial.suggest_int('n_steps', 16, 12000, step=16) # what if we allow this to be much higher?
        # n_steps = 4000 // env_kwargs['steps_u'] # expect around 4000 when interacts every timestep
        batch_size = trial.suggest_int('batch_size', 32, 256, step=32)
        learn_kwargs = dict(
            steps = trial.suggest_categorical('training_interactions', [50000, 100000, 150000, 200000, 250000]),
            # steps = 5000,
            n_steps = n_steps,
            learning_rate = learning_rate,
            n_epochs = n_epochs,
            batch_size = batch_size,
            seed=0,
            log_env_params = ('steps_u', 'scaling_factor') if args.env_kind=='traj' else (),
            tensorboard_log = env.base_env.name + ('/optstudy/%s/%03d' % (args.study_name, trial.number)),
            policy_kwargs=dict(squash_output=False,
                                net_arch=[dict(pi=[policy_size]*policy_layers, vf=[policy_size]*policy_layers)]),
            callback = Callback(trial=trial)
        )

        agent = learn_rl(env, progress_bar=True, **learn_kwargs)
        agent.save(agent.logger.dir + '/agent')
        agent = load_agent(agent.logger.dir + '/agent')

        if all_directions: # if training on cardinal wind, you probably want to evaluate hyperparameters on cardinal wind
            all_wind_ranges = [[(0,0), (0,0), (0,0)],
                               [(0,0), (5,5), (0,0)],
                               [(0,0), (7,7), (0,0)],
                               [(0,0), (10,10), (0,0)],
                               [(0,0), (-5,-5), (0,0)],
                               [(0,0), (-7,-7), (0,0)],
                               [(0,0), (-10,-10), (0,0)],
                               [(5,5), (0,0), (0,0)],
                               [(7,7), (0,0), (0,0)],
                               [(10,10), (0,0), (0,0)],
                               [(-5,-5), (0,0), (0,0)],
                               [(-7,-7), (0,0), (0,0)],
                               [(-10,-10), (0,0), (0,0)]] 
            #  all_wind_ranges = [[(0,0), (5,5), (0,0)],
            #                    [(0,0), (6,6), (0,0)],
            #                    [(0,0), (7,7), (0,0)],
            #                    [(0,0), (-5,-5), (0,0)],
            #                    [(0,0), (-6,-6), (0,0)],
            #                    [(0,0), (-7,-7), (0,0)],
            #                    [(5,5), (0,0), (0,0)],
            #                    [(6,6), (0,0), (0,0)],
            #                    [(7,7), (0,0), (0,0)],
            #                    [(-5,-5), (0,0), (0,0)],
            #                    [(-6,-6), (0,0), (0,0)],
            #                    [(-7,-7), (0,0), (0,0)]] 
        else: # if not training on cardinal wind, what wind magnitudes do you want to be considered for evaluating hyperparameters?
            # all_wind_ranges = [[(0,0), (0,0), (0,0)],
            #                [(0,0), (5,5), (0,0)],
            #                 [(0,0), (7,7), (0,0)],
            #                [(0,0), (10,10), (0,0)],]
            all_wind_ranges = [[(0,0), (10,10), (0,0)],]
            

        rewards = []
        for wind_range in all_wind_ranges:
            env = LongTrajEnv(
                waypoints = square_wpts,
                base_env = get_env(wind_ranges = wind_range, **env_kwargs),
                initial_waypoints=square_np,
                window_distance=wind_d,
                randomize_direction=False, # during evaluation, fix the directionf or consistency,
                has_turbulence=True
            )
            
            env.start_alt = 30
            env.base_env.vehicle.position[2] = 30

            episode_reward = 0
            state = np.array(env.reset(), dtype=np.float32)
            done = False
            while not done:
                action = agent.predict(state, deterministic=True)[0]
                state, reward, done, info = env.step(action)
                episode_reward += reward

            rewards.append(episode_reward)

        return np.mean(rewards)
    return objective



def optimize(args: Namespace=DEFAULTS, seed: int=0, queue=[]):
    study = get_study(args.study_name, seed=seed, args=args)
    for params in queue:
        study.enqueue_trial(params, skip_if_exists=True)
    study.optimize(
        make_objective(args),
        n_trials=args.ntrials//args.nprocs,
    )
    return study



def apply_params(env: MultirotorTrajEnv, **params):
    prefix = 'pid-'
    env_params = ('steps_u', 'scaling_factor')
    rl_params = ('learning_rate', 'n_epochs', 'n_steps', 'batch_size')
    env_dict = {k: params[k] for k in env_params if k in params}
    rl_dict = {k: params[k] for k in rl_params}

    pid_dict = {}
    for k, v in params.items():
        if k in env_params or k in rl_params:
            continue
        elif k.startswith(prefix):
            k = k[len(prefix):]
            pid_dict[k] = v
    apply_params_pid(env.ctrl, **pid_dict)

    env.scaling_factor = env_dict.get('scaling_factor', env.scaling_factor)
    env.steps_u = env_dict.get('steps_u', env.steps_u)
    return dict(rl=rl_dict, pid=pid_dict, env=env_dict)


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

    if not args.append:
        import shutil
        path = local_path / ('tensorboard/MultirotorTrajEnv/optstudy/' + args.study_name)
        shutil.rmtree(path=path, ignore_errors=True)
        try:
            os.remove(local_path / ('studies/' + args.study_name + '.db'))
        except OSError:
            pass
    
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