import os
import multiprocessing as mp
from argparse import ArgumentParser, Namespace
from typing import Callable, Dict, Union
from pprint import pprint as print
import pickle

import numpy as np
import optuna
from systems.multirotor import Multirotor, MultirotorTrajEnv, DEFAULTS as DEFAULTS_UAV
from systems.long_multirotor import LongTrajEnv
from multirotor.trajectories import Trajectory, eight_curve
from multirotor.helpers import DataLog
from multirotor.coords import direction_cosine_matrix, inertial_to_body
from multirotor.controller import (
    AltController, AltRateController,
    PosController, AttController,
    VelController, RateController,
    Controller,
    SCurveController
)

from .setup import local_path

DEFAULTS = Namespace(
    ntrials = 1000,
    nprocs = 5,
    safety_radius = DEFAULTS_UAV.safety_radius,
    max_velocity = DEFAULTS_UAV.max_velocity,
    max_acceleration = DEFAULTS_UAV.max_acceleration,
    max_tilt = DEFAULTS_UAV.max_tilt,
    scurve = False,
    leashing = True,
    sqrt_scaling = True,
    use_yaw = False,
    wind = '0@0',
    fault = '0@0',
    num_sims = 10,
    study_name = 'MultirotorPIDController',
    env_kind = 'traj',
)

def get_study(study_name: str=DEFAULTS.study_name, seed: int=0):
    storage_name = "sqlite:///studies/{}.db".format(study_name)
    study = optuna.create_study(
        study_name=study_name, direction='maximize',
        storage=storage_name,
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=seed)
    )
    return study



def run_sim(
    env, traj: Trajectory,
    ctrl_fn: Callable[[np.ndarray], np.ndarray]
) -> DataLog:
    log = DataLog(env.vehicle, env.ctrl,
                  other_vars=('reward',))
        
    for i, (pos, feed_forward_vel) in enumerate(traj):
        # Get prescribed normalized action for system as thrust and torques
        action = ctrl_fn(env.state)
        # Send speeds to environment
        state, r, done, *_ = env.step(action)
        log.log(reward=r)
        if done:
            break

    log.done_logging()
    return log



def get_controller(m: Multirotor, scurve=False, leashing=False, args: Namespace=DEFAULTS):
    assert m.simulation.dt <= 0.1, 'Simulation time step too large.'
    pos = PosController( # PD
        0.8, 0., 3.75,
        max_err_i=args.max_velocity, vehicle=m,
        max_velocity=args.max_velocity,
        max_acceleration=args.max_acceleration,
        square_root_scaling=args.sqrt_scaling,
        leashing=leashing
    )
    vel = VelController( # P
        1, 0., 0,
        max_err_i=args.max_acceleration,
        vehicle=m,
        max_tilt=args.max_tilt)
    att = AttController( # P
        [1., 1., 0.], 0, 0., # yaw param is set to 0, in case use_yaw=False
        max_err_i=1., vehicle=m)
    rat = RateController( # PD
        [4, 4, 0], 0, [40, 40, 0], # yaw param is set to 0, in case use_yaw=False
        max_err_i=0.5,
        vehicle=m)

    alt = AltController(
        1, 0, 0,
        max_err_i=1, vehicle=m,
        max_velocity=args.max_velocity)
    alt_rate = AltRateController(
        10, 0, 0,
        max_err_i=1, vehicle=m)

    ctrl = Controller(
        pos, vel, att, rat, alt, alt_rate,
        period_p=0.1, period_a=0.01, period_z=0.1,
    )
    if scurve:
        return SCurveController(ctrl)
    return ctrl



def make_controller_from_trial(trial: optuna.Trial, args: Namespace=DEFAULTS, prefix=''):
    r_pitch_roll_p = trial.suggest_float(prefix + 'r_pitch_roll.k_p', 0.1, 50)
    r_pitch_roll_i = trial.suggest_float(prefix + 'r_pitch_roll.k_i', 0.1, 10)
    r_pitch_roll_d = trial.suggest_float(prefix + 'r_pitch_roll.k_d', 1, 250)
    r_pitch_roll_max_acc = trial.suggest_float(prefix + 'r_pitch_roll.max_acceleration', 0.1, 25)
    if args.use_yaw:
        r_yaw_p = trial.suggest_float(prefix + 'r_yaw.k_p', 0.1, 50)
        r_yaw_i = trial.suggest_float(prefix + 'r_yaw.k_i', 0.1, 10)
        r_yaw_d = trial.suggest_float(prefix + 'r_yaw.k_d', 1, 250)
        r_yaw_max_acc = trial.suggest_float(prefix + 'r_yaw.max_acceleration', 0.1, 5)
    else:
        r_yaw_p, r_yaw_i, r_yaw_d, r_yaw_max_acc = 0, 0, 0, 0

    a_pitch_roll_p = trial.suggest_float(prefix + 'a_pitch_roll.k_p', 0.1, 50)
    a_pitch_roll_i = trial.suggest_float(prefix + 'a_pitch_roll.k_i', 0.1, 10)
    a_pitch_roll_d = trial.suggest_float(prefix + 'a_pitch_roll.k_d', 1, 250)
    if args.use_yaw:
        a_yaw_p = trial.suggest_float(prefix + 'a_yaw.k_p', 0.1, 5)
        a_yaw_i = trial.suggest_float(prefix + 'a_yaw.k_i', 0.1, 1)
        a_yaw_d = trial.suggest_float(prefix + 'a_yaw.k_d', 1, 25)
    else:
        a_yaw_p, a_yaw_i, a_yaw_d = 0, 0, 0

    params = dict(
        ctrl_p = dict(
            k_p = trial.suggest_float(prefix + 'p.k_p', 0.1, 50),
            k_i = trial.suggest_float(prefix + 'p.k_i', 0.1, 10),
            k_d = trial.suggest_float(prefix + 'p.k_d', 1, 250),
        ),
        ctrl_v = dict(
            k_p = trial.suggest_float(prefix + 'v.k_p', 0.1, 50),
            k_i = trial.suggest_float(prefix + 'v.k_i', 0.1, 10),
            k_d = trial.suggest_float(prefix + 'v.k_d', 1, 250),
        ),
        ctrl_a = dict(
            k_p = np.asarray((a_pitch_roll_p, a_pitch_roll_p, a_yaw_p)),
            k_i = np.asarray((a_pitch_roll_i, a_pitch_roll_i, a_yaw_i)),
            k_d = np.asarray((a_pitch_roll_d, a_pitch_roll_d, a_yaw_d)),
        ),
        ctrl_r = dict(
            k_p = np.asarray((r_pitch_roll_p, r_pitch_roll_p, r_yaw_p)),
            k_i = np.asarray((r_pitch_roll_i, r_pitch_roll_i, r_yaw_i)),
            k_d = np.asarray((r_pitch_roll_d, r_pitch_roll_d, r_yaw_d)),
            max_acceleration = np.asarray((r_pitch_roll_max_acc, r_pitch_roll_max_acc, r_yaw_max_acc)),
            max_err_i = np.asarray((r_pitch_roll_max_acc, r_pitch_roll_max_acc, r_yaw_max_acc)),
        ),
        # ctrl_z = dict(
        #     k_p = trial.suggest_float('z.k_p', 0.1, 50),
        #     k_i = trial.suggest_float('z.k_i', 0.1, 10),
        #     k_d = trial.suggest_float('z.k_d', 1, 250),
        # ),
        # ctrl_vz = dict(
        #     k_p = trial.suggest_float('vz.k_p', 0.1, 50),
        #     k_i = trial.suggest_float('vz.k_i', 0.1, 10),
        #     k_d = trial.suggest_float('vz.k_d', 1, 250),
        # ),
    )
    if args.scurve:
        params.update({prefix + 'feedforward_weight': trial.suggest_float(prefix + 'feedforward_weight', 0.1, 1.0, step=0.1)})
    return params



def make_disturbance_fn(heading, time=0.) -> Callable[[Multirotor], np.ndarray]:
    force_heading = heading.split('@')
    if len(force_heading) == 2:
        force, heading = force_heading
        force = float(force)
        try:
            force, heading = float(force), float(heading) * np.pi / 180
            wforce = force * np.asarray([-np.cos(heading), -np.sin(heading), 0])
            def wind_fn(m: Multirotor):
                if m.t >= time:
                    dcm = direction_cosine_matrix(*m.orientation).astype(m.dtype)
                    return inertial_to_body(wforce.astype(m.dtype), dcm)
                return np.zeros(3, m.dtype)
        except ValueError: # i.e. heading is not a number, random heading
            def wind_fn(m: Multirotor):
                if m.t == 0:
                    setattr(m, '_heading', np.random.choice([0, np.pi/2, np.pi, 3*np.pi/2]))
                    wforce = force * np.asarray([-np.cos(m._heading), -np.sin(m._heading), 0])
                    setattr(m, '_wforce', wforce)
                if m.t >= time:
                    dcm = direction_cosine_matrix(*m.orientation).astype(m.dtype)
                    return inertial_to_body(m._wforce.astype(m.dtype), dcm)
                return np.zeros(3, m.dtype)
    else:
        def wind_fn(m: Multirotor):
            return np.zeros(3, m.dtype)
    return wind_fn



def apply_fault(
    env: Union[Multirotor, MultirotorTrajEnv], fault: str
) -> MultirotorTrajEnv:
    vehicle = getattr(env, 'vehicle', env)
    loss, fault = fault.split('@')
    if fault == 'all' or fault == 'battery':
        motors = range(len(vehicle.propellers))
    else:
        motors = [int(fault)]
    loss = float(loss)
    for motor in motors:
        vehicle.params.propellers[motor].k_thrust *= (1 - loss)
        vehicle.params.propellers[motor].k_drag *= (1 - loss)
        # Since propeller class deepcopies params, need to change them too
        vehicle.propellers[motor].params.k_thrust *= (1 - loss)
        vehicle.propellers[motor].params.k_drag *= (1 - loss)
    return env



def make_env(env_params, args: Union[Namespace,Dict]=DEFAULTS):
    if isinstance(args, dict):
        args = Namespace(**args)
    if args.wind != DEFAULTS.wind and 'disturbance_fn' not in env_params.keys():
        env_params['disturbance_fn'] = make_disturbance_fn(args.wind)
    if args.env_kind == 'traj':
        env = LongTrajEnv(**env_params)
    if args.fault != DEFAULTS.fault:
        apply_fault(env, args.fault)
    return env



def make_objective(args: Namespace=DEFAULTS):
    env_kwargs = dict(
        safety_radius=args.safety_radius,
        seed=0,
        get_controller_fn=lambda m: get_controller(m, args.scurve, args)
    )
    def objective(trial: optuna.Trial):
        ctrl_params = make_controller_from_trial(trial=trial, args=args)
        env = make_env(env_kwargs, args)
        env.ctrl.set_params(**ctrl_params)
        errs = []
        for i in range(args.num_sims):
            env.reset()
            waypoints = np.asarray([[0,0,0]])
            traj = Trajectory(env.vehicle, waypoints, proximity=0.1)
            log = run_sim(
                env, traj,
                env.ctrl_fn
            )
            errs.append(log.reward.sum())
        return np.mean(errs)
    return objective



def optimize(args: Namespace=DEFAULTS, seed: int=0):
    study = get_study(args.study_name, seed=seed)
    study.optimize(make_objective(args), n_trials=args.ntrials//args.nprocs)
    return study



def apply_params(ctrl: Controller, **params) -> Dict[str, Dict[str, np.ndarray]]:
    p = dict(ctrl_p={}, ctrl_v={}, ctrl_a={}, ctrl_r={}, ctrl_z={}, ctrl_vz={})
    for name, param in params.items():
        if '.' in name:
            pre, post = name.split('.')
            # skip parameters such as r_pitch_roll.*/r_yaw.*, which need to be combined together
            if pre.startswith('r_p') or pre.startswith('a_p') or pre.startswith('r_y') or pre.startswith('a_y'):
                continue
            p['ctrl_' + pre][post] = param
        else:
            p[name] = param
    # special case for rate controller with differenr pitch/roll, and yaw params
    # if r_yaw or a_yaw are not present, assume that yaw was not being controlled, and
    # set yaw params to 0.
    p['ctrl_r']['k_p'] = np.asarray([params['r_pitch_roll.k_p'], params['r_pitch_roll.k_p'], params.get('r_yaw.k_p', 0)])
    p['ctrl_r']['k_i'] = np.asarray([params['r_pitch_roll.k_i'], params['r_pitch_roll.k_i'], params.get('r_yaw.k_i', 0)])
    p['ctrl_r']['k_d'] = np.asarray([params['r_pitch_roll.k_d'], params['r_pitch_roll.k_d'], params.get('r_yaw.k_d', 0)])
    p['ctrl_r']['max_acceleration'] = np.asarray([params['r_pitch_roll.max_acceleration'],
                                                  params['r_pitch_roll.max_acceleration'],
                                                  params.get('r_yaw.max_acceleration', 0)])
    p['ctrl_a']['k_p'] = np.asarray([params['a_pitch_roll.k_p'], params['a_pitch_roll.k_p'], params.get('a_yaw.k_p', 0)])
    p['ctrl_a']['k_i'] = np.asarray([params['a_pitch_roll.k_i'], params['a_pitch_roll.k_i'], params.get('a_yaw.k_i', 0)])
    p['ctrl_a']['k_d'] = np.asarray([params['a_pitch_roll.k_d'], params['a_pitch_roll.k_d'], params.get('a_yaw.k_d', 0)])
    if ctrl is not None:
        ctrl.set_params(**p)
    return p



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
    parser.add_argument('--safety_radius', default=DEFAULTS.safety_radius, type=float)
    parser.add_argument('--num_sims', default=DEFAULTS.num_sims, type=int)
    parser.add_argument('--append', action='store_true', default=False)
    parser.add_argument('--pid_params', help='File to save pid params to.', type=str, default='')
    parser.add_argument('--comment', help='Comments to attach to studdy.', type=str, default='')
    parser.add_argument('--env_kind', help='"[traj,alloc]"', default=DEFAULTS.env_kind)
    args = parser.parse_args()

    if not args.append:
        try:
            os.remove(local_path / ('studies/' + args.study_name + '.db'))
        except OSError:
            pass
    
    # create study if it doesn't exist. The study will be reused with a new seed
    # by each process
    study = get_study(args.study_name)

    for key in vars(args):
        study.set_user_attr(key, getattr(args, key))

    with mp.Pool(args.nprocs) as pool:
        pool.starmap(optimize, [(args, i) for i in range(args.nprocs)])
    print(study.best_trial.number)
    print(study.best_params)
    if len(args.pid_params) > 0:
        with open(args.pid_params + ('.pickle' if not args.pid_params.endswith('pickle') else ''), 'wb') as f:
            pickle.dump(apply_params(None, **study.best_params), f)
