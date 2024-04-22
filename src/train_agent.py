import sys
import os

import torch
import multiprocessing as mp
from argparse import ArgumentParser, Namespace
import numpy as np
from multirotor.trajectories import Trajectory
from rl import learn_rl, evaluate_rl, load_agent
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from systems.lstm_env import Multirotor, MultirotorTrajEnv, VP
from systems.long_multirotor_wind_estimation import LongTrajEnv
from environments import OctorotorEnvSelector
from trajectories import square_100, circle_100, nasa_wp


if __name__=='__main__':
    wp_options = [square_100, circle_100, nasa_wp]
    env_selector = OctorotorEnvSelector()
    best_params = {'steps_u':50, 'scaling_factor':4, 'window_distance':20}
    env = env_selector.get_env("lstm", best_params, [(0,5), (0,5), (0,0)], square_100, start_alt=30, has_turbulence=True)
    env.wp_options = wp_options
    env.base_env.fault_type="random"
    
    ppo = PPO(policy="MlpPolicy", env=env, learning_rate=1e-4, n_epochs=3, n_steps=100, 
              policy_kwargs=dict(squash_output=False,net_arch=[dict(pi=[128]*3, vf=[128]*3)]), verbose=1, tensorboard_log='./logs/agent_train')
    
    checkpoint_callback = CheckpointCallback(save_freq=5000, save_path="./saved_models/", name_prefix="lstm_agent_random")
    
    agent = ppo.learn(total_timesteps=250000, progress_bar=True, callback=checkpoint_callback)
