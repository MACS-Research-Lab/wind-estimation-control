from typing import Iterable

import numpy as np

from .blending import BlendingEnv
import gym

class LongBlendingEnv:

    def __init__(self, waypoints: Iterable[np.ndarray], initial_waypoints: Iterable[np.ndarray], base_env: BlendingEnv, randomize_direction=False, always_modify_wind=False, random_cardinal_wind=False, injection_data = None, window_distance = 10):
        self.waypoints = waypoints
        self.initial_waypoints = initial_waypoints

        # Whether to randomly reverse the direction of following the trajectory
        if randomize_direction:
            if np.random.uniform(0,1) > 0.5:
                self.waypoints = list(reversed(self.waypoints))
                self.initial_waypoints = list(reversed(self.initial_waypoints))
                
        # Setting up class variables 
        self.base_env = base_env
        self.base_env.always_modify_wind = always_modify_wind
        self.current_waypoint_idx = None
        self.observation_space = gym.spaces.Box(
            low=-1, high=1,
            shape=(21,), dtype=np.float32
        )
        self.action_space = gym.spaces.Box(
            low=-1, high=1, shape=(3,), dtype=np.float32
        )
        self.metadata = self.base_env.metadata
        self.seed = self.base_env.seed
        self.steps_u = self.base_env.steps_u
        self.scaling_factor = self.base_env.scaling_factor
        self.real_waypt_idx = None
        self.random_cardinal_wind = random_cardinal_wind
        self.base_env.window_distance = window_distance
        
        if injection_data is not None: # used for injecting wind mid-flight
            self.base_env.has_injection = True
            self.base_env.injection_start = injection_data['start']
            self.base_env.injection_end = injection_data['end']
            self.base_env.injected_wind = injection_data['wind']

        self.agents = self.get_agents()

    def reset(self):
        self.current_waypoint_idx = 0
        self.real_waypt_idx = 0
        self.base_env.completed_distance = 0
        self.base_env.random_cardinal_wind = self.random_cardinal_wind
        self.base_env.total_t = 0

        self.base_env.reset(uav_x=np.concatenate([np.zeros(15, np.float32)]), modify_wind=True)
        waypt_vec = self.waypoints[self.current_waypoint_idx+1] - np.array([0,0,0])
        self.base_env._des_unit_vec = waypt_vec / (np.linalg.norm(waypt_vec)+1e-6)
        
        # TODO: make sure this works for all trajectories
        self.base_env.prev_waypt = np.array([0,0,0])
        self.base_env.prev_real_waypt = np.array([0,0,0])
        self.base_env.next_waypt = self.initial_waypoints[self.real_waypt_idx]

        next_actions = self.get_actions(self.base_env.state, self.agents)
        self.current_actions = next_actions
        state = np.concatenate([self.base_env.state, next_actions[1], next_actions[2]])
        return state
    
    def step(self, u: np.ndarray):
        assert self.current_waypoint_idx is not None, "Make sure to call the reset() method first."
        # coming from a tanh NN policy function
        # u = np.clip(u, a_min=-1., a_max=1.)
        # u = u / (np.sum(u+1) + 1e-6) # normalize these to represent convex combination

        u =  self.current_actions.T @ (softmax(u*5)) # output of agent is the weight on each action

        done = False
        reward = 0
        s, reward, _, info = self.base_env.step(u)
        next_actions = self.get_actions(s, self.agents)
        state = np.concatenate([s, next_actions[1], next_actions[2]])

        self.current_actions = next_actions

        # if the sub gym env has reached its waypoint at the end of the bounding rectangle
        if info.get('reached'):
            # reward it if it was one of the original waypoints
            
            if self.real_waypt_idx < len(self.initial_waypoints) and np.linalg.norm(self.initial_waypoints[self.real_waypt_idx] - self.base_env.vehicle.position[:3]) < self.base_env._proximity:
                reward += 2500 # bonus for reaching one of the original waypoints
                self.real_waypt_idx += 1 
                if self.real_waypt_idx < len(self.initial_waypoints):
                    self.base_env.completed_distance += np.linalg.norm(self.base_env.prev_real_waypt - self.base_env.next_waypt)
                    self.base_env.next_waypt = self.initial_waypoints[self.real_waypt_idx]
                    self.base_env.prev_real_waypt = self.initial_waypoints[self.real_waypt_idx-1]
                
            self.current_waypoint_idx += 1

            # if full traj is finished
            if self.real_waypt_idx == len(self.initial_waypoints):
                done = True
            else:
                self.base_env.prev_waypt = self.initial_waypoints[self.real_waypt_idx-1]
                self.base_env.reset(uav_x=self.base_env.x, modify_wind=False) # may need to recalculate the 12:15 

                waypt_vec = self.initial_waypoints[self.real_waypt_idx] - self.initial_waypoints[self.real_waypt_idx-1]
                self.base_env._des_unit_vec = waypt_vec / np.linalg.norm(waypt_vec) 
                

        if not done:
            if info.get('tipped') or info.get('outoftime'):
                done = True
                reward -= 2500 # negative reward for not finishing (equivalent to 15 seconds outside the radius)

        return np.array(state, dtype=np.float32), reward, done, info
    
    def get_agents(self):
        highwind = get_agent("highwind")[1]
        midwind = get_agent("midwind")[1]

        agents = [midwind, highwind]
        return agents
    
    def get_actions(self, state, agents):
        actions = [[0,0,0]] # give zero action as option
        for agent in agents:
            actions.append(agent.predict(state, deterministic=True)[0])

        return np.array(actions)
    
        

# Various useful functions for this project
# from opt_multirotorenv import get_study
# from rl import load_agent

# Where the saved agent is stored during HPO
# def get_agent(name, agent_num=None):
#     log_root_path = './tensorboard/MultirotorTrajEnv/optstudy/%s/'
#     # study = get_study(name)
    
#     if agent_num is not None:
#         best_trial = agent_num
#     else:
#         best_trial = study.best_trial.number        
        
#     best_agent = load_agent((log_root_path + '%03d/run_1/agent') % (name, best_trial)) 
#     best_params = study.best_params
#     return study, best_agent, best_params


def softmax(x):
    e_x = np.exp(x - np.max(x))  # Subtracting the max value for numerical stability
    return e_x / e_x.sum(axis=0)