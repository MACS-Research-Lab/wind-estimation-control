from typing import Iterable

import numpy as np

from .multirotor import MultirotorTrajEnv



class LongTrajEnv:

    def __init__(self, waypoints: Iterable[np.ndarray], initial_waypoints: Iterable[np.ndarray], base_env: MultirotorTrajEnv, randomize_direction=False, always_modify_wind=False, random_cardinal_wind=False, injection_data = None):
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
        self.observation_space = self.base_env.observation_space
        self.action_space = self.base_env.action_space
        self.metadata = self.base_env.metadata
        self.seed = self.base_env.seed
        self.steps_u = self.base_env.steps_u
        self.scaling_factor = self.base_env.scaling_factor
        self.real_waypt_idx = None
        self.random_cardinal_wind = random_cardinal_wind
        
        if injection_data is not None: # used for injecting wind mid-flight
            self.base_env.has_injection = True
            self.base_env.injection_start = injection_data['start']
            self.base_env.injection_end = injection_data['end']
            self.base_env.injected_wind = injection_data['wind']


    def reset(self):
        self.current_waypoint_idx = 0
        self.real_waypt_idx = 0
        self.base_env.completed_distance = 0
        self.base_env.random_cardinal_wind = self.random_cardinal_wind
        self.base_env.total_t = 0
        waypt_vec = self.waypoints[self.current_waypoint_idx] - np.array([0,0,0])
        self.base_env._des_unit_vec = waypt_vec / (np.linalg.norm(waypt_vec)+1e-6)
        
        # TODO: make sure this works for all trajectories
        self.base_env.reset(uav_x=np.concatenate([np.array([0,0,0,0,0,0,0,0,0,0,0,0], np.float32), self.waypoints[self.current_waypoint_idx]]), modify_wind=True)
        #self.base_env.reset(uav_x=np.concatenate([np.array([0,0,0,0,0,0,0,0,0,0,0,0], np.float32), self.waypoints[self.current_waypoint_idx]]), modify_wind=True)
        self.base_env.prev_waypt = np.array([0,0,0])
        self.base_env.prev_real_waypt = np.array([0,0,0])
        self.base_env.next_waypt = self.initial_waypoints[self.real_waypt_idx]
        return self.base_env.state


    def step(self, u: np.ndarray):
        assert self.current_waypoint_idx is not None, "Make sure to call the reset() method first."
        # coming from a tanh NN policy function
        u = np.clip(u, a_min=-1., a_max=1.)

        # u = self.base_env.unnormalize_action(u)
        # u = u + self.waypoints[self.current_waypoint_idx]
        # u = self.base_env.normalize_action(u)

        done = False
        reward = 0
        s, reward, _, info = self.base_env.step(u)

        # if the sub gym env has reached its waypoint at the end of the bounding rectangle
        if info.get('reached'):
            # reward it if it was one of the original waypoints
            if self.real_waypt_idx < len(self.initial_waypoints) and np.linalg.norm(self.initial_waypoints[self.real_waypt_idx] - self.base_env.x[:3]) < self.base_env._proximity:
                reward += 1500 # bonus for reaching one of the original waypoints
                self.real_waypt_idx += 1 
                if self.real_waypt_idx < len(self.initial_waypoints):
                    self.base_env.completed_distance += np.linalg.norm(self.base_env.prev_real_waypt - self.base_env.next_waypt)
                    self.base_env.next_waypt = self.initial_waypoints[self.real_waypt_idx]
                    self.base_env.prev_real_waypt = self.initial_waypoints[self.real_waypt_idx-1]
                
            self.base_env.prev_waypt = self.waypoints[self.current_waypoint_idx]
            self.current_waypoint_idx += 1

            # if full traj is finished
            if self.current_waypoint_idx == len(self.waypoints):
                done = True
            else:
                self.base_env.reset(uav_x=np.concatenate([self.base_env.x[:12], self.waypoints[self.current_waypoint_idx]]), modify_wind=False)
                waypt_vec = self.waypoints[self.current_waypoint_idx] - self.waypoints[self.current_waypoint_idx-1]
                self.base_env._des_unit_vec = waypt_vec / np.linalg.norm(waypt_vec) 

        if not done:
            if info.get('tipped') or info.get('outoftime'):
                done = True
                reward -= 1500 # negative reward for not finishing (equivalent to 15 seconds outside the radius)

        # state is a 12+3 element vector, where last 3
        # elements are the normalized next waypoint
        return s, reward, done, info
    

    
        