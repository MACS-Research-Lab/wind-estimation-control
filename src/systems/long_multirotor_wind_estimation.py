from typing import Iterable

import numpy as np

from .multirotor import MultirotorTrajEnv
from multirotor.trajectories import Trajectory



class LongTrajEnv:

    def __init__(self, waypoints: Iterable[np.ndarray], initial_waypoints: Iterable[np.ndarray], base_env: MultirotorTrajEnv, randomize_direction=False, always_modify_wind=False, random_cardinal_wind=False, injection_data = None, window_distance = 10, has_turbulence=False):
        self.waypoints = waypoints
        self.initial_waypoints = initial_waypoints
        
        self.randomize_waypoints = True
        
        
        randomize_direction = False

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
        self.base_env.window_distance = window_distance
        self.start_alt = 0
        self.base_env.has_turbulence = has_turbulence
        
        if injection_data is not None: # used for injecting wind mid-flight
            self.base_env.has_injection = True
            self.base_env.injection_start = injection_data['start']
            self.base_env.injection_end = injection_data['end']
            self.base_env.injected_wind = injection_data['wind']


    def reset(self):
        if self.randomize_waypoints:
            chosen_idx = np.random.choice(len(self.wp_options))
            waypoints = self.wp_options[chosen_idx]
            traj = Trajectory(None, points=waypoints, resolution=1000) 
            wpts = traj.generate_trajectory(curr_pos=np.array([0,0,30]))
            self.initial_waypoints = waypoints
        
        self.current_waypoint_idx = 0
        self.real_waypt_idx = 0
        self.base_env.completed_distance = 0
        self.base_env.random_cardinal_wind = self.random_cardinal_wind
        self.base_env.total_t = 0
        self.base_env.lstm_input = [np.zeros(9)] * 10
        self.base_env.prev_pos = np.zeros(3)
        # TODO: make sure this works for all trajectories
        self.base_env.prev_waypt = np.array([0,0,self.start_alt])
        self.base_env.prev_real_waypt = np.array([0,0,self.start_alt])
        self.base_env.next_waypt = self.initial_waypoints[self.real_waypt_idx]
        # self.base_env.vehicle.position[2] = self.start_alt

        self.base_env.reset(uav_x=np.concatenate([np.zeros(17, np.float32)]), modify_wind=True)
        self.base_env.pos_pid.reset()
        self.base_env.vel_pid.reset()
        waypt_vec = self.waypoints[self.current_waypoint_idx+1] - np.array([0,0,self.start_alt]) # is this alt correct?
        self.base_env._des_unit_vec = waypt_vec / (np.linalg.norm(waypt_vec)+1e-6)
        
        # self.base_env.prev_waypt = np.array([0,0,self.start_alt])
        # self.base_env.prev_real_waypt = np.array([0,0,self.start_alt])
        # self.base_env.next_waypt = self.initial_waypoints[self.real_waypt_idx]
        self.base_env.vehicle.position[2] = self.start_alt
        
        self.base_env.fault_type = None
        if self.base_env.fault_type == "random":
            fault_idx = np.random.choice(8)
            fault_idx = 0
            self.base_env.fault_mult = np.ones(8)
            fault_magnitude = np.random.uniform(0,1)
            fault_magnitude = 0
            
            self.base_env.fault_mult[fault_idx] *= fault_magnitude
            self.base_env.fault_t = np.random.uniform(0, 10)
        elif self.base_env.fault_type == "fixed":
            fault_idx = 1
            self.base_env.fault_mult = np.ones(8)
            fault_magnitude = 0
            
            self.base_env.fault_mult[fault_idx] *= fault_magnitude
            self.base_env.fault_t = 5
        else:
            pass

        return self.base_env.state


    def step(self, u: np.ndarray):
        assert self.current_waypoint_idx is not None, "Make sure to call the reset() method first."
        # coming from a tanh NN policy function
        u = np.clip(u, a_min=-1., a_max=1.)

        done = False
        reward = 0
        s, reward, _, info = self.base_env.step(u)

        # if the sub gym env has reached its waypoint at the end of the bounding rectangle
        if info.get('reached'):
            # reward it if it was one of the original waypoints
            
            if self.real_waypt_idx < len(self.initial_waypoints) and np.linalg.norm(self.initial_waypoints[self.real_waypt_idx] - self.base_env.vehicle.position[:3]) < self.base_env._proximity:
                # reward += 2500 # bonus for reaching one of the original waypoints
                reward += 500 # bonus for reaching one of the original waypoints
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
            if info.get('tipped') or info.get('outoftime') or info.get('crashed'):
                done = True
                reward -= 5000 # negative reward for not finishing (equivalent to 15 seconds outside the radius)

        return s, reward, done, info
    
        