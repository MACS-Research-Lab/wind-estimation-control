# This contains common code for loading envs
import numpy as np
from multirotor.trajectories import Trajectory
from scripts.opt_multirotorenv import get_established_controller
from systems.multirotor import VP

from systems.multirotor_sliding_error import MultirotorTrajEnv as SlidingBaseEnv
from systems.long_multirotor_sliding_error import LongTrajEnv as SlidingLongEnv
from systems.long_blending import LongBlendingEnv
from systems.blending import BlendingEnv
from systems.sliding_error_leash import MultirotorTrajEnv as LeashBaseEnv


def setup_base_params(wind_ranges, **kwargs):
     kw = dict(
        safety_radius=kwargs['safety_radius'],
        vp=VP,get_controller_fn=kwargs['get_controller_fn'],
        steps_u=kwargs['steps_u'],
        scaling_factor=kwargs['scaling_factor'],
        wind_ranges=wind_ranges,
        proximity=5, # have to get within 5m of waypoint
        seed=kwargs['seed'])
     
     return kw
 
class OctorotorEnvSelector():
    def __init__(self):
        self.envs = {
            "sliding": (SlidingBaseEnv, SlidingLongEnv),
            "blending": (BlendingEnv, LongBlendingEnv),
            "leashed": (LeashBaseEnv, SlidingLongEnv) 
        }
    
    def get_env(self, env_name: str, params: dict, wind_range: list, waypts: np.ndarray):
        base_env_class, long_env_class = self.envs[env_name]
        
        leash = env_name == "leashed"
            
        env_kwargs = dict(
            safety_radius=5, 
            seed=0,
            get_controller_fn=lambda m: get_established_controller(m, leash=leash),
            vp = VP,
        )

        env_kwargs['steps_u'] = params['steps_u']
        env_kwargs['scaling_factor'] = params['scaling_factor']
            
        base_params = setup_base_params(wind_range, **env_kwargs)

        bounding_len = params['bounding_rect_length'] if 'bounding_rect_length' in params.keys() else 1000
        
        traj = Trajectory(None, points=waypts, resolution=bounding_len) 
        wpts = traj.generate_trajectory(curr_pos=np.array([0,0,0]))

        
        long_env = long_env_class( # can clean this up by adding all to a dict then just passing the **dict
            waypoints = wpts,
            base_env = base_env_class(**base_params),
            initial_waypoints = waypts,
            randomize_direction= False,
            window_distance = params['window_distance']
        )
       
        
        return long_env