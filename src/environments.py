# This contains common code for loading envs
import numpy as np
from multirotor.trajectories import Trajectory
# from opt_multirotorenv import get_established_controller
from systems.multirotor import VP

from systems.multirotor_sliding_error import MultirotorTrajEnv as SlidingBaseEnv
from systems.long_multirotor_sliding_error import LongTrajEnv as SlidingLongEnv
from systems.long_blending import LongBlendingEnv
from systems.blending import BlendingEnv
from systems.sliding_error_leash import MultirotorTrajEnv as LeashBaseEnv
from systems.long_multirotor_naive import LongTrajEnv as NaiveLongEnv
from systems.multirotor_naive import MultirotorTrajEnv as NaiveBaseEnv
from systems.long_multirotor_oracle import LongTrajEnv as OracleLongEnv
from systems.multirotor_oracle import MultirotorTrajEnv as OracleBaseEnv
from systems.long_multirotor_wind_estimation import LongTrajEnv as LSTMLongEnv
from systems.lstm_env import MultirotorTrajEnv as LSTMBaseEnv
from systems.long_dji_sliding_error import LongTrajEnv as DJILongSliding
from systems.dji_sliding_error import MultirotorTrajEnv as DJIBaseSliding
from systems.long_dji_wind_estimation import LongTrajEnv as LSTMDJILongEnv
from systems.dji_wind_estimation import MultirotorTrajEnv as LSTMDJIBaseEnv
from systems.ardupilot_wind import MultirotorTrajEnv as ArduPilotBaseEnv
from systems.long_ardupilot import LongTrajEnv as ArduPilotLongEnv
from systems.ardupilot_400hz import MultirotorTrajEnv as ArduPilotBase400Env
from systems.long_ardupilot_400hz import LongTrajEnv as ArduPilotLong400Env
from systems.default_env import MultirotorTrajEnv as DefaultEnv


def setup_base_params(wind_ranges, **kwargs):
     kw = dict(
        safety_radius=kwargs['safety_radius'],
        vp=VP,#get_controller_fn=kwargs['get_controller_fn'],
        steps_u=kwargs['steps_u'],
        scaling_factor=kwargs['scaling_factor'],
        wind_ranges=wind_ranges,
        proximity=2, # have to get within 5m of waypoint
        seed=kwargs['seed'],
        pid_parameters=kwargs['pid_parameters'])
     
     return kw
 
class OctorotorEnvSelector():
    def __init__(self):
        self.envs = {
            "sliding": (SlidingBaseEnv, SlidingLongEnv),
            "blending": (BlendingEnv, LongBlendingEnv),
            "leashed": (LeashBaseEnv, SlidingLongEnv),
            "naive": (NaiveBaseEnv, NaiveLongEnv),
            "oracle": (OracleBaseEnv, OracleLongEnv),
            "lstm": (LSTMBaseEnv, LSTMLongEnv) ,
            "dji_sliding": (DJIBaseSliding, DJILongSliding),
            "dji_lstm": (LSTMDJIBaseEnv, LSTMDJILongEnv),
            "ardupilot": (ArduPilotBaseEnv, ArduPilotLongEnv),
            "ardupilot_400hz": (ArduPilotBase400Env, ArduPilotLong400Env),
            "default": (DefaultEnv, SlidingLongEnv)
        }
    
    def get_env(self, env_name: str, params: dict, wind_range: list, waypts: np.ndarray, start_alt: int = 0, has_turbulence: bool = False, cardinal_wind=False):
        base_env_class, long_env_class = self.envs[env_name]
        
        leash = env_name == "leashed"
            
        env_kwargs = dict(
            safety_radius=2, 
            seed=0,
            vp = VP,
        )

        env_kwargs['steps_u'] = params['steps_u']
        env_kwargs['scaling_factor'] = params['scaling_factor']
        env_kwargs['pid_parameters'] = params['pid_parameters']
            
        base_params = setup_base_params(wind_range, **env_kwargs)

        bounding_len = params['bounding_rect_length'] if 'bounding_rect_length' in params.keys() else 1000
        
        traj = Trajectory(None, points=waypts, resolution=bounding_len) 
        wpts = traj.generate_trajectory(curr_pos=np.array([0,0,start_alt]))
        
        long_env = long_env_class( # can clean this up by adding all to a dict then just passing the **dict
            waypoints = wpts,
            base_env = base_env_class(**base_params),
            initial_waypoints = waypts,
            randomize_direction= False,
            window_distance = params['window_distance'],
            has_turbulence = has_turbulence,
            random_cardinal_wind=cardinal_wind
        )

        long_env.start_alt = start_alt
        long_env.base_env.vehicle.position[2] = start_alt

        return long_env