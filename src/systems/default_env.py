import inspect
from typing import Literal, Callable, Type, Union, Iterable
from argparse import Namespace

from copy import deepcopy
from multirotor.coords import direction_cosine_matrix, inertial_to_body, body_to_inertial
import numpy as np
from stable_baselines3.ppo import PPO
import control
import gym
from tqdm.autonotebook import tqdm
from multirotor.simulation import Multirotor
from multirotor.helpers import DataLog
from multirotor.trajectories import Trajectory
from multirotor.vehicle import BatteryParams, MotorParams, PropellerParams, VehicleParams, SimulationParams
from multirotor.controller import (
    AltController, AltRateController,
    PosController, AttController,
    VelController, RateController,
    Controller, PIDController
)
from matlab_control import *
from parameters import pid_params

from systems.dryden_python_implementation import Wind_Model

from .base import SystemEnv

DEFAULTS = Namespace(
    safety_radius = 5,
    max_velocity = 15,
    max_acceleration = 2.5,
    max_tilt = np.pi / 12,
    max_rads = 670
)

BP = BatteryParams(max_voltage=22.2)
MP = MotorParams(
    moment_of_inertia=5e-5,
    resistance=0.27,
    k_emf=0.0265,
    k_motor=0.0932, 
    speed_voltage_scaling=0.0347,
    max_current=38.
)
PP = PropellerParams(
    moment_of_inertia=1.86e-6,
    use_thrust_constant=True,
    k_thrust=9.8419e-05, # 18-inch propeller
    k_drag=1.8503e-06, # 18-inch propeller
    motor=MP
    # motor=None
)
VP = VehicleParams(
    propellers=[deepcopy(PP) for _ in range(8)],
    battery=BP,
    # angles in 45 deg increments, rotated to align with
    # model setup in gazebo sim (not part of this repo)
    angles=np.linspace(0, -2*np.pi, num=8, endpoint=False) + 0.375 * np.pi, # np.pi / 2
    # angles=np.linspace(0, -2*np.pi, num=8, endpoint=False) + np.pi / 2, # np.pi / 2
    distances=np.ones(8) * 0.635,
    clockwise=[-1,1,-1,1,-1,1,-1,1],
    mass=10.66,
    inertia_matrix=np.asarray([
        [0.2506, 0, 0],
        [0, 0.2506, 0.],
        [0, 0, 0.4538]
    ])
)
SP = SimulationParams(dt=0.01, g=0*9.81, dtype=np.float32)

def create_multirotor(
    vp=VP, sp=SP, name='multirotor', xformA=np.eye(12), xformB=np.eye(4),
    return_mult_ctrl=False,
    kind: Literal['speeds', 'dynamics', 'waypoints']='dynamics',
    max_rads: float=DEFAULTS.max_rads,
    disturbance_fn: Callable[[Multirotor], np.ndarray]=lambda _: np.zeros(3, SP.dtype),
    multirotor_class: Type[Multirotor]=Multirotor,
    multirotor_kwargs: dict={}
):
    m = multirotor_class(vp, sp, **multirotor_kwargs)

    sys = None
    if return_mult_ctrl:
        return sys, dict(multirotor=m)
    return sys



class MultirotorTrajEnv(SystemEnv):
    def get_controller():
        return None

    def __init__(
        self, vp=VP, sp=SP,
        xformA=np.eye(12), xformB=np.eye(4),
        get_controller_fn: Callable[[Multirotor], Controller]=get_controller,
        disturbance_fn: Callable[[Multirotor], np.ndarray]=lambda m: np.zeros(3, np.float32),
        wind_ranges: list = [(0,0), (0,0), (0,0)],
        scaling_factor: float=1.,
        steps_u: int=1,
        max_rads: float=DEFAULTS.max_rads,
        safety_radius: float=DEFAULTS.safety_radius,
        random_disturbance_direction=False,
        proximity=0.65,
        multirotor_class=Multirotor, multirotor_kwargs={},
        seed=0,
        pid_parameters: pid_params = None 
    ):
        
        system, extra = create_multirotor(
            vp, sp, xformA=xformA, xformB=xformB,
            return_mult_ctrl=True,
            #get_controller_fn=get_controller_fn,
            disturbance_fn=self.random_wind,
            kind='velocities',
            max_rads=max_rads,
            multirotor_class=multirotor_class,
            multirotor_kwargs=multirotor_kwargs
        )
        super().__init__(system=system, q=[], r=[], dt=SP.dt, seed=seed, dtype=SP.dtype)
        self.vehicle: Multirotor = extra['multirotor']
        self.wind_ranges = wind_ranges
        self.wind_x = np.random.uniform(self.wind_ranges[0][0], self.wind_ranges[0][1])
        self.wind_y = np.random.uniform(self.wind_ranges[1][0], self.wind_ranges[1][1])
        self.wind_z = np.random.uniform(self.wind_ranges[2][0], self.wind_ranges[2][1])
        self.disturbance_fn = self.random_wind
        self.noise_correlation = np.zeros(6)
        self.pos_pid = PIDController(k_p=pid_parameters.pos_p, k_i=pid_parameters.pos_i, k_d=pid_parameters.pos_d, max_err_i=0)
        self.vel_pid = PIDController(k_p=pid_parameters.vel_p, k_i=pid_parameters.vel_i, k_d=pid_parameters.vel_d, max_err_i=15)

        self.observation_space = gym.spaces.Box(
            low=-1, high=1,
            shape=(17,), dtype=self.dtype
        )
        self.action_space = gym.spaces.Box(
            low=-1, high=1, shape=(3,), dtype=self.dtype
        )
        
        self.ekf = getattr(self.vehicle, 'ekf', False)
        self.random_disturbance_direction = random_disturbance_direction
        self.max_rads = max_rads
        self.scaling_factor = scaling_factor
        self.safety_radius = safety_radius
        self.overshoot_factor = 0.5
        self.state_range = np.empty(self.observation_space.shape, self.dtype)
        self.action_range = np.empty(self.action_space.shape, self.dtype)
        self.x = np.zeros(self.observation_space.shape, self.dtype)
        self.steps_u = steps_u

        # self.period = 150*2 #TODO: CHANGED THIS # seconds
        self.period = 150 #TODO: CHANGED THIS # seconds
        self._proximity = proximity
        self.always_modify_wind = False
        self.random_cardinal_wind = False
        self.total_t = 0
        self.has_injection = False
        self.injected = False
        self.has_turbulence = False

        self.wind_forces = []
        self.vehicle.all_forces = []
        self.vehicle.all_torques = []
        self.vehicle.all_dxdt = []

        self.max_velocity = DEFAULTS.max_velocity

        self.max_tilt = (22.5*np.pi) / 180
        self.pid_parameters = pid_parameters

    

    @property
    def state(self) -> np.ndarray:
        if self.ekf:
            x = np.asarray(self.ekf.x, self.dtype)
        else:
            x = self.x
        # return self.normalize_state(x)
        return x

    def random_wind(self, m):
        if self.random_cardinal_wind: # if cardinal winds
            if self.direction_rand > 0.75: # N
                self.wind_x = 0
                self.wind_y = np.abs(self.wind_y)
            elif self.direction_rand > 0.5: # S
                self.wind_x = 0
                self.wind_y = -np.abs(self.wind_y)
            elif self.direction_rand > 0.25: # E
                self.wind_x = np.abs(self.wind_x)
                self.wind_y = 0
            else: # W
                self.wind_x = -np.abs(self.wind_x)
                self.wind_y = 0
                
        # Drag force calculation, copied from Matlab code
        rho = 1.2
        cd = 1
        Axy, Axz, Ayz = 0.403, 0.403, 0.403
        const = -0.5 * rho * cd

        # Make sure this wind vector can change
        wind_vector = np.array([self.wind_x, self.wind_y, self.wind_z], dtype=np.float32)
        dcm = direction_cosine_matrix(m.orientation[0], m.orientation[1], m.orientation[2])
        v_wb = inertial_to_body(wind_vector, dcm)

        Vb = m.velocity
        # v_a = v_wb + Vb
        v_a = Vb - v_wb
        newtons = const * np.array([Ayz * v_a[0]*np.abs(v_a[0]), Axz * v_a[1]*np.abs(v_a[1]), Axy * v_a[2]*np.abs(v_a[2])]) 
        # newtons = const * Axy * (np.square(v_a))
        self.wind_forces.append(newtons)
        
        return newtons

    def normalize_state(self, state):
        return state * 2 / (self.state_range+1e-6)
    def unnormalize_state(self, state):
        state *= self.state_range / 2
        return state
    def normalize_action(self, u):
        return u * 2 / (self.action_range)
    def unnormalize_action(self, u):
        u *= self.action_range / 2
        return u


    def reset(self, uav_x=None, modify_wind=False):
        super().reset(uav_x)
        # self.ctrl.reset()
        # self.vehicle.reset()

        if self.always_modify_wind:
            modify_wind = True

        if self.random_cardinal_wind and modify_wind:
            self.direction_rand = np.random.uniform(0,1)
        
        if modify_wind or self.always_modify_wind:
            self.wind_x = np.random.uniform(self.wind_ranges[0][0], self.wind_ranges[0][1])
            self.wind_y = np.random.uniform(self.wind_ranges[1][0], self.wind_ranges[1][1])
            self.wind_z = np.random.uniform(self.wind_ranges[2][0], self.wind_ranges[2][1])
        # Nominal range of state, not accounting for overshoot due to process dynamics
        self.state_range[0] = 500
        self.state_range[1] = 500
        self.state_range[2] = 100 
        self.state_range[3:6] = 2 * self.max_velocity
        self.state_range[6:9] = 2 * self.max_tilt
        self.state_range[9:12] = 2 * self.max_tilt # k_p also here TODO
        self.state_range[12:15] = self.state_range[:3]
        self.state_range[15:17] = 30 # -15 to 15

        self.action_range = self.scaling_factor
        # Max overshoot allowed, which will cause episode to terminate
        self._max_pos = self.safety_radius * (1 + self.overshoot_factor) / 2
        # self._max_angle = self.ctrl.ctrl_v.max_tilt * (1 + self.overshoot_factor)
        # self._max_angle = self.ctrl.ctrl_v.max_tilt * 2
        self._max_angle = 120
        
        self.time_penalty = self.dt * self.steps_u

        err_wp = np.asarray(uav_x[0:3]) 
        vel = np.asarray(uav_x[3:6]) 
        ori = np.asarray(uav_x[6:9])
        rat = np.asarray(uav_x[9:12]) 
        err_proj = np.asarray(uav_x[12:15]) 
        disturbance = np.asarray(uav_x[15:17])

        self.x = np.concatenate((err_wp, vel, ori, rat, err_proj, disturbance), dtype=self.dtype) 

        # Manually set underlying vehicle's state
        # self.vehicle.state = self.x #TODO: do I need this?
        self.vehicle.speeds = np.array([363.52]*8, dtype=np.float32) # initialize in hovering

        if self.has_turbulence:
            self.get_turbulence(self.prev_waypt, self.next_waypt)

        return self.state
    
    def step_spd(
        self, action: np.ndarray, disturb_forces: np.ndarray=0.,
        disturb_torques: np.ndarray=0.):
        """
        Step environment by providing speed signal.

        Parameters
        ----------
        action : np.ndarray
            An array of speed signals.
        disturb_forces : np.ndarray, optional
            Disturbinng x,y,z forces in the velicle's local frame, by default 0.
        disturb_torques : np.ndarray, optional
            Disturbing x,y,z torques in the vehicle's local frame, by default 0.

        Returns
        -------
        Tuple[np.ndarray, float, bool, dict]
            The state and other environment variables.
        """
        state = self.state
        forces, torques = self.vehicle.get_forces_torques(action, state.astype(self.vehicle.dtype))
        self.vehicle.all_forces.append(forces)
        self.vehicle.all_torques.append(torques)
            
        nstate = self.vehicle.step_speeds(
            u=action,
            disturb_forces=disturb_forces,
            disturb_torques=disturb_torques
        )
        reward = self.reward(state, action, nstate)
        return nstate, reward, False, {}

    
    def step(self, u: np.ndarray, **kwargs):
        u = np.clip(u, a_min=-1., a_max=1.)
        u = self.unnormalize_action(u)
        reward = 0
        for _ in range(self.steps_u):
            self.total_t += 1
            if self.has_injection: # dealing with injecting wind mid-flight, have do so some hacky logic because this is a gym env inside a gym env
                if self.total_t >= self.injection_end and self.injected:
                    self.wind_x = self.tmp_wind_x
                    self.wind_y = self.tmp_wind_y
                    self.wind_z = self.tmp_wind_z

                if self.total_t >= self.injection_start and not self.injected:
                    self.injected = True
                    self.tmp_wind_x = self.wind_x
                    self.tmp_wind_y = self.wind_y
                    self.tmp_wind_z = self.wind_z
                    
                    self.wind_x = self.injected_wind[0]
                    self.wind_y = self.injected_wind[1]
                    self.wind_z = self.injected_wind[2]
            
            
            prev_v = self.vehicle.position[:3] - self.prev_waypt
            # print("v", prev_v)
            # Calculate the scalar factor
            prev_scalar_factor = np.dot(prev_v, self._des_unit_vec) / (np.dot(self._des_unit_vec, self._des_unit_vec)+1e-8)
            # print("sc", prev_scalar_factor)

            # Calculate the intersection point coordinates
            prev_intersection_point = self.prev_waypt + prev_scalar_factor * self._des_unit_vec
            # x, r, d, *_, i = super().step(np.concatenate(([self.calculate_safe_sliding_bound(self.next_waypt, prev_intersection_point, distance=self.window_distance),u])))
            
            # dynamics = self.ctrl.step(np.concatenate(([self.calculate_safe_sliding_bound(self.next_waypt, prev_intersection_point, distance=self.window_distance),u])), ref_is_error=False, is_velocity=True)
            
            # m.all_forces.append(dynamics[0])
            # m.all_torques.append(dynamics[1:])


            # speeds = self.vehicle.allocate_control(dynamics[0], dynamics[1:4]) # see if the inverse works with small eps
            speeds = self.cascade_pid(self.next_waypt, self.vehicle.inertial_velocity, self.vehicle.position, self.vehicle.orientation, self.vehicle.angular_rate, self.pos_pid, self.vel_pid)
            
            # if self.t > 5:
            #     print("Has fault")
            #     fault_mult = np.array([1,1,1,1,1,1,1,1])
            #     speeds *= fault_mult 

                # self.vehicle.propellers[0].k_thrust = 0
                # self.vehicle.propellers[0].k_drag = 0

                # print(self.vehicle.propellers[0].k_thrust)

            # speeds *= fault_mult
            speeds = np.clip(speeds, a_min=0, a_max=670) 
            self.vehicle.speeds = speeds
            x, r, d, *_, i = self.step_spd(speeds, self.random_wind(self.vehicle))
            self.x = x
            
            # self.x[15] = self.wind_x + np.random.normal(0, 0.5)
            # self.x[16] = self.wind_y + np.random.normal(0, 0.5)
            
            dist = np.linalg.norm(self.next_waypt - self.x[:3])
            reached = dist <= self._proximity 
            current_v = self.vehicle.position[:3] - self.prev_waypt
            cross_v = np.cross(current_v, self._des_unit_vec)
            normal_distance = np.linalg.norm(cross_v) # because it is norm 1
            
            # Calculate the scalar factor
            scalar_factor = np.dot(current_v, self._des_unit_vec) / (np.dot(self._des_unit_vec, self._des_unit_vec)+1e-8)

            # Calculate the intersection point coordinates
            intersection_point = self.prev_waypt + scalar_factor * self._des_unit_vec
            
            # self.x[12:15] = self.vehicle.position[:3] - intersection_point
            
            self.vehicle.state = self.x[0:12]
            self.vehicle.t = self.t 

            self.vehicle.state += self.generate_noise_vector() 

            if self.has_turbulence:
                self.update_wind_with_turbulence(intersection_point, self.prev_waypt, self.next_waypt)

            outofbounds = normal_distance > self.safety_radius 
            
            reward -= normal_distance / 5
            
            outoftime = self.t >= self.period
            # tipped = np.any(np.abs(self.x[6:9]) > self._max_angle)
            tipped = np.any(self.vehicle.velocity > 50)
            self.t += 0.01
            
            crashed = self.vehicle.position[2] <= 0
            done = outoftime or reached or tipped or crashed

            if done:
                i.update(dict(reached=reached, outofbounds=outofbounds, outoftime=outoftime, tipped=tipped, crashed=crashed))
                break

        # observed_state = np.concatenate([self.next_waypt - self.x[:3], self.x[3:17]], dtype=np.float32)
        # return self.normalize_state(observed_state), reward, done, *_, i
        return self.x, reward, done, *_, i


    def cascade_pid(self, ref_pos, vel, pos, eul, rate, pos_pid, vel_pid):
        max_vel = self.max_velocity # make a parameter
        if self.total_t % 2 - 1 == 0:
            # Then we should get an update from the position and velocity controller
            # otherwise, just use our previous angle reference
            inert_ref_vel = pos_controller(ref_pos, pos, pos_pid)
            inert_ref_vel = np.clip(inert_ref_vel, -max_vel, max_vel)
            inert_ref_vel_leashed = vel_leash(inert_ref_vel, eul, max_vel)
            ref_vel = np.array([inert_ref_vel_leashed[0], inert_ref_vel_leashed[1], inert_ref_vel[2]])

            angle_ref = vel_controller(ref_vel, vel, vel_pid)
            self.angle_ref = angle_ref
        
        zforce_ref = self.angle_ref[2]

        theta_phi_ref = self.angle_ref[[1,0]] # swap roll and pitch
        rate_ref = angle_controller(theta_phi_ref, eul, self.pid_parameters.att_p)

        torque_ref = rate_controller(rate_ref, rate, self.pid_parameters.rate_p)

        return self.vehicle.allocate_control(zforce_ref, torque_ref)

    def ctrl_fn(self, x):
        return np.zeros(3, self.dtype)
    
    # should be added to self.vehicle.state
    def generate_noise_vector(self):
        noise_vector = np.zeros_like(self.vehicle.state)
        
        # noise_vector[0] = np.random.normal(0, 0.9)
        # noise_vector[1] = np.random.normal(0, 0.9)
        # noise_vector[2] = 0
        

        noise_vector[3] = np.random.normal(0, 0.0167/10) # vx
        noise_vector[4] = np.random.normal(0, 0.0167/10) # vy
        noise_vector[5] = 0

        w = 0.001
        tau = 0.002
        v = 0.001

        for i in range(6):
            self.noise_correlation[i] = self.noise_correlation[i] * np.exp(-self.dt / tau) + np.random.normal(0,w)
            noise_vector[i+6] = self.noise_correlation[i] + np.random.normal(0,v)

        return noise_vector

        
    def calculate_safe_sliding_bound(self, reference_point, intersection_point, distance=5):
        # Convert points to numpy arrays for vector calculations
        reference_point = np.array(reference_point)
        intersection_point = np.array(intersection_point)
        
        # Calculate the vector from the point to the reference point
        vector_to_reference = reference_point - intersection_point
        
        # Calculate the distance between the point and the reference point
        distance_to_reference = np.linalg.norm(vector_to_reference)
        
        if distance_to_reference <= distance:
            # If the distance is within the specified range, return the reference point
            return reference_point
        else:
            # Calculate the intermediate point that is 'distance' units along the vector_to_reference
            intermediate_point = intersection_point + (distance / distance_to_reference) * vector_to_reference
            return intermediate_point
        
    def get_turbulence(self, prev_waypt, curr_waypt):
        wind_vec = np.array([self.wind_x, self.wind_y, self.wind_z])

        wind_model = Wind_Model()
        
        if np.array_equal(prev_waypt, curr_waypt):
            turbulent_wind = [wind_vec]
        else:
            time, locs, turbulent_wind = wind_model.get_wind_vector_waypoint(start_wp=prev_waypt, end_wp=curr_waypt, veh_speed=1, turbulence=7.7, base_wind_vec=wind_vec)
        self.turbulent_wind = turbulent_wind

    def update_wind_with_turbulence(self, intersection_point, prev_waypt, next_waypt):
        waypt_vec = next_waypt - prev_waypt
        progress_vec = intersection_point - prev_waypt

        percent_completed =  np.clip(np.linalg.norm(progress_vec) / (np.linalg.norm(waypt_vec)+1e-6), 0, 1)
        index = int(len(self.turbulent_wind) * percent_completed) - 1

        self.wind_x = self.turbulent_wind[index][0]
        self.wind_y = self.turbulent_wind[index][1]
        self.wind_z = self.turbulent_wind[index][2]

