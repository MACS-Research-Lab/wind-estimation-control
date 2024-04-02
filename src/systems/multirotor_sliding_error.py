import inspect
from typing import Literal, Callable, Type, Union, Iterable
from argparse import Namespace

from copy import deepcopy
from multirotor.coords import direction_cosine_matrix, inertial_to_body
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
    Controller
)

from systems.dryden_python_implementation import Wind_Model

from .base import SystemEnv

DEFAULTS = Namespace(
    safety_radius = 5,
    max_velocity = 15,
    max_acceleration = 2.5,
    max_tilt = np.pi / 12,
    max_rads = 670
)

fault_mult: np.ndarray = np.array([0,1,1,1,1,1,1,1])

BP = BatteryParams(max_voltage=22.2)
MP = MotorParams(
    moment_of_inertia=5e-5,
    resistance=0.27,
    # resistance=0.081,
    k_emf=0.0265,
    k_motor=0.0932,
    speed_voltage_scaling=0.0347,
    max_current=38.
)
PP = PropellerParams(
    moment_of_inertia=1.86e-6,
    use_thrust_constant=True,
    k_thrust=9.8419e-05, # 18-inch propeller
    # k_thrust=5.28847e-05, # 15 inch propeller
    k_drag=1.8503e-06, # 18-inch propeller
    # k_drag=1.34545e-06, # 15-inch propeller
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
    # inertia_matrix=np.asarray([
    #     [0.2206, 0, 0],
    #     [0, 0.2206, 0.],
    #     [0, 0, 0.4238]
    # ])
    inertia_matrix=np.asarray([
        [0.2506, 0, 0],
        [0, 0.2506, 0.],
        [0, 0, 0.4538]
    ])
)
# VP.propellers[0].k_thrust = 0
# VP.propellers[0].k_drag = 0
# SP = SimulationParams(dt=0.01, g=9.81, dtype=np.float32)
SP = SimulationParams(dt=0.01, g=9.81, dtype=np.float32)




def get_controller(
        m: Multirotor, max_velocity=DEFAULTS.max_velocity,
        max_acceleration=DEFAULTS.max_acceleration,
        max_tilt=DEFAULTS.max_tilt,
        leash=False
    ) -> Controller:
    assert m.simulation.dt <= 0.1, 'Simulation time step too large.'
    pos = PosController(
        1.0, 0., 0.,
        max_err_i=DEFAULTS.max_velocity, vehicle=m,
        max_velocity=max_velocity,
        max_acceleration=max_acceleration  ,
        square_root_scaling=False,
        leashing=leash  
    )
    vel = VelController(
        2.0, 1.0, 0.5,
        max_err_i=DEFAULTS.max_acceleration, vehicle=m, max_tilt=max_tilt)
    att = AttController(
        [2.6875, 4.5, 4.5],
        0, 0.,
        max_err_i=1., vehicle=m)
    rat = RateController(
        [4., 4., 4.],
        0, 0, # purely P control
        # [0.1655, 0.1655, 0.5],
        # [0.135, 0.135, 0.018],
        # [0.01234, 0.01234, 0.],
        max_err_i=0.5,
        vehicle=m)

    alt = AltController(
        1, 0, 0,
        max_err_i=1, vehicle=m, max_velocity=max_velocity)
    alt_rate = AltRateController(
        10, 0, 0,
        max_err_i=1, vehicle=m)
    ctrl = Controller(
        pos, vel, att, rat, alt, alt_rate,
        period_p=0.1, period_a=0.01, period_z=0.1
    )
    return ctrl



def create_multirotor(
    vp=VP, sp=SP, name='multirotor', xformA=np.eye(12), xformB=np.eye(4),
    return_mult_ctrl=False,
    kind: Literal['speeds', 'dynamics', 'waypoints']='dynamics',
    max_rads: float=DEFAULTS.max_rads,
    get_controller_fn: Callable[[Multirotor], Controller]=get_controller,
    disturbance_fn: Callable[[Multirotor], np.ndarray]=lambda _: np.zeros(3, SP.dtype),
    multirotor_class: Type[Multirotor]=Multirotor,
    multirotor_kwargs: dict={}
):
    m = multirotor_class(vp, sp, **multirotor_kwargs)
    ctrl = get_controller_fn(m)

    if kind=='dynamics':
        inputs=['fz','tx','ty','tz']
        # NOTE: This is not a pure function due to setting m.speeds
        def update_fn(t, x, u, params):
            speeds = m.allocate_control(u[0], u[1:4])
            speeds = np.clip(speeds, a_min=0, a_max=max_rads)
            dxdt = m.dxdt_speeds(t, x.astype(m.dtype), speeds,
            disturb_forces=disturbance_fn(m))
            m.speeds = speeds
            return dxdt
    elif kind=='speeds':
        inputs = [('w%02d' % n) for n in range(len(m.propellers))]
        def update_fn(t, x, u, params):
            # speeds = np.clip(u, a_min=0, a_max=max_rads)
            # m.step_speeds(speeds, disturb_forces=disturbance_fn(m))
            # # here, waypoint supervision can be added
            # old_dynamics = ctrl.action
            # new_dynamics = ctrl.step(np.zeros(4, m.dtype), ref_is_error=False)
            # return (new_dynamics - old_dynamics) / m.simulation.dt
            return None # integration of dynamics is done directly in MultirotorAllocEnv.step()
    # NOTE: This is not a pure function due to setting m.speeds
    elif kind=='waypoints':
        inputs=['x','y','z','yaw']
        def update_fn(t, x, u, params):
            dynamics = ctrl.step(u, ref_is_error=False)
            speeds = m.allocate_control(dynamics[0], dynamics[1:4])
            speeds = np.clip(speeds, a_min=0, a_max=max_rads)
            dxdt = m.dxdt_speeds(t, x.astype(m.dtype), speeds,
                disturb_forces=disturbance_fn(m))
            m.speeds = speeds
            return dxdt
    elif kind=='velocities':
        inputs=['vx','vy','vz']
        def update_fn(t, x, u, params):
            dynamics = ctrl.step(u, ref_is_error=False, is_velocity=True)
            # m.all_forces.append(dynamics[0])
            # m.all_torques.append(dynamics[1:])
            speeds = m.allocate_control(dynamics[0], dynamics[1:4]) # see if the inverse works with small eps
            if t > 5:
                print("Has fault")
                speeds *= fault_mult 

            # speeds *= fault_mult 
            speeds = np.clip(speeds, a_min=0, a_max=max_rads) 
            forces, torques = m.get_forces_torques(speeds, x.astype(m.dtype))
            m.all_forces.append(forces)
            m.all_torques.append(torques)
            dxdt = m.dxdt_speeds(t, x.astype(m.dtype), speeds,
                disturb_forces=disturbance_fn(m))
            m.all_dxdt.append(dxdt)
            # m.speeds = speeds
            return dxdt
    # elif kind=='velocities':
    #     inputs=['vx','vy','vz']
    #     def update_fn(t, x, u, params):
    #         dynamics = ctrl.step(u, ref_is_error=False, is_velocity=True)
    #         # m.all_forces.append(dynamics[0])
    #         # m.all_torques.append(dynamics[1:])
    #         speeds = m.allocate_control(dynamics[0], dynamics[1:4]) # see if the inverse works with small eps
    #         # if t > 5:
    #         #     print("Has fault")
    #         #     speeds *= fault_mult # maybe do tha in the alocate control function
    #         # speeds *= fault_mult # maybe do tha in the alocate control function
    #         speeds = np.clip(speeds, a_min=0, a_max=max_rads) 
    #         forces, torques = m.get_forces_torques(speeds, x.astype(m.dtype))
    #         m.all_forces.append(forces)
    #         m.all_torques.append(torques)
    #         next_state = m.step_speeds(speeds, disturb_forces=disturbance_fn(m))
    #         # m.speeds = speeds
    #         return next_state

    sys = control.NonlinearIOSystem(
        updfcn=update_fn,
        inputs=inputs,
        states=['x','y','z',
                'vx','vy','vz',
                'roll','pitch','yaw',
                'xrate', 'yrate', 'zrate']
    )
    if return_mult_ctrl:
        return sys, dict(multirotor=m, ctrl=ctrl)
    return sys



class MultirotorTrajEnv(SystemEnv):


    def __init__(
        self, vp=VP, sp=SP,
        q=np.diagflat([1,1,1,0.25,0.25,0.25,0.5,0.5,0.5,0.1,0.1,0.1,1,1,1,1,1,1,1,1,1]),
        r = np.diagflat([1,1,1,0]) * 1e-4,
        dt=None, seed=None,
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
        multirotor_class=Multirotor, multirotor_kwargs={}
    ):
        system, extra = create_multirotor(
            vp, sp, xformA=xformA, xformB=xformB,
            return_mult_ctrl=True,
            get_controller_fn=get_controller_fn,
            disturbance_fn=self.random_wind,
            kind='velocities',
            max_rads=max_rads,
            multirotor_class=multirotor_class,
            multirotor_kwargs=multirotor_kwargs
        )
        self.vehicle: Multirotor = extra['multirotor']
        self.ctrl: Controller = extra['ctrl']
        self.wind_ranges = wind_ranges
        self.wind_x = np.random.uniform(self.wind_ranges[0][0], self.wind_ranges[0][1])
        self.wind_y = np.random.uniform(self.wind_ranges[1][0], self.wind_ranges[1][1])
        self.wind_z = np.random.uniform(self.wind_ranges[2][0], self.wind_ranges[2][1])
        self.disturbance_fn = self.random_wind
        self.noise_correlation = np.zeros(6)
        super().__init__(system=system, q=q, r=r, dt=sp.dt, seed=seed, dtype=sp.dtype)

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
        self.period = 30 #TODO: CHANGED THIS # seconds
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

    @property
    def state(self) -> np.ndarray:
        if self.ekf:
            x = np.asarray(self.ekf.x, self.dtype)
        else:
            x = self.x
        return self.normalize_state(x)

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
        v_a = v_wb + Vb
        newtons = const * np.array([Ayz * v_a[0]*np.abs(v_a[0]), Axz * v_a[1]*np.abs(v_a[1]), Axy * v_a[2]*np.abs(v_a[2])]) 
        self.wind_forces.append(newtons)
        return np.zeros(3, np.float32)
        # return newtons

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
        self.state_range[3:6] = 2 * self.ctrl.ctrl_p.max_velocity
        self.state_range[6:9] = 2 * self.ctrl.ctrl_v.max_tilt
        self.state_range[9:12] = 2 * self.ctrl.ctrl_v.max_tilt * self.ctrl.ctrl_a.k_p
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
        wind = np.asarray(uav_x[15:17])

        self.x = np.concatenate((err_wp, vel, ori, rat, err_proj, wind), dtype=self.dtype) 

        # Manually set underlying vehicle's state
        self.vehicle.state = self.x
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



    # def step(self, u: np.ndarray, **kwargs):
    #     u = np.clip(u, a_min=-1., a_max=1.)
    #     u = self.unnormalize_action(u)
    #     reward = 0
    #     for _ in range(self.steps_u):
    #         self.total_t += 1
    #         if self.has_injection: # dealing with injecting wind mid-flight, have do so some hacky logic because this is a gym env inside a gym env
    #             if self.total_t >= self.injection_end and self.injected:
    #                 self.wind_x = self.tmp_wind_x
    #                 self.wind_y = self.tmp_wind_y
    #                 self.wind_z = self.tmp_wind_z
                    
    #             if self.total_t >= self.injection_start and not self.injected:
    #                 self.injected = True
    #                 self.tmp_wind_x = self.wind_x
    #                 self.tmp_wind_y = self.wind_y
    #                 self.tmp_wind_z = self.wind_z
                    
    #                 self.wind_x = self.injected_wind[0]
    #                 self.wind_y = self.injected_wind[1]
    #                 self.wind_z = self.injected_wind[2]
            
            
    #         prev_v = self.vehicle.position[:3] - self.prev_waypt
    #         # print("v", prev_v)
    #         # Calculate the scalar factor
    #         prev_scalar_factor = np.dot(prev_v, self._des_unit_vec) / (np.dot(self._des_unit_vec, self._des_unit_vec)+1e-8)
    #         # print("sc", prev_scalar_factor)

    #         # Calculate the intersection point coordinates
    #         prev_intersection_point = self.prev_waypt + prev_scalar_factor * self._des_unit_vec
    #         x, r, d, *_, i = super().step(np.concatenate(([self.calculate_safe_sliding_bound(self.next_waypt, prev_intersection_point, distance=self.window_distance),u])))
            
    #         self.x[15] = self.wind_x + np.random.normal(0, 0.5)
    #         self.x[16] = self.wind_y + np.random.normal(0, 0.5)
            
    #         dist = np.linalg.norm(self.next_waypt - self.x[:3])
    #         reached = dist <= self._proximity 
    #         current_v = self.vehicle.position[:3] - self.prev_waypt
    #         cross_v = np.cross(current_v, self._des_unit_vec)
    #         normal_distance = np.linalg.norm(cross_v) # because it is norm 1
            
    #         # Calculate the scalar factor
    #         scalar_factor = np.dot(current_v, self._des_unit_vec) / (np.dot(self._des_unit_vec, self._des_unit_vec)+1e-8)

    #         # Calculate the intersection point coordinates
    #         intersection_point = self.prev_waypt + scalar_factor * self._des_unit_vec
            
    #         self.x[12:15] = self.vehicle.position[:3] - intersection_point
            
    #         self.vehicle.state = self.x
    #         self.vehicle.t = self.t 

    #         # self.vehicle.state += self.generate_noise_vector() 

    #         if self.has_turbulence:
    #             self.update_wind_with_turbulence(intersection_point, self.prev_waypt, self.next_waypt)

    #         outofbounds = normal_distance > self.safety_radius 
            
    #         reward -= normal_distance / 5
            
    #         outoftime = self.t >= self.period
    #         # tipped = np.any(np.abs(self.x[6:9]) > self._max_angle)
    #         tipped = False
            
    #         crashed = self.vehicle.position[2] <= 0
    #         done = outoftime or reached or tipped or crashed

    #         if done:
    #             i.update(dict(reached=reached, outofbounds=outofbounds, outoftime=outoftime, tipped=tipped, crashed=crashed))
    #             break

    #     observed_state = np.concatenate([self.next_waypt - self.x[:3], self.x[3:17]], dtype=np.float32)
    #     return self.normalize_state(observed_state), reward, done, *_, i
    
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
            dynamics = self.ctrl.step(np.concatenate(([self.calculate_safe_sliding_bound(self.next_waypt, prev_intersection_point, distance=self.window_distance),u])), ref_is_error=False, is_velocity=True)
            # m.all_forces.append(dynamics[0])
            # m.all_torques.append(dynamics[1:])
            speeds = self.vehicle.allocate_control(dynamics[0], dynamics[1:4]) # see if the inverse works with small eps
            # if t > 5:
            #     print("Has fault")
            #     speeds *= fault_mult 

            fault_mult = np.array([0,1,1,1,1,1,1,1])
            speeds *= fault_mult
            speeds = np.clip(speeds, a_min=0, a_max=670) 
            x, r, d, *_, i = self.step_spd(speeds, self.random_wind(self.vehicle))
            self.x = x
            
            self.x[15] = self.wind_x + np.random.normal(0, 0.5)
            self.x[16] = self.wind_y + np.random.normal(0, 0.5)
            
            dist = np.linalg.norm(self.next_waypt - self.x[:3])
            reached = dist <= self._proximity 
            current_v = self.vehicle.position[:3] - self.prev_waypt
            cross_v = np.cross(current_v, self._des_unit_vec)
            normal_distance = np.linalg.norm(cross_v) # because it is norm 1
            
            # Calculate the scalar factor
            scalar_factor = np.dot(current_v, self._des_unit_vec) / (np.dot(self._des_unit_vec, self._des_unit_vec)+1e-8)

            # Calculate the intersection point coordinates
            intersection_point = self.prev_waypt + scalar_factor * self._des_unit_vec
            
            self.x[12:15] = self.vehicle.position[:3] - intersection_point
            
            self.vehicle.state = self.x
            self.vehicle.t = self.t 

            # self.vehicle.state += self.generate_noise_vector() 

            if self.has_turbulence:
                self.update_wind_with_turbulence(intersection_point, self.prev_waypt, self.next_waypt)

            outofbounds = normal_distance > self.safety_radius 
            
            reward -= normal_distance / 5
            
            outoftime = self.t >= self.period
            # tipped = np.any(np.abs(self.x[6:9]) > self._max_angle)
            tipped = False
            self.t += 0.01
            
            crashed = self.vehicle.position[2] <= 0
            done = outoftime or reached or tipped or crashed

            if done:
                i.update(dict(reached=reached, outofbounds=outofbounds, outoftime=outoftime, tipped=tipped, crashed=crashed))
                break

        observed_state = np.concatenate([self.next_waypt - self.x[:3], self.x[3:17]], dtype=np.float32)
        return self.normalize_state(observed_state), reward, done, *_, i

    def ctrl_fn(self, x):
        return np.zeros(3, self.dtype)
    
    # should be added to self.vehicle.state
    def generate_noise_vector(self):
        noise_vector = np.zeros_like(self.vehicle.state)

        # noise_vector[0] = np.random.normal(0, 0.0167) # x
        # noise_vector[1] = np.random.normal(0, 0.0167) # y
        # noise_vector[2] = np.random.normal(0, 0.0167/2) # z
        noise_vector[3] = np.random.normal(0, 0.0167/10) # vx
        noise_vector[4] = np.random.normal(0, 0.0167/10) # vy
        # noise_vector[5] = np.random.normal(0, 0.0167/100) # vz
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