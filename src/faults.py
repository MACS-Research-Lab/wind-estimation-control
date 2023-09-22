from systems.long_multirotor_sliding import LongTrajEnv
from systems.multirotor_sliding import fault_mult
import numpy as np

class FaultInjector():
    def __init__(self, env: LongTrajEnv):
        self.env = env

    def inject_full_loss(self, motor_index):
        self.env.base_env.vehicle.params.propellers[motor_index].k_thrust *= 0 #TODO: make sure these are not reset
        self.env.base_env.vehicle.params.propellers[motor_index].k_drag *= 0
        self.env.base_env.vehicle.propellers[motor_index].params.k_thrust *= 0
        self.env.base_env.vehicle.propellers[motor_index].params.k_drag *= 0
        return self.env

    def inject_partial_loss(self, motor_index, percent_loss):
        fault_mult[motor_index] = percent_loss
        return self.env

    def inject_saturation_fault(self, motor_index, rpm):
        pass