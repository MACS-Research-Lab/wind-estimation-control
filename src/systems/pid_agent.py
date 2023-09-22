import numpy as np

# Agent to act as a PID controller only (no modification to reference velocity) in the Stablebaselines interface
class PIDAgent():
    def predict(self, state, deterministic=True):
        return np.array([0,0,0]), 0