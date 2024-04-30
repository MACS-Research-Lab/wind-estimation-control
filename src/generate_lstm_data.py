import numpy as np
import pandas as pd
from tqdm import tqdm

from environments import OctorotorEnvSelector
from multirotor.helpers import DataLog
from trajectories import square_100, circle_100
from parameters import pid_params

def run_trajectory(env_selector, wind_ranges, speed, trajectory):
    winds = []
    
    pid_parameters = pid_params()
    params = {'steps_u':1, 'scaling_factor':0, 'window_distance':1000, 'pid_parameters': pid_parameters}
    env = env_selector.get_env("lstm", params, wind_ranges, trajectory, start_alt=30, has_turbulence=True)
    env.wp_options = [trajectory]
    env.base_env.fault_type = None
    
    env.base_env.max_velocity = speed
    
    done = False
    state = np.array(env.reset(), dtype=np.float32)
    log = DataLog(env.base_env.vehicle)
    while not done:
        action = [0,0,0]
        state, reward, done, info = env.step(action)
        state = np.array(state, dtype=np.float32)
        log.log()
        winds.append([env.base_env.wind_x, env.base_env.wind_y, env.base_env.wind_z])

    log.done_logging()
    return log, info, winds, env.base_env.wind_forces

# Generates a wind range used by environment of a certain magnitude
def sample_wind_range(magnitude):
    random_angle_degrees = np.random.uniform(0, 360)

    # Step 2: Convert the angle to radians
    random_angle_radians = np.deg2rad(random_angle_degrees)
    
    # Step 3: Calculate the components of the unit vector
    x_component = np.cos(random_angle_radians)
    y_component = np.sin(random_angle_radians)
    
    # Step 4: Create the unit vector
    unit_vector = np.array([x_component, y_component]) 
    wind_vector = unit_vector * magnitude

    wind_range = [(wind_vector[0], wind_vector[0]), (wind_vector[1], wind_vector[1]), (0,0)]
    return wind_range, wind_vector

def save_data(filename, data, columns):
    df = pd.DataFrame(data, columns=columns)
    df.to_csv(f'./data/{filename}')

if __name__=="__main__":

    sim_params = {
        "trajectories": [square_100, circle_100],
        "winds": [0,1,2,3,4,5,6,7,8,9,10,11,12],
        "speeds": [3, 5, 7, 9, 11, 13, 15]
    }

    saved_columns = ["X", "Y", "Z", "VX", "VY", "VZ", "Roll", "Pitch", "Yaw", "Rate Roll", "Rate Pitch", "Rate Yaw",
                     "Wind X", "Wind Y", "Wind Z", "Disturbance X", "Disturbance Y", "Disturbance Z", "Flight Num"]
    
    data = []
    env_selector = OctorotorEnvSelector()
    flight_no = 0

    for traj in sim_params["trajectories"]:
        for wind in tqdm(sim_params["winds"]):
            for vehicle_speed in sim_params['speeds']:
                for i in range(5):
                    wind_range, vec = sample_wind_range(wind)
                    log, info, winds, disturbances = run_trajectory(env_selector, wind_range, vehicle_speed, traj)
                    winds = np.array(winds)
                    disturbances = np.array(disturbances)


                    flight_nums = [flight_no]*len(log.x)
                    result = np.vstack([log.x, log.y, log.z, log.velocity[:,0], log.velocity[:,1], log.velocity[:,2],
                                        log.roll, log.pitch, log.yaw, log.angular_rate[:,0], log.angular_rate[:,1], log.angular_rate[:,2],
                                        winds[:,0], winds[:,1], winds[:,2], disturbances[:,0], disturbances[:,1], disturbances[:,2], flight_nums]).T
                    flight_no += 1

                    data.extend(result.tolist())

                    save_data('lstm_disturbance.csv', data, saved_columns)