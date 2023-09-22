import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

from utilities import get_agent
from environments import OctorotorEnvSelector
from systems.pid_agent import PIDAgent
from multirotor.helpers import DataLog
from systems.long_blending import softmax

if __name__=="__main__":

    def get_tte(initial_pos: tuple, waypoints: np.ndarray, x: np.ndarray, y:np.ndarray, z:np.ndarray) -> np.ndarray:
            """
            Calculates the trajectory tracking error. 
            The distance between the current point and the vector between previous and next wp. Uses ||v1 x v2|| / ||v1||.

            Parameters
            ----------
            initial_pos : tuple  
                the initial position of the UAV.
            waypoints : np.ndarray 
                the reference positions at each point in time.
            x : np.ndarray 
                the x positions of the UAV.
            y : np.ndarray 
                the y positions of the UAV.
            z : np.ndarray
                the z positions of the UAV.

            Returns
            -------
            np.ndarray 
                the trajectory tracking error at each point in time.
            """
            ttes = []
            prev = initial_pos
            for i, waypoint in enumerate(waypoints):
                if i > 0 and not np.array_equal(waypoints[i-1], waypoints[i]):
                    prev = waypoints[i-1]

                v1 = waypoint - prev
                v2 = np.array([x[i],y[i],z[i]]) - prev
                tte = np.linalg.norm(np.cross(v1, v2)) / (np.linalg.norm(v1) + 1e-6)
                ttes.append(tte)
                    
            return np.array(ttes)

    def toc(tte: np.ndarray):
        corridor = 5
        return len(tte[tte > corridor]) / 2

    def completed_mission(waypoints: np.ndarray, x: np.ndarray, y: np.ndarray, z: np.ndarray, radius: float = 0.65):
            for waypoint in waypoints:
                reached_waypoint = False

                for position in zip(x,y,z):
                    dist = np.linalg.norm(waypoint - position)

                    if dist <= radius:
                        reached_waypoint = True
                        break

                if not reached_waypoint:
                    return False
                
            return True
        
    env_selector = OctorotorEnvSelector()
    pid_sl_params = {'steps_u':50, 'scaling_factor':0, 'window_distance':10}
    pid_sl_agent = PIDAgent()

    pid_params = {'steps_u':50, 'scaling_factor':0, 'window_distance':1000}
    pid_agent = PIDAgent()

    study, blending_agent, blending_params = get_agent('blending@softmax@scaled', filepath='BlendingEnv')
    blending_params['steps_u'] = 50
    blending_params['bounding_rect_length']=1000
    blending_params['window_distance']=10
    blending_params['scaling_factor']=5

    study, full_agent, full_params = get_agent('allwind')
    full_params['steps_u'] = 50
    full_params['bounding_rect_length']=1000
    full_params['window_distance']=10

    all_agents = [pid_agent, pid_sl_agent, blending_agent, full_agent]
    all_params = [pid_params, pid_sl_params, blending_params, full_params]
    types = ["sliding", "sliding", "blending", "sliding"]
    names = ['PID', 'PID SL', 'Blending', 'Full Agent']

    nasa_wp = np.asarray([ # defines a real trajectory shown in a NASA paper
        [164.0146725649829, -0.019177722744643688, 0],
        [165.6418055187678, 111.5351051245816, 0],
        [127.3337449710234, 165.73576059611514, 0],
        [-187.28170707810204, 170.33217775914818, 10],
        [-192.03130502498243, 106.30660058604553, 10],
        [115.89920266153058, 100.8644210617058, 0],
        [114.81859536317643, 26.80923518165946, 0],
        [-21.459931490011513, 32.60508110653609, 0]
    ])

    traj_len = 500 # trajectory seems to be around 500 seconds to complete
    direction_changes = [1,2,3]
    num_repeat = 25

    def sample_wind():
        wind_vec = np.random.uniform(0,10,2)
        mag = np.linalg.norm(wind_vec)
        
        if mag > 10:
            wind_vec = wind_vec * (10/mag)

        return wind_vec

    # Evaluates all saved agents with their params on a wind range
    def wind_injection(agents, params, types, names, traj_len, direction_changes, num_repeat):
        results = pd.DataFrame(columns=['Agent', 'Changes', 'Total TTE', 'Mean TTE', 'Completed Mission', 'Reward', 'Time Outside Corridor'])
        for num_changes in tqdm(direction_changes):
            for agent, param, env_type, name in tqdm(zip(agents, params, types, names), total=len(agents)):
                for i in range(num_repeat):

                    changes_triggered = 0
                    split_time = traj_len // num_changes
                    random_time = int(np.random.uniform(0,split_time))
                    change_time = changes_triggered * split_time + random_time
                    
                    env = env_selector.get_env(env_type, param, [(0,0),(0,0),(0,0)], nasa_wp)
                    done = False
                    state = env.reset()
                    state = np.array(state, dtype=np.float32)
                    log = DataLog(env.base_env.vehicle, env.base_env.ctrl,
                                    other_vars=('reward',))
                    index = 0
                    while not done:
                        if index == change_time:
                            changes_triggered += 1
                            change_time = changes_triggered * split_time + random_time
                            wind_vec = sample_wind()
                            env.base_env.wind_x = wind_vec[0]
                            env.base_env.wind_y = wind_vec[1]
                            
                        action = agent.predict(state, deterministic=True)[0] 
                        state, reward, done, info = env.step(action)
                        state = np.array(state, dtype=np.float32)
                        log.log(reward=reward)
                        index += 1
                    
                    log.done_logging()
                    traj_err = get_tte(np.array([0,0,0]), log.target.position, log.x, log.y, log.z)
                    new_result = {
                        'Agent': name,
                        'Changes': num_changes,
                        'Mean TTE': np.mean(traj_err),
                        'Total TTE': np.sum(traj_err),
                        'Completed Mission': completed_mission(nasa_wp, log.x, log.y, log.z, radius=5),
                        'Reward': np.sum(log.reward),
                        'Time Outside Corridor': toc(traj_err)
                    }
                    results = pd.concat([results, pd.DataFrame([new_result])], ignore_index=True)

        return results

    results = wind_injection(all_agents, all_params, types, names, traj_len, direction_changes, num_repeat)
    results.to_csv('./data/wind_changing.csv')