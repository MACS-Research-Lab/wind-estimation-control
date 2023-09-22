# Various useful functions for this project
from scripts.opt_multirotorenv import apply_params, get_study, get_established_controller
from rl import learn_rl, transform_rl_policy, evaluate_rl, PPO, load_agent

# Where the saved agent is stored during HPO
def get_agent(name, agent_num=None, filepath='MultirotorTrajEnv'):
    log_root_path = './tensorboard/' + filepath + '/optstudy/%s/'
    study = get_study(name)
    
    if agent_num is not None:
        best_trial = agent_num
    else:
        best_trial = study.best_trial.number        
        
    best_agent = load_agent((log_root_path + '%03d/run_1/agent') % (name, best_trial)) 
    best_params = study.best_params
    return study, best_agent, best_params

