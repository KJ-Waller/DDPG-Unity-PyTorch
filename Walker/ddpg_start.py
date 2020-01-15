from mlagents.envs.environment import UnityEnvironment
from ddpg_agent_unity import DDPGAgent
import mlagents
import sys
import random

# Specify environment location
env_name = '../../envs/Walker/Walker'

# Initializes a Unity Environment without openai gym interface
def init_unity_env(env_path, show_visuals=True):

    # Find a worker_id < 64 that's not in use
    worker_id = 0
    done = False

    while not done:
        if worker_id > 64:
            sys.exit()
        try:
            env = UnityEnvironment(env_path, worker_id=worker_id, no_graphics=not show_visuals)
            done = True
        except mlagents.envs.exception.UnityWorkerInUseException:
            worker_id += 1

    # Get state and action space, as well as multiagent and multibrain info from environment
    env.reset(train_mode=not show_visuals)
    # brain_name = list(env.brains.keys())[0]
    brain_names = list(env.brains.keys())

    if len(brain_names) > 1:
        multibrain = True
        n_agents = env._n_agents[brain_names[0]] + env._n_agents[brain_names[1]]
    else:
        multibrain = False
        n_agents = env._n_agents[brain_names[0]]

    # WalkerVis is a version of the Walker environment with one brain 'WalkerVis'
    # having visual observations, whereas 'Walker' brain does not.
    # The visual observations are used for recording episodes
    state_space = env.brains[brain_names[0]].vector_observation_space_size
    action_space = env.brains[brain_names[0]].vector_action_space_size

    multiagent = True if n_agents > 1 else False

    return env, state_space, action_space, n_agents, multiagent, brain_names, multibrain

# Train or test 
Train = True

# Initialize Unity Environment
env, state_space, action_space, n_agents, multiagent, brain_name, multibrain = init_unity_env(env_name, show_visuals=not Train)
    
# Create an agent
agent = DDPGAgent(alpha=1e-7, beta=1e-6, tau=1e-3, gamma=.99, state_space=state_space, 
                l1_size=512, l2_size=512, l3_size=256, l4_size=256, 
                action_space=action_space[0], env=env, brain_name=brain_name, 
                multibrain=multibrain, version='walker', mem_capacity=1e6, 
                batch_size=128, multiagent=multiagent, n_agents=n_agents, eval=not Train)

# Train num_eps amount of times and save onnx model
agent.train(num_eps=1000)
# agent.save_onnx_model()