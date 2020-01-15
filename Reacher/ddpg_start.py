from mlagents.envs.environment import UnityEnvironment
from ddpg_agent_unity import DDPGAgent
import mlagents
import sys
import random

# Specify environment location
env_name = '../../envs/Reacher/Reacher'

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

    # Get state and action space, as well as multiagent info from environment
    env.reset(train_mode=True)
    brain_name = list(env.brains.keys())[0]

    state_space = env.brains[brain_name].vector_observation_space_size
    action_space = env.brains[brain_name].vector_action_space_size

    n_agents = env._n_agents[brain_name]

    multiagent = True if n_agents > 1 else False

    return env, state_space, action_space, n_agents, multiagent, brain_name

# Initialize Unity Environment
env, state_space, action_space, n_agents, multiagent, brain_name = init_unity_env(env_name, show_visuals=True)
    
# Create an agent
agent = DDPGAgent(alpha=1e-6, beta=1e-5, input_dims=state_space, tau=1e-3, env=env, brain_name=brain_name,
                batch_size=128, l1_size=64, l2_size=64, l3_size=64, n_actions=action_space[0], mem_capacity=1e5, 
                multiagent=multiagent, n_agents=n_agents, game_name='reacher_v2')

# Train num_eps amount of times and save onnx model
agent.train(num_eps=75000)
agent.save_onnx_model()