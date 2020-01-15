from gym_unity.envs import UnityEnv
from mlagents.envs.environment import UnityEnvironment
from ddpg_agent_unity import DDPGAgent
import mlagents
import sys

# Specify environment location
env_name = '../../envs/3DBall/3DBall'

# Initializes a Unity Environment and returns variables such as state and action space
def init_unity_env(env_path, show_visuals=True):
    # Function tries different worker_id's in case one is still in use.
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

    env.reset(train_mode=True)
    brain_name = list(env.brains.keys())[0]

    state_space = env.brains[brain_name].vector_observation_space_size
    action_space = env.brains[brain_name].vector_action_space_size

    n_agents = env._n_agents[brain_name]

    multiagent = True if n_agents > 1 else False

    return env, state_space, action_space, n_agents, multiagent, brain_name

# Initialize Unity Environment
env, state_space, action_space, n_agents, multiagent, brain_name = init_unity_env(env_name, True)

# Create an agent
agent = DDPGAgent(alpha=1e-4, beta=1e-3, tau=1e-3, gamma=.99, state_space=state_space, 
                l1_size=64, l2_size=64, l3_size=32, action_space=action_space[0], env=env, 
                brain_name=brain_name, version='3dball', mem_capacity=1e6, 
                batch_size=128, multiagent=multiagent, n_agents=n_agents)

# Train num_eps amount of times
agent.train(num_eps=10000)