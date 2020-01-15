import torch
import torch.nn as nn
import numpy as np
from rl_plots import RLPlots
from noise import OUActionNoise
from replay_mem import ReplayBuffer
from ddpg_models import ActorNet, CriticNet

# Agent class that contains actor and critic models that can train on and play games
class DDPGAgent(object):
    def __init__(self, alpha, beta, input_dims, tau, env, brain_name, gamma=.99,
                 n_actions=2, mem_capacity=1e6, layer1_size=400,
                 layer2_size=300, batch_size=64, multiagent=False,
                 n_agents=None, game_name='Rollerball'):

        # Initialize memory
        self.batch_size = batch_size
        self.memory = ReplayBuffer(mem_capacity)
        
        # Initialize noise
        self.noise = OUActionNoise(np.zeros(n_actions))

        # Setup device used for torch computations
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        # Create actor critic and target networks
        self.actor = ActorNet(alpha, input_dims, layer1_size, layer2_size, n_actions, name='actor_' + game_name + '_ddpg_model').to(self.device)
        self.target_actor = ActorNet(alpha, input_dims, layer1_size, layer2_size, n_actions).to(self.device)

        self.critic = CriticNet(beta, input_dims, layer1_size, layer2_size, n_actions, name='critic_' + game_name + '_ddpg_model').to(self.device)
        self.target_critic = CriticNet(beta, input_dims, layer1_size, layer2_size, n_actions).to(self.device)
        
        # Initialize target nets to be identical to actor and critic networks
        self.init_networks()

        # Target networks set to eval, since they are not 
        # trained but simply updated with the target_network_update function
        self.target_actor.eval()
        self.target_critic.eval()

        # Set global parameters
        self.gamma = gamma
        self.env = env
        self.tau = tau
        self.state_space = input_dims
        self.action_space = n_actions
        self.multiagent = multiagent
        self.brain_name = brain_name
        if self.multiagent:
            self.n_agents = n_agents

        # Plotter object for showing live training graphs and saving them
        self.plotter = RLPlots('ddpg_training')

    def init_networks(self):
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())

    def save_models(self):
        self.actor.save_model()
        self.critic.save_model()

    def save_onnx_model(self):
        dummy_input = torch.autograd.Variable(torch.randn(1, self.state_space)).to(self.device)
        torch.onnx.export(self.actor, dummy_input, './models/ddpg_onnx_model.onnx', verbose=True)

    # Passes the state through the actor network to get actions
    def choose_action(self, state):
        # Assure actor is set to eval mode as we do not want to train when taking actions
        self.actor.eval()

        # Convert state to tensor and send to device
        state = torch.tensor(state).float().to(self.device)

        # Pass state through actor network to get actions
        actions = self.actor(state)

        # In case of multiagent environment, stack the noise vector
        # to have the same shape as the actions matrix (n_agents x num_actions)
        if self.multiagent:
            noise = np.tile(self.noise(), (self.n_agents, 1))
            noise = torch.tensor(noise).float().to(self.device)
        else:
            noise = torch.tensor(self.noise()).float().to(self.device)

        # Add noise to actions to get final mu action values
        actions = actions + noise

        # Send to cpu, detach from computation graph and convert to list
        # for the gym environment
        return actions.cpu().detach().tolist()

    # Stores transitions (single or multiple depending on multiagent environment)
    def store_transitions(self, state, action, reward, state_, done):
        if self.multiagent:
            for i in range(self.n_agents):
                self.memory.add_transition(state[i], action[i], reward[i], state_[i], int(done[i]))
        else:
            self.memory.add_transition(state, action, reward, state_, int(done))

    # This function samples a batch of transitions from memory, and 
    # puts them onto the device defined by self.device
    def sample_transitions(self):
        batch = self.memory.sample_batch(self.batch_size)

        # 'reward' and 'done' are unsequeezed to get the right dimensions, (batch_size, 1) instead of (batch_size)
        state = torch.tensor(batch.state).float().to(self.device)
        action = torch.tensor(batch.action).float().to(self.device)
        reward = torch.tensor(batch.reward).unsqueeze(dim=1).float().to(self.device)
        state_ = torch.tensor(batch.state_).float().to(self.device)
        done = torch.tensor(batch.done).unsqueeze(dim=1).float().to(self.device)

        return state, action, reward, state_, done

    def learn(self):
        # Only start learning when there's enough transitions stored for a full batch
        if self.memory.pointer < self.batch_size:
            return

        # Sample random batch of transitions
        state, action, reward, state_, done = self.sample_transitions()

        # We first handle the critic update, then the actor update
        # To evaluate the critic net, we set actor to eval, critic to train mode
        self.actor.eval()
        self.critic.train()

        # Order of operation is: 
        # 1) zero_grad > 2) forward pass > 3) loss > 4) backward pass > 5) optimizer step
        
        # 1) Zero grad on Critic
        self.critic.optim.zero_grad()

        # 2) Forward pass 
        # Calculate critic predicted and target values
        critic_pred = self.critic(state, action)
        critic_target = reward + self.gamma * self.target_critic(state_, self.target_actor(state_)) * (1 - done)
        
        # 3) Calculate loss
        # Calulcate critic loss, then perform backprop
        critic_loss = self.critic.loss(critic_pred, critic_target)
        
        # 4) Backward pass
        critic_loss.backward()
        
        # 5) Optimizer step
        self.critic.optim.step()

        # Now switch train and eval mode again on actor and critic to
        # do the actor update
        self.actor.train()
        self.critic.eval()

        # 1) Zero grad on actor
        self.actor.optim.zero_grad()
        
        # 2) Forward pass
        # Calculate actor loss and perform backprop
        mu = self.actor(state)
        
        # 3) Calculate loss
        actor_loss = torch.mean(-self.critic(state, mu))
        
        # 4) Backward pass
        actor_loss.backward()
        
        # 5) Optimizer step
        self.actor.optim.step()

        # Set actor back to eval mode
        self.actor.eval()

        # Update the target networks
        self.update_target_networks()

    # Steps through the environment with the given action and returns the next state, reward, done flag 
    # whether or not the max_steps has been reached
    def env_step(self, actions):
        env_info = self.env.step(actions)[self.brain_name]
        state_ = env_info.vector_observations
        rewards = env_info.rewards
        done = env_info.local_done
        max_reached = env_info.max_reached

        return state_, rewards, done, max_reached

    # Trains either a multi or single agent environment
    # eval is whether or not to simply play the game or to actually train the agent
    def train(self, num_eps, eval=False):
        if eval:
            self.actor.eval()
            self.critic.eval()

        if self.multiagent:
            self.train_multi(num_eps, eval)
        else:
            self.train_single(num_eps, eval)

    # Training loop for environment with single agent
    def train_single(self, num_eps, eval):
        
        # Keep track of scores
        scores = []
        avg_ph_scores = []

        # Save interval
        save_interval = 100

        # Plot interval
        plot_interval = 25

        # Play number of episodes
        for ep_num in range(num_eps):
            done = False
            state = self.env.reset()[self.brain_name].vector_observations

            score = 0

            # Keep playing until done
            while not done:
                # Pick action using actor network
                actions = self.choose_action(state)
                # Take action and observe next state and reward
                state_, rewards, done, _ = self.env_step(actions)
                # Store transition into memory
                self.store_transitions(state, actions, rewards, state_, done)
                # Sample batch of transitions and train networks (if eval mode is off)
                if not eval:
                    self.learn()

                if self.multiagent:
                    score += np.mean(rewards)
                else:
                    score += rewards

                state = state_

            print(f'Episode: {ep_num}\n\tScore: {score}\n\tAvg past 100 score: {np.mean(scores[-100:])}')

            scores.append(score)
            avg_ph_scores.append(np.mean(scores[-100:]))

            if ep_num % save_interval == 0:
                self.save_models()

            if ep_num % plot_interval == 0:
                self.plot_rewards(avg_ph_scores)
        
        self.plot_rewards(avg_ph_scores, save=True)

        self.env.close()

    # Training loop for environment with multiple agents
    def train_multi(self, num_eps, eval):
        
        # Keep track of scores
        scores = []
        avg_ph_scores = []

        # Save interval
        save_interval = 100

        # Plot interval
        plot_interval = 20

        # Play number of episodes
        for ep_num in range(num_eps):
            done = [False for i in range(self.n_agents)]

            state = self.env.reset()[self.brain_name].vector_observations

            score = 0

            # Keep playing until one of the agents is done
            while True not in done:
                # Pick action using actor network
                actions = self.choose_action(state)
                # Take action and observe next state and reward
                state_, rewards, done, _ = self.env_step(actions)
                # Store transition into memory
                self.store_transitions(state, actions, rewards, state_, done)
                # Sample batch of transitions and train networks (if eval mode is off)
                if not eval:
                    self.learn()
                
                score += np.mean(rewards)

                state = state_

            print(f'Episode: {ep_num}\n\tScore: {score}\n\tAvg past 100 score: {np.mean(scores[-100:])}')

            scores.append(score)
            avg_ph_scores.append(np.mean(scores[-100:]))

            # Save models every save_interval steps
            if ep_num % save_interval == 0:
                self.save_models()

            # Plots average rewards every plot_interval steps
            if ep_num % plot_interval == 0:
                self.plot_rewards(avg_ph_scores)
        
        # Save the final plot
        self.plot_rewards(avg_ph_scores, save=True)

        # Close environment
        self.env.close()

    # This function updates the target networks according to the DDPG algorithm
    # theta^q' = tau * theta^q + (1-tau) * theta^q'
    # theta^mu' = tau * theta^mu + (1-tau) * theta^mu'
    # Where q and q' are the critic and target critic networks
    # and mu and mu' are the actor and target actor networks respectively
    def update_target_networks(self):
        # Load all four state dictionaries for the networks
        actor_params = self.actor.state_dict()
        target_actor_params = self.target_actor.state_dict()

        critic_params = self.critic.state_dict()
        target_critic_params = self.target_critic.state_dict()

        # Load the a new state dict using a dictionary comprehension
        self.target_actor.load_state_dict({ key:
            (self.tau * params) + (1 - self.tau) * target_actor_params[key].clone()
            for key, params in actor_params.items()
        })

        self.target_critic.load_state_dict({ key:
            (self.tau * params) + (1 - self.tau) * target_critic_params[key].clone()
            for key, params in critic_params.items()
        })

    # For plotting learning progress
    def plot_rewards(self, scores, save=False):
        self.plotter.plot_rewards(scores, save)