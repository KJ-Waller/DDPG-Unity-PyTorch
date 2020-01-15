import os
import torch
import torch.nn as nn
import numpy as np

# Takes state and outputs actions mu
class ActorNet(nn.Module):

    # Params: Input dimensions (state space), output dimensions (actions)
    def __init__(self, lr, state_dim, l1_size, l2_size, l3_size, action_dim, name=None):
        super(ActorNet, self).__init__()

        # Setup network which takes state as input, and outputs the action means
        self.fc1 = nn.Linear(state_dim, l1_size)
        self.bn1 = nn.LayerNorm(l1_size)
        self.fc2 = nn.Linear(l1_size, l2_size)
        self.bn2 = nn.LayerNorm(l2_size)
        self.fc3 = nn.Linear(l2_size, l3_size)
        self.bn3 = nn.LayerNorm(l3_size)
        self.mu = nn.Linear(l3_size, action_dim)
        self.relu = nn.LeakyReLU()
        self.tanh = nn.Tanh()

        # Initialize network weights
        self.init_weights()

        # Setup the optimizer used for the actor net (loss is calculated by the critic)
        self.optim = torch.optim.Adam(self.parameters(), lr=lr)

        # Set directory for saving models
        if not os.path.isdir('./models/'):
            os.mkdir('./models/')

        # Set path for saving the model and check if pretrained model exists & load if so
        if name is not None:
            self.model_path = './models/' + name + '.pth'
            self.load_model()
        
    def load_model(self):
        try:
            self.load_state_dict(torch.load(self.model_path))
            print('Pretrained actor network found & loaded')
        except:
            print('No pretrained actor model found, creating new model')

    def save_model(self):
        torch.save(self.state_dict(), self.model_path)
        print(f'Actor model saved to: {self.model_path}')

    # Initializes the weights and biases to be of uniform distribution between -0.003 and 0.003
    def init_weights(self):
        f1 = 1 / np.sqrt(self.fc1.weight.data.size()[0])
        nn.init.uniform_(self.fc1.weight.data, -f1, f1)
        nn.init.uniform_(self.fc1.bias.data, -f1, f1)
        f2 = 1 / np.sqrt(self.fc2.weight.data.size()[0])
        nn.init.uniform_(self.fc2.weight.data, -f2, f2)
        nn.init.uniform_(self.fc2.bias.data, -f2, f2)
        f3 = 1 / np.sqrt(self.fc3.weight.data.size()[0])
        nn.init.uniform_(self.fc3.weight.data, -f3, f3)
        nn.init.uniform_(self.fc3.bias.data, -f3, f3)
        f4 = .003
        nn.init.uniform_(self.mu.weight.data, -f4, f4)
        nn.init.uniform_(self.mu.bias.data, -f4, f4)

    # Forwards the state through the actor network to get the actions/mu
    def forward(self, state):
        x = self.relu(self.bn1(self.fc1(state)))
        x = self.relu(self.bn2(self.fc2(x)))
        x = self.relu(self.bn3(self.fc3(x)))
        actions = self.tanh(self.mu(x))

        return actions

# Takes state and actions taken, and evaluates them/gives it a score
class CriticNet(nn.Module):
    def __init__(self, lr, state_dim, l1_size, l2_size, l3_size, action_dim, name=None):
        super(CriticNet, self).__init__()

        # Setup state layers, which transforms the state into a lower dimensional representation
        self.fc1 = nn.Linear(state_dim, l1_size)
        self.bn1 = nn.LayerNorm(l1_size)
        self.fc2 = nn.Linear(l1_size, l2_size)
        self.bn2 = nn.LayerNorm(l2_size)
        self.fc3 = nn.Linear(l2_size, l3_size)
        self.bn3 = nn.LayerNorm(l3_size)
        self.relu = nn.LeakyReLU()

        # Setup action layers, which transforms the actions into a lower dimensional representation
        self.action_value = nn.Linear(action_dim, l3_size)

        # Setup final layers, which combines state and action layers outputs,
        # and outputs the final critic value
        self.q = nn.Linear(l3_size, 1)

        # Initialize the weights to be within a small range of values
        self.init_weights()

        # Setup the loss and optimizer used for the critic network
        self.loss = nn.MSELoss()
        self.optim = torch.optim.Adam(self.parameters(), lr=lr)

        # Set directory for saving models
        if not os.path.isdir('./models/'):
            os.mkdir('./models/')

        # Set path for saving the model and check if pretrained model exists & load if so
        if name is not None:
            self.model_path = './models/' + name + '.pth'
            self.load_model()
        
    def load_model(self):
        try:
            self.load_state_dict(torch.load(self.model_path))
            print('Pretrained critic network found & loaded')
        except:
            print('No pretrained critic model found, creating new model')

    def save_model(self):
        torch.save(self.state_dict(), self.model_path)
        print(f'Critic model saved to: {self.model_path}')
    
    # Initializes the weights and biases to be of uniform distribution between -0.003 and 0.003
    def init_weights(self):
        f1 = 1 / np.sqrt(self.fc1.weight.data.size()[0])
        nn.init.uniform_(self.fc1.weight.data, -f1, f1)
        nn.init.uniform_(self.fc1.bias.data, -f1, f1)
        f2 = 1 / np.sqrt(self.fc2.weight.data.size()[0])
        nn.init.uniform_(self.fc2.weight.data, -f2, f2)
        nn.init.uniform_(self.fc2.bias.data, -f2, f2)
        f3 = 1 / np.sqrt(self.fc3.weight.data.size()[0])
        nn.init.uniform_(self.fc3.weight.data, -f3, f3)
        nn.init.uniform_(self.fc3.bias.data, -f3, f3)
        f4 = .003
        nn.init.uniform_(self.q.weight.data, -f4, f4)
        nn.init.uniform_(self.q.bias.data, -f4, f4)

    # Forwards the state and actions through the critic network to get q
    def forward(self, state, action):
        # Pass state through layers
        state_value = self.relu(self.bn1(self.fc1(state)))
        state_value = self.bn2(self.fc2(state_value))
        state_value = self.bn3(self.fc3(state_value))

        # Pass action through a separate layers to get action value
        action_value = self.relu(self.action_value(action))

        # Get state action value by adding state and action value and passing it
        # through the final q layer which is a single value evaluating the state_action pair
        state_action_value = self.relu(torch.add(state_value, action_value))
        state_action_value = self.q(state_action_value)

        return state_action_value