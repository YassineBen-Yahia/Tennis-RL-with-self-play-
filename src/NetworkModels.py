import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)


class Critic(nn.Module):
    
    def __init__(self, state_size, action_size, seed=1, fcs1_units=400, fc2_units=300):
        """
        Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fcs1_units (int): Number of nodes in the first hidden layer
            fc2_units (int): Number of nodes in the second hidden layer
        """
        super(Critic, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fcs1 = nn.Linear(state_size, fcs1_units)
        self.fc2 = nn.Linear(fcs1_units+action_size, fc2_units)
        self.fc3 = nn.Linear(fc2_units, 1)
        self.reset_parameters()
    
    def reset_parameters(self):
        self.fcs1.weight.data.uniform_(*hidden_init(self.fcs1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

        self.fcs1.bias.data.fill_(0.1)
        self.fc2.bias.data.fill_(0.1)
        self.fc3.bias.data.fill_(0.1)

    def forward(self, state, action):
        
        xs = F.relu(self.fcs1(state))
        x = torch.cat((xs, action), dim=1)
        x = F.relu(self.fc2(x))
        return self.fc3(x)



class Actor(nn.Module):
    
    def __init__(self, input_size = 24, output_size = 2, hidden_sizes = [400, 300]):
          """
        input_size: the dimension of the state vector of a single agent
        output_size: the dimension of the action vector of a single agent
        hidden_sizes: the sizes of the input and output units of the hidden layer
        for example, hidden_sizes = [400, 300] means the hidden layer has input_size = 400, and output_size = 300
          """
          

          super(Actor, self).__init__()
          self.hidden_layers = nn.ModuleList([])
          self.hidden_layers.append(nn.Linear(input_size, hidden_sizes[0])) 

          for h1, h2 in zip(hidden_sizes[:-1], hidden_sizes[1:]): # from the sencond layer to the (last-1) layer
              self.hidden_layers.append(nn.Linear(h1, h2))
            
          self.output_layer = nn.Linear(hidden_sizes[-1], output_size) 

          self.reset_parameters()

       

    def reset_parameters(self):
        for layer in self.hidden_layers:
            layer.weight.data.uniform_(*hidden_init(layer))
            layer.bias.data.fill_(0.1)
        self.output_layer.weight.data.uniform_(-3e-3, 3e-3)
        self.output_layer.bias.data.fill_(0.1)
    
    def forward(self, state):

        for layer in self.hidden_layers:
            x = F.relu(layer(state))
        return F.tanh(self.output_layer(x))
