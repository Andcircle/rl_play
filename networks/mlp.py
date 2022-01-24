import torch
import torch.nn as nn
import torch.nn.functional as F 

class MLP(nn.Module):
    def __init__(self, input_size, layers, output_size, lr=1e-4, activation=''):
        super(MLP, self).__init__()
        
        self.layers = list(layers)
        self.layers.insert(0, input_size)
        
        ops=[]
        
        for i, o in zip(self.layers, self.layers[1:]):
            ops.append(nn.Linear(i,o))
            ops.append(nn.ReLU())
        ops.append(nn.Linear(self.layers[-1], output_size))
        
        if activation == 'sigmoid':
            ops.append(nn.Sigmoid())
        elif activation == 'tanh':
            ops.append(nn.Tanh())
            
        self.model = nn.Sequential(*ops)
        
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)
        
    def forward(self, *arg):
        return self.model(torch.cat(arg, 1))
    
    def save_checkpoint(self, file_path):
        print('... saving checkpoint ... {}'.format(file_path))
        torch.save(self.state_dict(), file_path)

    def load_checkpoint(self, file_path):
        print('... loading checkpoint ... {}'.format(file_path))
        self.load_state_dict(torch.load(file_path))

class Critic(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Critic, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, output_size)

    def forward(self, state, action):
        """
        Params state and actions are torch tensors
        """
        x = torch.cat([state, action], 1)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)

        return x
    
    def save_checkpoint(self, file_path):
        print('... saving checkpoint ... {}'.format(file_path))
        torch.save(self.state_dict(), file_path)

    def load_checkpoint(self, file_path):
        print('... loading checkpoint ... {}'.format(file_path))
        self.load_state_dict(torch.load(file_path))
        

class Actor(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, learning_rate = 3e-4):
        super(Actor, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, output_size)
        
    def forward(self, state):
        """
        Param state is a torch tensor
        """
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        x = torch.tanh(self.linear3(x))

        return x
    
    def save_checkpoint(self, file_path):
        print('... saving checkpoint ... {}'.format(file_path))
        torch.save(self.state_dict(), file_path)

    def load_checkpoint(self, file_path):
        print('... loading checkpoint ... {}'.format(file_path))
        self.load_state_dict(torch.load(file_path))