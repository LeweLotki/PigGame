import torch
import torch.nn as nn
import torch.nn.functional as F

class Actor(nn.Module):
    def __init__(self, input_size=7, output_size=2):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)       
        self.fc3 = nn.Linear(64, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))             
        x = F.relu(self.fc2(x))            
        action_probs = F.softmax(self.fc3(x), dim=-1)  
        return action_probs


class Critic(nn.Module):
    def __init__(self, input_size=7):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)  
        self.fc2 = nn.Linear(128, 64)         
        self.fc3 = nn.Linear(64, 1)          

    def forward(self, x):
        x = F.relu(self.fc1(x))             
        x = F.relu(self.fc2(x))            
        value = self.fc3(x)              
        return value

