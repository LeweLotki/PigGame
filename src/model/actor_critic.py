import torch
import torch.nn as nn
import torch.nn.functional as F

# Actor Network: Outputs action probabilities (Pass or Roll)
class Actor(nn.Module):
    def __init__(self, input_size, output_size=2):
        """
        Initialize the Actor network.
        - input_size: Size of the observation space (number of features).
        - output_size: Number of actions (binary: pass or roll, hence 2).
        """
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)  # Fully connected layer 1
        self.fc2 = nn.Linear(128, 64)         # Fully connected layer 2
        self.fc3 = nn.Linear(64, output_size) # Output layer for action probabilities

    def forward(self, x):
        """
        Forward pass to calculate action probabilities.
        """
        x = F.relu(self.fc1(x))               # Apply ReLU activation after first layer
        x = F.relu(self.fc2(x))               # Apply ReLU activation after second layer
        action_probs = F.softmax(self.fc3(x), dim=-1)  # Softmax for action probabilities
        return action_probs


# Critic Network: Outputs state-value (V(s))
class Critic(nn.Module):
    def __init__(self, input_size):
        """
        Initialize the Critic network.
        - input_size: Size of the observation space (number of features).
        """
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)  # Fully connected layer 1
        self.fc2 = nn.Linear(128, 64)          # Fully connected layer 2
        self.fc3 = nn.Linear(64, 1)            # Output layer for state-value (V(s))

    def forward(self, x):
        """
        Forward pass to calculate state-value.
        """
        x = F.relu(self.fc1(x))               # Apply ReLU activation after first layer
        x = F.relu(self.fc2(x))               # Apply ReLU activation after second layer
        value = self.fc3(x)                   # Output state-value (no activation)
        return value

