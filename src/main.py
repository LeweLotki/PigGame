import torch.optim as optim
from env.pig_game import PigGame  # Import the PigGame environment
from env.dummy_player import DummyPlayer  # Import the Dummy player
from model.actor_critic import Actor, Critic  # Import Actor and Critic models
from loop.training_loop import training_loop  # Import the combined training loop

# Initialize the environment
env = PigGame()

# Initialize the actor and critic networks
input_dim = 7
actor = Actor(input_dim)  # Assuming input_dim is 7
critic = Critic(input_dim)  # Assuming input_dim is 7

# Initialize the dummy player
dummy = DummyPlayer()

# Optimizers for actor and critic
actor_optimizer = optim.Adam(actor.parameters(), lr=1e-3)
critic_optimizer = optim.Adam(critic.parameters(), lr=1e-3)

# Set hyperparameters
num_episodes = 1000
gamma = 0.99

# Run the training loop with debug prints
training_loop(env, actor, critic, dummy, actor_optimizer, critic_optimizer, gamma=gamma, num_episodes=num_episodes, debug=True)

