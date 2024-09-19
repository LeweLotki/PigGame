import torch.optim as optim
from env.pig_game import PigGame
from env.dummy_player import DummyPlayer  
from model.actor_critic import Actor, Critic
from loop.training_loop import training_loop

env = PigGame()

actor = Actor() 
critic = Critic()

dummy = DummyPlayer()

actor_optimizer = optim.Adam(actor.parameters(), lr=1e-3)
critic_optimizer = optim.Adam(critic.parameters(), lr=1e-3)

num_episodes = 1e4
gamma = 0.99

training_loop(
    env=env, 
    actor=actor, 
    critic=critic, 
    dummy=dummy, 
    actor_optimizer=actor_optimizer, 
    critic_optimizer=critic_optimizer, 
    gamma=gamma, 
    num_episodes=int(num_episodes), 
    debug=False
)

