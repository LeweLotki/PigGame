# main.py

from env.pig_game import PigGame  # Import PigGame environment
from model.actor_critic import Actor, Critic  # Import Actor and Critic networks
from loop.training_loop import train_actor_critic  # Import training loop

def main():
    # Create the PigGame environment
    env = PigGame()
    
    # Define the size of the observation space
    obs_size = 7  # 2 dice, 2 current stacks, 2 permanent stacks, and current player
    
    # Instantiate the Actor and Critic models
    actor = Actor(input_size=obs_size)
    critic = Critic(input_size=obs_size)
    
    # Train the models using the training loop
    train_actor_critic(env, actor, critic, num_episodes=1000)

if __name__ == "__main__":
    main()

