import torch
import torch.optim as optim

def train_actor_critic(env, actor, critic, num_episodes=1000, gamma=0.99, lr=1e-3):
    """
    Training loop for Actor-Critic method.
    
    Parameters:
    - env: The PigGame environment.
    - actor: The Actor network.
    - critic: The Critic network.
    - num_episodes: Number of episodes to train.
    - gamma: Discount factor for future rewards.
    - lr: Learning rate for optimizers.
    """
    # Optimizers for both networks
    actor_optimizer = optim.Adam(actor.parameters(), lr=lr)
    critic_optimizer = optim.Adam(critic.parameters(), lr=lr)
    
    for episode in range(num_episodes):
        state = env.reset()  # Reset the environment at the start of each episode
        state = torch.tensor(state, dtype=torch.float32)  # Convert to tensor
        
        done = False
        total_reward = 0  # Track total reward for the episode
        
        while not done:
            # Forward pass through Actor to get action probabilities
            action_probs = actor(state)
            # Sample an action based on the probabilities
            action = torch.multinomial(action_probs, 1).item()
            
            # Take the action and get the next state, reward, and done flag
            next_state, reward, done = env.step(action)
            next_state = torch.tensor(next_state, dtype=torch.float32)  # Convert to tensor
            reward = torch.tensor([reward], dtype=torch.float32)  # Convert reward to tensor
            
            # Forward pass through Critic to get state-value estimates
            state_value = critic(state)
            next_state_value = critic(next_state)
            
            # Calculate the advantage (reward + gamma * next state value - current state value)
            advantage = reward + gamma * next_state_value * (1 - int(done)) - state_value
            
            # Calculate the Critic loss (mean squared error)
            critic_loss = advantage.pow(2)
            
            # Calculate the Actor loss (using the advantage as weight for the log-probability)
            log_prob = torch.log(action_probs[action])
            actor_loss = -log_prob * advantage.detach()
            
            # Perform backpropagation and update the networks
            actor_optimizer.zero_grad()
            actor_loss.backward()
            actor_optimizer.step()
            
            critic_optimizer.zero_grad()
            critic_loss.backward()
            critic_optimizer.step()
            
            # Move to the next state
            state = next_state
            total_reward += reward.item()
        
        # Print the total reward every 100 episodes
        if episode % 100 == 0:
            print(f"Episode {episode}, Total Reward: {total_reward}")

