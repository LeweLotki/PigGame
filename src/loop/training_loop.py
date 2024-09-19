import torch

# Define min and max values based on the known range of each observation
min_vals = torch.tensor([0, 0, 0, 0, 0, 0, 0], dtype=torch.float32)
max_vals = torch.tensor([6, 6, 100, 100, 100, 100, 1], dtype=torch.float32)

def training_loop(env, actor, critic, dummy, actor_optimizer, critic_optimizer, gamma=0.99, num_episodes=1000, debug=True, entropy_beta=0.01):
    """
    A combined loop for playing the game, storing game data, and training the neural networks (Actor and Critic).
    
    Parameters:
    - env: The PigGame environment.
    - actor: The Actor network.
    - critic: The Critic network.
    - dummy: The Dummy player (part of the environment, no reward influence).
    - actor_optimizer: Optimizer for the Actor network.
    - critic_optimizer: Optimizer for the Critic network.
    - gamma: Discount factor for future rewards.
    - num_episodes: Number of episodes to run for training.
    - debug: Boolean flag to enable debug prints (default True).
    - entropy_beta: Coefficient for entropy regularization to encourage exploration.
    
    Returns:
    - rewards_per_episode: A list of total rewards collected per episode.
    - game_scores: A list of lists, where each element is a list of two lists (dummy scores, NN scores) for each game.
    - actor_losses: A list of actor losses for each episode.
    - critic_losses: A list of critic losses for each episode.
    """
    rewards_per_episode = []  # To store total rewards for each episode
    game_scores = []  # To store scores of each game (dummy's permanent stack, NN's permanent stack)
    actor_losses = []  # To store actor losses for each episode
    critic_losses = []  # To store critic losses for each episode
    
    for episode in range(num_episodes):
        state = env.reset()  # Reset the environment at the start of the game
        
        # Normalize the initial state
        state = normalize_observation(state, min_vals, max_vals)
        
        done = False
        total_reward = 0  # Track total reward for the episode (only NN's actions)
        episode_actor_loss = 0  # Accumulate actor loss for the episode
        episode_critic_loss = 0  # Accumulate critic loss for the episode

        while not done:
            current_player = state[-1].item()  # Check whose turn it is

            # Network-controlled player (NN's turn)
            while current_player == 0 and not done:
                # Forward pass through Actor to get action probabilities
                action_probs = actor(state)
                
                if debug:
                    print(f"Action probabilities from NN: {action_probs}")
                
                # Sample action based on probabilities
                action = torch.multinomial(action_probs, 1).item()  # Sample action
                if debug:
                    print(f"Network player's action: {action}")
                
                # Take action and get next state, reward, done
                next_state, reward, done = env.step(action)  # `done` should be set by the environment
                
                # Normalize the next state
                next_state = normalize_observation(next_state, min_vals, max_vals)

                # Convert reward to tensor if it's an int
                reward_tensor = torch.tensor([reward], dtype=torch.float32) if isinstance(reward, int) else reward

                # Critic update
                state_value = critic(state)
                next_state_value = critic(next_state)
                advantage = reward_tensor + gamma * next_state_value * (1 - int(done)) - state_value

                # Critic loss (MSE)
                critic_loss = advantage.pow(2).mean()
                episode_critic_loss += critic_loss.item()  # Accumulate critic loss

                # Actor loss (policy gradient)
                log_prob = torch.log(action_probs[action])
                actor_loss = -(log_prob * advantage.detach()).mean()
                episode_actor_loss += actor_loss.item()  # Accumulate actor loss

                # Entropy regularization to encourage exploration
                entropy = -(action_probs * torch.log(action_probs + 1e-10)).sum()
                actor_loss -= entropy_beta * entropy  # Add entropy to the loss to encourage exploration

                # Backpropagation for Actor-Critic
                actor_optimizer.zero_grad()
                actor_loss.backward()
                actor_optimizer.step()

                critic_optimizer.zero_grad()
                critic_loss.backward()
                critic_optimizer.step()

                # Accumulate the total reward for NN's actions only
                total_reward += reward_tensor.item()

                # If action is 0 (pass) or result is 6, switch to Dummy's turn
                if action == 0 or 6 in next_state[:2]:
                    current_player = 1  # Switch to Dummy's turn
                    state = next_state
                    break  # Exit the NN's loop
                else:
                    state = next_state  # NN continues to act

                game_scores.append([env.permanent_stack.tolist()])  # Permanent stack of both players

            # Dummy player's turn
            while current_player == 1 and not done:
                # Dummy always rolls first (action 1)
                action = 1
                if debug:
                    print(f"Dummy player's action: {action}")
                
                next_state, reward, done = env.step(action)  # `done` should be set by the environment
                
                # Normalize the next state for dummy as well
                next_state = normalize_observation(next_state, min_vals, max_vals)

                # If Dummy rolls a 6, switch back to NN's turn immediately
                if 6 in next_state[:2]:
                    current_player = 0  # Switch to NN's turn
                    state = next_state
                    continue  # Skip the pass and give NN the turn immediately

                # If no 6, Dummy passes (action 0)
                action = 0
                if debug:
                    print(f"Dummy player's action: {action}")
                
                next_state, reward, done = env.step(action)  # `done` should be set by the environment

                # Normalize the state again
                next_state = normalize_observation(next_state, min_vals, max_vals)

                # Switch back to NN's turn after passing
                current_player = 0
                state = next_state
                game_scores.append([env.permanent_stack.tolist()])  # Permanent stack of both players

            # Debugging information for every step
            if debug:
                print(f"Episode: {episode}, NN's Reward: {reward_tensor.item()}, Total Reward: {total_reward}")
                print(f"Observation: {next_state.numpy()}")
                print(f"Done: {done}")
                print("-" * 30)

        # End of episode: append the total reward and the scores (permanent stack for both players)
        rewards_per_episode.append(total_reward)

        # Append the losses
        actor_losses.append(episode_actor_loss)
        critic_losses.append(episode_critic_loss)

        # End of episode debugging information
        if episode % 100 == 0 or debug:
            print(f"Episode {episode} finished, Total Reward: {total_reward}")

    # Return the rewards for each episode, scores, and losses
    return rewards_per_episode, game_scores, actor_losses, critic_losses


def normalize_observation(state, min_vals, max_vals):
    """
    Normalizes the observation using Min-Max scaling to the range [0, 1].

    Parameters:
    - state: The current observation (numpy array or tensor).
    - min_vals: The minimum values for each feature in the observation.
    - max_vals: The maximum values for each feature in the observation.

    Returns:
    - Normalized state.
    """
    # Min-Max normalization (scaled to [0, 1])
    normalized_state = (state - min_vals) / (max_vals - min_vals)
    return normalized_state

