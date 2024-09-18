import torch

def training_loop(env, actor, critic, dummy, actor_optimizer, critic_optimizer, gamma=0.99, num_episodes=1000, debug=True):
    """
    A combined loop for playing the game, storing game data, and training the neural networks (Actor and Critic).
    
    Parameters:
    - env: The PigGame environment.
    - actor: The Actor network.
    - critic: The Critic network.
    - dummy: The Dummy player.
    - actor_optimizer: Optimizer for the Actor network.
    - critic_optimizer: Optimizer for the Critic network.
    - gamma: Discount factor for future rewards.
    - num_episodes: Number of episodes to run for training.
    - debug: Boolean flag to enable debug prints (default True).
    """
    for episode in range(num_episodes):
        state = env.reset()  # Reset the environment at the start of the game
        state = torch.tensor(state, dtype=torch.float32)  # Convert state to tensor
        done = False
        total_reward = 0  # Track total reward for the episode

        while not done:
            current_player = state[-1].item()  # Check whose turn it is

            # Network-controlled player (NN's turn)
            while current_player == 0:
                # Forward pass through Actor to get action probabilities
                action_probs = actor(state)
                action = torch.multinomial(action_probs, 1).item()  # Sample action
                if debug:
                    print(f"Network player's action: {action}")
                
                # Take action and get next state, reward, done
                next_state, reward, done = env.step(action)
                next_state = torch.tensor(next_state, dtype=torch.float32)  # Convert to tensor

                # Convert reward to tensor if it's an int
                if isinstance(reward, int):
                    reward = torch.tensor([reward], dtype=torch.float32)

                # Critic update
                state_value = critic(state)
                next_state_value = critic(next_state)
                advantage = reward + gamma * next_state_value * (1 - int(done)) - state_value

                # Critic loss (MSE)
                critic_loss = advantage.pow(2).mean()

                # Actor loss (policy gradient)
                log_prob = torch.log(action_probs[action])
                actor_loss = -(log_prob * advantage.detach()).mean()

                # Backpropagation for Actor-Critic
                actor_optimizer.zero_grad()
                actor_loss.backward()
                actor_optimizer.step()

                critic_optimizer.zero_grad()
                critic_loss.backward()
                critic_optimizer.step()

                # Accumulate the total reward
                total_reward += reward.item()

                # If action is 0 (pass) or result is 6, switch to Dummy's turn
                if action == 0 or 6 in next_state[:2]:
                    current_player = 1  # Switch to Dummy's turn
                    state = next_state
                    break  # Exit the NN's loop
                else:
                    state = next_state  # NN continues to act

            # Dummy player's turn
            if current_player == 1:
                # Dummy always rolls first (action 1)
                action = 1
                if debug:
                    print(f"Dummy player's action: {action}")
                
                next_state, reward, done = env.step(action)
                next_state = torch.tensor(next_state, dtype=torch.float32)  # Convert to tensor

                # Convert reward to tensor if it's an int
                if isinstance(reward, int):
                    reward = torch.tensor([reward], dtype=torch.float32)

                # Accumulate the total reward
                total_reward += reward.item()

                # If Dummy rolls a 6, switch back to NN's turn immediately
                if 6 in next_state[:2]:
                    current_player = 0  # Switch to NN's turn
                    state = next_state
                    continue  # Skip the pass and give NN the turn immediately

                # If no 6, Dummy passes (action 0)
                action = 0
                if debug:
                    print(f"Dummy player's action: {action}")
                
                next_state, reward, done = env.step(action)
                next_state = torch.tensor(next_state, dtype=torch.float32)  # Convert to tensor

                # Convert reward to tensor if it's an int
                if isinstance(reward, int):
                    reward = torch.tensor([reward], dtype=torch.float32)

                # Accumulate the total reward
                total_reward += reward.item()

                # Switch back to NN's turn after passing
                current_player = 0
                state = next_state

            # Debugging information for every step
            if debug:
                print(f"Episode: {episode}, Reward: {reward.item()}, Total Reward: {total_reward}")
                print(f"Observation: {next_state.numpy()}")
                print("-" * 30)

        # End of episode debugging information
        if episode % 100 == 0 or debug:
            print(f"Episode {episode} finished, Total Reward: {total_reward}")

