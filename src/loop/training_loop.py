import torch

min_vals = torch.tensor([0, 0, 0, 0, 0, 0, 0], dtype=torch.float32)
max_vals = torch.tensor([6, 6, 100, 100, 100, 100, 1], dtype=torch.float32)

def training_loop(env, actor, critic, dummy, actor_optimizer, critic_optimizer, gamma=0.99, num_episodes=1000, debug=True, entropy_beta=0.001):
    rewards_per_episode = []  
    game_scores = [] 
    actor_losses = [] 
    critic_losses = []

    actor_scheduler = torch.optim.lr_scheduler.StepLR(actor_optimizer, step_size=100, gamma=0.95)
    critic_scheduler = torch.optim.lr_scheduler.StepLR(critic_optimizer, step_size=100, gamma=0.95)

    for episode in range(num_episodes):
        state = env.reset()  
        
        state = normalize_observation(state, min_vals, max_vals)
        
        done = False
        total_reward = 0 
        episode_actor_loss = 0
        episode_critic_loss = 0

        while not done:
            current_player = state[-1].item()

            while current_player == 0 and not done:
                action_probs = actor(state)
                
                if debug:
                    print(f"Action probabilities from NN: {action_probs}")
                
                action = torch.multinomial(action_probs, 1).item()
                if debug:
                    print(f"Network player's action: {action}")
                
                next_state, reward, done = env.step(action) 
                
                next_state = normalize_observation(next_state, min_vals, max_vals)

                reward_tensor = torch.tensor([reward], dtype=torch.float32) if isinstance(reward, int) else reward

                state_value = critic(state)
                next_state_value = critic(next_state)
                advantage = reward_tensor + gamma * next_state_value * (1 - int(done)) - state_value

                critic_loss = advantage.pow(2).mean()
                episode_critic_loss += critic_loss.item()  

                log_prob = torch.log(action_probs[action])
                actor_loss = -(log_prob * advantage.detach()).mean()
                episode_actor_loss += actor_loss.item()  

                entropy = -(action_probs * torch.log(action_probs + 1e-10)).sum()
                actor_loss -= entropy_beta * entropy  

                actor_optimizer.zero_grad()
                actor_loss.backward()
                actor_optimizer.step()

                critic_optimizer.zero_grad()
                critic_loss.backward()
                critic_optimizer.step()

                total_reward += reward_tensor.item()

                if action == 0 or 1 in next_state[:2]:
                    current_player = 1  
                    state = next_state
                    break  
                else:
                    state = next_state  

                game_scores.append([env.permanent_stack.tolist()])

            while current_player == 1 and not done:
                action = 1
                if debug:
                    print(f"Dummy player's action: {action}")
                
                next_state, reward, done = env.step(action)
                
                next_state = normalize_observation(next_state, min_vals, max_vals)

                if 1 in next_state[:2]:
                    current_player = 0  
                    state = next_state
                    continue  

                action = 0
                if debug:
                    print(f"Dummy player's action: {action}")
                
                next_state, reward, done = env.step(action) 

                next_state = normalize_observation(next_state, min_vals, max_vals)

                current_player = 0
                state = next_state
                game_scores.append([env.permanent_stack.tolist()]) 

            if debug:
                print(f"Episode: {episode}, NN's Reward: {reward_tensor.item()}, Total Reward: {total_reward}")
                print(f"Observation: {next_state.numpy()}")
                print(f"Done: {done}")
                print("-" * 30)

        rewards_per_episode.append(total_reward)

        actor_losses.append(episode_actor_loss)
        critic_losses.append(episode_critic_loss)

        actor_scheduler.step()  
        critic_scheduler.step()

        if episode % 100 == 0 or debug:
            print(f"Episode {episode} finished, Total Reward: {total_reward}")

    return rewards_per_episode, game_scores, actor_losses, critic_losses


def normalize_observation(state, min_vals, max_vals):
    normalized_state = (state - min_vals) / (max_vals - min_vals)
    return normalized_state

