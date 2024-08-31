import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

### Initialize the training environment and define parameters to be used ###

env_train = gym.make("LunarLander-v2") # No render_mode = "human" when training
observation, info = env_train.reset()

episodes = 300000               # Total episodes in the run
discount = 0.6
stat_episode = 250              # Amount of episodes over which the data is processed
save_episode = episodes//5      # Every so many episodes, save the Q-table

learning_rate = 0.3
lr_decay_start = 1
lr_decay_stop = 80000
lr_decay = 0.9*learning_rate/(lr_decay_stop-lr_decay_start)     # Calculate decay per step of the learning rate

discrete_size = [10, 10, 20, 20, 20, 10]                        # Amount of windows in which to discretize in            
discrete_step = (env_train.observation_space.high[:6] - env_train.observation_space.low[:6])/discrete_size

q_table = np.random.uniform(low=-2, high=0, size=([11,11,21,21,21,11,2,2,4]))       # Initialize Q-table

rewards = []
rewards_stats = {'episode':[], 'average':[], 'max':[], 'min':[], 'positive':[], 'solution':[], 'landed':[]}     # Stats will be saved in this
positive = 0
solution = 0
landed = 0

epsilon = 1
e_decay_start = 1
e_decay_stop = episodes//3
epsilon_decay = 0.9*(epsilon/(e_decay_stop-e_decay_start))      # Calculate decay per step of epsilon


### Discretize the incoming state ###
def get_discrete_state(observation):                    
    state = ((observation[:6] - env_train.observation_space.low[:6])/discrete_step)
    s1 = int(state[0])
    if s1 > 10:             # Used to fix bug that sometimes made the state go outside of its maximum values for one step when crashed
        s1 = 10
    s2 = int(state[1])
    if s2 > 10:             # Idem
        s2 = 10
    s3 = int(state[2])
    if s3 > 20:             # Idem
        s3 = 20
    s4 = int(state[3])
    if s4 > 20:             # Idem
        s4 = 20
    s5 = int(state[4])
    if s5 > 20:             # Idem
        s5 = 20
    s6 = int(state[5])
    if s6 > 10:             # Idem
        s6 = 10
    s7 = int(observation[6])    # No discretization needed
    s8 = int(observation[7])    # Idem
    discrete_state = (s1,s2,s3,s4,s5,s6,s7,s8)
    return discrete_state

### Start of the run ###

for episode in range(episodes):
    
    current_state = get_discrete_state(observation)     # Initial state
    truncated = False                                   # -----
    terminated = False                                  # Initial settings 
    total_reward = 0                                    # -----

    while not truncated and not terminated:                 # Start of one episode
        if np.random.random() > epsilon:                    # -----
            action = np.argmax(q_table[current_state])      # Get Action
        else:                                               # 
            action = np.random.randint(0, 4)                # -----

        observation, reward, terminated, truncated, info = env_train.step(action)   # Update environment
        new_state = get_discrete_state(observation)                                 # New state
        
        max_future_q = np.max(q_table[new_state])                                                   # -----
        current_q = q_table[current_state + (action,)]                                              # Calculate new Q and replace old Q
        new_q =  (1-learning_rate) * current_q + learning_rate * (reward + discount * max_future_q) # 
        q_table[current_state + (action,)] = new_q                                                  # -----

        current_state = new_state

        total_reward += reward

        if reward == 100:               # Check if landing was succes and end to end when landed safely
            succes = 1
            # print(f'Succesfull landing!, total reward: {total_reward}')
            truncated = True            
        else:
            succes = 0

    if terminated or truncated:         # End of an episode
        if total_reward > 0:            # -----
            positive += 1               # Check if positive, solution or landed this episode
        if total_reward > 200:          #
            solution += 1               #
        if succes ==1:                  #
            landed += 1                 # -----
        rewards.append(total_reward)
        if episode % stat_episode == 0 and episode > 0:                             # -----
            avg_re = sum(rewards[-stat_episode:])/stat_episode                      # Every stat_episode process results of only the last stat_episode rewards
            rewards_stats['episode'].append(episode)                                #
            rewards_stats['average'].append(avg_re)                                 #
            rewards_stats['max'].append(max(rewards[-stat_episode:]))               #
            rewards_stats['min'].append(min(rewards[-stat_episode:]))               #
            rewards_stats['positive'].append((positive/stat_episode)*100)           #
            rewards_stats['solution'].append((solution/stat_episode)*100)           #
            rewards_stats['landed'].append((landed/stat_episode)*100)               # -----
            print(f'Episode: {episode:>5d}, average reward: {avg_re:>4.1f}, # of ep postives: {positive:>2}, # of ep solutions: {solution:>2}, # of ep landed: {landed:>2}, current epsilon: {epsilon:>1.2f}')
            positive = 0                                                            # -----
            solution = 0                                                            # Reset counters
            landed = 0                                                              # -----

        observation, info = env_train.reset()                                       # Reset environment to starting state
    
    if e_decay_stop >= episode >= e_decay_start:                                    # Decay episilon
        epsilon -= epsilon_decay                                                    #
    if episode == 200000:                                                           # When not using full decay to zero, set it at a specific episode
        epsilon = 0                                                                 #
    
    if lr_decay_stop >= episode >= lr_decay_start:                                  # Decay learning rate
        learning_rate -= lr_decay                                                   #

    if episode % save_episode == 0:                                                 # Save Q-table every save_episode
        np.save(f'qtables/{episode}-qtable.npy',q_table)

np.save(f'qtables/{episode}-qtable.npy',q_table)                                    # Save last Q-table 

env_train.close()                                                                   # Close training environment

env_test = gym.make("LunarLander-v2", render_mode='human')                          # -----
observation, info = env_test.reset()                                                # Initialize testing environment to see last Q-table behavior
current_state = get_discrete_state(observation)                                     #
terminated = False                                                                  #
truncated = False                                                                   # -----

while not terminated and not truncated:                                             # -----
    action = np.argmax(q_table[current_state])                                      # Run one episode without learning
    observation, reward, terminated, truncated, info = env_test.step(action)        #
    new_state = get_discrete_state(observation)                                     #
    current_state = new_state                                                       # -----

env_test.close()                                                                    # Close Test environment

plt.plot(rewards_stats['episode'], rewards_stats['average'], label='average rewards')   # Plot average, max and min reward
plt.plot(rewards_stats['episode'], rewards_stats['max'], label='max rewards')
plt.plot(rewards_stats['episode'], rewards_stats['min'], label='min rewards')
plt.xlabel('Number of episodes')
plt.ylabel('Reward value')
plt.title('Reward statistics')
plt.legend(loc=2)
plt.show()

plt.plot(rewards_stats['episode'], rewards_stats['average'], label='average rewards')   # Plot average alone to be more clear
plt.xlabel("Number of episodes")
plt.ylabel('Reward value')
plt.title('Average reward')
plt.show()

plt.plot(rewards_stats['episode'], rewards_stats['positive'], label='positive reward')  # Plot episode statistics
plt.plot(rewards_stats['episode'], rewards_stats['solution'], label='solution')
plt.plot(rewards_stats['episode'], rewards_stats['landed'], label='landed')
plt.xlabel('Number of episodes')
plt.ylabel('% of episodes in window size')
plt.title('Episode statistics')
plt.legend(loc=2)
plt.show()