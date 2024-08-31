import gymnasium as gym
import numpy as np

### This script is used to test a specific agent's Q-table ###
### For explanation on set-up check lander_trainer ###

env_test = gym.make("LunarLander-v2", render_mode='human')
observation, info = env_test.reset()

test_episodes = 5                               # Amount of test episodes to be done                           
discrete_size = [10, 10, 20, 20, 20, 10]              
discrete_step = (env_test.observation_space.high[:6] - env_test.observation_space.low[:6])/discrete_size

def get_discrete_state(observation):
    state = ((observation[:6] - env_test.observation_space.low[:6])/discrete_step)
    s1 = int(state[0])
    if s1 > 10:
        s1 = 10
    s2 = int(state[1])
    if s2 > 10:
        s2 = 10
    s3 = int(state[2])
    if s3 > 20:
        s3 = 20
    s4 = int(state[3])
    if s4 > 20:
        s4 = 20
    s5 = int(state[4])
    if s5 > 20:
        s5 = 20
    s6 = int(state[5])
    if s6 > 10:
        s6 = 10
    s7 = int(observation[6])
    s8 = int(observation[7])
    discrete_state = (s1,s2,s3,s4,s5,s6,s7,s8)
    return discrete_state


q_table = np.load('') ### Load Q-table


current_state = get_discrete_state(observation)
terminated = False
truncated = False

for episodes in range(test_episodes):
    current_state = get_discrete_state(observation)
    truncated = False
    terminated = False
    step = 0

    while not terminated and not truncated:
        print(q_table[current_state])
        action = np.argmax(q_table[current_state])
        observation, reward, terminated, truncated, info = env_test.step(action)
        new_state = get_discrete_state(observation)
        current_state = new_state  
        step += 1
    if terminated or truncated:
        observation, info = env_test.reset()
        print(step)

env_test.close()
