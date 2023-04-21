

import gym
import pandas as pd
import random
import numpy as np
import matplotlib.pyplot as plt
import collections
import json
#env = gym.make('FrozenLake-v1')

env = gym.make('LunarLander-v2')

# =============================================================================
# def state_extractor(s):
# 
#     state = (min(5, max(-5, int((s[0]) / 0.05))), \
#             min(5, max(-1, int((s[1]) / 0.1))), \
#             min(3, max(-3, int((s[2]) / 0.1))), \
#             min(3, max(-3, int((s[3]) / 0.1))), \
#             min(3, max(-3, int((s[4]) / 0.1))), \
#             min(3, max(-3, int((s[5]) / 0.1))), \
#             int(s[6]), \
#             int(s[7]))
# 
#     return state
# =============================================================================

def state_extractor(s):

    state = (int((s[0] - 0.3 / 2.0) / 0.3), \
            int((s[1] - 0.3 / 2.0) / 0.3), \
            int((s[2] - 0.2 / 2.0) / 0.2), \
            int((s[3] - 0.2 / 2.0) / 0.2), \
            int((s[4] - 0.2 / 2.0) / 0.2), \
            int((s[5] - 0.2 / 2.0) / 0.2), \
            int(s[6]), \
            int(s[7]))

    return state


# =============================================================================
# # INitialize a Q-function, Q(s,a) with random initial values. 
# 
# # when will the episode end?
# # when will we update the Q-function? what exactly gets updated in the Q-function?
# # all of them, or just the next states, form where we are standing?
# 
# # frozen lake have 4 actions, and 15 states. 
# Q = {}
# total_states_lunar_lander = 8
# for s in range(total_states_lunar_lander):
#     for a in range(env.action_space.n):
#         
#         print("s,a",s,a)
#         Q[(s,a)] = 0.0
#         
#         
# print("Q", Q)
# =============================================================================

# Define a policy. 
def making_key_for_Q_hashmap(state, action):
    
    return str(state) + " " + str(action)
    
    
# =============================================================================
# def policyExtractor(current_state, Q_value, episode_number, epsilon):
#     
#     max_Q_value_per_action = -float('inf')
#     max_action = None
#     
#     
#     if random.uniform(0,1) < epsilon:
#         return np.random.randint(0, 4)  # random action. 
#     
#     else:
#         
#         action_list = list(range(env.action_space.n))
#         
#         # in a particular state, there is a range of action available.
#         # out of that range of actions, select an action, which gives 
#         # max Q value for a particular state. 
#         # in a grid if state is (4,3) and there are actions 0,1,2,3 with 
#         # action 2 giving the highest Q-value for state(4,3) select that. 
#         # policy is an action, so return action corresponding to max Q valuue
#         for action in action_list:
#             
#             #hashmap_key = str(current_state) + " " + str(action)
#             
#             Qv = np.array(Q_value[making_key_for_Q_hashmap(current_state, action)])
#             
#             
#         # rewriting the above in short. 
#         
#             
#         return Qv
# =============================================================================
            
# epsiolon greedy policy extractor. 
def policyExtractor(current_state, Q_value, episode_number, epsilon):
    rand = np.random.randint(0, 100)

    threshold = 20

    if rand >= threshold:
        Qv = np.array([ Q_value[making_key_for_Q_hashmap(current_state, action)] for action in [0, 1, 2, 3]])
        return np.argmax(Qv)
    
    # 20 percent of times only it will go random. because threshold = 20
    else:
        return np.random.randint(0, 4)

# upper confidence bound:
# The optimal value of the exploration constant c in the UCB algorithm can depend on the 
# specific problem and the distribution of the rewards. In general, larger values of c 
# encourage more exploration and smaller values encourage more exploitation.
# =============================================================================
# def policyExtractor(current_state, Q_value, episode_number,epsilon):
#     
#     c = 2.5
#     n_actions = 4
#     
#     counts = np.array([Q_value.get(making_key_for_Q_hashmap(current_state, action)[-1]) for action in range(n_actions)])
#     #counts = np.array([ Q_value.get(making_key_for_Q_hashmap(current_state, action), 0) for action in range(n_actions) ])
#     total_counts = counts.sum()
#     #kkkkkkkkkkkk
#     if total_counts == 0:
#         return np.random.randint(0, n_actions)
#     else:
#         exploration = np.sqrt(c * np.log(total_counts) / counts)
#         Qv = np.array([ Q_value.get(making_key_for_Q_hashmap(current_state, action)) for action in range(n_actions) ])
#         Qv_ucb = Qv + exploration
#         return np.argmax(Qv_ucb)
# =============================================================================
    
    
    
    
# After completing an episode, we extract
#a new policy from the updated Q function and uses this new policy to act in the
#environment. (Remember that our policy is always an epsilon-greedy policy).


# Now, let's define the epsilon-greedy policy. We generate a 
# random number from the uniform distribution and if the random number is 
# less than epsilon we select the random action else we select the best action
#  which has the maximum Q value:

# =============================================================================
# def epsilon_greedy(state, epsilon):
#     max_Q_value_per_action = -float('inf')
#     max_action = None
#     if random.uniform(0,1) < epsilon:
#         return env.action_space.sample()  # random action. 
#     
#     # otherwise, action with maximum Q value. 
#     
#     # max Q
#     else:
#         action_list = list(range(env.action_space.n))
#         
#         # in a particular state, there is a range of action available.
#         # out of that range of actions, select an action, which gives 
#         # max Q value for a particular state. 
#         # in a grid if state is (4,3) and there are actions 0,1,2,3 with 
#         # action 2 giving the highest Q-value for state(4,3) select that. 
#         # policy is an action, so return action corresponding to max Q valuue
#         for action in action_list:
#             
#             if Q[(state, action)] > max_Q_value_per_action:
#                 max_Q_value_per_action = Q[(state, action)]
#                 max_action = action
#                 
#             
#             #max_Q_value_per_action = max(max_Q_value_per_action,Q[(state, action)])
#         
#         print("max_Q_value_per_action",max_Q_value_per_action)
#         return max_action
# =============================================================================
    
    
# =============================================================================
#         max_value = -float('inf')
#         max_action = None
#         for action in range(env.action_space.n):
#             if Q[(state, action)] > max_value:
#                 max_value = Q[(state, action)]
#                 max_action = action
#         return max_action
# =============================================================================
    
        #return max(list(range(env.action_space.n)), key = lambda x: Q[(state,x)])
    
# The step size parameter, alpha, controls the weight given to new experiences 
# versus past experiences. It determines how much the agent should update its 
# Q-values based on new information obtained during training. The higher the 
# alpha value, the more the agent will weight new information in its updates. 
alpha = 0.85
# The higher the gamma value, the more the agent values future rewards.
gamma = 0.9   # discounted rewards
epsilon = 0.8 # for epsilon greedy policy
    
# total number of episodes. 
num_episodes = 10000
num_episodes = 10000

# time steps per episode. 
num_timesteps = 1000
seg = 100

r_seq = []
it_reward = []
scores=[]
avg_score_seq = []
Q_value = collections.defaultdict(float)
# we cannot directly impose the epsilon_greedy policy like in 
# frozen lake environment, 
# so define policy extractor. 

# start an episode using a for loop. 
for i in range(num_episodes):
    
    total_reward = 0
    # in each episode, we reset the environment. 
    s = env.reset()
    
    # extract state, since s consist of 8 values.
    ds = state_extractor(s)
    
    
    
    a = policyExtractor(ds, Q_value,i, epsilon)
    
    #print("Line 177:action", a)
    # in each episode we select an initial action using the epsilon-greedy policy. 
    #a = epsilon_greedy(s, epsilon)
    
    # each time step in the episode. 
    for each_time_step in range(num_timesteps):
        
        #perform the selected action and store the next state information: 
        
        ns, reward, done, _ = env.step(a)
        
        nextState = state_extractor(ns)
        # select the nextaction in the nextState based on epsilon-greedy policy. 
        nextAction = int(policyExtractor(nextState, Q_value, i, epsilon))
        
        # compute the Q-values of the current state-action pair. 
        # based on next action and next state. 
        #print("nextState, nextAction", nextState, nextAction)
        
        current_key = making_key_for_Q_hashmap(s, a)
        next_state_key = making_key_for_Q_hashmap(nextState, nextAction)
        Q_value[current_key] += alpha*(reward+gamma*Q_value[next_state_key]- Q_value[current_key])
        
        #Q_value[(s,a)] += alpha*(reward+gamma*Q[(nextState, nextAction)]- Q_value[(s,a)])
        
        #update next state to current state
        s = nextState
        
        #update next action to current action
        a = nextAction
        
        total_reward += reward
        if done:
            
            it_reward.append(total_reward)
            break
        
    if i % seg  == 0:
        avg_rwd = np.mean(np.array(it_reward))
        print("#It: ", i, " avg reward: ", avg_rwd, " out of ", len(it_reward), " trials")
        it_reward = []
        r_seq.append(avg_rwd)
    
    scores.append(total_reward)
    avg_score = np.mean(scores[-100:])
    print('episode: ', i,'score: %.2f' % total_reward,
              ' average score %.2f' % avg_score)
    avg_score_seq.append(avg_score)
    
y = np.array(r_seq)
y1 = np.array(avg_score_seq)
x = np.linspace(0, num_episodes, y.shape[0])

plt.plot(x, y, label='naive_sarsa_reward_10Kepisodes_V4_NaiveSarsa_v25')
plt.savefig("naive_sarsa_reward_10Kepisodes_V4_NaiveSarsa_v25.png")

# =============================================================================
# plt.plot(x, y1, label='Naive Sarsa reward_10Kepisodes_V4_LessStateSpace_withUCBPolicyExtractor')
# plt.savefig("naive_sarsa_reward_10Kepisodes_V4_LessStateSpace_withUCBPolicyExtractor_y1.png")
# =============================================================================

#np.savetxt("naive_sarsa_reward_10Kepisodes_V4_LessStateSpace_withUCBPolicyExtractor.txt", y)
np.savetxt("naive_sarsa_reward_10Kepisodes_V4_NaiveSarsa_v25.txt", y1)
q = json.dumps(Q_value, indent=4)
f = open("sarsa_Q_withUCBPolicyExtractor.json","w")
f.write(q)
f.close()
        
    
    
    