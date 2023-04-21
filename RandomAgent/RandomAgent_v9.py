
import collections
import collections
import random
import numpy as np
import matplotlib.pyplot as plt

import lunar_lander as lander


num_iter = 10000

env = lander.LunarLander()

alpha = 0.85
# The higher the gamma value, the more the agent values future rewards.
gamma = 0.9   # discounted rewards
epsilon = 0.8 # for epsilon greedy policy
    
# total number of episodes. 

num_episodes = 10000

# time steps per episode. 
num_timesteps = 1000
seg = 100

r_seq = []
it_reward = []
scores=[]
avg_score_seq = []
Q_value = collections.defaultdict(float)
seg = 100

for i in range(num_episodes):
    total_reward = 0
    # in each episode, we reset the environment. 
    s = env.reset()
    steps = 0
    
    while True:
        # use a policy generator to guide sarsa exploration
        # step and get feedback
        a = np.random.randint(0, 4)

        sp, r, done, info = env.step(a)

        total_reward += r



        # if steps % 20 == 0 or done:
        #     print("observations:", " ".join(["{:+0.2f}".format(x) for x in s]))
        #     print("step {} total_reward {:+0.2f}".format(steps, total_reward))

        steps += 1

        if done or steps > 1000:
            # if total_reward > 50:
            #     print(ds, a, total_reward)
            it_reward.append(total_reward)
            break

    if i % seg == 0:
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

plt.plot(x, y, label='random_agent_reward_10Kepisodes_v26')
plt.savefig("random_agent_reward_10Kepisodes_v26.png")

# =============================================================================
# plt.plot(x, y1, label='Naive Sarsa reward_10Kepisodes_V4_LessStateSpace_withUCBPolicyExtractor')
# plt.savefig("naive_sarsa_reward_10Kepisodes_V4_LessStateSpace_withUCBPolicyExtractor_y1.png")
# =============================================================================

#np.savetxt("naive_sarsa_reward_10Kepisodes_V4_LessStateSpace_withUCBPolicyExtractor.txt", y)
np.savetxt("random_agent_reward_10Kepisodes_v26.txt", y1)