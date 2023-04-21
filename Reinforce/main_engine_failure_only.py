

# if you have more than 1 gpu, use device '0' or '1' to assign to a gpu
#import os
#os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
#os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import gym
import numpy as np
from reinforce_tf2 import Agent
import matplotlib.pyplot as plt
import lunar_lander as lander


def plotLearning(scores, filename, x=None, window=5):   
    N = len(scores)
    running_avg = np.empty(N)
    for t in range(N):
	    running_avg[t] = np.mean(scores[max(0, t-window):(t+1)])
    if x is None:
        x = [i for i in range(N)]
    plt.ylabel('Score')       
    plt.xlabel('Game')                     
    plt.plot(x, running_avg)
    plt.savefig(filename)
    
    

if __name__ == '__main__':
    agent = Agent(alpha=0.0005,  gamma=0.99,n_actions=4)
    
    env = lander.LunarLander()
    
    score_history = []
    
    average_score_accumumation = []
    average_score_accumumation2 = []
    num_episodes = 4000
    np.random.seed(0)
    average_score_accumumation = []
    
    for i in range(num_episodes):
        done = False
        score = 0
       
        observation = env.reset()
        
        while not done:
            action = agent.choose_action(observation)
            
            if np.random.random() > 0.2:
                observation_, reward, done, info = env.step(action)
                
            else:
                observation_, reward, done, info = env.step(0)
            
            agent.store_transition(observation, action, reward)
            
            observation = observation_
            score += reward
        
        score_history.append(score)

        agent.learn()
        avg_score = np.mean(score_history[-100:])
        average_score_accumumation.append(avg_score)
        print('episode: ', i,'score: %.1f' % score,
            'average score %.1f' % avg_score)

    filename = 'lunar-lander.png'
    plotLearning(score_history, filename=filename, window=100)
    np.savetxt("PolicyGradient_LunarLander_No_Perturbations_AtAll_v2.txt", average_score_accumumation)