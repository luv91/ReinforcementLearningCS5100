
from dueling_dqn import Agent
import numpy as np
import gym
from utils import plotLearning
import lunar_lander as lander


if __name__ == '__main__':
    
    env = lander.LunarLander()
    
    n_games = 500
    agent = Agent(gamma=0.99, epsilon=1, lr=1e-3, input_dims=[8], 
                  epsilon_dec=1e-3, mem_size=100000, batch_size=64, eps_end=0.01,
                  fc1_dims=128, fc2_dims=128, replace=100, n_actions=4)

    scores, eps_history = [], []
    average_score_accumumation = []
    for i in range(n_games):
        done = False
        score = 0
        
        observation = env.reset()
       
        
        while not done:
            action = agent.choose_action(observation)
            
            if np.random.random() > 0.2:
                
                observation_, reward, done, info = env.step(action)
                
            else:
                observation_, reward, done, info = env.step(0)
                
                
            score += reward
            agent.store_transition(observation, action, reward, observation_, done)
            observation = observation_
            agent.learn()
            
        eps_history.append(agent.epsilon)
        scores.append(score)

        avg_score = np.mean(scores[-100:])
        average_score_accumumation.append(avg_score)
        print('episode ', i, 'score %.1f' % score,
                'average score %.1f' % avg_score,
                'epsilon %.2f' % agent.epsilon)
    

    np.savetxt("DuelingDeepQLearning_LunarLander_No_Perturbations.txt", average_score_accumumation)
    
    filename='keras_lunar_landerNoPerturbation.png'
    x = [i+1 for i in range(n_games)]
    plotLearning(x, scores, eps_history, filename)