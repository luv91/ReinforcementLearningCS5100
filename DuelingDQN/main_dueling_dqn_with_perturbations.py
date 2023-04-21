



from dueling_dqn import Agent
import numpy as np
import gym
from utils import plotLearning
import lunar_lander as lander

def get_obs(true_loc):
    tX = true_loc[0]
    tY = true_loc[1]

    nX = np.random.normal(loc=tX, scale=0.1)

    return (nX, tY)

if __name__ == '__main__':
    
    env = lander.LunarLander()    
    n_games = 4000
    agent = Agent(gamma=0.99, epsilon=1, lr=1e-3, input_dims=[8], 
                  epsilon_dec=1e-3, mem_size=100000, batch_size=64, eps_end=0.01,
                  fc1_dims=128, fc2_dims=128, replace=100, n_actions=4)

    scores, eps_history = [], []

    for i in range(n_games):
        done = False
        score = 0        
        observation = env.reset()
        noisy_obs_x_y = get_obs((observation[0], observation[1]))
        observation = (noisy_obs_x_y[0], noisy_obs_x_y[1], observation[2], observation[3],
                       observation[4], observation[5], observation[6], observation[7])
        
        
        while not done:
            action = agent.choose_action(observation)
            
            if np.random.random() > 0.2:
                
                observation_, reward, done, info = env.step(action)
                new_state_noisy_obs_x_y = get_obs((observation_[0], observation_[1]))
                observation_ = (new_state_noisy_obs_x_y[0], new_state_noisy_obs_x_y[1], 
                                observation_[2], observation_[3],
                               observation_[4], observation_[5], 
                               observation_[6], observation_[7])
              

            else:
                observation_, reward, done, info = env.step(0)
                
            score += reward
            agent.store_transition(observation, action, reward, observation_, done)
            observation = observation_
            agent.learn()
        eps_history.append(agent.epsilon)
        scores.append(score)

        avg_score = np.mean(scores[-100:])
        print('episode ', i, 'score %.1f' % score,
                'average score %.1f' % avg_score,
                'epsilon %.2f' % agent.epsilon)

    filename='keras_lunar_lander.png'
    x = [i+1 for i in range(n_games)]
    plotLearning(x, scores, eps_history, filename)