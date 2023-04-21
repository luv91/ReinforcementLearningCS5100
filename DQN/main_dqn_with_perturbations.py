

import matplotlib.pyplot as plt
from dqn_keras import Agent
import numpy as np
import gym
from utils import plotLearning
from gym import wrappers
import lunar_lander as lander
def get_obs(true_loc):
    tX = true_loc[0]
    tY = true_loc[1]

    nX = np.random.normal(loc=tX, scale=0.1)

    return (nX, tY)

if __name__ == '__main__':
    
    r_seq = []
	
    #env = gym.make('LunarLander-v2')
    env = lander.LunarLander()
    lr = 0.0005
    n_games = 500
	
    agent = Agent(gamma=0.99, epsilon=1, lr=1e-3, input_dims=[8], 
                  epsilon_dec=1e-3, mem_size=10000, batch_size=64, eps_end=0.01,
                  fc1_dims=128, fc2_dims=128, replace=100, n_actions=4)

    #agent.load_model()
    scores = []
    eps_history = []

    #env = wrappers.Monitor(env, "tmp/lunar-lander-6",
    #                         video_callable=lambda episode_id: True, force=True)

    np.random.seed(0)
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
            agent.store_transition(observation, action, reward, observation_, int(done))
            observation = observation_
            agent.learn()

        eps_history.append(agent.epsilon)
        scores.append(score)

        #avg_score = np.mean(scores[max(0, i-100):(i+1)])
        avg_score = np.mean(scores[-100:])
        print('episode: ', i,'score: %.2f' % score,
              ' average score %.2f' % avg_score)

        #if i % 10 == 0 and i > 0:
        #    agent.save_model()

    filename = 'lunarlander.png'

    x = [i+1 for i in range(n_games)]
    plotLearning(x, scores, eps_history, filename)
    
    y = np.array(scores)
    x = np.linspace(0, n_games, y.shape[0])

    plt.plot(x, y, label='DeepQLearning_LunarLander')
    plt.savefig("DeepQLearning_LunarLander.png")

    np.savetxt("DeepQLearning_LunarLander.txt", y)

# =============================================================================
#     q = json.dumps(Q_value, indent=4)
#     f = open("DeepQLearning_LunarLander.json","w")
#     f.write(q)
#     f.close()
# =============================================================================
    
    
    
    
    
    