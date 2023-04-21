


import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.optimizers import Adam
import numpy as np



class DeepQNetwork(keras.Model):
    def __init__(self, n_actions, fc1_dims, fc2_dims):
        super(DeepQNetwork, self).__init__()
        self.dense1 = keras.layers.Dense(fc1_dims, activation='relu')
        self.dense2 = keras.layers.Dense(fc2_dims, activation='relu')
        self.Q = keras.layers.Dense(n_actions, activation=None)
        
        
        

    def call(self, state):
        x = self.dense1(state)
        x = self.dense2(x)
        Q = self.Q(x)

        return Q


# Same as DQN : Check out :L C:\Users\luvve\FAICoding\DQN_Keras\simple_dqn_keras_with_comments
# for detailed comments
class ReplayBuffer():
    def __init__(self, max_size, input_shape):
        self.mem_size = max_size
        self.mem_cntr = 0

        self.state_memory = np.zeros((self.mem_size, *input_shape),
                                        dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size, *input_shape),
                                        dtype=np.float32)
        self.action_memory = np.zeros(self.mem_size, dtype=np.int32)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool)

    def store_transition(self, state, action, reward, state_, done):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = done

        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        
        # cant learn anything from array of zeros, 
        # so only learn from the set which is filled and have some
        # values.. 
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, batch_size, replace=False)


        states = self.state_memory[batch]
        new_states = self.new_state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        dones = self.terminal_memory[batch]

        return states, actions, rewards, new_states, dones

class Agent():
    def __init__(self, lr, gamma, n_actions, epsilon, batch_size,
                 input_dims, epsilon_dec=1e-3, eps_end=0.01, 
                 mem_size=10000, fname='dqn.h5', fc1_dims=128,
                 fc2_dims=128, replace=100):
        
        self.action_space = [i for i in range(n_actions)]
        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_dec = epsilon_dec
        self.eps_min = eps_end
        self.fname = fname
        self.replace = replace
        self.batch_size = batch_size

        self.learn_step_counter = 0
        self.memory = ReplayBuffer(mem_size, input_dims)
        self.q_eval = DeepQNetwork(n_actions, fc1_dims, fc2_dims)
        self.q_next = DeepQNetwork(n_actions, fc1_dims, fc2_dims)

        self.q_eval.compile(optimizer=Adam(learning_rate=lr),
                            loss='mean_squared_error')
        self.q_next.compile(optimizer=Adam(learning_rate=lr),
                            loss='mean_squared_error')

    def store_transition(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)
        

    """
    The purpose of the target network(self.q_next) in DDQN is to provide a stable estimate
    of the Q-values for the selected actions. However, during the selection of 
    actions, we do not need to use the target network because we are only interested 
    in selecting the action that maximizes the Q-value (self.q__eval) according to the current policy.
    Using the evaluation network for action selection ensures that the selected action 
    reflects the most up-to-date information available to the agent.
    """
    def choose_action(self, observation):
        if np.random.random() < self.epsilon:
            action = np.random.choice(self.action_space)
        else:
            
            
            
            state = np.array([observation])
            
            
            actions = self.q_eval(state)
            
            action = tf.math.argmax(actions, axis=1).numpy()[0]

        return action

    def learn(self):
        
        if self.memory.mem_cntr < self.batch_size:
            return
    
        if self.learn_step_counter % self.replace == 0:
            self.q_next.set_weights(self.q_eval.get_weights())
    
        states, actions, rewards, states_, dones = \
                                    self.memory.sample_buffer(self.batch_size)
    
        q_pred = self.q_eval(states)
        q_next = tf.math.reduce_max(self.q_next(states_), axis=1, keepdims=True).numpy()
        q_target = np.copy(q_pred)
    
        #max_actions = tf.math.argmax(q_next, axis=1)
    
        for idx, terminal in enumerate(dones):
            
            # for each state in the batch, taking look at the done fflag. 
            # if state is done, then set its value to zero. 
            # if in terminal there are no future rewards. 
            
            if terminal:  # means sate is done, set the next state to 0. 
                q_next[idx] = 0.0
                
            # what is happening?
            
            q_target[idx, actions[idx]] = rewards[idx] + self.gamma*q_next[idx]

    
        self.q_eval.train_on_batch(states, q_target)
    
        self.epsilon = self.epsilon - self.eps_dec if self.epsilon > \
                        self.eps_min else self.eps_min
    
        self.learn_step_counter += 1

    def save_model(self):
        self.q_eval.save(self.model_file)
        
    def load_model(self):
        self.q_eval = self.load_model(self.model_file)

# =============================================================================
#     def load_model(self):
#         self.q_eval = load_model(self.model_file)
# =============================================================================
