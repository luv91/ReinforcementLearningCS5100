


import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.optimizers import Adam
import numpy as np

"""
The DuelingDeepQNetwork class inherits from the keras.Model class, 
which is the base class for all Keras models. The super(DuelingDeepQNetwork, self).__init__() 
call initializes the base class, keras.Model, which is required for the DuelingDeepQNetwork to 
function correctly.

"""


class DuelingDeepQNetwork(keras.Model):
    def __init__(self, n_actions, fc1_dims, fc2_dims):
        super(DuelingDeepQNetwork, self).__init__()
        self.dense1 = keras.layers.Dense(fc1_dims, activation='relu')
        self.dense2 = keras.layers.Dense(fc2_dims, activation='relu')
        
        # self.Q ==> will be defined directly for DQN, but not in this architecture.. 
        
        # No activation below.. ==> because we want the raw value. 
        self.V = keras.layers.Dense(1, activation=None)
        
        self.A = keras.layers.Dense(n_actions, activation=None)

        """
        In the Dueling Deep Q-Network (Dueling DQN) architecture, the Q-values are 
        decomposed into two parts: a state-value function (V) and an advantage function (A).
        
        The state-value function (V) represents the value of being in a certain state,
         regardless of the action taken. In contrast, the advantage function (A) represents 
         the advantage of taking a specific action in a certain state compared to other actions 
         that could be taken in that state.
        
        In the code provided, self.V and self.A are two separate dense layers of the model. 
        self.V represents the state-value function (V), and self.A represents the advantage 
        function (A). By separating the Q-value into these two components, the model can learn 
        to estimate the value of each action in a more efficient way.
        
        The final Q-value is then calculated by combining the two components with the formula: 
            Q = V + A - mean(A), where the mean(A) term is subtracted to ensure that the model remains unbiased.
        
        
        In contrast to the Dueling DQN architecture, the traditional Deep Q-Network (DQN) 
        architecture does not decompose the Q-value into state-value and advantage components.
        
        In the DQN architecture, the Q-value is directly estimated for each action in a 
        given state using a single neural network. The neural network takes the state as 
        input and outputs the Q-value for each possible action.
        
        Therefore, in a DQN, we don't need to define separate layers for the state-value 
        function and the advantage function, as we do in a Dueling DQN. Instead, a single 
        output layer of the neural network is used to estimate the Q-value for each action in the given state.
        
        """
        
        
        

    def call(self, state):
        
        """
        The call() method in the above code is necessary to define the 
        forward pass of the Dueling DQN model. 
        
        In the call() method, the state is first passed through two fully connected 
        layers (self.dense1 and self.dense2) to generate a feature representation of the input. 
        This feature representation is then passed through two separate layers, 
        self.V and self.A, to generate the state-value function (V) and the advantage 
        function (A), respectively.

        Finally, the Q-value is computed by combining the state-value and 
        advantage functions using the formula Q = V + (A - mean(A)), where 
        mean(A) is the mean of the advantage function across all actions. 
        This formula implements the Dueling DQN architecture by decomposing
        the Q-value into two components and then recombining them in a way that
        allows the model to learn more efficiently.
        
        """
        x = self.dense1(state)
        x = self.dense2(x)
        V = self.V(x)
        A = self.A(x)

        # tf.math.reduce_mean(A
        
        # tf.math.reduce_mean(A): When combining the two components to compute the Q-value,
        # it is important to remove any bias that may be present in the advantage function.
        #In particular, the advantage function can be biased if the values of the different 
        #actions are not centered around zero, meaning that the advantage of some actions may 
        #be consistently overestimated or underestimated.
        
        Q = (V + (A - tf.math.reduce_mean(A, axis=1, keepdims=True)))

        return Q

    # may be unnecessary, please experiment with call()
    def advantage(self, state):
        """
        
        The advantage() method in the provided code is not strictly
        necessary for the functioning of the Dueling DQN model. 
        
        The advantage() method simply computes the advantage function (A) 
        for a given state, without computing the state-value function (V)
        or combining them to compute the Q-value. The advantage function 
        represents the advantage of taking a specific action in a certain
        state compared to other actions that could be taken in that state.
        
        We will use it for choosing actions. 
        
        """
        x = self.dense1(state)
        x = self.dense2(x)
        A = self.A(x)

        return A

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
                 mem_size=100000, fname='dueling_dqn.h5', fc1_dims=128,
                 fc2_dims=128, replace=100):
        
        self.action_space = [i for i in range(n_actions)]
        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_dec = epsilon_dec
        self.eps_min = eps_end
        self.fname = fname
        
        """
        In the implementation of the Agent class for Dueling Deep Q Network (DDQN), 
        the self.replace parameter is used to determine how often the weights of the
         target network, self.q_next, are updated with the weights of the evaluation network, self.q_eval.
         
         Updating the target network less frequently (i.e., with a larger value of self.replace) 
         can make the learning process more stable, but may slow down the learning process. 
         Updating the target network more frequently (i.e., with a smaller value of self.replace) 
         can speed up the learning process, but may lead to instability due to fluctuations
         in the Q-value estimates.
         
         
         replace defulat value is 100. in DQN we use epsilon greedy policy, and
         learns at every step. 
         
         after every 100 moves, we are going to replace the moves in target network from
         the moves in online evalaution network. 
         
         
         # 
        """
        self.replace = replace
        self.batch_size = batch_size

        self.learn_step_counter = 0
        self.memory = ReplayBuffer(mem_size, input_dims)
        
        """
        In Dueling Deep Q Network (DDQN), we define two separate networks, self.q_eval 
        and self.q_next, to implement the double Q-learning algorithm. The purpose of using 
        two separate networks is to reduce the overestimation of the Q-values that can occur 
        in standard Q-learning algorithms.

        In DDQN, the self.q_eval network is used to select the actions to take, while the 
        self.q_next network is used to evaluate the Q-values of the selected actions. The 
        self.q_next network is updated less frequently than the self.q_eval network to ensure
        that the network's Q-value estimates remain stable during the learning process.

        By using two separate networks in DDQN, we avoid the overestimation of Q-values 
        that can occur when using a single network to both select and evaluate actions. 
        This overestimation can occur when the Q-value estimates for certain actions are
        consistently higher than their true values, leading to suboptimal policies.
        
        """
        
        
        self.q_eval = DuelingDeepQNetwork(n_actions, fc1_dims, fc2_dims)
        
        self.q_next = DuelingDeepQNetwork(n_actions, fc1_dims, fc2_dims)

        self.q_eval.compile(optimizer=Adam(learning_rate=lr),
                            loss='mean_squared_error')
        # just a formality, won't optimize network
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
            
            
            actions = self.q_eval.advantage(state)
            
            action = tf.math.argmax(actions, axis=1).numpy()[0]

        return action

    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return
        
        # learn_step_counter is divisible by replace.. 
        # set the weights from q_eval into q_next. 
        if self.learn_step_counter % self.replace == 0:
            self.q_next.set_weights(self.q_eval.get_weights())

        states, actions, rewards, states_, dones = \
                                    self.memory.sample_buffer(self.batch_size)

        q_pred = self.q_eval(states)
        
        """
        In the learn method of the Agent class provided for Dueling Deep Q Network (DDQN),
        we update the weights of the target network (self.q_next) periodically with the 
        weights of the evaluation network (self.q_eval) to ensure that the target network's
        Q-value estimates remain stable during the learning process.

        However, we still need to use the target network to compute the Q-values for the
        selected actions during the learning process. This is because we want to avoid 
        overestimation of the Q-values and improve the stability of the Q-value estimates.

        In particular, we use the target network to compute the Q-values for the next
        state (states_) and select the action with the highest Q-value. We then use 
        
        the Q-value of this action to update the Q-value of the current state-action 
        pair in the Q-table. This process is known as the "target network update" and
        is a key component of the DDQN algorithm.
        
        Here, self.q_next(states_) computes the Q-values for all 
        possible actions in the next state using the current weights of the 
        target network. We then use tf.math.reduce_max to select the highest
        Q-value for each state in the batch, and keepdims=True to preserve the 
        shape of the output as a column vector.
        
        We then use q_next to update the Q-value of the current state-action pair in the 
        Q-table, as shown in the following code:

\
        q_target[idx, actions[idx]] = rewards[idx] + self.gamma*q_next[idx]
        This line updates the Q-value of the current state-action pair by adding the
        immediate reward and the discounted estimate of the next state's value (self.gamma*q_next[idx]).
        
        """
        
        q_next = tf.math.reduce_max(self.q_next(states_), axis=1, keepdims=True).numpy()
        
        
        # copy the predicted network q_pred from q_eval
        q_target = np.copy(q_pred)

        # improve on my solution!
        for idx, terminal in enumerate(dones):
            
            # for each state in the batch, taking look at the done fflag. 
            # if state is done, then set its value to zero. 
            # if in terminal there are no future rewards. 
            
            if terminal:  # means sate is done, set the next state to 0. 
                q_next[idx] = 0.0
                
            # what is happening?
            
            q_target[idx, actions[idx]] = rewards[idx] + self.gamma*q_next[idx]

        """
        Yes, self.q_eval.train_on_batch(states, q_target) calculates the loss 
        and performs a single update step to minimize the loss using stochastic gradient descent.

        The loss function being minimized is the mean squared error (MSE) between 
        the predicted Q-values and the target Q-values. This is done by 
        computing the element-wise squared difference between q_pred and q_target,
        and then taking the mean over all elements in the batch:
        
        
        loss = tf.keras.losses.mean_squared_error(q_target, q_pred)
        The train_on_batch method then performs a single update step using stochastic 
        gradient descent to minimize the loss with respect to the weights of the q_eval network.
        
        Therefore, during each call to self.q_eval.train_on_batch(states, q_target), 
        the loss is calculated and the network weights are updated to minimize the loss.
        
        """
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
