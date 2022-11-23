import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow_probability as tfp
from tensorflow.keras.layers import Dense, Flatten, InputLayer


class REINFORCE(keras.Model):
    def __init__(self, state_size, action_size):
        super(REINFORCE, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        # Define model shape
        self.model = keras.Sequential(
            layers=[
                InputLayer(input_shape=(self.state_size,)),
                Dense(32, activation='elu'),
                Dense(64, activation='elu'),
                Dense(64, activation='elu'),
                Dense(32, activation='elu'),
                Dense(16, activation='elu'),
                Flatten(),
                Dense(4, activation='softmax')
            ])

    def action_distribution(self, observations):
        probs = self.model(observations)
        return tfp.distributions.Categorical(logits=probs)

    def call(self, s):
        return self.model(s)


class Agent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = 0.99
        self.learning_rate = 1e-6

        self.network = REINFORCE(self.state_size, self.action_size)
        self.opt = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)

    def preprocess(self, s):
        s = np.array(s).astype('float32')
        s[0] = s[0]/6
        s[1:9] = s[1:9]/69
        s[9:17] = s[9:17]/138
        return s

    def act(self, state, action_list):

        state = self.preprocess(state)
        
        # Reshape input and call policy
        state = np.reshape(state, [1, self.state_size])
        state = tf.convert_to_tensor(state)
        probs = self.network.call(state)
        # Filter valid actions and sample action from distribution
        valid = []
        for a in action_list:
            valid.append(probs[0][a])
        action_probabilities = tfp.distributions.Categorical(probs=valid, dtype=tf.float32)
        action = action_probabilities.sample()
        # return chosen action
        return action_list[int(action.numpy())]

    def act_test(self, state, action_list):

        state = self.preprocess(state)

        # Convert state and query network for probabilities
        state = np.reshape(state, [1, self.state_size])
        state = tf.convert_to_tensor(state, dtype=tf.float32)
        probs = self.network.call(state)
        # filter out invalid actions and create distribution
        valid = []
        for a in action_list:
            valid.append(probs[0][a])

        # store and return action and log_prob of action
        return action_list[np.argmax(valid)]

    def learn(self, traj, dr):
        observations = tf.convert_to_tensor(traj[0])
        actions = tf.convert_to_tensor(traj[1])

        with tf.GradientTape() as tape:
            # Calculate log_prob and loss for each action-state and related discounted reward
            log_prob = self.network.action_distribution(observations).log_prob(actions)
            loss = -tf.math.reduce_mean(log_prob * tf.cast(np.array(dr), tf.float32))
        # Fit the model
        grads = tape.gradient(loss, self.network.model.trainable_weights)
        self.opt.apply_gradients(zip(grads, self.network.model.trainable_weights))

    def load_model(self, name):
        self.network.built = True
        self.network.model.load_weights(name)

    def save_model(self, name):
        try:
            self.network.model.save_weights(name)
        except RuntimeError:
            print("Error: Exception when saving model weights")

    def get_returns(self, batch):
        # Calculate discounted reward for each episode in the batch
        all_returns = []
        for rewards in batch:
            returns = []
            reversed_rewards = np.flip(rewards, 0)
            g_t = 0
            for r in reversed_rewards:
                g_t = r + self.gamma * g_t
                returns.insert(0, g_t)
            all_returns.append(returns)
        returns = np.concatenate(all_returns)
        return returns
