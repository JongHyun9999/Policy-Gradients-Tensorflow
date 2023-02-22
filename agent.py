import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow import keras
import numpy as np
from network import *


class Agent:
    def __init__(self, alpha=0.003, gamma=0.99, n_actions=4, n_obs=4,
                 neurons1=256, neurons2=256):
        self.gamma = gamma
        self.alpha = alpha
        self.n_actions = n_actions
        self.state_memory = []
        self.action_memory = []
        self.reward_memory = []
        self.policy = PolicyGradientNetwork(n_actions, n_obs, neurons1, neurons2).NeuralNetwork
        self.policy.compile(optimizer=keras.optimizers.Adam(learning_rate=alpha))
        
    def choose_action(self, observation):
        state = tf.convert_to_tensor([observation], dtype=tf.float32)
        probs = self.policy(state)
        distribuition_actions = tfp.distributions.Categorical(probs=probs)
        action_choosed = distribuition_actions.sample()
        return action_choosed.numpy()[0]

    def store_transition(self, observation, action, reward):
        self.state_memory.append(observation)
        self.action_memory.append(action)
        self.reward_memory.append(reward)
    
    def learn(self):
        actions = tf.convert_to_tensor(self.action_memory, dtype=tf.float32)
        reward = tf.convert_to_tensor(self.reward_memory)
        
        G = np.zeros_like(reward)
        for t in range(len(reward)):
            G_sum = 0
            discount = 1
            for k in range(t, len(reward)):
                G_sum += reward[k] * discount
                discount *= self.gamma
            G[t] = G_sum
        with tf.GradientTape() as tape:
            loss = 0
            for idx, (g, state) in enumerate(zip(G, self.state_memory)):
                state = tf.convert_to_tensor([state], dtype=tf.float32)
                probs = self.policy(state)
                action_probs = tfp.distributions.Categorical(probs=probs)
                log_prob = action_probs.log_prob(actions[idx])
                loss += - g * tf.squeeze(log_prob)
            gradient = tape.gradient(loss, self.policy.trainable_variables)
            self.policy.optimizer.apply_gradients(zip(gradient, self.policy.trainable_variables))
            self.state_memory = []
            self.action_memory = []
            self.reward_memory = []
    
    def load_model(self, path):
        self.policy = keras.models.load_model(path)
    def save_model(self, path):
        self.policy.save(path)
        
        

