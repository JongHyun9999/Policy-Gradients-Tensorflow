# I will create a agent class that will represent a agent in the environment on reinforcement learning framework
from tensorflow import keras


class PolicyGradientNetwork(keras.Model):
    def __init__(self, n_actions, n_obs, neurons1=256, neurons2=256):
        super(PolicyGradientNetwork, self).__init__()
        self.neurons1 = neurons1
        self.neurons2 = neurons2
        self.n_actions = n_actions 
        self.NeuralNetwork = keras.models.Sequential([
                                keras.layers.Dense(neurons1, activation='relu', input_shape=[n_obs]),
                                keras.layers.Dense(neurons2, activation='relu'),
                                keras.layers.Dense(2, activation='softmax'),
                            ])
    
    def __call__(self, state):
        return self.NeuralNetwork(state)