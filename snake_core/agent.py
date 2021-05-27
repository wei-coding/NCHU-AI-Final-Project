import numpy as np
import random
import time
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Embedding, Reshape, Dense


class Agent:
    def __init__(self, env):
        # init attr
        self._state_size = 12  # (up, down, right, left) * (wall, food, self's body)
        self._action_size = 4  # up, down, right ,left
        self._optimizer = 'Adamax'

        self.exp_replay = list()

        # hyper parameter
        self.gamma = 0.7
        self.epsilon = 0.2

        # Build networks
        self.q_network = self._build_compiled_model()
        self.target_network = self._build_compiled_model()
        self.alighn_target_model()

    def store(self, state, action, reward, next_state, terminated):
        if len(self.exp_replay) > 200:
            self.exp_replay.pop(0)
        self.exp_replay.append((state, action, reward, next_state, terminated))

    def _build_compiled_model(self):
        model = Sequential()
        model.add(Dense(self._state_size, input_shape=(12, 1)))
        model.add(Dense(50, activation='relu'))
        model.add(Dense(50, activation='relu'))
        model.add(Dense(self._action_size, activation='linear'))

        model.summary()
        model.compile(loss='mse', optimizer=self._optimizer)
        return model

    def alighn_target_model(self):
        self.target_network.set_weights(self.q_network.get_weights())

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(0, 4)

        q_values = self.q_network.predict(state)
        return np.argmax(q_values[0])

    def retrain(self, batch_size):
        minibatch = random.sample(self.exp_replay, batch_size)

        for state, action, reward, next_state, terminated in minibatch:
            target = self.q_network.predict(state)
            if terminated:
                target[action] = reward
            else:
                t = self.target_network.predict(next_state)
                target[action] = reward + self.gamma * np.amax(t)

            print('training...')
            self.q_network.fit(state, target, epochs=1, verbose=0)
