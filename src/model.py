import tensorflow as tf
from tensorflow.keras.layers import Dense
import numpy as np


class QTrainer:
    def __init__(self, n_state, lr, input_size, hidden_size, output_size, gamma):
        self.n_state = n_state
        self.gamma = gamma
        self.model = tf.keras.Sequential()
        
        self.model.add(Dense(input_size, activation='relu', input_shape=(n_state, )))
        for layers in hidden_size:
            self.model.add(Dense(layers, activation='relu'))
        self.model.add(Dense(output_size, activation='linear'))

        self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr), loss='mse')

    def save_model(self, path='model.h5'):
        self.model.save_weights(path)

    def load_model(self, path='model.h5'):
        self.model.load_weights(path)

    def train_step(self, state, action, reward, next_state, done):
        state = np.array(state, dtype=np.float)
        action = np.array(action, dtype=np.float)
        reward = np.array(reward, dtype=np.float)
        next_state = np.array(next_state, dtype=np.float)
        done = np.array(done, dtype=np.float)

        # predicted Q values
        pred = self.model.predict(state)

        # Q_new = r + y * max(next_predicted Q value)
        target = np.copy(pred)
        for idx in range(done.shape[0]):
            Q_new = reward[idx]
            if not done[idx]:
                state_t = np.array(next_state[idx])
                Q_new = reward[idx] + self.gamma * np.max(self.model.predict(state_t))

            target[idx, 0, np.argmax(action)] = Q_new

        self.model.fit(np.reshape(state, (-1, self.n_state)), target, verbose=1)



