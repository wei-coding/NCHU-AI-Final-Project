import tensorflow as tf
from collections import deque
from environment import SnakeGameAI, Direction, Point, BLOCK_SIZE
import numpy as np
import random
from model import QTrainer
import mylogging
import time

MAX_MEM = 100_000
BATCH_SIZE = 500
LR = 0.001


class Agent:

    def __init__(self):
        self.n_games = 0
        self.n_state = (640 // BLOCK_SIZE) * (480 // BLOCK_SIZE)
        self.frame_to_read = 1
        self.epsilon = 0.3
        self.gamma = 0.8
        self.memory = deque(maxlen=MAX_MEM)
        self.states = np.zeros((3, 32, 24), dtype=np.float32)
        self.trainer = QTrainer(self.n_state * self.frame_to_read, LR, self.n_state * self.frame_to_read, [256, 512, 256], 3, self.gamma)
        random.seed(time.time())

    def get_state(self, game):
        state = np.zeros((32, 24), dtype=np.float32)
        for body in game.snake:
            try:
                state[int(body.x // BLOCK_SIZE)][int(body.y // BLOCK_SIZE)] = 1
            except IndexError:
                pass
        state[int(game.food.x // BLOCK_SIZE)][int(game.food.y // BLOCK_SIZE)] = 2
        state = np.array(state, dtype=np.float32)
        temp = np.copy(self.states)
        temp[0, :, :] = self.states[1, :, :]
        temp[1, :, :] = self.states[2, :, :]
        temp[2, :, :] = state
        self.states = temp

        return np.transpose(temp, (1, 2, 0))

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            minibatch = random.sample(self.memory, BATCH_SIZE)
        else:
            minibatch = self.memory

        states, actions, rewards, next_states, dones = zip(*minibatch)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        print('reward =', reward)
        self.trainer.train_step((state, ), (action,), (reward,), (next_state,), (done,))

    def get_action(self, state):
        # random move
        state = np.reshape(state, (-1, 640 // BLOCK_SIZE, 480 // BLOCK_SIZE, 3))
        final_move = [0, 0, 0]
        if random.random() < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
            print('random move')
        else:
            prediction = self.trainer.model.predict(state)
            move = np.argmax(prediction)
            final_move[move] = 1

        return final_move


def train(record=0, n_games=0, filename='logs.csv'):
    tot_score = 0
    print(tf.test.is_gpu_available())
    agent = Agent()
    game = SnakeGameAI()
    # agent.trainer.model.predict(np.zeros((1, agent.n_state * agent.frame_to_read)))
    # agent.trainer.load_model('models/model_11state_1hidden_noreward_30_19.h5')
    agent.n_games = n_games
    while True:
        # get old state
        state_old = agent.get_state(game)

        # get move
        final_move = agent.get_action(state_old)

        # perform move and get new state
        reward, done, score = game.play_step(final_move)
        state_new = agent.get_state(game)

        # train short memory
        agent.train_short_memory(state_old, final_move, reward, state_new, done)

        # remember
        agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            # train long memory
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()
            tot_score += score
            agent.epsilon *= 0.99

            if score > record:
                record = score
                agent.trainer.save_model(f'model_{record}_{agent.n_games}.h5')
            if agent.n_games % 10 == 0:
                agent.trainer.save_model(f'model_{record}_{agent.n_games}.h5')
            mylogging.save_logs(record, score, agent.n_games, filename)

            print('*'*10)
            print('Game', agent.n_games, 'Score', score, 'Record', record)
            print('AVG score', tot_score / agent.n_games)
            print('*'*10)


if __name__ == '__main__':
    mylogging.save_logs(0, 0, 0, init=True, filename='logs_fullstate.csv')
    # record, n_games = mylogging.load_logs(filename='logs/logs_11state_1hidden_noreward.csv')
    train(record=0, n_games=0, filename='logs_fullstate.csv')
