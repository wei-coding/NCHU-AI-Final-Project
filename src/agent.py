import tensorflow as tf
from collections import deque
from environment import SnakeGameAI, Direction, Point, BLOCK_SIZE
import numpy as np
import random
from model import QTrainer
from mylogging import *
import os

MAX_MEM = 100_000
BATCH_SIZE = 10000
LR = 1e-5


class Agent:

    def __init__(self):
        self.n_games = 0
        self.n_state = 14
        self.frame_to_read = 1
        self.epsilon = 0.4
        self.gamma = 0.8
        self.memory = deque(maxlen=MAX_MEM)
        self.states = deque(maxlen=self.frame_to_read)
        for _ in range(self.frame_to_read):
            self.states.append([0 for _ in range(self.n_state)])
        self.trainer = QTrainer(self.n_state * self.frame_to_read, LR, self.n_state * self.frame_to_read, [256, 256], 3, self.gamma)

    def get_state(self, game):
        head = game.snake[0]
        point_l = Point(head.x - BLOCK_SIZE, head.y)
        point_r = Point(head.x + BLOCK_SIZE, head.y)
        point_u = Point(head.x, head.y - BLOCK_SIZE)
        point_d = Point(head.x, head.y + BLOCK_SIZE)

        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        wall_u = 1 / abs(game.head.y + 21)
        wall_d = 1 / abs(game.h - game.head.y + 1)
        wall_l = 1 / abs(game.head.x + 21)
        wall_r = 1 / abs(game.w - game.head.x + 1)

        food_u = 1 if game.food.y < game.head.y else 0
        food_d = 1 if game.food.y > game.head.y else 0
        food_l = 1 if game.food.x < game.head.x else 0
        food_r = 1 if game.food.x > game.head.x else 0

        self_s = float('inf')
        self_l = float('inf')
        self_r = float('inf')
        for i, b in enumerate(game.snake):
            if i == 0: continue
            if dir_l:
                if b.x <= game.head.x and b.y == game.head.y:
                    self_s = min(self_s, game.head.x - b.x)
                if b.y <= game.head.y and b.x == game.head.x:
                    self_r = min(self_r, game.head.y - b.y)
                if b.y >= game.head.y and b.x == game.head.x:
                    self_l = min(self_l, b.y - game.head.y)
            if dir_r:
                if b.x >= game.head.x and b.y == game.head.y:
                    self_s = min(self_s, b.x - game.head.x)
                if b.y >= game.head.y and b.x == game.head.x:
                    self_r = min(self_r, b.y - game.head.y)
                if b.y <= game.head.y and b.x == game.head.x:
                    self_l = min(self_l, game.head.y - b.y)
            if dir_u:
                if b.y <= game.head.y and b.x == game.head.x:
                    self_s = min(self_s, game.head.y - b.y)
                if b.x >= game.head.x and b.y == game.head.y:
                    self_r = min(self_r, b.x - game.head.x)
                if b.x <= game.head.x and b.y == game.head.y:
                    self_l = min(self_l, game.head.x - b.x)
            if dir_d:
                if b.y >= game.head.y and b.x == game.head.x:
                    self_s = min(self_s, b.y - game.head.y)
                if b.x <= game.head.x and b.y == game.head.y:
                    self_r = min(self_r, game.head.x - b.x)
                if b.x >= game.head.x and b.y == game.head.y:
                    self_l = min(self_l, b.x - game.head.x)

        self_s = 1 / (self_s + 1)
        self_r = 1 / (self_r + 1)
        self_l = 1 / (self_l + 1)

        state = [
            # danger straight
            (dir_r and game.is_collision(point_r)) or
            (dir_l and game.is_collision(point_l)) or
            (dir_u and game.is_collision(point_u)) or
            (dir_d and game.is_collision(point_d)),

            # danger right
            (dir_u and game.is_collision(point_r)) or
            (dir_d and game.is_collision(point_l)) or
            (dir_l and game.is_collision(point_u)) or
            (dir_r and game.is_collision(point_d)),

            # danger left
            (dir_d and game.is_collision(point_r)) or
            (dir_u and game.is_collision(point_l)) or
            (dir_r and game.is_collision(point_u)) or
            (dir_l and game.is_collision(point_d)),

            # to body coefficient
            self_s,
            self_r,
            self_l,

            # move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,

            # food loc
            game.food.x < game.head.x,
            game.food.x > game.head.x,
            game.food.y < game.head.y,
            game.food.y > game.head.y,
        ]
        print(state)
        self.states.append(state)

        return np.array(list(self.states), dtype=np.float32).reshape((-1, self.n_state * self.frame_to_read))

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
        final_move = [0, 0, 0]
        if random.random() < self.epsilon:
            print('random behavior')
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            prediction = self.trainer.model.predict(state)
            move = np.argmax(prediction)
            final_move[move] = 1

        return final_move


def train(model_name='model.h5', log_name='log.csv'):

    print(tf.test.is_gpu_available())
    min_ep = 0.02
    agent = Agent()
    game = SnakeGameAI()
    record, n_games = load_logs(log_name)
    if record is None:
        record = 0
        n_games = 0
    else:
        agent.n_games = n_games
    tot_score = 0
    if os.path.exists(model_name):
        print('*'*10 + '\nload model...\n' + '*'*10)
        agent.trainer.load_model(model_name)
        agent.epsilon = min_ep
    new_model = True
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
            if new_model:
                agent.epsilon *= 0.99
            if agent.epsilon < min_ep:
                agent.epsilon = min_ep

            if score > record:
                record = score
                agent.trainer.save_model(f'{model_name}')
            if agent.n_games % 10 == 0:
                agent.trainer.save_model(f'{model_name}')
            save_logs(record, score, agent.n_games, log_name)

            print('*'*10)
            print('Game', agent.n_games, 'Score', score, 'Record', record)
            print('AVG score', tot_score / agent.n_games)
            print('*'*10)


if __name__ == '__main__':
    filename = 'model5'
    train(model_name=f'models/{filename}.h5', log_name=f'logs/{filename}.csv')
