import tensorflow as tf
from collections import deque
from environment import SnakeGameAI, Direction, Point, BLOCK_SIZE
import numpy as np
import random
from model import QTrainer

MAX_MEM = 100_000
BATCH_SIZE = 500
LR = 0.001


class Agent:

    def __init__(self):
        self.n_games = 0
        self.n_state = 11
        self.epsilon = 0.1
        self.gamma = 0.8
        self.memory = deque(maxlen=MAX_MEM)
        self.trainer = QTrainer(self.n_state, LR, self.n_state, [256, 512, 256], 3, self.gamma)

    def get_state(self, game):
        head = game.snake[0]
        point_l = Point(head.x - BLOCK_SIZE, head.y)
        point_r = Point(head.x + BLOCK_SIZE, head.y)
        point_u = Point(head.x, head.y - BLOCK_SIZE)
        point_d = Point(head.x, head.y + BLOCK_SIZE)

        point_l_2 = Point(head.x - 2 * BLOCK_SIZE, head.y)
        point_r_2 = Point(head.x + 2 * BLOCK_SIZE, head.y)
        point_u_2 = Point(head.x, head.y - 2 * BLOCK_SIZE)
        point_d_2 = Point(head.x, head.y + 2 * BLOCK_SIZE)

        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        state = [
            # danger straight
            (dir_r and game.is_collision(point_r)) or
            (dir_l and game.is_collision(point_l)) or
            (dir_u and game.is_collision(point_u)) or
            (dir_d and game.is_collision(point_d)),

            # danger straight 2 step
            (dir_r and game.is_collision(point_r_2)) or
            (dir_l and game.is_collision(point_l_2)) or
            (dir_u and game.is_collision(point_u_2)) or
            (dir_d and game.is_collision(point_d_2)),

            # danger right
            (dir_u and game.is_collision(point_r)) or
            (dir_d and game.is_collision(point_l)) or
            (dir_l and game.is_collision(point_u)) or
            (dir_r and game.is_collision(point_d)),

            # danger right 2 step
            (dir_u and game.is_collision(point_r_2)) or
            (dir_d and game.is_collision(point_l_2)) or
            (dir_l and game.is_collision(point_u_2)) or
            (dir_r and game.is_collision(point_d_2)),

            # danger left
            (dir_d and game.is_collision(point_r)) or
            (dir_u and game.is_collision(point_l)) or
            (dir_r and game.is_collision(point_u)) or
            (dir_l and game.is_collision(point_d)),

            # danger left 2 step
            (dir_d and game.is_collision(point_r_2)) or
            (dir_u and game.is_collision(point_l_2)) or
            (dir_r and game.is_collision(point_u_2)) or
            (dir_l and game.is_collision(point_d_2)),

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

        return np.array(state, dtype=np.uint8)

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
        state = np.array(state)
        state = np.reshape(state, (1, self.n_state))
        final_move = [0, 0, 0]
        epsilon = self.epsilon * 0.99
        self.epsilon = epsilon
        if random.random() < epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            prediction = self.trainer.model.predict(state)
            move = np.argmax(prediction)
            final_move[move] = 1

        return final_move


def train():
    tot_score = 0
    print(tf.test.is_gpu_available())
    record = 0
    agent = Agent()
    game = SnakeGameAI()
    # agent.trainer.load_model('model_11state_2hidden_260.h5')
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

            if score > record:
                record = score
                agent.trainer.save_model(f'model_11state_2hidden_{record}.h5')

            print('*'*10)
            print('Game', agent.n_games, 'Score', score, 'Record', record)
            print('AVG score', tot_score / agent.n_games)
            print('*'*10)


if __name__ == '__main__':
    train()
