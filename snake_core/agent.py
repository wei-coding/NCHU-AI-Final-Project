import tensorflow as tf
from collections import deque
from environment import SnakeGameAI, Direction, Point, BLOCK_SIZE
import numpy as np
import random
from model import QTrainer

MAX_MEM = 100_000
BATCH_SIZE = 1000
LR = 0.001


class Agent:

    def __init__(self):
        self.n_games = 0
        self.epsilion = 0
        self.gamma = 0.9
        self.memory = deque(maxlen=MAX_MEM)
        self.trainer = QTrainer(LR, 11, 256, 3, self.gamma)

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

            # move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,

            # food loc
            game.food.x < game.head.x,
            game.food.x > game.head.x,
            game.food.y < game.head.y,
            game.food.y > game.head.y
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
        self.trainer.train_step((state, ), (action,), (reward,), (next_state,), (done,))

    def get_action(self, state):
        # random move
        state = np.array(state)
        state = np.reshape(state, (1, 11))
        self.epsilion = 0.01
        final_move = [0, 0, 0]
        if random.random() <= self.epsilion:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            print(state)
            prediction = self.trainer.model.predict(state)
            move = np.argmax(prediction)
            final_move[move] = 1

        return final_move


def train():
    print(tf.test.is_gpu_available())
    record = 0
    with open('log.txt', 'w+') as f:
        try:
            record = int(f.readline().strip())
        except ValueError:
            print('log.txt not found.')
    agent = Agent()
    agent.trainer.load_model()
    game = SnakeGameAI()
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

            if score >= record:
                record = score
                with open('log.txt', 'w+') as f:
                    f.write(str(record))
                agent.trainer.save_model()

            print('Game', agent.n_games, 'Score', score, 'Record', record)


if __name__ == '__main__':
    train()
