from snake import Game
from helper import *
from agent import Agent
import numpy as np
import threading

EPISODES = 500
TIMESTEP_PER_EPISODES = 300
BATCH_SIZE = 40

env = Game(ai=True)
agent = Agent(env)

def training():
    for e in range(EPISODES):
        state, _, _ = env.reset()

        state = np.reshape(state, (-1, 1))

        for timestep in range(TIMESTEP_PER_EPISODES):
            reward = 0
            terminated = False

            # run action
            action = agent.act(state)
            print('action=', action)

            # take action
            next_state, reward, terminated = env.step(action)
            next_state = np.reshape(next_state, (-1, 1))
            agent.store(state, action, reward, next_state, terminated)

            state = next_state

            if terminated:
                agent.retrain(BATCH_SIZE)
                agent.alighn_target_model()
                break

            # if len(agent.exp_replay) > BATCH_SIZE:
            #     agent.retrain(BATCH_SIZE)

        if (e + 1) % 10 == 0:
            print('*' * 10)
            print(f'Episode: {e + 1}')
            print('*' * 10)


def main():
    t_training = threading.Thread(target=training)
    t_training.start()
    env.start()
    t_training.join()


if __name__ == "__main__":
    main()
