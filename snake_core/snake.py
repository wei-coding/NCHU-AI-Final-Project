import random
import sys
import time
import numpy as np

import pygame
from pygame.locals import QUIT, KEYDOWN, K_UP, K_DOWN, K_RIGHT, K_LEFT

from helper import Action

RED = (255, 0, 0)
BLUE = (0, 0, 255)
GREEN = (0, 255, 0)
WINDOW_SIZE = (800, 800)
SNAKE_SIZE = 20


class Snake(pygame.sprite.Sprite):
    def __init__(self):
        super().__init__()
        self.body = [[WINDOW_SIZE[0] // 2, WINDOW_SIZE[1] // 2, SNAKE_SIZE, SNAKE_SIZE]]
        self.length = 1


class Game:
    def __init__(self, ai=False):
        self.snake = Snake()
        pygame.init()
        random.seed(time.time())
        self.window_surface = pygame.display.set_mode(WINDOW_SIZE)
        pygame.display.set_caption('Snake game by AI team')
        self.clock = pygame.time.Clock()

        self.food = self.rand_pos()
        self.game_over = False
        self.d_x, self.d_y = 0, 0
        self.refresh_food = False
        self.score = 200
        self.new_score = self.score
        self.reward = 0
        self.action = Action.NONE
        self.ai = ai

    def reset(self):
        self.snake = Snake()
        self.food = (WINDOW_SIZE[0] // 2, WINDOW_SIZE[1] // 2 - SNAKE_SIZE, SNAKE_SIZE, SNAKE_SIZE)
        self.game_over = False
        self.d_x, self.d_y = 0, 0
        self.refresh_food = False
        self.score = 200
        self.new_score = self.score
        self.reward = 0
        self.action = Action.NONE
        return self.step()

    def start(self):
        self.gameloop()

    def gameloop(self):
        while True:
            self.move_reset()
            # Movement
            # for human(Keyboard)
            for event in pygame.event.get():
                if event.type == QUIT:
                    pygame.quit()
                    sys.exit()
                if event.type == KEYDOWN:
                    if event.key == K_UP:
                        self.d_x = 0
                        self.d_y = -SNAKE_SIZE
                    elif event.key == K_DOWN:
                        self.d_x = 0
                        self.d_y = SNAKE_SIZE
                    elif event.key == K_LEFT:
                        self.d_x = -SNAKE_SIZE
                        self.d_y = 0
                    elif event.key == K_RIGHT:
                        self.d_x = SNAKE_SIZE
                        self.d_y = 0

            # Calculate reward
            if not self.ai:
                self.new_score -= 1
                self.reward = self.new_score - self.score
                self.score = self.new_score

            # ai movement
            if self.ai:
                if self.action != Action.NONE:
                    # Calculate reward
                    self.new_score -= 1
                    self.reward = self.new_score - self.score
                    self.score = self.new_score

                    # do action
                    if self.action == Action.UP:
                        self.move_u()
                    elif self.action == Action.DOWN:
                        self.move_d()
                    elif self.action == Action.RIGHT:
                        self.move_r()
                    elif self.action == Action.LEFT:
                        self.move_l()
                    self.action = Action.NONE


            # Refresh screen
            self.window_surface.fill((0, 0, 0))

            # Draw food
            if not self.refresh_food:
                pygame.draw.ellipse(self.window_surface, GREEN, self.food)
            else:
                self.food = self.rand_pos()
                pygame.draw.ellipse(self.window_surface, GREEN, self.food)
                self.refresh_food = False

            # Draw snake's body
            now_pos = self.snake.body[0]
            self.snake.body.insert(0, [now_pos[0] + self.d_x, now_pos[1] + self.d_y, SNAKE_SIZE, SNAKE_SIZE])
            if len(self.snake.body) > self.snake.length:
                self.snake.body.pop(-1)
            for part in self.snake.body:
                pygame.draw.rect(self.window_surface, BLUE, part)
                pygame.draw.rect(self.window_surface, (60, 106, 255),
                                 [part[0] + 5, part[1] + 5, SNAKE_SIZE - 10, SNAKE_SIZE - 10])

            # See if snake touch the wall
            if self.snake.body[0][0] > WINDOW_SIZE[0] or self.snake.body[0][0] < 0 \
                    or self.snake.body[0][1] > WINDOW_SIZE[1] or self.snake.body[0][1] < 0:
                self.game_over = True
                game_over_text = pygame.font.SysFont(None, 30)
                game_over_text_surface = game_over_text.render('Game Over!', True, RED)
                self.window_surface.blit(game_over_text_surface, (WINDOW_SIZE[0] / 2 - 60, WINDOW_SIZE[1] / 2))
                self.snake.kill()
                pygame.display.update()
                self.new_score -= 100000

            # When snake eats
            if self.snake.body[0][0] == self.food[0] and self.snake.body[0][1] == self.food[1]:
                self.refresh_food = True
                self.snake.length += 1
                self.new_score += 30

            # When snake touch it's body
            for i in range(1, len(self.snake.body)):
                if self.snake.body[0] == self.snake.body[i]:
                    self.game_over = True
                    self.new_score -= 100000

            # Write down score
            score_text = pygame.font.SysFont(None, 30)
            score_text_surface = score_text.render(f'score: {self.score}', True, (255, 255, 255))
            self.window_surface.blit(score_text_surface, (10, 10))
            pygame.display.update()

            if self.game_over:
                self.game_over = False
                pygame.event.pump()
                self.reset()
                print('----------------new game----------------------')

            # Refresh rate
            self.clock.tick(10)

    def step(self, action=Action.NONE):
        # do action
        self.action = action
        # return [distance to wall, distance to food, distance to self(min)],
        # reward
        # gameover
        head = (self.snake.body[0][0], self.snake.body[0][1])
        body_u, body_d, body_r, body_l = float('inf'), float('inf'), float('inf'), float('inf')
        for body in self.snake.body:
            if body == self.snake.body[0]:
                continue
            if body[0] == head[0]:
                if body[1] - head[1] > 0:
                    body_u = min(body_u, body[1] - head[1])
                else:
                    body_d = min(body_d, head[1] - body[1])
            elif body[1] == head[1]:
                if body[0] - head[0] > 0:
                    body_r = min(body_r, body[0] - head[0])
                else:
                    body_l = min(body_l, head[0] - body[0])
        state = [
            # distance to wall
            # up
            head[1] / WINDOW_SIZE[0],
            # down
            (WINDOW_SIZE[1] - head[1]) / WINDOW_SIZE[0],
            # left
            head[0] / WINDOW_SIZE[0],
            # right
            (WINDOW_SIZE[0] - head[0]) / WINDOW_SIZE[0],

            # distance to food
            # up
            (self.secure(head[1] - self.food[1])) / WINDOW_SIZE[0],
            # down
            (self.secure(self.food[1] - head[1])) / WINDOW_SIZE[0],
            # right
            (self.secure(self.food[0] - head[0])) / WINDOW_SIZE[0],
            # left
            (self.secure(head[0] - self.food[0])) / WINDOW_SIZE[0],

            # distance to self
            # up
            self.secure(body_u) / WINDOW_SIZE[0],
            # down
            self.secure(body_d) / WINDOW_SIZE[0],
            # right
            self.secure(body_r) / WINDOW_SIZE[0],
            # left
            self.secure(body_l) / WINDOW_SIZE[0]
        ]
        return np.array(state, dtype=np.float), self.reward, self.game_over

    def move_u(self):
        self.d_x, self.d_y = 0, -SNAKE_SIZE

    def move_d(self):
        self.d_x, self.d_y = 0, SNAKE_SIZE

    def move_r(self):
        self.d_x, self.d_y = SNAKE_SIZE, 0

    def move_l(self):
        self.d_x, self.d_y = -SNAKE_SIZE, 0

    def move_reset(self):
        self.d_x, self.d_y = 0, 0

    @staticmethod
    def secure(x):
        if x < 0 or x == float('inf'):
            return 0
        else:
            return x

    def rand_pos(self):
        rx = random.randint(0, WINDOW_SIZE[0] - SNAKE_SIZE)
        rx = rx // SNAKE_SIZE * SNAKE_SIZE
        ry = random.randint(0, WINDOW_SIZE[1] - SNAKE_SIZE)
        ry = ry // SNAKE_SIZE * SNAKE_SIZE
        for body in self.snake.body:
            while rx == body[0] and ry == body[1]:
                rx = random.randint(0, WINDOW_SIZE[0] - SNAKE_SIZE)
                rx = rx // SNAKE_SIZE * SNAKE_SIZE
                ry = random.randint(0, WINDOW_SIZE[1] - SNAKE_SIZE)
                ry = ry // SNAKE_SIZE * SNAKE_SIZE
        return rx, ry, SNAKE_SIZE, SNAKE_SIZE


def main():
    game = Game(ai=True)
    game.start()


if __name__ == "__main__":
    main()
