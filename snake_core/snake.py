import random
import sys
import time
import numpy as np

import pygame
from pygame.locals import QUIT, KEYDOWN, K_UP, K_DOWN, K_RIGHT, K_LEFT

from agent import AI

RED = (255, 0, 0)
BLUE = (0, 0, 255)
GREEN = (0, 255, 0)
WINDOW_SIZE = (800, 600)
SNAKE_SIZE = 20


class Snake(pygame.sprite.Sprite):
    def __init__(self):
        super().__init__()
        self.body = [[WINDOW_SIZE[0] // 2, WINDOW_SIZE[1] // 2, SNAKE_SIZE, SNAKE_SIZE]]
        self.length = 1


class Game:
    def __init__(self, ai=None):
        self.ai = None
        if ai:
            self.ai = ai
        self.difficulty = 5
        random.seed(time.time())
        self.food = rand_pos()
        pygame.init()
        self.window_surface = pygame.display.set_mode(WINDOW_SIZE)
        pygame.display.set_caption('Snake game by AI team')
        self.game_over = False
        self.clock = pygame.time.Clock()
        self.snake = Snake()
        self.hunger_event = pygame.USEREVENT + 1
        self.hunger_value = 100
        self.d_x, self.d_y = 0, 0
        self.refresh_food = False

    def start(self):
        self.gameloop()

    def gameloop(self):
        while True:
            while self.game_over:
                self.window_surface.fill(BLUE)
                game_over_text = pygame.font.SysFont(None, 30)
                game_over_text_surface = game_over_text.render('Game Over!', True, RED)
                self.window_surface.blit(game_over_text_surface, (WINDOW_SIZE[0] / 2 - 60, WINDOW_SIZE[1] / 2))
                self.snake.kill()
                pygame.display.update()
                for event in pygame.event.get():
                    if event.type == QUIT:
                        pygame.quit()
                        sys.exit()
            for event in pygame.event.get():
                if event.type == QUIT:
                    pygame.quit()
                    sys.exit()
                if event.type == KEYDOWN:
                    set_hunger_timer(self.hunger_event)
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
                if event.type == self.hunger_event:
                    self.hunger_value -= self.difficulty
                    # if self.hunger_value < 0:
                    #     self.game_over = True
            if self.ai:
                move = self.ai.get_movement()
                if move == 'u':
                    self.move_u()
                elif move == 'd':
                    self.move_d()
                elif move == 'r':
                    self.move_r()
                elif move == 'l':
                    self.move_l()
                else:
                    raise Exception('Wrong movement string.')
            self.window_surface.fill((0, 0, 0))
            if not self.refresh_food:
                pygame.draw.ellipse(self.window_surface, GREEN, self.food)
            else:
                self.food = rand_pos()
                pygame.draw.ellipse(self.window_surface, GREEN, self.food)
                self.refresh_food = False
            now_pos = self.snake.body[0]
            self.snake.body.insert(0, [now_pos[0] + self.d_x, now_pos[1] + self.d_y, SNAKE_SIZE, SNAKE_SIZE])
            if len(self.snake.body) > self.snake.length:
                self.snake.body.pop(-1)
            for part in self.snake.body:
                pygame.draw.rect(self.window_surface, BLUE, part)
                pygame.draw.rect(self.window_surface, (60, 106, 255),
                                 [part[0] + 5, part[1] + 5, SNAKE_SIZE - 10, SNAKE_SIZE - 10])
            if self.snake.body[0][0] >= WINDOW_SIZE[0] or self.snake.body[0][0] <= 0 \
                    or self.snake.body[0][1] >= WINDOW_SIZE[1] or self.snake.body[0][1] <= 0:
                self.game_over = True
                game_over_text = pygame.font.SysFont(None, 30)
                game_over_text_surface = game_over_text.render('Game Over!', True, RED)
                self.window_surface.blit(game_over_text_surface, (WINDOW_SIZE[0] / 2 - 60, WINDOW_SIZE[1] / 2))
                self.snake.kill()
                pygame.display.update()
            if self.snake.body[0][0] == self.food[0] and self.snake.body[0][1] == self.food[1]:
                print('eat food')
                self.refresh_food = True
                self.snake.length += 1
                self.hunger_value += self.difficulty
            for i in range(1, len(self.snake.body)):
                if self.snake.body[0] == self.snake.body[i]:
                    self.game_over = True
            hunger_text = pygame.font.SysFont(None, 30)
            hunger_text_surface = hunger_text.render(f'hunger: {self.hunger_value}', True, (255, 255, 255))
            self.window_surface.blit(hunger_text_surface, (10, 10))
            pygame.display.update()
            print(self.get_state())
            self.clock.tick(5)

    def get_state(self):
        # return distance to wall, distance to food distance to self(min)
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
            head[1],
            # down
            WINDOW_SIZE[1] - head[1],
            # left
            head[0],
            # right
            WINDOW_SIZE[0] - head[0],

            # distance to food
            # up
            self.secure(head[1] - self.food[1]),
            # down
            self.secure(self.food[1] - head[1]),
            # right
            self.secure(self.food[0] - head[0]),
            # left
            self.secure(head[0] - self.food[0]),

            # distance to self
            # up
            self.secure(body_u),
            # down
            self.secure(body_d),
            # right
            self.secure(body_r),
            # left
            self.secure(body_l)
        ]
        return np.array(state, dtype=np.float)

    def move_u(self):
        self.d_x, self.d_y = 0, -SNAKE_SIZE

    def move_d(self):
        self.d_x, self.d_y = 0, SNAKE_SIZE

    def move_r(self):
        self.d_x, self.d_y = SNAKE_SIZE, 0

    def move_l(self):
        self.d_x, self.d_y = -SNAKE_SIZE, 0

    @staticmethod
    def secure(x):
        if x < 0 or x == float('inf'):
            return 0
        else:
            return x


def rand_pos():
    rx, ry = 0, 0
    while rx == 0 or ry == 0:
        rx = random.randint(0, WINDOW_SIZE[0] - SNAKE_SIZE)
        rx = rx // SNAKE_SIZE * SNAKE_SIZE
        ry = random.randint(0, WINDOW_SIZE[1] - SNAKE_SIZE)
        ry = ry // SNAKE_SIZE * SNAKE_SIZE
        print(rx, ry)
    return rx, ry, SNAKE_SIZE, SNAKE_SIZE


is_event_set = False


def set_hunger_timer(event):
    global is_event_set
    if not is_event_set:
        pygame.time.set_timer(event, 1000)
        is_event_set = True


def main():
    game = Game()
    game.start()


if __name__ == "__main__":
    main()
