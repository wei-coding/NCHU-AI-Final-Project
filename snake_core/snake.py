import pygame
from pygame.locals import QUIT, KEYDOWN, K_UP, K_DOWN, K_RIGHT, K_LEFT
import sys, time, math
import random
RED = (255, 0, 0)
BLUE = (0, 0, 255)
GREEN = (0, 255, 0)
WINDOW_SIZE = (800, 600)
SNAKE_SIZE = 20


class Snake(pygame.sprite.Sprite):
    def __init__(self):
        super().__init__()
        self.body = [[WINDOW_SIZE[0] / 2, WINDOW_SIZE[1] / 2, SNAKE_SIZE, SNAKE_SIZE]]
        self.length = 1


def rand_pos():
    rx, ry = 0, 0
    while rx == 0 or ry == 0:
        rx = random.randint(0, WINDOW_SIZE[0] - SNAKE_SIZE)
        rx = rx // SNAKE_SIZE * SNAKE_SIZE
        ry = random.randint(0, WINDOW_SIZE[1] - SNAKE_SIZE)
        ry = ry // SNAKE_SIZE * SNAKE_SIZE
        print(rx, ry)
    return rx, ry, SNAKE_SIZE, SNAKE_SIZE


def main():
    random.seed(time.time())
    food = rand_pos()
    pygame.init()
    window_surface = pygame.display.set_mode(WINDOW_SIZE)
    pygame.display.set_caption('Snake game by AI team')
    game_over = False
    clock = pygame.time.Clock()
    snake = Snake()
    d_x, d_y = 0, 0
    while True:
        while game_over:
            window_surface.fill(BLUE)
            game_over_text = pygame.font.SysFont(None, 30)
            game_over_text_surface = game_over_text.render('Game Over!', True, RED)
            window_surface.blit(game_over_text_surface, (WINDOW_SIZE[0] / 2 - 60, WINDOW_SIZE[1] / 2))
            snake.kill()
            pygame.display.update()
            for event in pygame.event.get():
                if event.type == QUIT:
                    pygame.quit()
                    sys.exit()
                if event.type == KEYDOWN:
                    if event.key == pygame.K_q:
                        pygame.quit()
                        sys.exit()
                    if event.key == pygame.K_c:
                        main()
        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()
                sys.exit()
            if event.type == KEYDOWN:
                if event.key == K_UP:
                    d_x = 0
                    d_y = -SNAKE_SIZE
                elif event.key == K_DOWN:
                    d_x = 0
                    d_y = SNAKE_SIZE
                elif event.key == K_LEFT:
                    d_x = -SNAKE_SIZE
                    d_y = 0
                elif event.key == K_RIGHT:
                    d_x = SNAKE_SIZE
                    d_y = 0
        window_surface.fill((0, 0, 0))
        if food:
            pygame.draw.ellipse(window_surface, GREEN, food)
        else:
            food = rand_pos()
            pygame.draw.ellipse(window_surface, GREEN, food)
        now_pos = snake.body[0]
        snake.body.insert(0, [now_pos[0] + d_x, now_pos[1] + d_y, SNAKE_SIZE, SNAKE_SIZE])
        if len(snake.body) > snake.length:
            snake.body.pop(-1)
        for part in snake.body:
            pygame.draw.rect(window_surface, BLUE, part)
        pygame.draw.rect(window_surface, RED, snake.body[0])
        if snake.body[0][0] >= WINDOW_SIZE[0] or snake.body[0][0] <= 0\
                or snake.body[0][1] >= WINDOW_SIZE[1] or snake.body[0][1] <= 0:
            game_over = True
            game_over_text = pygame.font.SysFont(None, 30)
            game_over_text_surface = game_over_text.render('Game Over!', True, RED)
            window_surface.blit(game_over_text_surface, (WINDOW_SIZE[0]/2 - 60, WINDOW_SIZE[1]/2))
            snake.kill()
            pygame.display.update()
        if snake.body[0][0] == food[0] and snake.body[0][1] == food[1]:
            print('eat food')
            food = None
            snake.length += 1
        pygame.display.update()
        clock.tick(10)


if __name__ == "__main__":
    main()
