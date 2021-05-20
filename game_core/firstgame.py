import pygame
from pygame.locals import QUIT, MOUSEBUTTONDOWN, USEREVENT
import sys, time
import random

WINDOW_WIDTH = 800
WINDOW_HEIGHT = 600
WHITE = (255, 255, 255)
IMAGEWIDTH = 300
IMAGEHEIGHT = 200
FPS = 60


class Mosquito(pygame.sprite.Sprite):
    def __init__(self, width, height, window_width, window_height, random_x, randox_y):
        super().__init__()
        self.raw_image = pygame.image.load('./mosquito.png').convert_alpha()
        self.image = pygame.transform.scale(self.raw_image, (width, height))
        self.rect = self.image.get_rect()

        self.rect.topleft = (random_x, randox_y)
        self.width = width
        self.height = height
        self.window_width = window_width
        self.window_height = window_height


def rand_pos(window_width, window_height, image_width, image_height):
    r_x = random.randint(0, window_width - image_height)
    r_y = random.randint(0, window_height - image_height)
    return r_x, r_y


def main():
    pygame.init()
    random.seed(time.time())
    window_surface = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption('Mosquito War')
    rx, ry = rand_pos(WINDOW_WIDTH, WINDOW_HEIGHT, IMAGEWIDTH, IMAGEHEIGHT)
    mosquito = Mosquito(IMAGEWIDTH, IMAGEHEIGHT, WINDOW_WIDTH, WINDOW_HEIGHT, rx, ry)
    reload = USEREVENT + 1
    pygame.time.set_timer(reload, 60)
    clear_hit = USEREVENT + 2
    pygame.time.set_timer(clear_hit, 2000)
    points = 0
    score_text = pygame.font.SysFont(None, 30)
    hit_text = pygame.font.SysFont(None, 40)
    hit_text_surface = None
    main_clock = pygame.time.Clock()

    while True:
        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == reload:
                mosquito.kill()
                rx, ry = rand_pos(WINDOW_WIDTH, WINDOW_HEIGHT, IMAGEWIDTH, IMAGEHEIGHT)
                mosquito = Mosquito(IMAGEWIDTH, IMAGEHEIGHT, WINDOW_WIDTH, WINDOW_HEIGHT, rx, ry)
            elif event.type == MOUSEBUTTONDOWN:
                mouse_pos = pygame.mouse.get_pos()
                if (mouse_pos[0] < rx + IMAGEWIDTH and mouse_pos[0] > rx) and (mouse_pos[1] < ry + IMAGEHEIGHT and mouse_pos[1] > ry):
                    mosquito.kill()
                    rx, ry = rand_pos(WINDOW_WIDTH, WINDOW_HEIGHT, IMAGEWIDTH, IMAGEHEIGHT)
                    mosquito = Mosquito(IMAGEWIDTH, IMAGEHEIGHT, WINDOW_WIDTH, WINDOW_HEIGHT, rx, ry)
                    hit_text_surface = hit_text.render('Hit!', True, (255, 0, 0))
                    points += 5
            elif event.type == clear_hit:
                hit_text_surface = None

        window_surface.fill(WHITE)

        text_surface = score_text.render(f'Points: {points}', True, (0, 0, 0))
        window_surface.blit(mosquito.image, mosquito.rect)
        window_surface.blit(text_surface, (10, 0))

        if hit_text_surface:
            window_surface.blit(hit_text_surface, (10, 20))

        pygame.display.update()
        main_clock.tick(FPS)


if __name__ == "__main__":
    main()
