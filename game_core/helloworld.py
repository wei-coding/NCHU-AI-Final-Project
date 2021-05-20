import sys
import pygame
from pygame.locals import QUIT

pygame.init()
window_surface = pygame.display.set_mode((800, 600))
pygame.display.set_caption('Hello World:>')
window_surface.fill((255, 255, 255))

head_font = pygame.font.SysFont(None, 60)
text_surface = head_font.render('Hello World!', True, (0, 0, 0))
window_surface.blit(text_surface, (10, 10))

pygame.display.update()

while True:
    for event in pygame.event.get():
        if event.type == QUIT:
            pygame.quit()
            sys.exit()
