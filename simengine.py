import pygame
import numpy as np
import geneticnn as gnn

pygame.init()

win_width = 800
win_height = 600

win = pygame.display.set_mode((win_width, win_height))
pygame.display.set_caption('Genetic Neural Network')

player_size = 30
vel = 10

large_font = pygame.font.SysFont('Calibri', 36)
small_font = pygame.font.SysFont('Calibri', 18)

gold_size = 20

show_player_scores = True


class Player(object):

    centered = True

    def __init__(self, c):
        self.player_x = (win_width / 2 - player_size / 2) if Player.centered else np.random.randint(0, win_width - player_size)
        self.player_y = win_height / 2 - player_size / 2 if Player.centered else np.random.randint(0, win_height - player_size)
        self.color = c
        self.score = 0
        self.best_fitness = 0
        self.last_fitness = 0

    def check_collision(self):
        p = pygame.Rect(self.player_x, self.player_y, player_size, player_size)
        g = pygame.Rect(gold_x, gold_y, gold_size, gold_size)
        return p.colliderect(g)

    def left(self):
        self.player_x -= vel

    def right(self):
        self.player_x += vel

    def up(self):
        self.player_y -= vel

    def down(self):
        self.player_y += vel

    def ensure_bounds(self):
        self.player_x = np.clip(self.player_x, 0, win_width - player_size)
        self.player_y = np.clip(self.player_y, 0, win_height - player_size)

    def draw(self):
        pygame.draw.rect(win, self.color, (self.player_x, self.player_y, player_size, player_size))
        fitness = self.fitness()
        if fitness > self.best_fitness:
            self.best_fitness = fitness

    def increment_score(self):
        self.score += 1

    def fitness(self):
        fitness = 10000 * self.score
        distance_squared = (self.player_x + player_size * 0.5 - gold_x - gold_size * 0.5) ** 2 + \
                           (self.player_y + player_size * 0.5 - gold_y - gold_size * 0.5) ** 2
        return fitness + np.sqrt(win_width ** 2 + win_height ** 2 - distance_squared)

    def set_last_fitness(self):
        self.last_fitness = self.fitness()


def new_gold_location():
    return np.random.uniform(0, win_width - gold_size, 1), np.random.uniform(0, win_height - gold_size, 1)


(gold_x, gold_y) = new_gold_location()


def spawn_gold():
    global gold_x, gold_y
    (gold_x, gold_y) = new_gold_location()


def draw(generation_score):
    # Clear screen
    win.fill((0, 0, 0))

    # Render generational statistics
    text = large_font.render('Generation: ' + str(gnn.generation_num), True, (255, 255, 255))
    current_score = large_font.render('Current Score: ' + str(generation_score), True, (255, 255, 255))
    win.blit(text, (win_width / 2 - text.get_rect().width / 2 - 200, 10))
    win.blit(current_score, (win_width / 2 - current_score.get_rect().width / 2 + 200, 10))

    pygame.draw.rect(win, (0, 255, 255), (gold_x, gold_y, gold_size, gold_size))

    # Render players
    for i in range(len(gnn.players)):
        player = gnn.players[i]
        player.draw()

        # Render player score
        if show_player_scores:
            player_score = small_font.render(str(player.score), True, (255, 255, 255))
            win.blit(player_score,
                     (player.player_x + (player_size / 2 - player_score.get_rect().width / 2),
                      (player.player_y + (player_size / 2 - player_score.get_rect().height / 2))))


def update_display():
    pygame.display.update()


def shutdown():
    pygame.quit()
