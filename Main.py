import pygame
import tensorflow as tf
import numpy as np
from pathlib import Path
import time
from datetime import datetime

pool_size = 20

current_pool = []


def create_model():
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(3, input_shape=(3,), activation='relu'))
    model.add(tf.keras.layers.Dense(4, activation='sigmoid'))
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    weights = model.get_weights()

    for i in range(len(weights)):
        for j in range(len(weights[i])):
            weights[i][j] = np.random.uniform(-1, 1)

    model.set_weights(weights)

    return model


def predict(dx, dy, model_num):
    neural_input = np.asarray([dx, dy, (dx ** 2 + dy ** 2) / 2])
    neural_input = np.atleast_2d(neural_input)

    all_output = current_pool[model_num].predict(neural_input)[0]
    return all_output


def model_crossover_weights(weight1, weight2):
    new_weight1 = weight1
    new_weight2 = weight2

    gene = np.random.randint(0, len(weight1))

    new_weight1[gene] = weight2[gene]
    new_weight2[gene] = weight1[gene]

    return np.asarray([new_weight1, new_weight2])


def model_crossover(parent1, parent2):
    global current_pool

    weight1 = current_pool[parent1].get_weights()
    weight2 = current_pool[parent2].get_weights()

    return model_crossover_weights(weight1, weight2)


def model_mutate(weights):

    for i in range(len(weights)):
        for j in range(len(weights[i])):
            if np.random.uniform(0, 1) < 0.15:
                change = np.random.uniform(-0.5, 0.5)
                weights[i][j] += change

    return weights


for i in range(pool_size):
    current_pool.append(create_model())


pygame.init()

win_width = 800
win_height = 600

win = pygame.display.set_mode((win_width, win_height))
pygame.display.set_caption('Genetic Neural Network')


player_size = 30
vel = 10

score = 0
font = pygame.font.SysFont('Calibri', 36)

gold_size = 20

ai = True


class Player(object):

    def __init__(self, c):
        self.player_x = win_width / 2 - player_size / 2
        self.player_y = win_height / 2 - player_size / 2
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
        fitness = 100000 * self.score
        distance_squared = (self.player_x + player_size * 0.5 - gold_x - gold_size * 0.5) ** 2 + (self.player_y + player_size * 0.5 - gold_y - gold_size * 0.5) ** 2
        return fitness + np.sqrt(win_width ** 2 + win_height ** 2 - distance_squared)

    def set_last_fitness(self):
        self.last_fitness = self.fitness()


def new_gold_location():
    return np.random.uniform(0, win_width - gold_size, 1), np.random.uniform(0, win_height - gold_size, 1)


(gold_x, gold_y) = new_gold_location()


def spawn_gold():
    (x, y) = new_gold_location()
    global gold_x, gold_y
    gold_x = x
    gold_y = y


players = []

op_ai_enabled = False
op_ai = Player((255, 255, 255)) if op_ai_enabled else None


def create_players():
    players.clear()
    for i in range(pool_size):
        color = np.random.uniform(0, 255, 3)
        players.append(Player(color))

    if op_ai_enabled:
        global op_ai
        op_ai = Player((255, 255, 255))


def select_parent(fitness, median):
    fitness_sum = np.sum(fitness)
    curr_lower = 0
    select = np.random.uniform(0, 1)

    for i in range(pool_size):
        if fitness[i] >= median:
            curr_fitness = fitness[i] / fitness_sum
            if curr_lower <= select < curr_lower + curr_fitness:
                return current_pool[i]
            curr_lower += curr_fitness


best_weights_ever = []
best_fitness_ever = 0
best_model_ever = None


def reset_round():
    fitness = []

    global current_pool, best_fitness_ever, best_weights_ever, best_model_ever

    for i in range(pool_size):
        fitness_score = players[i].fitness()
        fitness.append(fitness_score)
        if fitness_score > best_fitness_ever:
            best_fitness_ever = fitness_score
            best_weights_ever = current_pool[i].get_weights()
            best_model_ever = current_pool[i]

    median = np.median(fitness)

    '''
    print('fitness:', fitness)
    print('fit count:', fit_count)
    print('median:', median)
    print('max:', max_fitness)
    print('best score', best_fitness_ever)
    '''

    new_generation = current_pool.copy()

    for i in range(pool_size):

        parent1 = select_parent(fitness, median)
        parent2 = select_parent(fitness, median)

        if parent1 is None:
            parent1 = current_pool[np.random.randint(0, pool_size)]

        if parent2 is None:
            parent2 = current_pool[np.random.randint(0, pool_size)]

        new_weights = model_mutate(model_crossover_weights(parent1.get_weights(), parent2.get_weights())[0])
        new_generation[i].set_weights(new_weights)

    new_generation[np.random.randint(0, pool_size)].set_weights(best_weights_ever)
    current_pool = new_generation
    create_players()


round_num = 1
generation_score = 0


def next_round():
    global round_num
    reset_round()
    print(str(round_num), 'completed.', 'Score:', generation_score)
    round_num += 1


def current_total_fitness():
    total_fitness = 0
    for i in range(pool_size):
        total_fitness += players[i].fitness()

    return total_fitness


def should_stop():
    progress = False
    for i in range(pool_size):
        player = players[i]
        last_fitness = player.last_fitness
        curr_fitness = player.fitness()
        if curr_fitness > last_fitness:
            progress = True
            break
    return not progress


create_players()

best_total_fitness = 1
ticks = 0
running = True

while running:
    pygame.time.delay(50)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Calculate current fitness
    current_fitness = current_total_fitness()

    # Reset round
    if should_stop():
        next_round()
        best_total_fitness = 0
        generation_score = 0

    best_total_fitness = max(current_total_fitness(), best_total_fitness)

    keys = pygame.key.get_pressed()

    if keys[pygame.K_ESCAPE]:
        running = False

    if keys[pygame.K_SPACE]:
        next_round()
        best_total_fitness = 0
        generation_score = 0

    if keys[pygame.K_F1]:
        print('Saving current generation...')
        path = 'models/' + str(datetime.fromtimestamp(time.time())) + '/generation' + str(round_num)
        Path(path).mkdir(parents=True, exist_ok=True)
        for i in range(pool_size):
            model = current_pool[i]
            model.save(path + '/' + str(i) + '.h5')
            if best_model_ever is not None:
                best_model_ever.save(path + '/best.h5')
        print('Current generation saved successfully')

    if not ai:

        if keys[pygame.K_d]:
            players[0].right()

        if keys[pygame.K_a]:
            players[0].left()

        if keys[pygame.K_w]:
            players[0].up()

        if keys[pygame.K_s]:
            players[0].down()

    else:
        # '''

        if op_ai_enabled:
            op_ai_delta_x = gold_x - op_ai.player_x
            op_ai_delta_y = gold_y - op_ai.player_y

            if np.abs(op_ai_delta_x) > gold_size - 1:
                op_ai.right() if op_ai_delta_x > 0 else op_ai.left()

            if np.abs(op_ai_delta_y) > gold_size - 1:
                op_ai.down() if op_ai_delta_y > 0 else op_ai.up()

            if op_ai.check_collision():
                op_ai.increment_score()
                spawn_gold()

        for i in range(len(players)):
            player = players[i]
            x = player.player_x
            y = player.player_y

            delta_x = gold_x - x
            delta_y = gold_y - y

            player.set_last_fitness()

            output = predict(0 if delta_x < 0 else 1, 0 if delta_y < 0 else 1, i)

            if output[0] > 0.5:
                player.right()

            if output[1] > 0.5:
                player.left()

            if output[2] > 0.5:
                player.up()

            if output[3] > 0.5:
                player.down()

            players[i].ensure_bounds()

            if players[i].check_collision():
                generation_score += 1
                player.increment_score()
                spawn_gold()
        # '''

    win.fill((0, 0, 0))
    text = font.render('Generation: ' + str(round_num), False, (255, 255, 255))
    score = font.render('Current Score: ' + str(generation_score), False, (255, 255, 255))
    win.blit(text, (win_width / 2 - text.get_rect().width / 2 - 200, 10))
    win.blit(score, (win_width / 2 - score.get_rect().width / 2 + 200, 10))
    pygame.draw.rect(win, (0, 255, 255), (gold_x, gold_y, gold_size, gold_size))

    for i in range(len(players)):
        players[i].draw()

    if op_ai_enabled:
        op_ai.draw()

    pygame.display.update()
    ticks += 1

pygame.quit()