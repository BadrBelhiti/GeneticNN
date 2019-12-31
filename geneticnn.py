import tensorflow as tf
import numpy as np
from simengine import Player
from pathlib import Path
from datetime import datetime
import time


pool_size = 20
current_pool = []

players = []


def create_model():
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(2, input_shape=(2,), activation='relu'))
    model.add(tf.keras.layers.Dense(4, activation='sigmoid'))
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # weights = model.get_weights()

    '''
    for i in range(len(weights)):
        for j in range(len(weights[i])):
            weights[i][j] = np.random.uniform(-1, 1)
    '''

    # model.set_weights(weights)

    return model


def predict(dx, dy, model_num):
    neural_input = np.asarray([dx, dy])
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

generation_num = 1


def next_generation():
    fitness = []

    global current_pool, best_fitness_ever, best_weights_ever, best_model_ever, generation_num

    for i in range(pool_size):
        fitness_score = players[i].fitness()
        fitness.append(fitness_score)
        if fitness_score > best_fitness_ever:
            best_fitness_ever = fitness_score
            best_weights_ever = current_pool[i].get_weights()
            best_model_ever = current_pool[i]

    median = np.median(fitness)

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
    generation_num += 1


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


def create_players():
    players.clear()
    for i in range(pool_size):
        color = np.random.uniform(0, 255, 3)
        players.append(Player(color))


def save_current_generation():
    print('Saving current generation...')
    path = 'models/' + str(datetime.fromtimestamp(time.time())) + '/generation' + str(generation_num)
    Path(path).mkdir(parents=True, exist_ok=True)
    for i in range(pool_size):
        model = current_pool[i]
        model.save(path + '/' + str(i) + '.h5')
        if best_model_ever is not None:
            best_model_ever.save(path + '/best.h5')
    print('Current generation saved successfully')