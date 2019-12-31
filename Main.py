import pygame
import numpy as np
import geneticnn as gnn
import simengine as eng
from simengine import Player

op_ai_enabled = False
op_ai = Player((200, 200, 200)) if op_ai_enabled else None

best_total_fitness = 1
generation_score = 0

# gnn.load_generation('models/2019-12-30 21:52:52.233808/generation14', 20)
# gnn.load_player('models/2019-12-30 21:53:20.474513/generation15/best.h5')


def next_round():
    global best_total_fitness, generation_score

    print(str(gnn.generation_num), 'completed.', 'Score:', generation_score)
    gnn.next_generation()

    best_total_fitness = 0
    generation_score = 0


def act(player_num):
    curr_player = gnn.players[player_num]
    x = curr_player.player_x
    y = curr_player.player_y

    delta_x = eng.gold_x - x
    delta_y = eng.gold_y - y

    curr_player.set_last_fitness()

    output = gnn.predict(0 if delta_x < 0 else 1, 0 if delta_y < 0 else 1, player_num)

    if output[0] > 0.5:
        curr_player.right()

    if output[1] > 0.5:
        curr_player.left()

    if output[2] > 0.5:
        curr_player.up()

    if output[3] > 0.5:
        curr_player.down()


if not gnn.model_specified:
    gnn.create_players()

ticks = 0
running = True

while running:
    pygame.time.delay(50)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Reset round
    if not gnn.model_specified and gnn.should_stop():
        next_round()

    # Handle 'quit', 'skip', and 'save' keyboard shortcuts
    keys = pygame.key.get_pressed()

    if keys[pygame.K_ESCAPE]:
        running = False

    if keys[pygame.K_SPACE] and not gnn.model_specified:
        next_round()

    if keys[pygame.K_F1]:
        gnn.save_current_generation()

    if op_ai_enabled:
        op_ai_delta_x = eng.gold_x - op_ai.player_x
        op_ai_delta_y = eng.gold_y - op_ai.player_y

        # non-ML AI
        if np.abs(op_ai_delta_x) > eng.gold_size - 1:
            op_ai.right() if op_ai_delta_x > 0 else op_ai.left()

        if np.abs(op_ai_delta_y) > eng.gold_size - 1:
            op_ai.down() if op_ai_delta_y > 0 else op_ai.up()

        if op_ai.check_collision():
            op_ai.increment_score()
            eng.spawn_gold()

    for i in range(len(gnn.players)):
        player = gnn.players[i]

        act(i)

        player.ensure_bounds()

        if player.check_collision():
            generation_score += 1
            player.increment_score()
            eng.spawn_gold()

    eng.draw(generation_score)

    if op_ai_enabled:
        op_ai.draw()

    eng.update_display()
    ticks += 1

eng.shutdown()
