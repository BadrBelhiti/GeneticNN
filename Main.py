import pygame
import numpy as np
import geneticnn as gnn
import simengine as eng
from simengine import Player

ai = True

op_ai_enabled = False
op_ai = Player((255, 255, 255)) if op_ai_enabled else None


best_total_fitness = 1
generation_score = 0


def next_round():
    global best_total_fitness, generation_score

    print(str(gnn.generation_num), 'completed.', 'Score:', generation_score)
    gnn.next_generation()

    best_total_fitness = 0
    generation_score = 0


gnn.create_players()

ticks = 0
running = True

while running:
    pygame.time.delay(50)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Calculate current fitness
    current_fitness = gnn.current_total_fitness()

    # Reset round
    if gnn.should_stop():
        next_round()

    best_total_fitness = max(gnn.current_total_fitness(), best_total_fitness)

    keys = pygame.key.get_pressed()

    if keys[pygame.K_ESCAPE]:
        running = False

    if keys[pygame.K_SPACE]:
        next_round()

    if keys[pygame.K_F1]:
        gnn.save_current_generation()

    if not ai:

        if keys[pygame.K_d]:
            gnn.players[0].right()

        if keys[pygame.K_a]:
            gnn.players[0].left()

        if keys[pygame.K_w]:
            gnn.players[0].up()

        if keys[pygame.K_s]:
            gnn.players[0].down()

    else:
        # '''

        if op_ai_enabled:
            op_ai_delta_x = eng.gold_x - op_ai.player_x
            op_ai_delta_y = eng.gold_y - op_ai.player_y

            if np.abs(op_ai_delta_x) > eng.gold_size - 1:
                op_ai.right() if op_ai_delta_x > 0 else op_ai.left()

            if np.abs(op_ai_delta_y) > eng.gold_size - 1:
                op_ai.down() if op_ai_delta_y > 0 else op_ai.up()

            if op_ai.check_collision():
                op_ai.increment_score()
                eng.spawn_gold()

        for i in range(len(gnn.players)):
            player = gnn.players[i]
            x = player.player_x
            y = player.player_y

            delta_x = eng.gold_x - x
            delta_y = eng.gold_y - y

            player.set_last_fitness()

            output = gnn.predict(0 if delta_x < 0 else 1, 0 if delta_y < 0 else 1, i)

            if output[0] > 0.5:
                player.right()

            if output[1] > 0.5:
                player.left()

            if output[2] > 0.5:
                player.up()

            if output[3] > 0.5:
                player.down()

            gnn.players[i].ensure_bounds()

            if gnn.players[i].check_collision():
                generation_score += 1
                player.increment_score()
                eng.spawn_gold()
        # '''

    eng.draw(generation_score)

    if op_ai_enabled:
        op_ai.draw()

    eng.update_display()
    ticks += 1

pygame.quit()
