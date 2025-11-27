# main.py  # ver4

import sys
import os
import random
import pickle
import pygame

from env_maze import MazeEnv, CELL_SIZE, GRID_W, GRID_H, UP, DOWN, LEFT, RIGHT  # ver4
from ai2_rules import rule_based_ai2  # ver4


# --- 화면 설정 ---  # ver4
SIDEBAR_W = 260
MAZE_PIX_W = GRID_W * CELL_SIZE
MAZE_PIX_H = GRID_H * CELL_SIZE
WIN_W = MAZE_PIX_W + SIDEBAR_W
WIN_H = MAZE_PIX_H

COLOR_SIDEBAR = (30, 30, 30)
COLOR_TEXT = (230, 230, 230)
COLOR_BTN_PAUSE = (150, 60, 60)
COLOR_BTN_RESUME = (60, 150, 60)

ACTIONS = [UP, DOWN, LEFT, RIGHT]  # ver4

# 학습된 AI1을 쓸지 여부 (u): True면 Q-policy, False면 랜덤  # ver4
USE_TRAINED_AI1 = True  # 나중에 필요하면 False로 바꾸면 됨  # ver4


def load_q_table(path: str):
    """학습된 Q-table 로드. 없으면 None."""  # ver4
    if not os.path.exists(path):
        print(f"[WARN] Q-table file not found: {path}. AI1 will use RANDOM policy.")
        return None
    with open(path, "rb") as f:
        Q = pickle.load(f)
    print(f"[INFO] Q-table loaded from {path}. Entries: {len(Q)}")
    return Q


def greedy_action_from_q(Q, state) -> int:
    """Q-table 기준으로 greedy 행동 선택. state = (x,y)."""  # ver4
    qs = [Q.get((state, a), 0.0) for a in ACTIONS]
    max_q = max(qs)
    candidates = [a for a, q in zip(ACTIONS, qs) if q == max_q]
    return random.choice(candidates)


def draw_sidebar(screen, font, info: dict, paused: bool,
                 pause_button_rect: pygame.Rect, use_trained: bool, q_loaded: bool):  # ver4
    sidebar_rect = pygame.Rect(MAZE_PIX_W, 0, SIDEBAR_W, WIN_H)
    pygame.draw.rect(screen, COLOR_SIDEBAR, sidebar_rect)

    x0 = MAZE_PIX_W + 20
    y = 20

    def line(text: str):
        nonlocal y
        surf = font.render(text, True, COLOR_TEXT)
        screen.blit(surf, (x0, y))
        y += 26

    line(f"Episode : {info.get('episode', 1)}")
    line(f"Step    : {info.get('step', 0)}")
    line(f"AI1 pos : {info.get('ai1_pos')}")
    line(f"AI2 pos : {info.get('ai2_pos')}")
    y += 10

    # AI1 모드 표시  # ver4
    if use_trained and q_loaded:
        line("AI1 mode: Q-learning")
    else:
        line("AI1 mode: RANDOM")

    y += 10
    line("[Controls]")
    line("P : Pause / Resume")
    line("ESC : Quit")

    # 버튼  # ver4
    btn_color = COLOR_BTN_RESUME if paused else COLOR_BTN_PAUSE
    pygame.draw.rect(screen, btn_color, pause_button_rect, border_radius=8)
    label = "Resume" if paused else "Pause"
    label_surf = font.render(label, True, (255, 255, 255))
    label_rect = label_surf.get_rect(center=pause_button_rect.center)
    screen.blit(label_surf, label_rect)


def random_action() -> int:
    return random.choice(ACTIONS)


def main():
    pygame.init()
    screen = pygame.display.set_mode((WIN_W, WIN_H))
    pygame.display.set_caption("Maze AI1 vs AI2 (UI + Pause + Q-policy)  ver4")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("consolas", 18)

    env = MazeEnv()
    obs1, obs2 = env.reset()

    # Q-table 로드 (있으면 사용)  # ver4
    Q = load_q_table("qtable.pkl") if USE_TRAINED_AI1 else None
    q_loaded = Q is not None

    episode = 1
    step = 0
    paused = False
    running = True

    pause_button_rect = pygame.Rect(MAZE_PIX_W + 40, 260, 140, 40)

    while running:
        # --- 이벤트 처리 ---  # ver4
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                break
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                    break
                if event.key == pygame.K_p:
                    paused = not paused
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if pause_button_rect.collidepoint(event.pos):
                    paused = not paused

        if not running:
            break

        if not paused:
            step += 1

            # --- AI1 행동 선택 ---  # ver4
            if USE_TRAINED_AI1 and q_loaded:
                state = env.ai1_pos
                action1 = greedy_action_from_q(Q, state)
            else:
                action1 = random_action()

            # --- AI2: 규칙 기반 ---  # ver4
            obs2_dict = {
                "ai2_pos": env.ai2_pos,
                "ai1_pos": env.ai1_pos,
            }
            action2 = rule_based_ai2(obs2_dict, env, eps=0.1)

            (obs1, obs2), done, info = env.step(action1, action2)

            if done:
                episode += 1
                step = 0
                obs1, obs2 = env.reset()

        # --- 렌더링 ---  # ver4
        env.render(screen)

        info_dict = {
            "episode": episode,
            "step": step,
            "ai1_pos": env.ai1_pos,
            "ai2_pos": env.ai2_pos,
        }
        draw_sidebar(screen, font, info_dict, paused,
                     pause_button_rect, USE_TRAINED_AI1, q_loaded)

        pygame.display.flip()
        clock.tick(10)  # (u) FPS

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()
