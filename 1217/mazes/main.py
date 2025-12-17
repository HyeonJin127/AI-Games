# main.py  # ver5  (AI vs PLAYER: AI1=Q-learning, AI2=PLAYER)

import sys
import os
import random
from typing import Tuple, Dict, Any

import pygame
from dotenv import load_dotenv
from supabase import create_client, Client

from env_maze import MazeEnv, CELL_SIZE, GRID_W, GRID_H, UP, DOWN, LEFT, RIGHT
from ai2_rules import rule_based_ai2


# --- 화면/색상 ---
SIDEBAR_W = 320
HUD_H = 64

MAZE_PIX_W = GRID_W * CELL_SIZE
MAZE_PIX_H = GRID_H * CELL_SIZE

WIN_W = MAZE_PIX_W + SIDEBAR_W
WIN_H = MAZE_PIX_H + HUD_H

COLOR_SIDEBAR = (30, 30, 30)
COLOR_TEXT = (230, 230, 230)

COLOR_BTN_PAUSE = (150, 60, 60)
COLOR_BTN_RESUME = (60, 150, 60)

COLOR_HUD_BG = (18, 18, 18)
COLOR_SLOT = (200, 200, 200)

COLOR_AI1_LABEL = (200, 80, 80)
COLOR_AI2_LABEL = (80, 120, 220)

ACTIONS = [UP, DOWN, LEFT, RIGHT]

STATE_TITLE = "TITLE"
STATE_DIFFICULTY = "DIFFICULTY"
STATE_GAME = "GAME"

MODE_AI_VS_AI = "AI_VS_AI"
MODE_AI_VS_PLAYER = "AI_VS_PLAYER"

USE_TRAINED_AI1 = True

Action = int
State = Tuple[int, int]

DIFFICULTY_SETTINGS: Dict[str, float] = {
    "easy": 0.7,
    "normal": 0.3,
    "hard": 0.05,
}


# --- Supabase ---
def init_supabase_client() -> Client | None:
    load_dotenv()
    url: str | None = os.environ.get("SUPABASE_URL")
    key: str | None = os.environ.get("SUPABASE_ANON_KEY")
    if not url or not key:
        print("[ERROR] SUPABASE_URL 또는 SUPABASE_ANON_KEY 환경 변수가 설정되지 않았습니다.")
        return None
    return create_client(url, key)


def load_q_table(table_name: str = "q_table_maze_v4") -> Dict[Tuple[State, Action], float] | None:
    Q: Dict[Tuple[State, Action], float] = {}
    try:
        supabase = init_supabase_client()
        if supabase is None:
            return None

        response = supabase.table(table_name).select("*").execute()
        if not response.data:
            print(f"[WARN] No data found in Supabase table: {table_name}.")
            return None

        for row in response.data:
            state = (row["state_x"], row["state_y"])
            action = row["action"]
            q_value = row["q_value"]
            Q[(state, action)] = q_value

        print(f"[INFO] Q-table loaded from Supabase. Entries: {len(Q)}")
        return Q

    except Exception as e:
        print(f"[ERROR] Supabase Q-table loading failed: {e}. AI will use fallback policy.")
        return None


# --- 정책 ---
def choose_action_with_difficulty(Q: Dict[Tuple[State, Action], float] | None, state: State, epsilon: float) -> Action:
    if Q is None or random.random() < epsilon:
        return random.choice(ACTIONS)
    qs = [Q.get((state, a), 0.0) for a in ACTIONS]
    max_q = max(qs)
    candidates = [a for a, q in zip(ACTIONS, qs) if q == max_q]
    return random.choice(candidates)


def random_action() -> int:
    return random.choice(ACTIONS)


# --- 버튼 UI 유틸 ---
def draw_centered_button(screen: pygame.Surface, rect: pygame.Rect, text: str, font: pygame.font.Font, hover: bool = False):
    color = (80, 150, 230) if hover else (60, 120, 200)
    pygame.draw.rect(screen, color, rect, border_radius=8)
    surf = font.render(text, True, COLOR_TEXT)
    srect = surf.get_rect(center=rect.center)
    screen.blit(surf, srect)


def build_vertical_buttons(center_x: int, center_y: int, labels: list[str]) -> Dict[str, pygame.Rect]:
    btn_w, btn_h = 260, 56
    spacing = 26
    total_h = len(labels) * btn_h + (len(labels) - 1) * spacing
    start_y = center_y - total_h // 2
    rects: Dict[str, pygame.Rect] = {}
    for i, label in enumerate(labels):
        r = pygame.Rect(0, 0, btn_w, btn_h)
        r.centerx = center_x
        r.y = start_y + i * (btn_h + spacing)
        rects[label] = r
    return rects


def draw_title_screen(screen: pygame.Surface, font_big: pygame.font.Font, font_small: pygame.font.Font, btn_rects: Dict[str, pygame.Rect]):
    screen.fill((10, 10, 30))
    title_y = int(WIN_H * 0.18)
    title_surf = font_big.render("Maze RL Battle", True, COLOR_TEXT)
    title_rect = title_surf.get_rect(center=(WIN_W // 2, title_y))
    screen.blit(title_surf, title_rect)

    sub_surf = font_small.render("AI vs AI / AI vs PLAYER", True, COLOR_TEXT)
    sub_rect = sub_surf.get_rect(center=(WIN_W // 2, title_y + 32))
    screen.blit(sub_surf, sub_rect)

    mx, my = pygame.mouse.get_pos()
    for label, rect in btn_rects.items():
        draw_centered_button(screen, rect, label, font_small, rect.collidepoint(mx, my))


def draw_difficulty_screen(screen: pygame.Surface, font_big: pygame.font.Font, font_small: pygame.font.Font, btn_rects: Dict[str, pygame.Rect], selected_mode: str | None):
    screen.fill((10, 10, 30))
    title_y = int(WIN_H * 0.18)
    title_surf = font_big.render("Select Difficulty", True, COLOR_TEXT)
    title_rect = title_surf.get_rect(center=(WIN_W // 2, title_y))
    screen.blit(title_surf, title_rect)

    mode_surf = font_small.render(f"Mode: {selected_mode or 'N/A'}", True, COLOR_TEXT)
    mode_rect = mode_surf.get_rect(center=(WIN_W // 2, title_y + 32))
    screen.blit(mode_surf, mode_rect)

    mx, my = pygame.mouse.get_pos()
    for label, rect in btn_rects.items():
        draw_centered_button(screen, rect, label, font_small, rect.collidepoint(mx, my))


# --- 사이드바 ---
def draw_sidebar(screen: pygame.Surface, font: pygame.font.Font, maze_width: int, info_dict: Dict[str, Any], paused: bool, fast_mode: bool, skip_rounds: int) -> pygame.Rect:
    sidebar_rect = pygame.Rect(maze_width, 0, SIDEBAR_W, MAZE_PIX_H)
    screen.fill(COLOR_SIDEBAR, sidebar_rect)

    start_x = maze_width + 18
    line_y = 14

    def line(text: str):
        nonlocal line_y
        surf = font.render(text, True, COLOR_TEXT)
        screen.blit(surf, (start_x, line_y))
        line_y += 17

    line("Maze Q-Learning Info")
    line_y += 6
    line(f"Episode : {info_dict.get('episode', 1)}")
    line(f"Step    : {info_dict.get('step', 0)}")
    line(f"Remain  : {info_dict.get('turn_remaining', 0)}")
    line_y += 6
    line(f"AI1 Pos : {info_dict.get('ai1_pos')}")
    line(f"AI2 Pos : {info_dict.get('ai2_pos')}")
    line_y += 6
    line(f"AI1 Score : {info_dict.get('score_ai1', 0)}")
    line(f"AI2 Score : {info_dict.get('score_ai2', 0)}")
    line(f"Coins(AI1): {info_dict.get('items_ai1', 0)}")
    line(f"Coins(AI2): {info_dict.get('items_ai2', 0)}")
    line_y += 6
    line(f"AI1 Mode  : {info_dict.get('ai1_mode', 'N/A')}")
    line(f"Difficulty: {info_dict.get('difficulty', 'UNKNOWN')}")
    line_y += 6
    line(f"Fast Mode : {'ON' if fast_mode else 'OFF'}")
    line(f"Skip Rnds : {skip_rounds}")
    line_y += 10
    line("[Controls]")
    line("P:Pause / Q:Quit / ESC:Title")
    line("F,M,N: Fast/Skip30/Next (AIvsAI)")
    line("WASD/Arrows: Move (PLAYER=AI2)")
    line("E: Place Trap (PLAYER=AI2)")

    btn_w, btn_h = 100, 40
    btn_x = maze_width + (SIDEBAR_W - btn_w) // 2
    btn_y = MAZE_PIX_H - 60
    pause_button_rect = pygame.Rect(btn_x, btn_y, btn_w, btn_h)

    btn_color = COLOR_BTN_RESUME if paused else COLOR_BTN_PAUSE
    btn_text = "RESUME" if paused else "PAUSE"
    pygame.draw.rect(screen, btn_color, pause_button_rect, border_radius=5)
    text_surface = font.render(btn_text, True, COLOR_TEXT)
    screen.blit(text_surface, text_surface.get_rect(center=pause_button_rect.center))
    return pause_button_rect


# --- 하단 HUD ---
def draw_item_slot(surface: pygame.Surface, rect: pygame.Rect, icon: pygame.Surface | None):
    pygame.draw.rect(surface, COLOR_SLOT, rect, 2, border_radius=6)
    if icon is not None:
        ir = icon.get_rect(center=rect.center)
        surface.blit(icon, ir)


def draw_bottom_hud(
    screen: pygame.Surface,
    font: pygame.font.Font,
    ai1_label: str,
    ai2_label: str,
    ai1_icon_trap: pygame.Surface | None,
    ai1_icon_tranq: pygame.Surface | None,
    ai2_icon_trap: pygame.Surface | None,
    ai2_icon_tranq: pygame.Surface | None,
):
    hud_rect = pygame.Rect(0, MAZE_PIX_H, WIN_W, HUD_H)
    pygame.draw.rect(screen, COLOR_HUD_BG, hud_rect)

    pad = 12
    slot = 44
    gap = 10

    # AI1 라벨(왼쪽)
    label1 = font.render(ai1_label, True, COLOR_AI1_LABEL)
    screen.blit(label1, (pad, MAZE_PIX_H + 8))

    s1_trap = pygame.Rect(pad, MAZE_PIX_H + 28, slot, slot)
    s1_tranq = pygame.Rect(pad + slot + gap, MAZE_PIX_H + 28, slot, slot)
    draw_item_slot(screen, s1_trap, ai1_icon_trap)
    draw_item_slot(screen, s1_tranq, ai1_icon_tranq)

    # AI2 라벨(오른쪽)
    label2 = font.render(ai2_label, True, COLOR_AI2_LABEL)
    label2_rect = label2.get_rect(topright=(WIN_W - pad, MAZE_PIX_H + 8))
    screen.blit(label2, label2_rect)

    s2_tranq = pygame.Rect(WIN_W - pad - slot, MAZE_PIX_H + 28, slot, slot)
    s2_trap = pygame.Rect(WIN_W - pad - (slot * 2) - gap, MAZE_PIX_H + 28, slot, slot)
    draw_item_slot(screen, s2_trap, ai2_icon_trap)
    draw_item_slot(screen, s2_tranq, ai2_icon_tranq)


def decide_trap_use_ai1(env: MazeEnv) -> bool:
    """AI1 자동 트랩 사용(간단): 보유 중이고 AI2가 인접하면 설치."""
    if not getattr(env, "has_trap_ai1", False):
        return False
    if getattr(env, "trap_positions", {}).get(env.ai1_pos) is not None:
        return False
    ax, ay = env.ai1_pos
    bx, by = env.ai2_pos
    return abs(ax - bx) + abs(ay - by) == 1


def main():
    pygame.init()
    screen = pygame.display.set_mode((WIN_W, WIN_H))
    pygame.display.set_caption("Maze RL Battle ver5")

    font_small = pygame.font.Font(None, 24)
    font_big = pygame.font.Font(None, 40)

    clock = pygame.time.Clock()

    title_btn_rects = build_vertical_buttons(
        center_x=WIN_W // 2,
        center_y=int(WIN_H * 0.6),
        labels=["AI vs AI", "AI vs PLAYER", "QUIT"],
    )
    diff_btn_rects = build_vertical_buttons(
        center_x=WIN_W // 2,
        center_y=int(WIN_H * 0.6),
        labels=["EASY", "NORMAL", "HARD", "BACK"],
    )

    game_state = STATE_TITLE
    selected_mode: str | None = None
    selected_difficulty: str = "normal"

    env = MazeEnv()
    obs1, obs2 = env.reset()

    Q: Dict[Tuple[State, Action], float] | None = None
    q_loaded = False
    epsilon = DIFFICULTY_SETTINGS["normal"]

    fast_mode = False
    skip_rounds = 0
    render_enabled = True

    fps_normal = 10
    fps_fast = 1000

    episode = 1
    step = 0
    paused = False
    running = True

    # 플레이어는 AI2
    pending_player_move_ai2: Action | None = None
    pending_player_place_trap_ai2 = False

    ai1_mode_text = "N/A"
    pause_button_rect: pygame.Rect | None = None

    while running:
        # ---- 이벤트 처리 ----
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                break

                # ESC: GAME -> TITLE 복귀
                if event.key == pygame.K_ESCAPE:
                    if game_state == STATE_GAME:
                        game_state = STATE_TITLE
                        paused = False
                        fast_mode = False
                        skip_rounds = 0
                        render_enabled = True
                        pending_player_move_ai2 = None
                        pending_player_place_trap_ai2 = False
                    continue

            if not running:
                break

            # TITLE
            if game_state == STATE_TITLE:
                if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    mx, my = event.pos
                    if title_btn_rects["AI vs AI"].collidepoint(mx, my):
                        selected_mode = MODE_AI_VS_AI
                        game_state = STATE_DIFFICULTY
                    elif title_btn_rects["AI vs PLAYER"].collidepoint(mx, my):
                        selected_mode = MODE_AI_VS_PLAYER
                        game_state = STATE_DIFFICULTY
                    elif title_btn_rects["QUIT"].collidepoint(mx, my):
                        running = False
                        break
                continue

            # DIFFICULTY
            if game_state == STATE_DIFFICULTY:
                if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    mx, my = event.pos

                    def enter_game(difficulty: str):
                        nonlocal env, obs1, obs2, epsilon, Q, q_loaded, episode, step, paused, fast_mode, skip_rounds, render_enabled, selected_difficulty
                        selected_difficulty = difficulty
                        env = MazeEnv()
                        obs1, obs2 = env.reset()
                        epsilon = DIFFICULTY_SETTINGS[selected_difficulty]
                        Q = load_q_table()
                        q_loaded = Q is not None
                        episode = 1
                        step = 0
                        paused = False
                        fast_mode = False
                        skip_rounds = 0
                        render_enabled = True
                        return STATE_GAME

                    if diff_btn_rects["EASY"].collidepoint(mx, my):
                        game_state = enter_game("easy")
                    elif diff_btn_rects["NORMAL"].collidepoint(mx, my):
                        game_state = enter_game("normal")
                    elif diff_btn_rects["HARD"].collidepoint(mx, my):
                        game_state = enter_game("hard")
                    elif diff_btn_rects["BACK"].collidepoint(mx, my):
                        game_state = STATE_TITLE
                continue

            # GAME
            if game_state == STATE_GAME:
                if event.type == pygame.KEYDOWN:
                    # pause
                    if event.key == pygame.K_p:
                        paused = not paused

                    # AI vs AI: fast/skip/next
                    if selected_mode == MODE_AI_VS_AI:
                        if event.key == pygame.K_f:
                            fast_mode = not fast_mode
                        elif event.key == pygame.K_m:
                            skip_rounds += 30
                            render_enabled = False
                            fast_mode = True
                        elif event.key == pygame.K_n:
                            episode += 1
                            step = 0
                            obs1, obs2 = env.reset()

                    # AI vs PLAYER: player controls AI2
                    elif selected_mode == MODE_AI_VS_PLAYER:
                        if event.key in (pygame.K_w, pygame.K_UP):
                            pending_player_move_ai2 = UP
                        elif event.key in (pygame.K_s, pygame.K_DOWN):
                            pending_player_move_ai2 = DOWN
                        elif event.key in (pygame.K_a, pygame.K_LEFT):
                            pending_player_move_ai2 = LEFT
                        elif event.key in (pygame.K_d, pygame.K_RIGHT):
                            pending_player_move_ai2 = RIGHT
                        elif event.key == pygame.K_e:
                            pending_player_place_trap_ai2 = True

                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    if pause_button_rect is not None and pause_button_rect.collidepoint(event.pos):
                        paused = not paused

        if not running:
            break

        # ---- 상태별 렌더/업데이트 ----
        if game_state == STATE_TITLE:
            draw_title_screen(screen, font_big, font_small, title_btn_rects)
            pygame.display.flip()
            clock.tick(60)
            continue

        if game_state == STATE_DIFFICULTY:
            draw_difficulty_screen(screen, font_big, font_small, diff_btn_rects, selected_mode)
            pygame.display.flip()
            clock.tick(60)
            continue

        # GAME 업데이트
        if game_state == STATE_GAME:
            if not paused:
                # --- AI vs AI ---
                if selected_mode == MODE_AI_VS_AI:
                    step += 1

                    # AI1
                    if USE_TRAINED_AI1 and q_loaded and Q is not None:
                        state_ai1: State = env.ai1_pos
                        action1 = choose_action_with_difficulty(Q, state_ai1, epsilon)
                        ai1_mode_text = "Q-learning"
                    else:
                        action1 = random_action()
                        ai1_mode_text = "RANDOM"

                    use_trap1 = decide_trap_use_ai1(env)

                    # AI2 규칙 기반
                    obs2_dict = {"ai2_pos": env.ai2_pos, "ai1_pos": env.ai1_pos}
                    action2, use_trap2 = rule_based_ai2(obs2_dict, env, eps=0.05)

                    (obs1, obs2), done, info = env.step(action1, action2, use_trap1, use_trap2)
                    if done:
                        episode += 1
                        step = 0
                        obs1, obs2 = env.reset()
                        if skip_rounds > 0:
                            skip_rounds -= 1
                            if skip_rounds == 0:
                                render_enabled = True

                # --- AI vs PLAYER (AI1=AI, AI2=PLAYER) ---
                elif selected_mode == MODE_AI_VS_PLAYER:
                    # 플레이어(AI2) 입력이 있을 때만 한 턴 진행
                    if pending_player_move_ai2 is not None:
                        step += 1

                        # AI1: Q-learning(학습 AI)
                        if USE_TRAINED_AI1 and q_loaded and Q is not None:
                            state_ai1: State = env.ai1_pos
                            action1 = choose_action_with_difficulty(Q, state_ai1, epsilon)
                            ai1_mode_text = "Q-learning"
                        else:
                            action1 = random_action()
                            ai1_mode_text = "RANDOM"

                        # AI2: 플레이어 입력
                        action2 = pending_player_move_ai2
                        pending_player_move_ai2 = None

                        use_trap1 = decide_trap_use_ai1(env)  # AI1도 자동 트랩 쓰게 하고 싶으면 유지
                        use_trap2 = bool(pending_player_place_trap_ai2)
                        pending_player_place_trap_ai2 = False

                        (obs1, obs2), done, info = env.step(action1, action2, use_trap1, use_trap2)
                        if done:
                            episode += 1
                            step = 0
                            obs1, obs2 = env.reset()

            # ---- 렌더 ----
            if render_enabled:
                maze_surf = pygame.Surface((MAZE_PIX_W, MAZE_PIX_H))
                env.render(maze_surf)
                screen.blit(maze_surf, (0, 0))

                info_dict = {
                    "episode": episode,
                    "step": step,
                    "turn_remaining": max(0, env.turn_limit - env.turn_count),
                    "ai1_pos": env.ai1_pos,
                    "ai2_pos": env.ai2_pos,
                    "ai1_mode": ai1_mode_text,
                    "difficulty": selected_difficulty.upper(),
                    "score_ai1": env.score_ai1,
                    "score_ai2": env.score_ai2,
                    "items_ai1": env.items_ai1,
                    "items_ai2": env.items_ai2,
                }

                pause_button_rect = draw_sidebar(screen, font_small, MAZE_PIX_W, info_dict, paused, fast_mode, skip_rounds)

                # HUD 아이콘: trap은 pickup 아이콘이 보유표시용
                img_trap_pickup = getattr(env, "img_trap_pickup", None)
                img_tranq = getattr(env, "img_tranq", None)

                ai1_trap_icon = img_trap_pickup if getattr(env, "has_trap_ai1", False) else None
                ai1_tranq_icon = img_tranq if getattr(env, "has_tranq_ai1", False) else None
                ai2_trap_icon = img_trap_pickup if getattr(env, "has_trap_ai2", False) else None
                ai2_tranq_icon = img_tranq if getattr(env, "has_tranq_ai2", False) else None

                # 라벨: 모드에 따라 표기 명확히
                if selected_mode == MODE_AI_VS_PLAYER:
                    label_ai1 = "AI1 (AI)"
                    label_ai2 = "AI2 (PLAYER)"
                else:
                    label_ai1 = "AI1"
                    label_ai2 = "AI2"

                draw_bottom_hud(
                    screen,
                    font_small,
                    label_ai1,
                    label_ai2,
                    ai1_trap_icon,
                    ai1_tranq_icon,
                    ai2_trap_icon,
                    ai2_tranq_icon,
                )

                pygame.display.flip()

            # FPS
            if selected_mode == MODE_AI_VS_AI and (fast_mode or skip_rounds > 0):
                if skip_rounds > 0:
                    clock.tick(0)
                else:
                    clock.tick(fps_fast)
            else:
                clock.tick(fps_normal)

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()
