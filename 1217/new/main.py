# main.py  # ver5-fixed

import sys
import os
import random
from typing import Tuple, Dict, Any, Optional

import pygame
from dotenv import load_dotenv
from supabase import create_client, Client

from env_maze import (
    MazeEnv,
    CELL_SIZE, GRID_W, GRID_H,
    UP, DOWN, LEFT, RIGHT,
)

from ai2_rules import rule_based_ai2

# ----------------------------
# 화면/레이아웃
# ----------------------------
SIDEBAR_W = 320
HUD_H = 96  # TESTver4: 하단 아이템 HUD 높이

MAZE_PIX_W = GRID_W * CELL_SIZE
MAZE_PIX_H = GRID_H * CELL_SIZE

WIN_W = MAZE_PIX_W + SIDEBAR_W
WIN_H = MAZE_PIX_H + HUD_H  # TESTver4: 하단 HUD 포함

# ----------------------------
# 색상
# ----------------------------
COLOR_SIDEBAR = (30, 30, 30)
COLOR_TEXT = (230, 230, 230)
COLOR_BTN_PAUSE = (150, 60, 60)
COLOR_BTN_RESUME = (60, 150, 60)

COLOR_TITLE_BG = (10, 10, 30)
COLOR_TITLE_BTN = (60, 120, 200)
COLOR_TITLE_BTN_HOVER = (80, 150, 230)

COLOR_HUD_BG = (18, 18, 18)
COLOR_HUD_LINE = (60, 60, 60)

COLOR_AI1 = (200, 80, 80)
COLOR_AI2 = (80, 120, 220)

COLOR_SLOT_OFF = (55, 55, 55)
COLOR_SLOT_ON = (200, 200, 200)

ACTIONS = [UP, DOWN, LEFT, RIGHT]

# ----------------------------
# 상태/모드
# ----------------------------
STATE_TITLE = "TITLE"
STATE_DIFFICULTY = "DIFFICULTY"
STATE_GAME = "GAME"

MODE_AI_VS_AI = "AI_VS_AI"
MODE_AI_VS_PLAYER = "AI_VS_PLAYER"

# AI1을 Q-table로 쓸지
USE_TRAINED_AI1 = True

Action = int
State = Tuple[int, int]

DIFFICULTY_SETTINGS: Dict[str, float] = {
    "easy": 0.7,
    "normal": 0.3,
    "hard": 0.05,
}

# ----------------------------
# Supabase
# ----------------------------
def init_supabase_client() -> Client | None:
    load_dotenv()
    url: str | None = os.environ.get("SUPABASE_URL")
    key: str | None = os.environ.get("SUPABASE_ANON_KEY")
    if not url or not key:
        print("[ERROR] SUPABASE_URL 또는 SUPABASE_ANON_KEY 환경 변수가 설정되지 않았습니다.")
        return None
    return create_client(url, key)


def load_q_table(table_name: str = "q_table_maze_v4") -> Dict[Tuple[State, Action], float] | None:
    """
    Supabase에서 Q-table 로드.
    row가 state_x/state_y/action/q_value 컬럼을 가진다고 가정.
    """
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
            # 여기서 'state_x' KeyError가 뜨면 DB 스키마가 다름
            state = (row["state_x"], row["state_y"])
            action = row["action"]
            q_value = row["q_value"]
            Q[(state, action)] = q_value

        print(f"[INFO] Q-table loaded from Supabase. Entries: {len(Q)}")
        return Q

    except Exception as e:
        print(f"[ERROR] Supabase Q-table loading failed: {e}. AI will use fallback policy.")
        return None


# ----------------------------
# 정책
# ----------------------------
def choose_action_with_difficulty(
    Q: Dict[Tuple[State, Action], float] | None, state: State, epsilon: float
) -> Action:
    if Q is None or random.random() < epsilon:
        return random.choice(ACTIONS)

    qs = [Q.get((state, a), 0.0) for a in ACTIONS]
    max_q = max(qs)
    candidates = [a for a, q in zip(ACTIONS, qs) if q == max_q]
    return random.choice(candidates)


def random_action() -> int:
    return random.choice(ACTIONS)


# ----------------------------
# UI: 버튼
# ----------------------------
def draw_centered_button(
    screen: pygame.Surface,
    rect: pygame.Rect,
    text: str,
    font: pygame.font.Font,
    hover: bool = False,
):
    color = COLOR_TITLE_BTN_HOVER if hover else COLOR_TITLE_BTN
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


def draw_title_screen(
    screen: pygame.Surface,
    font_big: pygame.font.Font,
    font_small: pygame.font.Font,
    btn_rects: Dict[str, pygame.Rect],
):
    screen.fill(COLOR_TITLE_BG)

    title_y = int(WIN_H * 0.18)
    title_surf = font_big.render("Maze RL Battle", True, COLOR_TEXT)
    title_rect = title_surf.get_rect(center=(WIN_W // 2, title_y))
    screen.blit(title_surf, title_rect)

    subtitle = "AI vs AI / AI vs PLAYER"
    sub_surf = font_small.render(subtitle, True, COLOR_TEXT)
    sub_rect = sub_surf.get_rect(center=(WIN_W // 2, title_y + 32))
    screen.blit(sub_surf, sub_rect)

    mx, my = pygame.mouse.get_pos()
    for label, rect in btn_rects.items():
        hover = rect.collidepoint(mx, my)
        draw_centered_button(screen, rect, label, font_small, hover)


def draw_difficulty_screen(
    screen: pygame.Surface,
    font_big: pygame.font.Font,
    font_small: pygame.font.Font,
    btn_rects: Dict[str, pygame.Rect],
    selected_mode: str | None,
):
    screen.fill(COLOR_TITLE_BG)

    title_y = int(WIN_H * 0.18)
    title_surf = font_big.render("Select Difficulty", True, COLOR_TEXT)
    title_rect = title_surf.get_rect(center=(WIN_W // 2, title_y))
    screen.blit(title_surf, title_rect)

    mode_text = f"Mode: {selected_mode or 'N/A'}"
    mode_surf = font_small.render(mode_text, True, COLOR_TEXT)
    mode_rect = mode_surf.get_rect(center=(WIN_W // 2, title_y + 32))
    screen.blit(mode_surf, mode_rect)

    mx, my = pygame.mouse.get_pos()
    for label, rect in btn_rects.items():
        hover = rect.collidepoint(mx, my)
        draw_centered_button(screen, rect, label, font_small, hover)


# ----------------------------
# HUD: 우측 사이드바
# ----------------------------
def draw_sidebar(screen, font, maze_width, info: Dict[str, Any], paused, fast_mode, skip_rounds):
    sidebar_rect = pygame.Rect(maze_width, 0, SIDEBAR_W, WIN_H)
    screen.fill(COLOR_SIDEBAR, sidebar_rect)

    x = maze_width + 18
    y = 14

    def line(txt):
        nonlocal y
        surf = font.render(txt, True, COLOR_TEXT)
        screen.blit(surf, (x, y))
        y += 18

    line("Maze Q-Learning Info")
    y += 6

    line(f"Episode : {info.get('episode', 1)}")
    line(f"Step    : {info.get('step', 0)}")
    line(f"Turns   : {info.get('turn_remaining', 0)}")
    y += 6

    line(f"AI1 Pos : {info.get('ai1_pos')}")
    line(f"AI2 Pos : {info.get('ai2_pos')}")
    y += 6

    line(f"AI1 Score : {info.get('score_ai1', 0)}")
    line(f"AI2 Score : {info.get('score_ai2', 0)}")
    line(f"Coins AI1 : {info.get('items_ai1', 0)}/{info.get('required_coins', 3)}")
    line(f"Coins AI2 : {info.get('items_ai2', 0)}/{info.get('required_coins', 3)}")
    y += 6

    line(f"AI1 used item : {info.get('used_item_ai1', False)}")
    line(f"AI2 used item : {info.get('used_item_ai2', False)}")
    line(f"AI1 checkpoint: {info.get('reached_cp_ai1', False)}")
    line(f"AI2 checkpoint: {info.get('reached_cp_ai2', False)}")
    line(f"AI1 unlocked  : {info.get('unlocked_ai1', False)}")
    line(f"AI2 unlocked  : {info.get('unlocked_ai2', False)}")
    y += 6

    line(f"AI1 Mode  : {info.get('ai1_mode', 'N/A')}")
    line(f"Difficulty: {info.get('difficulty', 'UNKNOWN')}")
    y += 6

    line(f"Fast Mode : {'ON' if fast_mode else 'OFF'}")
    line(f"Skip Rnds : {skip_rounds}")
    y += 8

    line("[Controls]")
    line("ESC: Title  P: Pause")
    line("F: Fast  M: Skip30  N: Next (AIvsAI)")
    line("Player(AI2): WASD/Arrows, E: Trap")

    # Pause button
    btn_w, btn_h = 120, 42
    btn_x = maze_width + (SIDEBAR_W - btn_w) // 2
    btn_y = WIN_H - 60
    pause_rect = pygame.Rect(btn_x, btn_y, btn_w, btn_h)

    col = COLOR_BTN_RESUME if paused else COLOR_BTN_PAUSE
    txt = "RESUME" if paused else "PAUSE"
    pygame.draw.rect(screen, col, pause_rect, border_radius=6)
    t = font.render(txt, True, COLOR_TEXT)
    screen.blit(t, t.get_rect(center=pause_rect.center))

    return pause_rect


# ----------------------------
# HUD: 하단 아이템 바
# ----------------------------
def _draw_item_slot(screen: pygame.Surface, rect: pygame.Rect, on: bool):
    pygame.draw.rect(screen, COLOR_SLOT_ON if on else COLOR_SLOT_OFF, rect, border_radius=6)
    pygame.draw.rect(screen, COLOR_HUD_LINE, rect, 2, border_radius=6)


def draw_bottom_item_hud(screen: pygame.Surface, font: pygame.font.Font, env: MazeEnv):
    # TESTver4: 반드시 y = MAZE_PIX_H 기준
    hud_y = MAZE_PIX_H
    pygame.draw.rect(screen, COLOR_HUD_BG, pygame.Rect(0, hud_y, WIN_W, HUD_H))
    pygame.draw.line(screen, COLOR_HUD_LINE, (0, hud_y), (WIN_W, hud_y), 2)

    # 좌: AI1 / 우: AI2(플레이어)
    pad = 14
    mid = WIN_W // 2

    # 라벨
    t1 = font.render("AI1", True, COLOR_AI1)
    t2 = font.render("AI2 / PLAYER", True, COLOR_AI2)
    screen.blit(t1, (pad, hud_y + 10))
    screen.blit(t2, (mid + pad, hud_y + 10))

    # 슬롯 2개: Trap, Tranq (나중에 이미지로 교체)
    slot_w, slot_h = 44, 44
    gap = 10
    base_y = hud_y + 38

    # AI1 slots
    s1_trap = pygame.Rect(pad, base_y, slot_w, slot_h)
    s1_tranq = pygame.Rect(pad + slot_w + gap, base_y, slot_w, slot_h)

    _draw_item_slot(screen, s1_trap, bool(getattr(env, "has_trap_ai1", False)))
    _draw_item_slot(screen, s1_tranq, bool(getattr(env, "has_tranq_ai1", False)))

    # AI2 slots
    s2_trap = pygame.Rect(mid + pad, base_y, slot_w, slot_h)
    s2_tranq = pygame.Rect(mid + pad + slot_w + gap, base_y, slot_w, slot_h)

    _draw_item_slot(screen, s2_trap, bool(getattr(env, "has_trap_ai2", False)))
    _draw_item_slot(screen, s2_tranq, bool(getattr(env, "has_tranq_ai2", False)))


# ----------------------------
# main
# ----------------------------
def main():
    pygame.init()
    screen = pygame.display.set_mode((WIN_W, WIN_H))
    pygame.display.set_caption("Maze RL Battle ver5-fixed")

    # TESTver4: 미로는 별도 surface에만 그린다(하단 HUD/사이드바 보존)
    maze_surf = pygame.Surface((MAZE_PIX_W, MAZE_PIX_H))

    try:
        font_small = pygame.font.Font(None, 24)
        font_big = pygame.font.Font(None, 40)
    except Exception:
        font_small = pygame.font.SysFont(None, 24)
        font_big = pygame.font.SysFont(None, 40)

    clock = pygame.time.Clock()

    # 버튼 Rect
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

    # 상태 변수
    game_state = STATE_TITLE
    selected_mode: Optional[str] = None
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

    # 플레이어(AI2) 조작
    pending_player_move: Optional[Action] = None
    pending_player_place_trap = False  # TESTver4

    ai1_mode_text = "N/A"

    pause_button_rect = None

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

    # ------------------ loop ------------------
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                break

            # 공통: ESC = 타이틀(게임 중)
            if event.type == pygame.KEYDOWN:
                if game_state == STATE_GAME and event.key == pygame.K_ESCAPE:
                    game_state = STATE_TITLE
                    paused = False
                    fast_mode = False
                    skip_rounds = 0
                    render_enabled = True
                    pending_player_move = None
                    pending_player_place_trap = False
                    continue

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

            # DIFFICULTY
            elif game_state == STATE_DIFFICULTY:
                if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    mx, my = event.pos
                    if diff_btn_rects["EASY"].collidepoint(mx, my):
                        enter_game("easy")
                        game_state = STATE_GAME
                    elif diff_btn_rects["NORMAL"].collidepoint(mx, my):
                        enter_game("normal")
                        game_state = STATE_GAME
                    elif diff_btn_rects["HARD"].collidepoint(mx, my):
                        enter_game("hard")
                        game_state = STATE_GAME
                    elif diff_btn_rects["BACK"].collidepoint(mx, my):
                        game_state = STATE_TITLE

            # GAME
            elif game_state == STATE_GAME:
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_p:
                        paused = not paused

                    if selected_mode == MODE_AI_VS_AI:
                        if event.key == pygame.K_f:
                            fast_mode = not fast_mode
                        if event.key == pygame.K_m:
                            skip_rounds += 30
                            render_enabled = False
                            fast_mode = True
                        if event.key == pygame.K_n:
                            episode += 1
                            step = 0
                            obs1, obs2 = env.reset()

                    if selected_mode == MODE_AI_VS_PLAYER:
                        # 플레이어는 AI2
                        if event.key in (pygame.K_w, pygame.K_UP):
                            pending_player_move = UP
                        elif event.key in (pygame.K_s, pygame.K_DOWN):
                            pending_player_move = DOWN
                        elif event.key in (pygame.K_a, pygame.K_LEFT):
                            pending_player_move = LEFT
                        elif event.key in (pygame.K_d, pygame.K_RIGHT):
                            pending_player_move = RIGHT

                        # TESTver4: 트랩 설치(E) (AI2=플레이어)
                        if event.key == pygame.K_e:
                            pending_player_place_trap = True

                if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    if pause_button_rect is not None and pause_button_rect.collidepoint(event.pos):
                        paused = not paused

        if not running:
            break

        # ------------------ state render ------------------
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

        # ------------------ GAME update ------------------
        if game_state == STATE_GAME:
            if not paused:
                # AI vs AI
                if selected_mode == MODE_AI_VS_AI:
                    step += 1

                    # AI1
                    if USE_TRAINED_AI1 and q_loaded and env.ai1_pos is not None:
                        action1 = choose_action_with_difficulty(Q, env.ai1_pos, epsilon)
                        ai1_mode_text = "Q-learning"
                    else:
                        action1 = random_action()
                        ai1_mode_text = "RANDOM"

                    # AI2 (rule-based)
                    obs2_dict = {"ai2_pos": env.ai2_pos, "ai1_pos": env.ai1_pos}
                    action2, use_trap2 = rule_based_ai2(obs2_dict, env, eps=0.07)

                    (obs1, obs2), done, info = env.step(action1, action2, False, use_trap2)

                    if done:
                        episode += 1
                        step = 0
                        obs1, obs2 = env.reset()
                        if skip_rounds > 0:
                            skip_rounds -= 1
                            if skip_rounds == 0:
                                render_enabled = True

                # AI vs PLAYER (AI1=Q, AI2=Player)
                elif selected_mode == MODE_AI_VS_PLAYER:
                    # 플레이어가 키를 눌렀거나 트랩 설치 눌렀을 때만 진행
                    do_step = (pending_player_move is not None) or pending_player_place_trap
                    if do_step:
                        step += 1

                        # AI1 (Q or fallback)
                        if USE_TRAINED_AI1 and q_loaded and env.ai1_pos is not None:
                            action1 = choose_action_with_difficulty(Q, env.ai1_pos, epsilon)
                            ai1_mode_text = "Q-learning"
                        else:
                            action1 = random_action()
                            ai1_mode_text = "RANDOM"

                        # Player(AI2)
                        action2 = pending_player_move if pending_player_move is not None else random_action()
                        use_trap2 = pending_player_place_trap

                        pending_player_move = None
                        pending_player_place_trap = False

                        (obs1, obs2), done, info = env.step(action1, action2, False, use_trap2)

                        if done:
                            episode += 1
                            step = 0
                            obs1, obs2 = env.reset()

            # ------------------ GAME render (순서 중요) ------------------
            if render_enabled:
                # 1) 미로는 maze_surf에만 렌더
                env.render(maze_surf)
                screen.blit(maze_surf, (0, 0))

                # 2) 사이드바
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
                    "required_coins": getattr(env, "required_coins", 3),
                    "used_item_ai1": getattr(env, "used_item_ai1", False),
                    "used_item_ai2": getattr(env, "used_item_ai2", False),
                    "reached_cp_ai1": getattr(env, "reached_checkpoint_ai1", getattr(env, "checkpoint_cleared_ai1", False)),
                    "reached_cp_ai2": getattr(env, "reached_checkpoint_ai2", getattr(env, "checkpoint_cleared_ai2", False)),
                    "unlocked_ai1": getattr(env, "unlocked_ai1", False),
                    "unlocked_ai2": getattr(env, "unlocked_ai2", False),
                }

                pause_button_rect = draw_sidebar(
                    screen, font_small, MAZE_PIX_W,
                    info_dict, paused, fast_mode, skip_rounds
                )

                # 3) 하단 아이템 HUD는 반드시 마지막
                draw_bottom_item_hud(screen, font_small, env)

                pygame.display.flip()

            # FPS control
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
