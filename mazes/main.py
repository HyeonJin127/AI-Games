# main.py  # ver4

import sys
import os
import random
import pickle
import pygame
from typing import Tuple, Dict, Any, List

from dotenv import load_dotenv
from supabase import create_client, Client

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

Action = int
State = Tuple[int, int]

# 난이도별 설정
DIFFICULTY_SETTINGS: Dict[str, float] = {
    "easy": 0.7,
    "normal": 0.3,
    "hard" : 0.05
}

CURRENT_DIFFICULTY = "normal"

# Supabase 클라이언트 초기화 함수
# main.py 파일 내

def init_supabase_client() -> Client:
    """Supabase 클라이언트 초기화 (Anon Key 사용)."""
    load_dotenv()
    url: str = os.environ.get("SUPABASE_URL")
    # 데이터 읽기만 하므로 anon_key 사용
    key: str = os.environ.get("SUPABASE_ANON_KEY") 
    
    # ⭐️ 환경 변수 누락 시 에러 처리
    if not url or not key:
        print("[ERROR] SUPABASE_URL 또는 SUPABASE_ANON_KEY 환경 변수가 설정되지 않았습니다.")
        return None
        
    return create_client(url, key)

# load_q_table 수정
def load_q_table(table_name: str = "q_table_maze_v4") -> Dict[Tuple[State, Action], float] | None:
    """Supabase에서 Q-table 로드 및 Dictionary로 재구성."""
    Q: Dict[Tuple[State, Action], float] = {}
    try:
        supabase = init_supabase_client()
        if supabase is None:
            return None
        
        # 테이블의 모든 데이터 조회
        response = supabase.table(table_name).select("*").execute()
        
        if not response.data:
            print(f"[WARN] No data found in Supabase table: {table_name}.")
            return None
        
        # Q-table 딕셔너리로 재구성: Key = ((state_x, state_y), action)
        for row in response.data:
            state = (row['state_x'], row['state_y'])
            action = row['action']
            q_value = row['q_value']
            Q[(state, action)] = q_value
            
        print(f"[INFO] Q-table loaded from Supabase. Entries: {len(Q)}")
        return Q
        
    except Exception as e:
        print(f"[ERROR] Supabase Q-table loading failed: {e}. AI1 will use RANDOM policy.")
        return None

# 난이도별 Epsilon-Greedy 행동 선택 함수 추가
def choose_action_with_difficulty(Q: Dict[Tuple[State, Action], float], state: State, epsilon: float) -> Action:
    """Epsilon-greedy 방식으로 행동 선택 (난이도 적용)."""
    if random.random() < epsilon:
        return random.choice(ACTIONS) # 탐험 (랜덤)
    
    # 활용 (Greedy)
    # Q-table에 값이 없으면 0.0으로 간주
    qs = [Q.get((state, a), 0.0) for a in ACTIONS] 
    max_q = max(qs)
    # 최대 Q 값을 가진 행동들 중에서 무작위로 선택
    candidates = [a for a, q in zip(ACTIONS, qs) if q == max_q]
    return random.choice(candidates)

# def greedy_action_from_q(Q, state) -> int:
#     """Q-table 기준으로 greedy 행동 선택. state = (x,y)."""  # ver4
#     qs = [Q.get((state, a), 0.0) for a in ACTIONS]
#     max_q = max(qs)
#     candidates = [a for a, q in zip(ACTIONS, qs) if q == max_q]
#     return random.choice(candidates)


def draw_sidebar(screen: pygame.Surface, font: pygame.font.Font, maze_width: int, info_dict: Dict[str, Any], paused: bool) -> pygame.Rect:
    
    # 사이드바 영역 (미로 너비부터 끝까지)
    sidebar_rect = pygame.Rect(maze_width, 0, SIDEBAR_W, WIN_H)
    screen.fill(COLOR_SIDEBAR, sidebar_rect)
    
    start_x = maze_width + 20
    line_y = 20

    # --- 텍스트 정보 출력 ---

    # 제목
    title_line = font.render("Maze Q-Learning Info", True, COLOR_TEXT)
    screen.blit(title_line, (start_x, line_y))
    line_y += 40

    # 에피소드 및 스텝
    ep_line = font.render(f"Episode: {info_dict['episode']}", True, COLOR_TEXT)
    screen.blit(ep_line, (start_x, line_y))
    line_y += 30
    
    step_line = font.render(f"Step: {info_dict['step']}", True, COLOR_TEXT)
    screen.blit(step_line, (start_x, line_y))
    line_y += 40

    # AI 위치
    ai1_pos_line = font.render(f"AI1 Pos: {info_dict['ai1_pos']}", True, COLOR_TEXT)
    screen.blit(ai1_pos_line, (start_x, line_y))
    line_y += 30

    ai2_pos_line = font.render(f"AI2 Pos: {info_dict['ai2_pos']}", True, COLOR_TEXT)
    screen.blit(ai2_pos_line, (start_x, line_y))
    line_y += 40
    
    # ⭐️ AI1 모드 (Q-learning vs Random) 출력
    mode_text = info_dict.get("ai1_mode", "N/A")
    mode_line = font.render(f"AI1 Mode: {mode_text}", True, COLOR_TEXT)
    screen.blit(mode_line, (start_x, line_y))
    line_y += 30
    
    # ⭐️ 현재 난이도 출력
    difficulty_line = font.render(f"Difficulty: {CURRENT_DIFFICULTY.upper()}", True, COLOR_TEXT)
    screen.blit(difficulty_line, (start_x, line_y))
    line_y += 40

    # --- 일시정지/재개 버튼 ---

    # 버튼 영역 설정
    btn_w, btn_h = 100, 40
    btn_x = maze_width + (SIDEBAR_W - btn_w) // 2
    btn_y = WIN_H - 60
    pause_button_rect = pygame.Rect(btn_x, btn_y, btn_w, btn_h)

    # 버튼 색상 및 텍스트 설정
    if paused:
        btn_color = COLOR_BTN_RESUME
        btn_text = "RESUME"
    else:
        btn_color = COLOR_BTN_PAUSE
        btn_text = "PAUSE"
        
    pygame.draw.rect(screen, btn_color, pause_button_rect, border_radius=5)

    # 버튼 텍스트 렌더링
    text_surface = font.render(btn_text, True, COLOR_TEXT)
    text_rect = text_surface.get_rect(center=pause_button_rect.center)
    screen.blit(text_surface, text_rect)

    return pause_button_rect


def random_action() -> int:
    return random.choice(ACTIONS)


def main():
    pygame.init()
    screen = pygame.display.set_mode((WIN_W, WIN_H))
    pygame.display.set_caption("Q-Learning Maze Demo")
    
    try:
        font = pygame.font.Font(None, 24)
    except:
        font = pygame.font.SysFont(None, 24)

    env = MazeEnv()
    obs1, obs2 = env.reset()

    # Q-table 로드 (Supabase에서)
    Q = load_q_table() if USE_TRAINED_AI1 else None
    q_loaded = Q is not None

    epsilon = DIFFICULTY_SETTINGS.get(CURRENT_DIFFICULTY, DIFFICULTY_SETTINGS['normal'])

    scale_factor = 1.0

    if CURRENT_DIFFICULTY == "easy":
        scale_factor = 0.5  # Q-값 차이를 줄여서 "확신도" 감소 -> 우연성 증가
    elif CURRENT_DIFFICULTY == "hard":
        scale_factor = 2.0  # Q-값 차이를 키워서 "확신도" 증가 -> 선택 명확화
        
    if Q is not None and scale_factor != 1.0:
        Q = {k: v * scale_factor for k, v in Q.items()}
        print(f"[INFO] Q-table scaled by {scale_factor} for difficulty '{CURRENT_DIFFICULTY}'")

    clock = pygame.time.Clock()

    episode = 1
    step = 0
    paused = False
    running = True

    # pause_button_rect = pygame.Rect(MAZE_PIX_W + 40, 260, 140, 40)

    while running:
        # --- 이벤트 처리 ---  # ver4
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                break
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    running = False
                    break
                if event.key == pygame.K_p:
                    paused = not paused
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if 'pause_button_rect' in locals() and pause_button_rect.collidepoint(event.pos):
                    paused = not paused

        if not running:
            break

        if not paused:
            step += 1

            # --- AI1 행동 선택 ---  # ver4
            if USE_TRAINED_AI1 and q_loaded:
                state = env.ai1_pos
                action1 = choose_action_with_difficulty(Q, state, epsilon)
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

        ai1_mode_text = "Q-learning" if USE_TRAINED_AI1 and q_loaded else "RANDOM"

        info_dict = {
            "episode": episode,
            "step": step,
            "ai1_pos": env.ai1_pos,
            "ai2_pos": env.ai2_pos,
            "ai1_mode": ai1_mode_text,
        }

        pause_button_rect = draw_sidebar(screen, font, MAZE_PIX_W, info_dict, paused)

        # draw_sidebar(screen, font, info_dict, paused,
        #              pause_button_rect, USE_TRAINED_AI1, q_loaded)

        pygame.display.flip()
        clock.tick(10)  # (u) FPS

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()
