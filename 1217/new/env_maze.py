# env_maze.py  # ver5-fixed

from __future__ import annotations

import os
import random
from typing import Any, Dict, Optional, Set, Tuple

import pygame

# --- 상수들 ---
CELL_SIZE = 48
GRID_W = 15
GRID_H = 11

# tile codes
T_FLOOR = 0
T_WALL = 1
T_PORTAL = 2
T_GATE = 3         # 평소 통과 가능, "골 가능 상태"가 되면 벽처럼 막힘
T_CHECKPOINT = 4   # 골 전 체크포인트
T_GOAL = 5         # 골

COLOR_BG = (10, 10, 30)
COLOR_WALL = (30, 30, 35)
COLOR_GRID = (90, 90, 100)

COLOR_GATE_OPEN = (120, 160, 220)   # gate(3) 평소 표시
COLOR_GATE_BLOCK = (60, 80, 110)    # gate(3) 잠김 상태 표시

COLOR_PORTAL = (160, 90, 230)
COLOR_CHECKPOINT = (120, 210, 120)
COLOR_GOAL = (230, 230, 230)

COLOR_COIN = (240, 210, 60)

COLOR_AI1 = (200, 80, 80)
COLOR_AI2 = (80, 120, 220)

COLOR_TRANQ = (230, 200, 60)
COLOR_TRAP = (200, 60, 200)

# actions
UP, DOWN, LEFT, RIGHT = 0, 1, 2, 3

# gameplay
STUN_TURNS = 5
REQUIRED_COINS_FOR_CHECKPOINT = 3  # <- 너가 말한 값
TURN_LIMIT_DEFAULT = 240           # 라운드 타이머(턴 제한)


def _base_dir() -> str:
    return os.path.dirname(os.path.abspath(__file__))


def _path(*parts: str) -> str:
    return os.path.join(_base_dir(), *parts)


def load_image(path: str, size: Tuple[int, int]) -> pygame.Surface:
    img = pygame.image.load(path).convert_alpha()
    return pygame.transform.scale(img, size)


class MazeEnv:
    """
    - grid: 0/1/2/3/4/5 타일
    - 2: 포탈(2개 페어)
    - 3: 평소 통과 가능, goal unlock 상태가 되면 벽처럼 막힘
    - 4: 체크포인트(고정)
    - 5: 골(고정)
    - 코인(여러개 랜덤 스폰)
    - 트랩(설치형, 보임)
    - 마취총(소지형)
    - goal unlock 조건(각 AI별):
        코인 >= REQUIRED_COINS_FOR_CHECKPOINT
        + 상대에게 아이템 적중 1회 이상(트랩/마취총)
        + 체크포인트(4) 도달
    """

    def __init__(self):
        # 요구한 grid
        self.grid = [
            [1, 2, 1, 1, 1, 1, 1, 5, 1, 1, 1, 1, 1, 1, 1],
            [1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1],
            [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
            [1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1],
            [1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1],
            [1, 0, 0, 0, 0, 1, 3, 3, 3, 1, 0, 0, 0, 0, 1],
            [1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1],
            [1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1],
            [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
            [1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1],
            [1, 1, 1, 1, 1, 1, 1, 4, 1, 1, 1, 1, 1, 2, 1],
        ]

        self.width = GRID_W
        self.height = GRID_H

        # --- 고정 타일 좌표 캐시 ---
        self.portal_positions: list[Tuple[int, int]] = []
        self.gate_positions: Set[Tuple[int, int]] = set()
        self.checkpoint_pos: Optional[Tuple[int, int]] = None
        self.goal_pos: Optional[Tuple[int, int]] = None

        for y in range(self.height):
            for x in range(self.width):
                t = self.grid[y][x]
                if t == T_PORTAL:
                    self.portal_positions.append((x, y))
                elif t == T_GATE:
                    self.gate_positions.add((x, y))
                elif t == T_CHECKPOINT:
                    self.checkpoint_pos = (x, y)
                elif t == T_GOAL:
                    self.goal_pos = (x, y)

        # portal pair (2개만 있다고 가정)
        self.portal_pair: Optional[Tuple[Tuple[int, int], Tuple[int, int]]] = None
        if len(self.portal_positions) >= 2:
            self.portal_pair = (self.portal_positions[0], self.portal_positions[1])

        # --- 라운드/턴 ---
        self.turn_limit = TURN_LIMIT_DEFAULT
        self.required_coins = REQUIRED_COINS_FOR_CHECKPOINT
        self.turn_count = 0

        # --- 에이전트 ---
        self.ai1_pos: Optional[Tuple[int, int]] = None
        self.ai2_pos: Optional[Tuple[int, int]] = None
        self.ai1_dir: int = UP
        self.ai2_dir: int = UP

        # --- 스코어/코인 카운트 ---
        self.score_ai1 = 0
        self.score_ai2 = 0
        self.items_ai1 = 0  # 코인 개수
        self.items_ai2 = 0

        # --- 코인(랜덤) ---
        self.coin_positions: Set[Tuple[int, int]] = set()

        # --- 트랩/마취총 ---
        self.trap_pickup_pos: Optional[Tuple[int, int]] = None  # 주울 수 있는 트랩 아이템(맵에 1개)
        self.tranq_pos: Optional[Tuple[int, int]] = None        # 주울 수 있는 마취총 아이템(맵에 1개)

        self.has_trap_ai1 = False
        self.has_trap_ai2 = False
        self.has_tranq_ai1 = False
        self.has_tranq_ai2 = False

        self.trap_positions: Dict[Tuple[int, int], str] = {}  # pos -> "AI1"/"AI2"

        self.stun_ai1 = 0
        self.stun_ai2 = 0

        # 아이템 적중 여부(언락 조건)
        self.used_item_ai1 = False
        self.used_item_ai2 = False

        # 체크포인트 도달 여부(각자)
        self.checkpoint_cleared_ai1 = False
        self.checkpoint_cleared_ai2 = False

        # goal unlock 상태(각자)
        self.unlocked_ai1 = False
        self.unlocked_ai2 = False

        # goal 상태(전역): 누가 언락해도 "골 가능 상태"가 되면 gate(3)은 벽처럼 막힘
        self.goal_active = False

        # --- 이미지 로딩 ---
        self.images_ok = False
        self._load_images()

        self.reset()

    # -------------------- 이미지 --------------------

    def _load_images(self) -> None:
        """assets 폴더에서 이미지 로딩. 실패하면 도형 렌더 fallback."""
        try:
            size_agent = (CELL_SIZE - 8, CELL_SIZE - 8)
            size_item = (CELL_SIZE - 8, CELL_SIZE - 8)

            # AI
            self.img_ai1_up = load_image(_path("assets", "ai", "ai1_up.png"), size_agent)
            self.img_ai1_down = load_image(_path("assets", "ai", "ai1_down.png"), size_agent)
            self.img_ai1_left = load_image(_path("assets", "ai", "ai1_left.png"), size_agent)
            self.img_ai1_right = load_image(_path("assets", "ai", "ai1_right.png"), size_agent)

            self.img_ai2_up = load_image(_path("assets", "ai", "ai2_up.png"), size_agent)
            self.img_ai2_down = load_image(_path("assets", "ai", "ai2_down.png"), size_agent)
            self.img_ai2_left = load_image(_path("assets", "ai", "ai2_left.png"), size_agent)
            self.img_ai2_right = load_image(_path("assets", "ai", "ai2_right.png"), size_agent)

            # items
            self.img_tranq = load_image(_path("assets", "items", "tranq.png"), size_item)
            self.img_trap_pickup = load_image(_path("assets", "items", "trap_pickup.png"), size_item)
            self.img_trap_set = load_image(_path("assets", "items", "trap_set.png"), size_item)

            self.images_ok = True

        except Exception as e:
            self.images_ok = False
            print(f"[WARN] 이미지 로드 실패, 도형 렌더 fallback 사용: {e}.")

    # -------------------- 기본 유틸 --------------------

    def in_bounds(self, pos: Tuple[int, int]) -> bool:
        x, y = pos
        return 0 <= x < self.width and 0 <= y < self.height

    def tile_at(self, pos: Tuple[int, int]) -> int:
        x, y = pos
        return self.grid[y][x]

    def is_passable_for(self, who: str, pos: Tuple[int, int]) -> bool:
        """who: 'AI1'/'AI2' - gate(3) 규칙 반영."""
        x, y = pos
        t = self.grid[y][x]
        if t == T_WALL:
            return False
        # gate(3): goal_active 되면 벽처럼 막힘
        if t == T_GATE and self.goal_active:
            return False
        return True

    def is_valid_action(self, who: str, pos: Tuple[int, int], action: int) -> bool:
        dx = {LEFT: -1, RIGHT: 1}.get(action, 0)
        dy = {UP: -1, DOWN: 1}.get(action, 0)
        nxt = (pos[0] + dx, pos[1] + dy)
        if not self.in_bounds(nxt):
            return False
        return self.is_passable_for(who, nxt)

    def _move(self, who: str, pos: Tuple[int, int], action: int, other: Optional[Tuple[int, int]]) -> Tuple[int, int]:
        if not self.is_valid_action(who, pos, action):
            return pos
        dx = {LEFT: -1, RIGHT: 1}.get(action, 0)
        dy = {UP: -1, DOWN: 1}.get(action, 0)
        nxt = (pos[0] + dx, pos[1] + dy)
        if other is not None and nxt == other:
            return pos
        return nxt

    def _apply_portal(self, pos: Tuple[int, int]) -> Tuple[int, int]:
        if self.portal_pair is None:
            return pos
        a, b = self.portal_pair
        if pos == a:
            return b
        if pos == b:
            return a
        return pos

    def _random_floor_cell(self, exclude: Set[Tuple[int, int]]) -> Optional[Tuple[int, int]]:
        candidates: list[Tuple[int, int]] = []
        for y in range(self.height):
            for x in range(self.width):
                t = self.grid[y][x]
                if t in (T_FLOOR, T_GATE, T_CHECKPOINT):  # goal(5), portal(2)는 고정 느낌이라 스폰 제외
                    p = (x, y)
                    if p not in exclude:
                        candidates.append(p)
        return random.choice(candidates) if candidates else None

    # -------------------- 스폰/리셋 --------------------

    def reset(self):
        # 시작 위치(원하는 대로 바꿔도 됨)
        self.ai1_pos = (1, 1)
        self.ai2_pos = (GRID_W - 2, GRID_H - 2)
        self.ai1_dir = UP
        self.ai2_dir = UP

        self.turn_count = 0

        self.score_ai1 = 0
        self.score_ai2 = 0
        self.items_ai1 = 0
        self.items_ai2 = 0

        self.stun_ai1 = 0
        self.stun_ai2 = 0

        self.used_item_ai1 = False
        self.used_item_ai2 = False

        self.checkpoint_cleared_ai1 = False
        self.checkpoint_cleared_ai2 = False

        self.unlocked_ai1 = False
        self.unlocked_ai2 = False
        self.goal_active = False

        self.has_trap_ai1 = False
        self.has_trap_ai2 = False
        self.has_tranq_ai1 = False
        self.has_tranq_ai2 = False

        self.trap_positions.clear()

        self.coin_positions.clear()
        self._spawn_coins(3)

        self.trap_pickup_pos = None
        self.tranq_pos = None
        self._spawn_trap_pickup()
        self._spawn_tranq_pickup()

        return self._get_obs()
    
    # --- main.py 호환용 별칭(legacy fields) ---
    @property
    def reached_checkpoint_ai1(self) -> bool:
        return bool(getattr(self, "checkpoint_cleared_ai1", False))

    @property
    def reached_checkpoint_ai2(self) -> bool:
        return bool(getattr(self, "checkpoint_cleared_ai2", False))


    def _spawn_coins(self, n: int) -> None:
        exclude: Set[Tuple[int, int]] = set()
        if self.ai1_pos: exclude.add(self.ai1_pos)
        if self.ai2_pos: exclude.add(self.ai2_pos)
        if self.checkpoint_pos: exclude.add(self.checkpoint_pos)
        if self.goal_pos: exclude.add(self.goal_pos)
        exclude |= set(self.portal_positions)

        for _ in range(n):
            p = self._random_floor_cell(exclude | self.coin_positions)
            if p is not None:
                self.coin_positions.add(p)

    def _spawn_trap_pickup(self) -> None:
        exclude: Set[Tuple[int, int]] = set(self.coin_positions) | set(self.portal_positions)
        if self.ai1_pos: exclude.add(self.ai1_pos)
        if self.ai2_pos: exclude.add(self.ai2_pos)
        if self.checkpoint_pos: exclude.add(self.checkpoint_pos)
        if self.goal_pos: exclude.add(self.goal_pos)
        p = self._random_floor_cell(exclude)
        self.trap_pickup_pos = p

    def _spawn_tranq_pickup(self) -> None:
        exclude: Set[Tuple[int, int]] = set(self.coin_positions) | set(self.portal_positions)
        if self.ai1_pos: exclude.add(self.ai1_pos)
        if self.ai2_pos: exclude.add(self.ai2_pos)
        if self.checkpoint_pos: exclude.add(self.checkpoint_pos)
        if self.goal_pos: exclude.add(self.goal_pos)
        if self.trap_pickup_pos: exclude.add(self.trap_pickup_pos)
        p = self._random_floor_cell(exclude)
        self.tranq_pos = p

    # -------------------- 관측 --------------------

    def _get_obs(self):
        obs1 = {"ai1_pos": self.ai1_pos, "ai2_pos": self.ai2_pos, "score_ai1": self.score_ai1, "score_ai2": self.score_ai2}
        obs2 = {"ai2_pos": self.ai2_pos, "ai1_pos": self.ai1_pos, "score_ai1": self.score_ai1, "score_ai2": self.score_ai2}
        return obs1, obs2

    # -------------------- 언락 --------------------

    def _check_unlock_for(self, who: str) -> bool:
        if who == "AI1":
            return (
                self.items_ai1 >= REQUIRED_COINS_FOR_CHECKPOINT
                and self.used_item_ai1
                and self.checkpoint_cleared_ai1
            )
        else:
            return (
                self.items_ai2 >= REQUIRED_COINS_FOR_CHECKPOINT
                and self.used_item_ai2
                and self.checkpoint_cleared_ai2
            )

    def is_goal_unlocked_ai1(self) -> bool:
        return self._check_unlock_for("AI1")

    def is_goal_unlocked_ai2(self) -> bool:
        return self._check_unlock_for("AI2")

    def _update_goal_active(self) -> None:
        self.unlocked_ai1 = self.is_goal_unlocked_ai1()
        self.unlocked_ai2 = self.is_goal_unlocked_ai2()
        self.goal_active = self.unlocked_ai1 or self.unlocked_ai2

    # -------------------- 아이템 처리 --------------------

    @staticmethod
    def _manhattan(a: Tuple[int, int], b: Tuple[int, int]) -> int:
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def _pickup_coin(self) -> None:
        # AI1
        if self.ai1_pos in self.coin_positions:
            self.coin_positions.remove(self.ai1_pos)
            self.items_ai1 += 1
            self.score_ai1 += 3
            self._spawn_coins(1)

        # AI2
        if self.ai2_pos in self.coin_positions:
            self.coin_positions.remove(self.ai2_pos)
            self.items_ai2 += 1
            self.score_ai2 += 3
            self._spawn_coins(1)

    def _pickup_trap_tranq(self) -> None:
        if self.trap_pickup_pos is not None:
            if self.ai1_pos == self.trap_pickup_pos:
                self.has_trap_ai1 = True
                self.trap_pickup_pos = None
            elif self.ai2_pos == self.trap_pickup_pos:
                self.has_trap_ai2 = True
                self.trap_pickup_pos = None

        if self.tranq_pos is not None:
            if self.ai1_pos == self.tranq_pos:
                self.has_tranq_ai1 = True
                self.tranq_pos = None
            elif self.ai2_pos == self.tranq_pos:
                self.has_tranq_ai2 = True
                self.tranq_pos = None

        # pickup이 사라졌으면 재스폰(너무 부족하면 재미없어서)
        if self.trap_pickup_pos is None:
            self._spawn_trap_pickup()
        if self.tranq_pos is None:
            self._spawn_tranq_pickup()

    def _place_trap(self, who: str) -> None:
        if who == "AI1":
            if not self.has_trap_ai1 or self.ai1_pos is None:
                return
            if self.ai1_pos not in self.trap_positions:
                self.trap_positions[self.ai1_pos] = "AI1"
                self.has_trap_ai1 = False
        else:
            if not self.has_trap_ai2 or self.ai2_pos is None:
                return
            if self.ai2_pos not in self.trap_positions:
                self.trap_positions[self.ai2_pos] = "AI2"
                self.has_trap_ai2 = False

    def _trigger_trap_if_any(self) -> None:
        # AI1이 AI2 트랩 밟으면
        owner = self.trap_positions.get(self.ai1_pos)
        if owner == "AI2" and self.stun_ai1 == 0:
            self.stun_ai1 = STUN_TURNS
            self.used_item_ai2 = True  # ✅ 적중 인정
        # AI2가 AI1 트랩 밟으면
        owner = self.trap_positions.get(self.ai2_pos)
        if owner == "AI1" and self.stun_ai2 == 0:
            self.stun_ai2 = STUN_TURNS
            self.used_item_ai1 = True  # ✅ 적중 인정

    def _fire_tranq_auto(self) -> None:
        """인접 시 자동 발사(적중 시 used_item 플래그)."""
        if self.ai1_pos is None or self.ai2_pos is None:
            return

        # AI1 -> AI2
        if self.has_tranq_ai1 and self.stun_ai2 == 0:
            if self._manhattan(self.ai1_pos, self.ai2_pos) == 1:
                self.stun_ai2 = STUN_TURNS
                self.has_tranq_ai1 = False
                self.used_item_ai1 = True

        # AI2 -> AI1
        if self.has_tranq_ai2 and self.stun_ai1 == 0:
            if self._manhattan(self.ai1_pos, self.ai2_pos) == 1:
                self.stun_ai1 = STUN_TURNS
                self.has_tranq_ai2 = False
                self.used_item_ai2 = True

    def _handle_checkpoint(self) -> None:
        if self.checkpoint_pos is None:
            return
        if self.ai1_pos == self.checkpoint_pos:
            self.checkpoint_cleared_ai1 = True
            self.score_ai1 += 5
        if self.ai2_pos == self.checkpoint_pos:
            self.checkpoint_cleared_ai2 = True
            self.score_ai2 += 5

    # -------------------- step --------------------

    def step(self, action1: int, action2: int, use_trap1: bool = False, use_trap2: bool = False):
        """
        main에서 호출: env.step(action1, action2, use_trap1, use_trap2)
        - 이동
        - 포탈
        - 코인/아이템 획득
        - 트랩 설치/발동
        - 마취총 자동 발사
        - 체크포인트
        - 언락/goal_active 갱신
        - 종료조건(goal 도달 or turn_limit)
        """
        self.turn_count += 1

        # 스턴 카운트 감소
        if self.stun_ai1 > 0:
            self.stun_ai1 -= 1
        if self.stun_ai2 > 0:
            self.stun_ai2 -= 1

        # trap 설치(현재 칸)
        if use_trap1:
            self._place_trap("AI1")
        if use_trap2:
            self._place_trap("AI2")

        # 이동(스턴이면 제자리)
        if self.ai1_pos is not None:
            if self.stun_ai1 == 0:
                self.ai1_pos = self._move("AI1", self.ai1_pos, action1, self.ai2_pos)
                self.ai1_pos = self._apply_portal(self.ai1_pos)
            # 방향 기록
            if action1 in (UP, DOWN, LEFT, RIGHT):
                self.ai1_dir = action1

        if self.ai2_pos is not None:
            if self.stun_ai2 == 0:
                self.ai2_pos = self._move("AI2", self.ai2_pos, action2, self.ai1_pos)
                self.ai2_pos = self._apply_portal(self.ai2_pos)
            if action2 in (UP, DOWN, LEFT, RIGHT):
                self.ai2_dir = action2

        # 아이템 획득/처리
        self._pickup_coin()
        self._pickup_trap_tranq()

        # 트랩 발동
        self._trigger_trap_if_any()

        # 마취총 자동 발사(적중 시 used_item True)
        self._fire_tranq_auto()

        # 체크포인트
        self._handle_checkpoint()

        # 언락 / goal_active 업데이트 (gate(3) 잠김 상태 결정)
        self._update_goal_active()

        done = False
        info: Dict[str, Any] = {}

        # goal 도달
        if self.goal_pos is not None:
            if self.ai1_pos == self.goal_pos and self.unlocked_ai1:
                done = True
                info["winner"] = "AI1"
            elif self.ai2_pos == self.goal_pos and self.unlocked_ai2:
                done = True
                info["winner"] = "AI2"

        # 턴 제한 무승부
        if not done and self.turn_count >= self.turn_limit:
            done = True
            info["winner"] = "DRAW"

        obs1, obs2 = self._get_obs()
        return (obs1, obs2), done, info

    # -------------------- render --------------------

    def render(self, surface: pygame.Surface) -> None:
        surface.fill(COLOR_BG)

        # 바닥/벽/격자
        for y in range(self.height):
            for x in range(self.width):
                rect = pygame.Rect(x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE)
                t = self.grid[y][x]

                if t == T_WALL:
                    pygame.draw.rect(surface, COLOR_WALL, rect)
                else:
                    pygame.draw.rect(surface, COLOR_WALL, rect, 1)
                    # 특수타일 강조
                    if t == T_GATE:
                        c = COLOR_GATE_BLOCK if self.goal_active else COLOR_GATE_OPEN
                        pygame.draw.rect(surface, c, rect, 2)
                    elif t == T_CHECKPOINT:
                        pygame.draw.rect(surface, COLOR_CHECKPOINT, rect, 2)
                    elif t == T_GOAL:
                        pygame.draw.rect(surface, COLOR_GOAL, rect, 2)
                    elif t == T_PORTAL:
                        pygame.draw.rect(surface, COLOR_PORTAL, rect, 2)

        # 코인
        for (cx, cy) in self.coin_positions:
            px = cx * CELL_SIZE + CELL_SIZE // 2
            py = cy * CELL_SIZE + CELL_SIZE // 2
            pygame.draw.circle(surface, COLOR_COIN, (px, py), CELL_SIZE // 4)

        # trap pickup
        if self.trap_pickup_pos is not None:
            x, y = self.trap_pickup_pos
            draw_x = x * CELL_SIZE + 4
            draw_y = y * CELL_SIZE + 4
            if self.images_ok and getattr(self, "img_trap_pickup", None) is not None:
                surface.blit(self.img_trap_pickup, (draw_x, draw_y))
            else:
                pygame.draw.rect(surface, COLOR_TRAP, pygame.Rect(draw_x, draw_y, CELL_SIZE - 8, CELL_SIZE - 8), 2)

        # tranq pickup
        if self.tranq_pos is not None:
            x, y = self.tranq_pos
            draw_x = x * CELL_SIZE + 4
            draw_y = y * CELL_SIZE + 4
            if self.images_ok and getattr(self, "img_tranq", None) is not None:
                surface.blit(self.img_tranq, (draw_x, draw_y))
            else:
                pygame.draw.rect(surface, COLOR_TRANQ, pygame.Rect(draw_x, draw_y, CELL_SIZE - 8, CELL_SIZE - 8), 2)

        # placed traps (visible)
        for (tx, ty), owner in self.trap_positions.items():
            draw_x = tx * CELL_SIZE + 4
            draw_y = ty * CELL_SIZE + 4
            if self.images_ok and getattr(self, "img_trap_set", None) is not None:
                surface.blit(self.img_trap_set, (draw_x, draw_y))
            else:
                pygame.draw.rect(surface, COLOR_TRAP, pygame.Rect(draw_x, draw_y, CELL_SIZE - 8, CELL_SIZE - 8), 2)

        # stun shake
        def _shake(stun: int) -> Tuple[int, int]:
            if stun <= 0:
                return (0, 0)
            return (random.randint(-2, 2), random.randint(-2, 2))

        shake1 = _shake(self.stun_ai1)
        shake2 = _shake(self.stun_ai2)

        def _pick_ai_img(is_ai1: bool, d: int) -> Optional[pygame.Surface]:
            if not self.images_ok:
                return None
            if is_ai1:
                return {UP: self.img_ai1_up, DOWN: self.img_ai1_down, LEFT: self.img_ai1_left, RIGHT: self.img_ai1_right}.get(d)
            return {UP: self.img_ai2_up, DOWN: self.img_ai2_down, LEFT: self.img_ai2_left, RIGHT: self.img_ai2_right}.get(d)

        # AI1
        if self.ai1_pos is not None:
            ax, ay = self.ai1_pos
            img = _pick_ai_img(True, self.ai1_dir)
            draw_x = ax * CELL_SIZE + 4 + shake1[0]
            draw_y = ay * CELL_SIZE + 4 + shake1[1]
            if img is not None:
                surface.blit(img, (draw_x, draw_y))
            else:
                pygame.draw.rect(surface, COLOR_AI1, pygame.Rect(draw_x, draw_y, CELL_SIZE - 8, CELL_SIZE - 8))

        # AI2
        if self.ai2_pos is not None:
            ax, ay = self.ai2_pos
            img = _pick_ai_img(False, self.ai2_dir)
            draw_x = ax * CELL_SIZE + 4 + shake2[0]
            draw_y = ay * CELL_SIZE + 4 + shake2[1]
            if img is not None:
                surface.blit(img, (draw_x, draw_y))
            else:
                pygame.draw.rect(surface, COLOR_AI2, pygame.Rect(draw_x, draw_y, CELL_SIZE - 8, CELL_SIZE - 8))
