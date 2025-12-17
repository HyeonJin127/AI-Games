# env_maze.py  # ver5

import os
import random
import pygame

# --- 상수들 ---
CELL_SIZE = 48
GRID_W = 15
GRID_H = 11

COLOR_TELEPORT = (140, 90, 220)     # 포탈
COLOR_GOAL_OFF = (100, 100, 100)    # 골 비활성
COLOR_GOAL_ON  = (80, 220, 180)     # 골 활성
COLOR_TRANQ    = (230, 200, 60)     # 마취총

COLOR_AI1 = (200, 80, 80)           # AI1
COLOR_AI2 = (80, 120, 220)          # AI2

COLOR_BG = (10, 10, 10)
COLOR_WALL = (40, 40, 60)
COLOR_FLOOR = (80, 80, 80)

COLOR_ITEM = (240, 210, 60)         # 코인(원)
COLOR_GOAL = (200, 200, 200)         # goal 테두리(보조용)
COLOR_CHECKPOINT = (120, 200, 120)   # 체크포인트 테두리
COLOR_PORTAL = (140, 90, 220)

COLOR_TRAP_DEBUG = (220, 120, 80)    # 이미지 없을 때 fallback
COLOR_TRANQ_DEBUG = (230, 200, 60)

STUN_TURNS = 5

UP, DOWN, LEFT, RIGHT = 0, 1, 2, 3
ACTIONS = [UP, DOWN, LEFT, RIGHT]


def load_image(path: str, size: tuple[int, int]) -> pygame.Surface:
    img = pygame.image.load(path).convert_alpha()
    size = (CELL_SIZE - 8, CELL_SIZE - 8)
    return pygame.transform.scale(img, size)

REQUIRED_COINS_FOR_CHECKPOINT = 3

class MazeEnv:
    """
    타일 의미:
    0: floor
    1: wall
    2: portal (2개가 한 쌍)
    3: soft-wall (평소엔 통과 가능, 'unlocked' 상태에서는 벽처럼 막힘)
    4: checkpoint (여기 도달하면 unlocked = True)
    5: goal (unlocked 상태에서만 골인 판정)
    """

    def __init__(self):
        # --- 맵(요구사항 반영) ---
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
        
        base = os.path.dirname(__file__)
        size = (CELL_SIZE - 8, CELL_SIZE - 8)
        
        # --- AI1 ---
        self.img_ai1_up = load_image(
            os.path.join(base, "assets", "ai", "ai1_up.png"), size
        )
        self.img_ai1_down = load_image(
            os.path.join(base, "assets", "ai", "ai1_down.png"), size
        )
        self.img_ai1_left = load_image(
            os.path.join(base, "assets", "ai", "ai1_left.png"), size
        )
        self.img_ai1_right = load_image(
            os.path.join(base, "assets", "ai", "ai1_right.png"), size
        )

        # --- AI2 (PLAYER) ---
        self.img_ai2_up = load_image(
            os.path.join(base, "assets", "ai", "ai2_up.png"), size
        )
        self.img_ai2_down = load_image(
            os.path.join(base, "assets", "ai", "ai2_down.png"), size
        )
        self.img_ai2_left = load_image(
            os.path.join(base, "assets", "ai", "ai2_left.png"), size
        )
        self.img_ai2_right = load_image(
            os.path.join(base, "assets", "ai", "ai2_right.png"), size
        )

        # --- 아이템 ---
        self.img_tranq = load_image(
            os.path.join(base, "assets", "items", "tranq.png"), size
        )
        self.img_trap_pickup = load_image(
            os.path.join(base, "assets", "items", "trap_pickup.png"), size
        )
        self.img_trap_set = load_image(
            os.path.join(base, "assets", "items", "trap_set.png"), size
        )

        self.width = GRID_W
        self.height = GRID_H
        
        # 아이템 사용 여부 (체크포인트 조건용)
        self.used_item_ai1 = False
        self.used_item_ai2 = False


        # 위치들(타일 스캔으로 고정)
        self.portal_positions: list[tuple[int, int]] = []
        self.goal_pos: tuple[int, int] | None = None
        self.checkpoint_pos: tuple[int, int] | None = None

        for y in range(self.height):
            for x in range(self.width):
                t = self.grid[y][x]
                if t == 2:
                    self.portal_positions.append((x, y))
                elif t == 5:
                    self.goal_pos = (x, y)
                elif t == 4:
                    self.checkpoint_pos = (x, y)

        # 포탈은 2개가 한 쌍이라고 가정
        if len(self.portal_positions) != 2:
            print("[WARN] portal tile(2)의 개수가 2개가 아닙니다. 현재:", len(self.portal_positions))

        # --- 이미지 로드 (없으면 fallback) ---
        base = os.path.dirname(__file__)
        try:
            self.img_ai1 = {
                UP: load_image(os.path.join(base, "assets", "ai", "ai1_up.png"), (40, 40)),
                DOWN: load_image(os.path.join(base, "assets", "ai", "ai1_down.png"), (40, 40)),
                LEFT: load_image(os.path.join(base, "assets", "ai", "ai1_left.png"), (40, 40)),
                RIGHT: load_image(os.path.join(base, "assets", "ai", "ai1_right.png"), (40, 40)),
            }
            self.img_ai2 = {
                UP: load_image(os.path.join(base, "assets", "ai", "ai2_up.png"), (40, 40)),
                DOWN: load_image(os.path.join(base, "assets", "ai", "ai2_down.png"), (40, 40)),
                LEFT: load_image(os.path.join(base, "assets", "ai", "ai2_left.png"), (40, 40)),
                RIGHT: load_image(os.path.join(base, "assets", "ai", "ai2_right.png"), (40, 40)),
            }

            # 아이템(현재는 trap.png 1장으로 HUD/설치 모두 사용)
            self.img_trap_pickup = load_image(os.path.join(base, "assets", "items", "trap_pickup.png"), (32, 32))
            self.img_trap_set    = load_image(os.path.join(base, "assets", "items", "trap_set.png"), (32, 32))
            self.img_tranq = load_image(os.path.join(base, "assets", "items", "tranq.png"), (32, 32))
            self.images_ok = True
        except Exception as e:
            print("[WARN] 이미지 로드 실패, 도형 렌더 fallback 사용:", e)
            self.images_ok = False
            self.img_ai1 = {}
            self.img_ai2 = {}
            self.img_trap = None
            self.img_tranq = None

        # 방향 상태
        self.ai1_dir = DOWN
        self.ai2_dir = UP

        # --- 상태 변수 ---
        self.ai1_pos: tuple[int, int] | None = None
        self.ai2_pos: tuple[int, int] | None = None

        self.score_ai1 = 0
        self.score_ai2 = 0

        self.items_ai1 = 0
        self.items_ai2 = 0

        # unlocked: 체크포인트(4) 통과 여부
        self.unlocked_ai1 = False
        self.unlocked_ai2 = False

        # 코인
        self.item_positions: set[tuple[int, int]] = set()

        # 마취총
        self.tranq_pos: tuple[int, int] | None = None
        self.has_tranq_ai1 = False
        self.has_tranq_ai2 = False
        self.stun_ai1 = 0
        self.stun_ai2 = 0

        # 트랩(설치형) + 보유
        self.trap_positions: dict[tuple[int, int], str] = {}   # {(x,y): "AI1"/"AI2"}
        self.has_trap_ai1 = False
        self.has_trap_ai2 = False
        self.trap_pickup_pos: tuple[int, int] | None = None

        # 라운드 제한(타이머 대용)
        self.turn_limit = 250
        self.turn_count = 0

        self.reset()

    # ---- 호환 프로퍼티(예전 main.py 참조 보호) ----
    @property
    def goal_active(self) -> bool:
        return bool(self.unlocked_ai1 or self.unlocked_ai2)

    @property
    def checkpoint_cleared_ai1(self) -> bool:
        return bool(self.unlocked_ai1)

    @property
    def checkpoint_cleared_ai2(self) -> bool:
        return bool(self.unlocked_ai2)

    # ---- 유틸 ----
    def in_bounds(self, pos: tuple[int, int]) -> bool:
        x, y = pos
        return 0 <= x < self.width and 0 <= y < self.height

    def tile_at(self, pos: tuple[int, int]) -> int:
        x, y = pos
        return self.grid[y][x]

    def is_passable_for(self, who: str, pos: tuple[int, int]) -> bool:
        """who에 따라 '가변벽(3)' 통행 여부가 바뀜."""
        if not self.in_bounds(pos):
            return False
        t = self.tile_at(pos)
        if t == 1:
            return False

        # 3: unlocked 상태에서는 벽처럼 막힘
        if t == 3:
            if who == "AI1" and self.unlocked_ai1:
                return False
            if who == "AI2" and self.unlocked_ai2:
                return False

        return True

    def is_valid_action(self, who: str, pos: tuple[int, int], action: int) -> bool:
        dx = {LEFT: -1, RIGHT: 1}.get(action, 0)
        dy = {UP: -1, DOWN: 1}.get(action, 0)
        nxt = (pos[0] + dx, pos[1] + dy)
        return self.is_passable_for(who, nxt)

    def _random_empty_cell(self, exclude: set[tuple[int, int]]) -> tuple[int, int] | None:
        candidates: list[tuple[int, int]] = []
        for y in range(self.height):
            for x in range(self.width):
                t = self.grid[y][x]
                if t in (0, 3):  # 기본 통과 가능 후보(3도 평소 floor 취급)
                    p = (x, y)
                    if p not in exclude:
                        candidates.append(p)
        return random.choice(candidates) if candidates else None

    def _apply_portal_if_needed(self, pos: tuple[int, int]) -> tuple[int, int]:
        if len(self.portal_positions) != 2:
            return pos
        a, b = self.portal_positions
        if pos == a:
            return b
        if pos == b:
            return a
        return pos

    def reset(self):
        self.ai1_pos = (1, 1)
        self.ai2_pos = (GRID_W - 2, GRID_H - 2)

        self.ai1_dir = DOWN
        self.ai2_dir = UP

        self.score_ai1 = 0
        self.score_ai2 = 0

        self.items_ai1 = 0
        self.items_ai2 = 0

        self.unlocked_ai1 = False
        self.unlocked_ai2 = False

        self.turn_count = 0
        
        self.used_item_ai1 = False
        self.used_item_ai2 = False


        # 코인 2개
        self.item_positions = set()
        self._spawn_items(2)

        # 트랩/마취총
        self.trap_positions = {}
        self.has_trap_ai1 = False
        self.has_trap_ai2 = False
        self.trap_pickup_pos = None
        self._spawn_trap_pickup()

        self.tranq_pos = None
        self.has_tranq_ai1 = False
        self.has_tranq_ai2 = False
        self.stun_ai1 = 0
        self.stun_ai2 = 0
        self._spawn_tranq()

        return self._get_obs()

    def _get_obs(self):
        obs1 = {
            "ai1_pos": self.ai1_pos,
            "ai2_pos": self.ai2_pos,
            "score_ai1": self.score_ai1,
            "score_ai2": self.score_ai2,
        }
        obs2 = {
            "ai2_pos": self.ai2_pos,
            "ai1_pos": self.ai1_pos,
            "score_ai1": self.score_ai1,
            "score_ai2": self.score_ai2,
        }
        return obs1, obs2

    def _spawn_items(self, n: int):
        exclude = {self.ai1_pos, self.ai2_pos, self.checkpoint_pos, self.goal_pos}
        exclude |= set(self.item_positions)
        exclude = {p for p in exclude if p is not None}

        for _ in range(n):
            pos = self._random_empty_cell(exclude)
            if pos:
                self.item_positions.add(pos)
                exclude.add(pos)

    def _spawn_tranq(self):
        exclude = {self.ai1_pos, self.ai2_pos, self.checkpoint_pos, self.goal_pos}
        exclude |= set(self.item_positions)
        exclude |= set(self.trap_positions.keys())
        if self.trap_pickup_pos is not None:
            exclude.add(self.trap_pickup_pos)
        exclude = {p for p in exclude if p is not None}

        self.tranq_pos = self._random_empty_cell(exclude)

    def _spawn_trap_pickup(self):
        exclude = {self.ai1_pos, self.ai2_pos, self.checkpoint_pos, self.goal_pos}
        exclude |= set(self.item_positions)
        exclude |= set(self.trap_positions.keys())
        if self.tranq_pos is not None:
            exclude.add(self.tranq_pos)
        exclude = {p for p in exclude if p is not None}

        self.trap_pickup_pos = self._random_empty_cell(exclude)

    @staticmethod
    def _manhattan(a: tuple[int, int], b: tuple[int, int]) -> int:
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def step(self, action1: int, action2: int, use_trap1: bool = False, use_trap2: bool = False):
        self.turn_count += 1

        # 스턴 감소
        if self.stun_ai1 > 0:
            self.stun_ai1 -= 1
        if self.stun_ai2 > 0:
            self.stun_ai2 -= 1

        # 방향 업데이트(스턴이어도 '보이는 방향'은 마지막 입력 기준 유지)
        if action1 in ACTIONS:
            self.ai1_dir = action1
        if action2 in ACTIONS:
            self.ai2_dir = action2

        # 트랩 설치(현재 위치에 설치) - 이동 전에 처리
        if use_trap1 and self.has_trap_ai1 and self.ai1_pos not in self.trap_positions:
            self.trap_positions[self.ai1_pos] = "AI1"
            self.has_trap_ai1 = False

        if use_trap2 and self.has_trap_ai2 and self.ai2_pos not in self.trap_positions:
            self.trap_positions[self.ai2_pos] = "AI2"
            self.has_trap_ai2 = False

        # 이동
        prev1 = self.ai1_pos
        prev2 = self.ai2_pos

        new1 = prev1
        new2 = prev2

        if self.stun_ai1 == 0:
            new1 = self._move_agent("AI1", prev1, action1, other=prev2)
            new1 = self._apply_portal_if_needed(new1)

        if self.stun_ai2 == 0:
            new2 = self._move_agent("AI2", prev2, action2, other=new1)
            new2 = self._apply_portal_if_needed(new2)

        self.ai1_pos = new1
        self.ai2_pos = new2

        # 트랩 발동(밟으면 스턴)
        self._handle_trap_trigger()

        # 코인 처리
        self._handle_items()

        # 체크포인트 처리(4)
        self._handle_checkpoint()

        # 트랩/마취총 픽업
        self._handle_trap_pickup()
        self._handle_tranq_pickup()

        # 마취총 발사(인접 자동)
        self._handle_tranq_fire()

        # 종료 판정(골 5는 unlocked 상태에서만)
        done = False
        info: dict = {}

        # 턴 제한(무승부 판정은 main.py에서 처리하는 게 보통이라 여기서는 info만 전달)
        if self.turn_count >= self.turn_limit:
            done = True
            info["winner"] = None
            info["reason"] = "TIMEOUT"

        if not done and self.goal_pos is not None:
            if self.ai1_pos == self.goal_pos and self.unlocked_ai1:
                done = True
                info["winner"] = "AI1"
                info["reason"] = "GOAL"
            elif self.ai2_pos == self.goal_pos and self.unlocked_ai2:
                done = True
                info["winner"] = "AI2"
                info["reason"] = "GOAL"

        obs1, obs2 = self._get_obs()
        return (obs1, obs2), done, info

    def _move_agent(self, who: str, pos: tuple[int, int], action: int, other: tuple[int, int] | None):
        dx = {LEFT: -1, RIGHT: 1}.get(action, 0)
        dy = {UP: -1, DOWN: 1}.get(action, 0)
        nxt = (pos[0] + dx, pos[1] + dy)
        if not self.is_passable_for(who, nxt):
            return pos
        if other is not None and nxt == other:
            return pos
        return nxt

    def _handle_items(self):
        if self.ai1_pos in self.item_positions:
            self.item_positions.remove(self.ai1_pos)
            self.score_ai1 += 3
            self.items_ai1 += 1
            self._spawn_items(1)

        if self.ai2_pos in self.item_positions:
            self.item_positions.remove(self.ai2_pos)
            self.score_ai2 += 3
            self.items_ai2 += 1
            self._spawn_items(1)

    def _handle_checkpoint(self):
        if self.checkpoint_pos is None:
            return

        # AI1
        if (
            self.ai1_pos == self.checkpoint_pos
            and not self.unlocked_ai1
            and self.items_ai1 >= REQUIRED_COINS_FOR_CHECKPOINT
            and self.used_item_ai1
        ):
            self.unlocked_ai1 = True

        # AI2
        if (
            self.ai2_pos == self.checkpoint_pos
            and not self.unlocked_ai2
            and self.items_ai2 >= REQUIRED_COINS_FOR_CHECKPOINT
            and self.used_item_ai2
        ):
            self.unlocked_ai2 = True


    def _handle_tranq_pickup(self):
        if self.tranq_pos is None:
            return
        if self.ai1_pos == self.tranq_pos:
            self.has_tranq_ai1 = True
            self.tranq_pos = None
        elif self.ai2_pos == self.tranq_pos:
            self.has_tranq_ai2 = True
            self.tranq_pos = None

    def _handle_tranq_fire(self) -> None:
        """마취총 보유 + 상하좌우 인접 시 자동 발사 → 상대 스턴 + 아이템 사용 플래그."""
        if self.ai1_pos is None or self.ai2_pos is None:
            return

        # AI1 → AI2
        if self.has_tranq_ai1 and self.stun_ai2 == 0:
            if self._manhattan(self.ai1_pos, self.ai2_pos) == 1:
                self.stun_ai2 = STUN_TURNS
                self.has_tranq_ai1 = False
                self.used_item_ai1 = True   # ✅ 아이템을 상대에게 1회 이상 사용 조건 충족

        # AI2 → AI1
        if self.has_tranq_ai2 and self.stun_ai1 == 0:
            if self._manhattan(self.ai1_pos, self.ai2_pos) == 1:
                self.stun_ai1 = STUN_TURNS
                self.has_tranq_ai2 = False
                self.used_item_ai2 = True   # ✅ 아이템을 상대에게 1회 이상 사용 조건 충족


    def _handle_trap_pickup(self):
        if self.trap_pickup_pos is None:
            return
        if self.ai1_pos == self.trap_pickup_pos:
            self.has_trap_ai1 = True
            self.trap_pickup_pos = None
        elif self.ai2_pos == self.trap_pickup_pos:
            self.has_trap_ai2 = True
            self.trap_pickup_pos = None

        # 트랩 아이템이 없어졌으면 다시 하나 스폰(원하면 확률로 바꿔도 됨)
        if self.trap_pickup_pos is None:
            self._spawn_trap_pickup()

    def _handle_trap_trigger(self):
        # AI1이 AI2 트랩 밟으면 스턴
        o = self.trap_positions.get(self.ai1_pos)
        if o == "AI2" and self.stun_ai1 == 0:
            self.stun_ai1 = 3
            self.used_item_ai2 = True

        # AI2가 AI1 트랩 밟으면 스턴
        o = self.trap_positions.get(self.ai2_pos)
        if o == "AI1" and self.stun_ai2 == 0:
            self.stun_ai2 = 3
            self.used_item_ai1 = True
        

        # ---- 렌더링 ----
    def render(self, surface):
        """맵 + 오브젝트 + 에이전트 렌더링 (이미지 우선, 실패 시 도형 fallback)."""
        surface.fill(COLOR_BG)

        # -----------------------------
        # TESTver4: 스턴 흔들림(시각 피드백)
        # -----------------------------
        shake_px = 3
        shake1 = (0, 0)
        shake2 = (0, 0)
        if getattr(self, "stun_ai1", 0) > 0:
            shake1 = (random.randint(-shake_px, shake_px), random.randint(-shake_px, shake_px))
        if getattr(self, "stun_ai2", 0) > 0:
            shake2 = (random.randint(-shake_px, shake_px), random.randint(-shake_px, shake_px))

        # -----------------------------
        # 1) 타일 렌더 (grid 코드 기반)
        #   0: 바닥, 1: 벽, 2: 포탈, 3: 가변벽(평소 통과/언락 후 벽), 4: 체크포인트, 5: 골
        # -----------------------------
        gate_closed = bool(getattr(self, "gate_closed", False))  # (u) env 로직에서 관리
        for y in range(self.height):
            for x in range(self.width):
                v = self.grid[y][x]
                rect = pygame.Rect(x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE)

                if v == 1:
                    pygame.draw.rect(surface, COLOR_WALL, rect)
                    continue

                # 기본 바닥
                pygame.draw.rect(surface, COLOR_FLOOR, rect, 1)

                # 포탈(2)
                if v == 2:
                    inner = pygame.Rect(rect.x + 4, rect.y + 4, CELL_SIZE - 8, CELL_SIZE - 8)
                    pygame.draw.rect(surface, COLOR_TELEPORT, inner, 3, border_radius=6)

                # 가변벽(3): 평소엔 통과 가능(표시만), gate_closed면 벽처럼 보이게
                elif v == 3:
                    if gate_closed:
                        pygame.draw.rect(surface, COLOR_WALL, rect)
                    else:
                        inner = pygame.Rect(rect.x + 6, rect.y + 6, CELL_SIZE - 12, CELL_SIZE - 12)
                        pygame.draw.rect(surface, (120, 120, 160), inner, 2, border_radius=6)

                # 체크포인트(4)
                elif v == 4:
                    inner = pygame.Rect(rect.x + 6, rect.y + 6, CELL_SIZE - 12, CELL_SIZE - 12)
                    pygame.draw.rect(surface, COLOR_CHECKPOINT, inner, 2)

                # 골(5)
                elif v == 5:
                    inner = pygame.Rect(rect.x + 4, rect.y + 4, CELL_SIZE - 8, CELL_SIZE - 8)
                    # goal_active 같은 변수가 있으면 반영, 없으면 기본 색으로
                    is_active = bool(getattr(self, "goal_active", True))
                    color = COLOR_GOAL_ON if is_active else COLOR_GOAL_OFF
                    pygame.draw.rect(surface, color, inner, 3)

        # -----------------------------
        # 2) 코인 아이템
        # -----------------------------
        for (ix, iy) in getattr(self, "item_positions", set()):
            cx = ix * CELL_SIZE + CELL_SIZE // 2
            cy = iy * CELL_SIZE + CELL_SIZE // 2
            r = CELL_SIZE // 4
            pygame.draw.circle(surface, COLOR_ITEM, (cx, cy), r)

        # -----------------------------
        # 3) 트랩 "줍기" 아이템 (있다면)
        #    - 네 코드에서 trap_item_pos / trap_pickup_pos 등 이름이 다를 수 있으니 getattr로 흡수
        # -----------------------------
        trap_pickup_pos = getattr(self, "trap_item_pos", None)
        if trap_pickup_pos is None:
            trap_pickup_pos = getattr(self, "trap_pickup_pos", None)

        if trap_pickup_pos is not None:
            tx, ty = trap_pickup_pos
            px = tx * CELL_SIZE + 8
            py = ty * CELL_SIZE + 8
            if getattr(self, "images_ok", False) and getattr(self, "img_trap_pickup", None) is not None:
                surface.blit(self.img_trap_pickup, (px, py))
            else:
                pygame.draw.rect(surface, (220, 140, 60), (px, py, CELL_SIZE - 16, CELL_SIZE - 16), 0)

        # -----------------------------
        # 4) 설치된 트랩(보이는 설정)
        #    - trap_positions: dict[(x,y)] = owner(1/2) 혹은 set 형태일 수 있음
        # -----------------------------
        trap_positions = getattr(self, "trap_positions", None)
        if trap_positions:
            if isinstance(trap_positions, dict):
                positions_iter = trap_positions.keys()
            else:
                positions_iter = trap_positions  # set/list 가정

            for (tx, ty) in positions_iter:
                px = tx * CELL_SIZE + 8
                py = ty * CELL_SIZE + 8
                if getattr(self, "images_ok", False) and getattr(self, "img_trap_set", None) is not None:
                    surface.blit(self.img_trap_set, (px, py))
                else:
                    pygame.draw.rect(surface, (240, 90, 90), (px, py, CELL_SIZE - 16, CELL_SIZE - 16), 0)

        # -----------------------------
        # 5) 마취총 아이템
        # -----------------------------
        if getattr(self, "tranq_pos", None) is not None:
            tx, ty = self.tranq_pos
            px = tx * CELL_SIZE + 8
            py = ty * CELL_SIZE + 8
            if getattr(self, "images_ok", False) and getattr(self, "img_tranq", None) is not None:
                surface.blit(self.img_tranq, (px, py))
            else:
                pygame.draw.circle(
                    surface,
                    COLOR_TRANQ,
                    (tx * CELL_SIZE + CELL_SIZE // 2, ty * CELL_SIZE + CELL_SIZE // 2),
                    CELL_SIZE // 3,
                )

        # -----------------------------
        # 6) 에이전트 렌더 (이미지 방향 + 스턴 흔들림)
        # -----------------------------
        def _pick_ai_img(is_ai1: bool, direction: int):
            # direction: UP/DOWN/LEFT/RIGHT
            if is_ai1:
                return {
                    UP: getattr(self, "img_ai1_up", None),
                    DOWN: getattr(self, "img_ai1_down", None),
                    LEFT: getattr(self, "img_ai1_left", None),
                    RIGHT: getattr(self, "img_ai1_right", None),
                }.get(direction, getattr(self, "img_ai1_up", None))
            else:
                return {
                    UP: getattr(self, "img_ai2_up", None),
                    DOWN: getattr(self, "img_ai2_down", None),
                    LEFT: getattr(self, "img_ai2_left", None),
                    RIGHT: getattr(self, "img_ai2_right", None),
                }.get(direction, getattr(self, "img_ai2_up", None))

        # AI1
        if self.ai1_pos is not None:
            ax1, ay1 = self.ai1_pos
            dir1 = getattr(self, "ai1_dir", UP)
            img1 = _pick_ai_img(True, dir1)

            draw_x1 = ax1 * CELL_SIZE + 4 + shake1[0]   # TESTver4
            draw_y1 = ay1 * CELL_SIZE + 4 + shake1[1]   # TESTver4

            if img1 is not None:
                surface.blit(img1, (draw_x1, draw_y1))
            else:
                r1 = pygame.Rect(draw_x1, draw_y1, CELL_SIZE - 8, CELL_SIZE - 8)
                pygame.draw.rect(surface, COLOR_AI1, r1)

        # AI2
        if self.ai2_pos is not None:
            ax2, ay2 = self.ai2_pos
            dir2 = getattr(self, "ai2_dir", UP)
            img2 = _pick_ai_img(False, dir2)

            draw_x2 = ax2 * CELL_SIZE + 4 + shake2[0]   # TESTver4
            draw_y2 = ay2 * CELL_SIZE + 4 + shake2[1]   # TESTver4

            if img2 is not None:
                surface.blit(img2, (draw_x2, draw_y2))
            else:
                r2 = pygame.Rect(draw_x2, draw_y2, CELL_SIZE - 8, CELL_SIZE - 8)
                pygame.draw.rect(surface, COLOR_AI2, r2)
