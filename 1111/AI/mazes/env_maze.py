# env_maze.py
"""
미로 환경 베이스 코드 (업그레이드 버전)
- 2인용 (AI1 vs AI2)
- Gym 스타일: reset(), step(), render()
- 상태에 미션/아이템 관련 정보 포함
"""

import random
from dataclasses import dataclass
from typing import Tuple, Dict, Any, Optional, Set

try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    pygame = None
    PYGAME_AVAILABLE = False


ACTION_UP = 0
ACTION_DOWN = 1
ACTION_LEFT = 2
ACTION_RIGHT = 3
NUM_ACTIONS = 4  # 일단 이동 4개만


@dataclass
class AgentState:
    row: int
    col: int
    score: float = 0.0
    items: int = 0  # 들고 있는 아이템 개수


class MazeEnv:
    """
    미로 환경 (2인용, AI1 vs AI2)
    - 격자 기반
    - 0: 빈칸, 1: 벽, 2: 미션/Goal
    - item_cells: 랜덤 아이템 위치
    """

    def __init__(self, cell_size: int = 40, use_pygame: Optional[bool] = None):
        if use_pygame is None:
            self.use_pygame = PYGAME_AVAILABLE
        else:
            self.use_pygame = use_pygame and PYGAME_AVAILABLE

        # 기본 미로 레이아웃
        base_grid = [
            [0, 0, 0, 0, 0, 0, 0],
            [0, 1, 1, 0, 1, 1, 0],
            [0, 0, 0, 0, 0, 1, 0],
            [0, 1, 0, 2, 0, 0, 0],  # (3,3)에 Goal(2)
            [0, 1, 0, 0, 0, 1, 0],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 1, 0, 0, 0, 1, 0],
        ]
        # reset마다 복사해서 사용
        self.base_grid = [row[:] for row in base_grid]
        self.grid = [row[:] for row in base_grid]

        self.rows = len(self.grid)
        self.cols = len(self.grid[0])

        self.start_pos_agent1 = (0, 0)
        self.start_pos_agent2 = (6, 6)

        # 리워드 설정
        self.step_penalty = -0.1
        self.hit_wall_penalty = -1.0
        self.goal_reward = 10.0
        self.item_reward = 2.0
        self.max_steps = 200

        # 아이템 관련
        self.num_random_items = 3
        self.item_cells: Set[Tuple[int, int]] = set()

        # 내부 상태
        self.agent1: AgentState = AgentState(*self.start_pos_agent1)
        self.agent2: AgentState = AgentState(*self.start_pos_agent2)
        self.current_step = 0
        self.done = False

        # 렌더링
        self.cell_size = cell_size
        self.screen = None
        self.clock = None

    # ---------- Gym 스타일 메서드 ----------

    def reset(self) -> Dict[str, Any]:
        """환경 리셋"""
        self.grid = [row[:] for row in self.base_grid]

        self.agent1 = AgentState(*self.start_pos_agent1)
        self.agent2 = AgentState(*self.start_pos_agent2)
        self.current_step = 0
        self.done = False

        # 랜덤 아이템 스폰
        self._spawn_items()

        return self._get_observation()

    def step(
        self,
        action_agent1: int,
        action_agent2: Optional[int] = None,
    ) -> Tuple[Dict[str, Any], Tuple[float, float], bool, Dict[str, Any]]:
        """
        한 스텝 진행
        반환:
            next_state, (reward1, reward2), done, info
        """
        if self.done:
            return self._get_observation(), (0.0, 0.0), True, {}

        if action_agent2 is None:
            action_agent2 = -1  # 아무 행동 안 함

        self.current_step += 1

        reward1, reached_goal1 = self._move_agent(self.agent1, action_agent1)
        reward2, reached_goal2 = self._move_agent(self.agent2, action_agent2)

        # 기본 스텝 패널티
        reward1 += self.step_penalty
        reward2 += self.step_penalty

        self.agent1.score += reward1
        self.agent2.score += reward2

        if reached_goal1 or reached_goal2 or self.current_step >= self.max_steps:
            self.done = True

        info = {
            "reached_goal_agent1": reached_goal1,
            "reached_goal_agent2": reached_goal2,
            "steps": self.current_step,
        }

        next_state = self._get_observation()
        return next_state, (reward1, reward2), self.done, info

    def render(self):
        """현재 상태 시각화"""
        if self.use_pygame and PYGAME_AVAILABLE:
            self._render_pygame()
        else:
            self._render_text()

    # ---------- 내부 유틸 ----------

    def _spawn_items(self):
        """빈 칸 중에서 랜덤 위치에 아이템 스폰"""
        self.item_cells.clear()
        empty_cells = [
            (r, c)
            for r in range(self.rows)
            for c in range(self.cols)
            if self.grid[r][c] == 0
        ]
        random.shuffle(empty_cells)
        for pos in empty_cells[: self.num_random_items]:
            self.item_cells.add(pos)

    def _in_bounds(self, row: int, col: int) -> bool:
        return 0 <= row < self.rows and 0 <= col < self.cols

    def _is_wall(self, row: int, col: int) -> bool:
        return self.grid[row][col] == 1

    def _is_goal(self, row: int, col: int) -> bool:
        return self.grid[row][col] == 2

    def _move_agent(self, agent: AgentState, action: int) -> Tuple[float, bool]:
        """에이전트 이동 및 리워드 계산"""
        if action not in (ACTION_UP, ACTION_DOWN, ACTION_LEFT, ACTION_RIGHT):
            return 0.0, False

        drow, dcol = 0, 0
        if action == ACTION_UP:
            drow, dcol = -1, 0
        elif action == ACTION_DOWN:
            drow, dcol = 1, 0
        elif action == ACTION_LEFT:
            drow, dcol = 0, -1
        elif action == ACTION_RIGHT:
            drow, dcol = 0, 1

        new_row = agent.row + drow
        new_col = agent.col + dcol

        if not self._in_bounds(new_row, new_col):
            return self.hit_wall_penalty, False

        if self._is_wall(new_row, new_col):
            return self.hit_wall_penalty, False

        # 실제 이동
        agent.row = new_row
        agent.col = new_col

        reward = 0.0
        reached_goal = False

        # 아이템 획득
        if (new_row, new_col) in self.item_cells:
            self.item_cells.remove((new_row, new_col))
            agent.items += 1
            reward += self.item_reward

        # Goal 도달
        if self._is_goal(new_row, new_col):
            reward += self.goal_reward
            # 한 번 밟은 Goal은 없앰 (미션 완료)
            self.grid[new_row][new_col] = 0
            reached_goal = True

        return reward, reached_goal

    def _get_observation(self) -> Dict[str, Any]:
        """현재 상태를 딕셔너리로 반환"""
        remaining_missions = sum(
            1 for r in range(self.rows) for c in range(self.cols) if self.grid[r][c] == 2
        )
        return {
            "agent1_pos": (self.agent1.row, self.agent1.col),
            "agent2_pos": (self.agent2.row, self.agent2.col),
            "agent1_score": self.agent1.score,
            "agent2_score": self.agent2.score,
            "steps": self.current_step,
            "remaining_missions": remaining_missions,
            "items_agent1": self.agent1.items,
            "items_agent2": self.agent2.items,
            "num_items_on_map": len(self.item_cells),
        }

    # ---------- 텍스트 렌더 ----------

    def _render_text(self):
        display_grid = [[cell for cell in row] for row in self.grid]

        # 아이템 표시
        for (ir, ic) in self.item_cells:
            if display_grid[ir][ic] == 0:
                display_grid[ir][ic] = "*"

        r1, c1 = self.agent1.row, self.agent1.col
        r2, c2 = self.agent2.row, self.agent2.col

        if (r1, c1) == (r2, c2):
            display_grid[r1][c1] = "B"
        else:
            display_grid[r1][c1] = "A"
            display_grid[r2][c2] = "E"

        print("Step:", self.current_step)
        for row in display_grid:
            line = ""
            for cell in row:
                if cell == 0:
                    line += ". "
                elif cell == 1:
                    line += "# "
                elif cell == 2:
                    line += "G "
                elif cell == "*":
                    line += "* "
                elif cell == "A":
                    line += "A "
                elif cell == "E":
                    line += "E "
                elif cell == "B":
                    line += "B "
                else:
                    line += "? "
            print(line)
        print(
            f"Agent1 score: {self.agent1.score:.2f}, items={self.agent1.items} | "
            f"Agent2 score: {self.agent2.score:.2f}, items={self.agent2.items}"
        )
        print("-" * 40)

    # ---------- pygame 렌더 ----------

    def _render_pygame(self):
        if self.screen is None:
            pygame.init()
            width = self.cols * self.cell_size
            height = self.rows * self.cell_size
            self.screen = pygame.display.set_mode((width, height))
            pygame.display.set_caption("MazeEnv - AI1 vs AI2")
            self.clock = pygame.time.Clock()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                self.screen = None
                return

        COLOR_EMPTY = (240, 240, 240)
        COLOR_WALL = (50, 50, 50)
        COLOR_GOAL = (0, 200, 0)
        COLOR_AGENT1 = (0, 0, 255)
        COLOR_AGENT2 = (255, 0, 0)
        COLOR_BOTH = (200, 0, 200)
        COLOR_ITEM = (255, 215, 0)

        self.screen.fill((0, 0, 0))

        for r in range(self.rows):
            for c in range(self.cols):
                cell = self.grid[r][c]
                x = c * self.cell_size
                y = r * self.cell_size

                if cell == 0:
                    color = COLOR_EMPTY
                elif cell == 1:
                    color = COLOR_WALL
                elif cell == 2:
                    color = COLOR_GOAL
                else:
                    color = COLOR_EMPTY

                pygame.draw.rect(
                    self.screen,
                    color,
                    pygame.Rect(x, y, self.cell_size, self.cell_size),
                )

        # 아이템 그리기
        for (ir, ic) in self.item_cells:
            cx = ic * self.cell_size + self.cell_size // 2
            cy = ir * self.cell_size + self.cell_size // 2
            pygame.draw.circle(self.screen, COLOR_ITEM, (cx, cy), self.cell_size // 4)

        # 에이전트
        r1, c1 = self.agent1.row, self.agent1.col
        r2, c2 = self.agent2.row, self.agent2.col

        if (r1, c1) == (r2, c2):
            color = COLOR_BOTH
            pygame.draw.rect(
                self.screen,
                color,
                pygame.Rect(c1 * self.cell_size, r1 * self.cell_size, self.cell_size, self.cell_size),
            )
        else:
            pygame.draw.rect(
                self.screen,
                COLOR_AGENT1,
                pygame.Rect(c1 * self.cell_size, r1 * self.cell_size, self.cell_size, self.cell_size),
            )
            pygame.draw.rect(
                self.screen,
                COLOR_AGENT2,
                pygame.Rect(c2 * self.cell_size, r2 * self.cell_size, self.cell_size, self.cell_size),
            )

        pygame.display.flip()
        self.clock.tick(10)
