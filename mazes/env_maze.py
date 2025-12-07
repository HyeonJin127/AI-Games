# env_maze.py  # ver4

import pygame

# --- 상수들 (u) ---  # ver4
CELL_SIZE = 32  # 격자 한 칸 크기 (픽셀)
GRID_W = 15     # 가로 칸 수
GRID_H = 11     # 세로 칸 수

COLOR_BG    = (10, 10, 10)
COLOR_WALL  = (40, 40, 60)
COLOR_FLOOR = (80, 80, 80)
COLOR_AI1   = (200, 80, 80)   # 빨강
COLOR_AI2   = (80, 120, 220)  # 파랑

UP, DOWN, LEFT, RIGHT = 0, 1, 2, 3


class MazeEnv:  # ver4
    """
    최소 미로 환경: 벽, 빈칸, AI1/AI2 위치, 이동만 처리 (학습/보상 없음).
    """

    def __init__(self):
        # 0 = 빈칸, 1 = 벽
        self.grid = [
            [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
            [1,0,0,0,0,0,0,1,0,0,0,0,0,0,1],
            [1,0,1,1,0,0,0,1,0,0,1,1,0,0,1],
            [1,0,0,1,0,0,0,0,0,0,1,0,0,0,1],
            [1,0,0,1,1,1,0,0,0,0,1,0,0,0,1],
            [1,0,0,0,0,1,0,1,1,0,0,0,1,0,1],
            [1,0,0,0,0,1,0,0,0,0,0,1,1,0,1],
            [1,0,0,1,0,0,0,0,0,1,0,0,0,0,1],
            [1,0,0,1,0,0,0,1,0,1,0,0,1,0,1],
            [1,0,0,0,0,0,0,1,0,0,0,0,0,0,1],
            [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
        ]
        self.width = GRID_W
        self.height = GRID_H

        self.ai1_pos = None  # (x,y)
        self.ai2_pos = None
        self.reset()

    # ----- 기본 유틸 -----  # ver4
    def reset(self):
        """AI1, AI2 위치 초기화."""
        self.ai1_pos = (1, 1)
        self.ai2_pos = (GRID_W - 2, GRID_H - 2)
        return self._get_obs()

    def _get_obs(self):
        """관측값 리턴 (지금은 단순히 위치 정보 dict)."""
        obs1 = {
            "ai1_pos": self.ai1_pos,
            "ai2_pos": self.ai2_pos,
        }
        obs2 = {
            "ai2_pos": self.ai2_pos,
            "ai1_pos": self.ai1_pos,
        }
        return obs1, obs2

    def is_wall(self, pos):
        x, y = pos
        return self.grid[y][x] == 1

    def in_bounds(self, pos):
        x, y = pos
        return 0 <= x < self.width and 0 <= y < self.height

    def is_valid_action(self, pos, action):
        """해당 위치에서 action이 벽/맵 밖이면 False."""
        dx = {LEFT: -1, RIGHT: 1}.get(action, 0)
        dy = {UP: -1, DOWN: 1}.get(action, 0)
        nx, ny = pos[0] + dx, pos[1] + dy
        npos = (nx, ny)
        if not self.in_bounds(npos):
            return False
        if self.is_wall(npos):
            return False
        return True

    def step(self, action1, action2):
        """
        액션 두 개 받아서 위치만 업데이트.
        보상/에피소드 끝 처리 없음 (나중에 강화학습 붙일 때 추가).
        """
        self.ai1_pos = self._move_agent(self.ai1_pos, action1, other=self.ai2_pos)
        self.ai2_pos = self._move_agent(self.ai2_pos, action2, other=self.ai1_pos)

        obs1, obs2 = self._get_obs()
        info = {}
        done = False  # 나중에 목표 도착/체력 등 구현하면서 True로 바꾸면 됨
        return (obs1, obs2), done, info

    def _move_agent(self, pos, action, other=None):
        if not self.is_valid_action(pos, action):
            return pos
        dx = {LEFT: -1, RIGHT: 1}.get(action, 0)
        dy = {UP: -1, DOWN: 1}.get(action, 0)
        npos = (pos[0] + dx, pos[1] + dy)
        if other is not None and npos == other:
            return pos
        return npos

    # ----- pygame 렌더링 -----  # ver4
    def render(self, surface):
        """미로 + 에이전트 그리기."""
        surface.fill(COLOR_BG)
        for y in range(self.height):
            for x in range(self.width):
                rect = pygame.Rect(x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE)
                if self.grid[y][x] == 1:
                    pygame.draw.rect(surface, COLOR_WALL, rect)
                else:
                    pygame.draw.rect(surface, COLOR_FLOOR, rect, 1)

        # AI1
        ax1, ay1 = self.ai1_pos
        r1 = pygame.Rect(
            ax1 * CELL_SIZE + 4, ay1 * CELL_SIZE + 4,
            CELL_SIZE - 8, CELL_SIZE - 8
        )
        pygame.draw.rect(surface, COLOR_AI1, r1)

        # AI2
        ax2, ay2 = self.ai2_pos
        r2 = pygame.Rect(
            ax2 * CELL_SIZE + 4, ay2 * CELL_SIZE + 4,
            CELL_SIZE - 8, CELL_SIZE - 8
        )
        pygame.draw.rect(surface, COLOR_AI2, r2)
