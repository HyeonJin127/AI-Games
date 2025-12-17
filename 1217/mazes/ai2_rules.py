# ai2_rules.py  # ver4

import random
from collections import deque
from typing import Dict, Any, List, Tuple, Optional

from env_maze import UP, DOWN, LEFT, RIGHT, MazeEnv

Action = int
Pos = Tuple[int, int]

ACTIONS: List[Action] = [UP, DOWN, LEFT, RIGHT]
DIRS = [(UP, 0, -1), (DOWN, 0, 1), (LEFT, -1, 0), (RIGHT, 1, 0)]


def rule_based_ai2(obs: Dict[str, Any], env: MazeEnv, eps: float = 0.05) -> Tuple[int, bool]:
    who = "AI2"
    my_pos: Pos = obs["ai2_pos"]
    other_pos: Pos = obs["ai1_pos"]

    # 트랩 설치 판단: 인접이고 트랩 보유중이면 현재 칸에 설치
    use_trap = False
    if getattr(env, "has_trap_ai2", False):
        if abs(my_pos[0] - other_pos[0]) + abs(my_pos[1] - other_pos[1]) == 1:
            if my_pos not in getattr(env, "trap_positions", {}):
                use_trap = True

    # 탐험
    if random.random() < eps:
        return _random_valid_action(env, my_pos, who), use_trap

    # 목표: unlocked 전엔 checkpoint(4), unlocked 후엔 goal(5)
    target = env.goal_pos if env.unlocked_ai2 else env.checkpoint_pos
    if target is None:
        return _random_valid_action(env, my_pos, who), use_trap

    a = _bfs_next_action(env, start=my_pos, goal=target, who=who)
    if a is not None:
        return a, use_trap

    return _random_valid_action(env, my_pos, who), use_trap


def _random_valid_action(env: MazeEnv, pos: Pos, who: str) -> int:
    actions = ACTIONS[:]
    random.shuffle(actions)
    for a in actions:
        if env.is_valid_action(who, pos, a):
            return a
    return random.choice(ACTIONS)


def _bfs_next_action(env: MazeEnv, start: Pos, goal: Pos, who: str) -> Optional[int]:
    if start == goal:
        return None

    trap_positions = getattr(env, "trap_positions", {})

    q = deque([start])
    prev: Dict[Pos, Pos] = {}
    prev_action: Dict[Pos, int] = {}
    visited = {start}

    while q:
        cur = q.popleft()
        if cur == goal:
            break

        for a, dx, dy in DIRS:
            nxt = (cur[0] + dx, cur[1] + dy)
            if nxt in visited:
                continue
            if not env.in_bounds(nxt):
                continue
            if not env.is_passable_for(who, nxt):
                continue

            # 상대 트랩 회피(보이는 트랩이면 우회)
            owner = trap_positions.get(nxt)
            if owner == "AI1":
                continue

            visited.add(nxt)
            prev[nxt] = cur
            prev_action[nxt] = a
            q.append(nxt)

    if goal not in visited:
        return None

    cur = goal
    while prev.get(cur) != start:
        cur = prev[cur]
        if cur not in prev:
            return None

    return prev_action[cur]
