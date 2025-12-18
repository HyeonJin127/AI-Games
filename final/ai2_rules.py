# ai2_rules.py  # ver5-fixed

import random
from collections import deque
from typing import Any, Dict, List, Optional, Tuple

from env_maze import UP, DOWN, LEFT, RIGHT, MazeEnv

Action = int
Pos = Tuple[int, int]

ACTIONS: List[Action] = [UP, DOWN, LEFT, RIGHT]
DIRS = [(UP, 0, -1), (DOWN, 0, 1), (LEFT, -1, 0), (RIGHT, 1, 0)]


def rule_based_ai2(obs: Dict[str, Any], env: MazeEnv, eps: float = 0.07) -> Tuple[int, bool]:
    """
    AI2 목표 우선순위(너가 말한 규칙 기준):
      1) 코인 REQUIRED 미만이면 -> 코인으로 감
      2) 코인 충분하지만 used_item_ai2 False면:
          - 아이템 없으면 -> (trap_pickup / tranq) 중 가까운 곳
          - 아이템 있으면 -> 상대에게 붙어서 적중 시도(인접)
      3) 체크포인트(4) 미도달이면 -> 체크포인트로 감
      4) unlock이면 -> goal로 감
    + 너무 똑똑해지지 않게:
      - eps 랜덤
      - 일정 확률로 '쓸데없는 이동' 섞음
    반환: (action, use_trap)
    """
    who = "AI2"
    my_pos: Pos = obs["ai2_pos"]
    other_pos: Pos = obs["ai1_pos"]

    # ---- 트랩 설치 판단(인접/유효 칸이면 설치) ----
    use_trap = False
    if getattr(env, "has_trap_ai2", False):
        if _manhattan(my_pos, other_pos) == 1:
            if my_pos not in getattr(env, "trap_positions", {}):
                use_trap = True

    # ---- 탐험 랜덤 ----
    if random.random() < eps:
        return _random_valid_action(env, my_pos, who), use_trap

    # ---- 목표 결정 ----
    required = getattr(env, "REQUIRED_COINS_FOR_CHECKPOINT", None)
    # env_maze.py에서 상수로만 들고 있으면 getattr로 못 읽으니, 여기선 직접 env 값으로 판단
    coins = getattr(env, "items_ai2", 0)
    used_item = bool(getattr(env, "used_item_ai2", False))
    cp_cleared = bool(getattr(env, "checkpoint_cleared_ai2", False))

    # env에 함수가 있으면 그걸 신뢰
    unlocked = bool(getattr(env, "unlocked_ai2", False))
    if hasattr(env, "is_goal_unlocked_ai2"):
        try:
            unlocked = bool(env.is_goal_unlocked_ai2())
        except Exception:
            pass

    # 1) 코인 부족 -> 코인 타겟
    if coins < 3:
        target = _nearest(my_pos, list(getattr(env, "coin_positions", set())))
        if target is not None:
            a = _bfs_next_action(env, my_pos, target, who, avoid_owner="AI1")
            return _maybe_waste(env, my_pos, who, a), use_trap

    # 2) 코인 충분하지만 아이템 적중 안함 -> 아이템/상대 접근
    if not used_item:
        has_item = bool(getattr(env, "has_trap_ai2", False) or getattr(env, "has_tranq_ai2", False))
        if not has_item:
            pickups: List[Pos] = []
            tp = getattr(env, "trap_pickup_pos", None)
            tq = getattr(env, "tranq_pos", None)
            if tp is not None:
                pickups.append(tp)
            if tq is not None:
                pickups.append(tq)
            target = _nearest(my_pos, pickups)
            if target is not None:
                a = _bfs_next_action(env, my_pos, target, who, avoid_owner="AI1")
                return _maybe_waste(env, my_pos, who, a), use_trap
        else:
            # 아이템이 있으면 상대에게 붙으러 감(인접)
            a = _bfs_next_action(env, my_pos, other_pos, who, avoid_owner="AI1")
            return _maybe_waste(env, my_pos, who, a), use_trap

    # 3) 체크포인트 미도달 -> 체크포인트
    if not cp_cleared and getattr(env, "checkpoint_pos", None) is not None:
        target = env.checkpoint_pos
        a = _bfs_next_action(env, my_pos, target, who, avoid_owner="AI1")
        return _maybe_waste(env, my_pos, who, a), use_trap

    # 4) unlock이면 goal
    if unlocked and getattr(env, "goal_pos", None) is not None:
        target = env.goal_pos
        a = _bfs_next_action(env, my_pos, target, who, avoid_owner="AI1")
        return _maybe_waste(env, my_pos, who, a), use_trap

    # fallback: 그냥 유효 이동
    return _random_valid_action(env, my_pos, who), use_trap


def _manhattan(a: Pos, b: Pos) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _nearest(start: Pos, points: List[Pos]) -> Optional[Pos]:
    if not points:
        return None
    best = None
    best_d = 10**9
    for p in points:
        d = _manhattan(start, p)
        if d < best_d:
            best_d = d
            best = p
    return best


def _random_valid_action(env: MazeEnv, pos: Pos, who: str) -> int:
    actions = ACTIONS[:]
    random.shuffle(actions)
    for a in actions:
        if env.is_valid_action(who, pos, a):
            return a
    return random.choice(ACTIONS)


def _bfs_next_action(env: MazeEnv, start: Pos, goal: Pos, who: str, avoid_owner: Optional[str] = None) -> Optional[int]:
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

            # 보이는 상대 트랩 회피
            if avoid_owner is not None:
                owner = trap_positions.get(nxt)
                if owner == avoid_owner:
                    continue

            visited.add(nxt)
            prev[nxt] = cur
            prev_action[nxt] = a
            q.append(nxt)

    if goal not in visited:
        return None

    cur = goal
    while prev.get(cur) != start:
        cur = prev.get(cur)
        if cur is None:
            return None

    return prev_action.get(goal) if prev.get(goal) == start else prev_action.get(cur)


def _maybe_waste(env: MazeEnv, pos: Pos, who: str, best_action: Optional[int]) -> int:
    """
    '너무 똑똑함' 방지:
    - 18% 확률로 엉뚱한 유효 행동
    - 그 외엔 best_action(없으면 랜덤)
    """
    if best_action is None:
        return _random_valid_action(env, pos, who)

    if random.random() < 0.18:
        valid = [a for a in ACTIONS if env.is_valid_action(who, pos, a)]
        if valid:
            # best_action과 다른 걸 우선 선택
            others = [a for a in valid if a != best_action]
            return random.choice(others) if others else random.choice(valid)

    return best_action
