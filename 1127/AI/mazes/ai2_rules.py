# ai2_rules.py  # ver4

import random
from typing import Dict, Any

from env_maze import UP, DOWN, LEFT, RIGHT, MazeEnv  # ver4


def rule_based_ai2(obs: Dict[str, Any], env: MazeEnv, eps: float = 0.1) -> int:  # ver4
    """
    아주 단순 규칙 기반 AI2:
      1) eps 확률로 랜덤 (탐험)
      2) AI1과 너무 붙어 있으면 멀어지는 방향으로 이동
      3) 그 외에는 그냥 랜덤이지만, 벽은 피함  # ver4
    """
    if random.random() < eps:
        return _random_valid_action(env, obs["ai2_pos"])

    my_pos = obs["ai2_pos"]
    other_pos = obs["ai1_pos"]

    # AI1과 맨해튼 거리 1이면 회피  # ver4
    if abs(my_pos[0] - other_pos[0]) + abs(my_pos[1] - other_pos[1]) == 1:
        away_candidates = []
        if other_pos[0] < my_pos[0]:
            away_candidates.append(RIGHT)
        if other_pos[0] > my_pos[0]:
            away_candidates.append(LEFT)
        if other_pos[1] < my_pos[1]:
            away_candidates.append(DOWN)
        if other_pos[1] > my_pos[1]:
            away_candidates.append(UP)
        for a in away_candidates:
            if env.is_valid_action(my_pos, a):
                return a

    # 그 외엔 유효한 방향 중 랜덤  # ver4
    return _random_valid_action(env, my_pos)


def _random_valid_action(env: MazeEnv, pos) -> int:  # ver4
    actions = [UP, DOWN, LEFT, RIGHT]
    random.shuffle(actions)
    for a in actions:
        if env.is_valid_action(pos, a):
            return a
    return random.choice([UP, DOWN, LEFT, RIGHT])
