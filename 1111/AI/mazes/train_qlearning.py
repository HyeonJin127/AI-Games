# train_qlearning.py
"""
MazeEnv + 탭형 Q-learning (업그레이드 버전)
- 상태: (agent1_pos, agent2_pos, remaining_missions, items_agent1)
- AI1: Q-learning
- AI2: 규칙 기반 봇
- 에피소드별 리워드 CSV 저장
"""

import random
import csv
from collections import defaultdict
from typing import Tuple, Dict, Any

from env_maze import (
    MazeEnv,
    NUM_ACTIONS,
    ACTION_UP,
    ACTION_DOWN,
    ACTION_LEFT,
    ACTION_RIGHT,
)


# ---------- 1. 상태 인코딩 ----------

def obs_to_state(obs: Dict[str, Any]) -> Tuple[int, int, int, int, int, int]:
    """
    관찰을 탭형 Q 테이블에서 쓸 수 있는 튜플 상태로 변환.
    포함 정보:
    - agent1 위치 (r1, c1)
    - agent2 위치 (r2, c2)
    - remaining_missions (0~여러 개)
    - items_agent1 (0,1,2+ → 2로 클램프)
    """
    r1, c1 = obs["agent1_pos"]
    r2, c2 = obs["agent2_pos"]
    remaining = int(obs.get("remaining_missions", 0))
    items1 = int(obs.get("items_agent1", 0))
    if items1 > 2:
        items1 = 2  # 상태 공간 폭발 방지
    return (r1, c1, r2, c2, remaining, items1)


# ---------- 2. ε-greedy 정책 (AI1) ----------

def select_action_epsilon_greedy(
    q_table: Dict[Tuple[int, ...], list],
    state: Tuple[int, ...],
    epsilon: float,
) -> int:
    if random.random() < epsilon:
        return random.randint(0, NUM_ACTIONS - 1)

    q_values = q_table[state]
    max_q = max(q_values)
    candidates = [a for a, q in enumerate(q_values) if q == max_q]
    return random.choice(candidates)


# ---------- 3. 규칙 기반 AI2 ----------

def _next_pos(row: int, col: int, action: int) -> Tuple[int, int]:
    if action == ACTION_UP:
        return row - 1, col
    if action == ACTION_DOWN:
        return row + 1, col
    if action == ACTION_LEFT:
        return row, col - 1
    if action == ACTION_RIGHT:
        return row, col + 1
    return row, col


def rule_based_action_agent2(env: MazeEnv, obs: Dict[str, Any]) -> int:
    """
    간단한 휴리스틱:
    1) 미션(Goal)이 남아 있으면 가장 가까운 Goal 로 이동
    2) 없으면 AI1에게 다가가도록 이동
    - 벽/범위는 피하려고 시도
    """
    r2, c2 = obs["agent2_pos"]

    # 1) 목표 위치 선택
    goals = []
    for r in range(env.rows):
        for c in range(env.cols):
            if env.grid[r][c] == 2:
                goals.append((r, c))

    if goals:
        target_r, target_c = min(
            goals, key=lambda g: abs(g[0] - r2) + abs(g[1] - c2)
        )
    else:
        # Goal 없으면 AI1 위치로
        target_r, target_c = obs["agent1_pos"]

    # 2) 타겟 방향으로 우선순위 행동 리스트 만들기
    actions_priority = []
    dr = target_r - r2
    dc = target_c - c2

    if dr < 0:
        actions_priority.append(ACTION_UP)
    elif dr > 0:
        actions_priority.append(ACTION_DOWN)
    if dc < 0:
        actions_priority.append(ACTION_LEFT)
    elif dc > 0:
        actions_priority.append(ACTION_RIGHT)

    # 나머지 방향도 랜덤하게 채워넣기 (막혔을 때 대비)
    others = [ACTION_UP, ACTION_DOWN, ACTION_LEFT, ACTION_RIGHT]
    random.shuffle(others)
    for a in others:
        if a not in actions_priority:
            actions_priority.append(a)

    # 3) 유효한 첫 번째 행동 선택
    for a in actions_priority:
        nr, nc = _next_pos(r2, c2, a)
        if env._in_bounds(nr, nc) and (not env._is_wall(nr, nc)):
            return a

    # 전부 막혀 있으면 랜덤
    return random.randint(0, NUM_ACTIONS - 1)


# ---------- 4. Q-learning 학습 루프 ----------

def train_q_learning(
    num_episodes: int = 1000,
    alpha: float = 0.1,
    gamma: float = 0.95,
    epsilon_start: float = 1.0,
    epsilon_end: float = 0.05,
    epsilon_decay_episodes: int = 700,
    render_every: int = 0,
    rewards_csv_path: str = "qlearning_rewards.csv",
):
    env = MazeEnv(cell_size=40, use_pygame=False)

    q_table = defaultdict(lambda: [0.0] * NUM_ACTIONS)
    rewards_per_episode = []

    for episode in range(1, num_episodes + 1):
        obs = env.reset()
        state = obs_to_state(obs)
        done = False
        total_reward_agent1 = 0.0

        # ε 감소
        if episode <= epsilon_decay_episodes:
            epsilon = epsilon_start - (epsilon_start - epsilon_end) * (
                episode - 1
            ) / max(1, epsilon_decay_episodes - 1)
        else:
            epsilon = epsilon_end

        while not done:
            # AI1 행동
            action1 = select_action_epsilon_greedy(q_table, state, epsilon)
            # AI2 규칙 기반
            action2 = rule_based_action_agent2(env, obs)

            next_obs, (reward1, reward2), done, info = env.step(action1, action2)
            next_state = obs_to_state(next_obs)

            total_reward_agent1 += reward1

            old_q = q_table[state][action1]
            if done:
                td_target = reward1
            else:
                next_max_q = max(q_table[next_state])
                td_target = reward1 + gamma * next_max_q

            td_error = td_target - old_q
            q_table[state][action1] = old_q + alpha * td_error

            state = next_state
            obs = next_obs

            if render_every > 0 and episode % render_every == 0:
                env.render()

        rewards_per_episode.append(total_reward_agent1)

        if episode % 50 == 0:
            recent = rewards_per_episode[-50:]
            avg_reward = sum(recent) / len(recent)
            print(
                f"[Episode {episode:4d}] epsilon={epsilon:.3f}, "
                f"recent_avg_reward={avg_reward:.3f}"
            )

    # CSV 저장
    with open(rewards_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["episode", "total_reward_agent1"])
        for i, r in enumerate(rewards_per_episode, start=1):
            writer.writerow([i, r])

    print(f"학습 완료, 리워드 로그: {rewards_csv_path}")
    return q_table


# ---------- 5. 학습된 정책으로 테스트 플레이 ----------

def play_with_trained_policy(q_table, num_episodes: int = 5):
    env = MazeEnv(cell_size=60, use_pygame=True)

    for episode in range(1, num_episodes + 1):
        obs = env.reset()
        state = obs_to_state(obs)
        done = False
        print(f"\n=== 테스트 Episode {episode} 시작 ===")

        while not done:
            action1 = select_action_epsilon_greedy(q_table, state, epsilon=0.0)
            action2 = rule_based_action_agent2(env, obs)

            next_obs, (reward1, reward2), done, info = env.step(action1, action2)
            next_state = obs_to_state(next_obs)

            env.render()
            state = next_state
            obs = next_obs

        print(
            f"Episode {episode} 종료. "
            f"Agent1 score={obs['agent1_score']:.2f}, "
            f"Agent2 score={obs['agent2_score']:.2f}"
        )


if __name__ == "__main__":
    q = train_q_learning(
        num_episodes=1000,
        alpha=0.1,
        gamma=0.95,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay_episodes=700,
        render_every=0,
    )

    play_with_trained_policy(q, num_episodes=3)
