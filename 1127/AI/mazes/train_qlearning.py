# train_qlearning.py  # ver4

import random
import pickle
import csv
from collections import defaultdict
from typing import Tuple

from env_maze import MazeEnv, GRID_W, GRID_H, UP, DOWN, LEFT, RIGHT  # ver4
from ai2_rules import rule_based_ai2  # ver4


Action = int
State = Tuple[int, int]  # (x, y) - AI1 위치를 상태로 사용  # ver4
ACTIONS = [UP, DOWN, LEFT, RIGHT]  # ver4


def choose_action_epsilon_greedy(Q, state: State, epsilon: float) -> Action:  # ver4
    if random.random() < epsilon:
        return random.choice(ACTIONS)
    qs = [Q[(state, a)] for a in ACTIONS]
    max_q = max(qs)
    candidates = [a for a, q in zip(ACTIONS, qs) if q == max_q]
    return random.choice(candidates)


def compute_reward(prev_pos: State, new_pos: State, goal_pos: State) -> Tuple[float, bool]:  # ver4
    """
    보상 설계 (임시 버전):
      - 목표 도달: +10, 에피소드 종료
      - 제자리(벽에 박힘/상대와 충돌 등): -0.5
      - 그냥 한 칸 이동: -0.1
    """  # ver4
    if new_pos == goal_pos:
        return 10.0, True
    if new_pos == prev_pos:
        return -0.5, False
    return -0.1, False


def save_q_table(Q, path: str) -> None:  # ver4
    """Q-table을 pickle로 저장."""  # ver4
    # defaultdict -> 일반 dict로 변환 (선택 사항이지만 깔끔하게)  # ver4
    plain_q = dict(Q)
    with open(path, "wb") as f:
        pickle.dump(plain_q, f)
    print(f"[INFO] Q-table saved to {path}  ({len(plain_q)} entries)")  # ver4


def save_rewards_csv(rewards, path: str) -> None:  # ver4
    """에피소드별 리워드를 CSV로 저장."""  # ver4
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["episode", "reward"])
        for i, r in enumerate(rewards, start=1):
            writer.writerow([i, r])
    print(f"[INFO] Rewards per episode saved to {path}")  # ver4


def main():
    env = MazeEnv()

    # 목표 위치 (u): 맵 오른쪽 위쪽 근처. 필요하면 수정해도 됨.  # ver4
    goal_pos: State = (GRID_W - 2, 1)

    # Q-table: key = (state, action), value = Q-value  # ver4
    Q = defaultdict(float)

    num_episodes = 300  # (u)
    max_steps = 200
    alpha = 0.2
    gamma = 0.95
    epsilon = 0.3
    epsilon_min = 0.05
    epsilon_decay = 0.98

    rewards_per_episode = []

    for ep in range(1, num_episodes + 1):
        obs1, obs2 = env.reset()
        total_reward = 0.0

        state: State = env.ai1_pos

        for step in range(max_steps):
            # --- AI1: Q-learning 정책 ---  # ver4
            action1 = choose_action_epsilon_greedy(Q, state, epsilon)

            # --- AI2: 규칙 기반 봇 ---  # ver4
            obs2_dict = {
                "ai2_pos": env.ai2_pos,
                "ai1_pos": env.ai1_pos,
            }
            action2 = rule_based_ai2(obs2_dict, env, eps=0.1)

            prev_pos = env.ai1_pos

            (obs1, obs2), done_env, info = env.step(action1, action2)
            new_pos = env.ai1_pos
            next_state: State = new_pos

            reward, done_goal = compute_reward(prev_pos, new_pos, goal_pos)
            done = done_env or done_goal
            total_reward += reward

            # Q-learning 업데이트  # ver4
            old_q = Q[(state, action1)]
            next_qs = [Q[(next_state, a)] for a in ACTIONS]
            max_next_q = max(next_qs) if next_qs else 0.0

            target = reward + (0.0 if done else gamma * max_next_q)
            Q[(state, action1)] = old_q + alpha * (target - old_q)

            state = next_state

            if done:
                break

        rewards_per_episode.append(total_reward)
        epsilon = max(epsilon_min, epsilon * epsilon_decay)

        if ep % 10 == 0:
            avg_last = sum(rewards_per_episode[-10:]) / min(len(rewards_per_episode), 10)
            print(
                f"[Episode {ep:3d}] total_reward={total_reward:6.2f}, "
                f"avg_last10={avg_last:6.2f}, epsilon={epsilon:5.3f}"
            )

    overall_avg = sum(rewards_per_episode) / len(rewards_per_episode)
    print(f"\nTraining finished. Episodes: {num_episodes}, "
          f"average reward: {overall_avg:.3f}")

    # --- 학습 결과 저장 (4.5용) ---  # ver4
    save_q_table(Q, "qtable.pkl")
    save_rewards_csv(rewards_per_episode, "rewards.csv")


if __name__ == "__main__":
    main()
