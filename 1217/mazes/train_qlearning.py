# train_qlearning.py  # ver4

import random
import csv
from collections import defaultdict
from typing import Tuple

from env_maze import MazeEnv, UP, DOWN, LEFT, RIGHT  # ver4
from ai2_rules import rule_based_ai2  # ver4

Action = int
State = Tuple[int, int]
ACTIONS = [UP, DOWN, LEFT, RIGHT]


def choose_action_epsilon_greedy(Q, state: State, epsilon: float) -> Action:  # ver4
    if random.random() < epsilon:
        return random.choice(ACTIONS)
    qs = [Q[(state, a)] for a in ACTIONS]
    max_q = max(qs)
    candidates = [a for a, q in zip(ACTIONS, qs) if q == max_q]
    return random.choice(candidates)


def save_rewards_csv(rewards, path: str) -> None:  # ver4
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["episode", "reward"])
        for i, r in enumerate(rewards, start=1):
            writer.writerow([i, r])


# TESTver4: 학습 중 AI1 트랩 사용은 휴리스틱(상태 폭증 방지)
def decide_trap_use_ai1(env: MazeEnv) -> bool:
    if not env.has_trap_ai1:
        return False
    g = env.goal_pos
    cp = env.checkpoint_pos
    on_tp = any(env.ai1_pos == a or env.ai1_pos == b for (a, b) in getattr(env, "teleport_pairs", []))
    near_goal = env.goal_active and g is not None and abs(env.ai1_pos[0] - g[0]) + abs(env.ai1_pos[1] - g[1]) <= 2
    near_cp = cp is not None and abs(env.ai1_pos[0] - cp[0]) + abs(env.ai1_pos[1] - cp[1]) <= 1
    return on_tp or near_goal or near_cp


def compute_shaping_reward(env: MazeEnv, info: dict) -> float:
    """
    TESTver4: 이벤트 기반 shaping(가볍게)
    - 기본 스텝: -0.05
    - AI1 goal 승리: +8
    - 아이템 획득(코인/체크포인트/트랩/마취총): + (소폭)
    - 상대 스턴 성공(트랩/마취총): +1
    - 내가 스턴: -1
    - 타임아웃 판정승: +2 / 패배: -2 / 무승부: 0
    """
    r = -0.05
    ev = set(info.get("events", []))

    if "ai1_tranq_pick" in ev:
        r += 0.5
    if "ai1_tranq_hit" in ev:
        r += 1.0
    if "ai2_tranq_hit" in ev:
        r -= 1.0

    if "ai1_trap_pick" in ev:
        r += 0.4
    if "ai1_trap_place" in ev:
        r += 0.2
    if "ai2_trap_hit" in ev:   # AI1이 깐 트랩에 AI2가 걸림
        r += 1.0
    if "ai1_trap_hit" in ev:
        r -= 1.0

    if info.get("winner") == "AI1" and "ai1_goal" in ev:
        r += 8.0

    if "timeout" in ev:
        w = info.get("winner")
        if w == "AI1":
            r += 2.0
        elif w == "AI2":
            r -= 2.0
        else:
            r += 0.0

    return r


def main():
    env = MazeEnv()
    Q = defaultdict(float)

    num_episodes = 800
    max_steps = 250
    alpha = 0.2
    gamma = 0.95
    epsilon = 0.4
    epsilon_min = 0.05
    epsilon_decay = 0.995

    rewards_per_episode = []

    for ep in range(1, num_episodes + 1):
        env.reset()
        total_reward = 0.0
        state: State = env.ai1_pos

        for _ in range(max_steps):
            action1 = choose_action_epsilon_greedy(Q, state, epsilon)
            use_trap1 = decide_trap_use_ai1(env)

            obs2_dict = {"ai2_pos": env.ai2_pos, "ai1_pos": env.ai1_pos}
            action2, use_trap2 = rule_based_ai2(obs2_dict, env, eps=0.05)

            (obs1, obs2), done, info = env.step(action1, action2, use_trap1=use_trap1, use_trap2=use_trap2)

            reward = compute_shaping_reward(env, info)
            total_reward += reward

            next_state: State = env.ai1_pos

            old_q = Q[(state, action1)]
            max_next_q = max(Q[(next_state, a)] for a in ACTIONS)
            target = reward + (0.0 if done else gamma * max_next_q)
            Q[(state, action1)] = old_q + alpha * (target - old_q)

            state = next_state
            if done:
                break

        rewards_per_episode.append(total_reward)
        epsilon = max(epsilon_min, epsilon * epsilon_decay)

        if ep % 20 == 0:
            avg_last = sum(rewards_per_episode[-20:]) / 20
            print(f"[Episode {ep:4d}] total_reward={total_reward:7.2f}, avg_last20={avg_last:7.2f}, epsilon={epsilon:6.3f}")

    save_rewards_csv(rewards_per_episode, "rewards.csv")
    print("[INFO] Training finished. rewards.csv saved.")
    print("[NOTE] Q-table 저장(Supabase)은 기존 로직이 있으면 그대로 붙여 사용하세요.")


if __name__ == "__main__":
    main()
