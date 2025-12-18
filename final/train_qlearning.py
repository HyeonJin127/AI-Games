# train_qlearning.py  # ver5-optimized

import numpy as np
import random
import csv
from collections import defaultdict
from typing import Tuple
import os
from dotenv import load_dotenv
from supabase import create_client, Client
import time

from env_maze import MazeEnv, UP, DOWN, LEFT, RIGHT
from ai2_rules import rule_based_ai2

# --- 설정 및 초기화 ---
load_dotenv()
url = os.environ.get("SUPABASE_URL")
key = os.environ.get("SUPABASE_SERVICE_KEY")
supabase = create_client(url, key)

ACTIONS = [UP, DOWN, LEFT, RIGHT]

def get_state(env: MazeEnv):
    """
    ai2_rules의 논리를 학습하기 위한 확장 상태 정의:
    (현재위치, 코인개수, 아이템적중여부, 체크포인트도달여부, 최종언락여부)
    """
    return (
        env.ai1_pos,
        env.items_ai1,
        env.used_item_ai1,
        env.checkpoint_cleared_ai1,
        env.unlocked_ai1
    )

def compute_shaping_reward(env, info, prev_state):
    """개선된 보상 함수 - 더 명확한 학습 신호"""
    _, p_items, p_used, p_cp, p_unlocked = prev_state
    
    reward = -0.05  # 기본 이동 패널티

    # 1. 코인 획득 (가장 기초 단계)
    if env.items_ai1 > p_items:
        return 80.0
    
    # 2. 아이템 적중 (두 번째 단계)
    if env.used_item_ai1 and not p_used:
        return 80.0
    
    # 3. 체크포인트 도달 (세 번째 단계)
    if env.checkpoint_cleared_ai1 and not p_cp:
        return 100.0

    # 4. 최종 언락 (모든 조건 충족 시)
    if env.unlocked_ai1 and not p_unlocked:
        return 100.0
    
    # 5. 골인 성공
    if info.get("winner") == "AI1":
        return 300.0
    
    # 6. 패배 페널티 추가
    if info.get("winner") == "AI2":
        return -100.0

    # --- 유도 보상 (ai2_rules의 우선순위 가이드) ---
    ax, ay = env.ai1_pos
    target = None

    if env.items_ai1 < 3:
        target = None
    elif not env.used_item_ai1:
        target = env.ai2_pos
    elif not env.checkpoint_cleared_ai1:
        target = env.checkpoint_pos if env.checkpoint_pos else env.goal_pos
        # target = getattr(env, "checkpoint_pos", (7, 5))
    else:
        target = env.goal_pos

    if target is not None:
        dist = abs(ax - target[0]) + abs(ay - target[1])
        reward += (1.0 / (dist + 1)) * 2.0
    
    return reward

def save_q_table_to_supabase(Q, table_name="q_table_maze_v4"):
    """학습된 Q-테이블을 Supabase에 업로드"""
    print(f"\n[INFO] Supabase 테이블 '{table_name}'에 데이터를 저장하는 중...")
    
    try:
        # 1. 기존 데이터 삭제
        supabase.table(table_name).delete().neq("action", -1).execute()
        
        records = []
        for (state, action), q_value in Q.items():
            # Q값이 0인 데이터는 제외 (용량 절약)
            if q_value == 0:
                continue
                
            records.append({
                "state": str(state),
                "action": int(action),
                "q_value": float(q_value)
            })

        # 2. 배치 업로드
        batch_size = 1000
        for i in range(0, len(records), batch_size):
            batch = records[i : i + batch_size]
            supabase.table(table_name).insert(batch).execute()
            print(f"  > {i + len(batch)} / {len(records)} 개 저장 완료...")

        print(f"[SUCCESS] 모든 데이터가 성공적으로 저장되었습니다! (총 {len(records)}개)")
        
    except Exception as e:
        print(f"[ERROR] Supabase 저장 중 오류 발생: {e}")

def main():
    env = MazeEnv()
    Q = defaultdict(float)

    # ========== 최적화된 학습 파라미터 ==========
    num_episodes = 50000      # 에피소드 증가 (더 많은 경험)
    max_steps = 300           # ✅ 240(turn_limit) + 여유 60
    alpha = 0.1               # 학습률
    gamma = 0.95              # 할인율
    epsilon = 1.0             # 초기 탐험율
    epsilon_min = 0.01        # ✅ 최소값 더 낮춤 (0.05 → 0.01)
    epsilon_decay = 0.9995    # ✅ 더 천천히 감소 (0.998 → 0.9995)
    
    # 학습 모니터링
    rewards_history = []
    win_history = []
    steps_history = []
    
    # 중간 저장 설정
    save_interval = 2000  # 2000 에피소드마다 저장
    
    print("\n" + "="*60)
    print("[START] 최적화된 Q-Learning 학습 시작")
    print("="*60)
    print(f"에피소드: {num_episodes}")
    print(f"Max Steps: {max_steps} (게임 제한: {env.turn_limit})")
    print(f"Epsilon: {epsilon} → {epsilon_min} (decay: {epsilon_decay})")
    print(f"중간 저장: {save_interval} 에피소드마다")
    print("="*60 + "\n")

    start_time = time.time()
    
    try:
        for ep in range(1, num_episodes + 1):
            env.reset()
            total_reward = 0
            steps = 0
            
            for step in range(max_steps):
                state = get_state(env)
                
                # 1. ε-greedy 정책
                if random.random() < epsilon:
                    action1 = random.choice(ACTIONS)
                else:
                    qs = [Q[(state, a)] for a in ACTIONS]
                    action1 = ACTIONS[np.argmax(qs)]

                # 2. 트랩 설치 휴리스틱
                dist_to_ai2 = abs(env.ai1_pos[0]-env.ai2_pos[0]) + abs(env.ai1_pos[1]-env.ai2_pos[1])
                use_trap1 = (dist_to_ai2 == 1 and env.has_trap_ai1)

                # 3. AI2 행동
                obs2_dict = {"ai2_pos": env.ai2_pos, "ai1_pos": env.ai1_pos}
                action2, use_trap2 = rule_based_ai2(obs2_dict, env)

                # 4. 환경 실행
                _, done, info = env.step(action1, action2, use_trap1=use_trap1, use_trap2=use_trap2)
                
                # 5. Q-learning 업데이트
                next_state = get_state(env)
                reward = compute_shaping_reward(env, info, state)
                
                max_next_q = max([Q[(next_state, a)] for a in ACTIONS])
                Q[(state, action1)] += alpha * (reward + gamma * max_next_q - Q[(state, action1)])
                
                total_reward += reward
                steps += 1
                
                if done:
                    break

            # Epsilon 감소
            epsilon = max(epsilon_min, epsilon * epsilon_decay)
            
            # 기록
            rewards_history.append(total_reward)
            steps_history.append(steps)
            win_history.append(1 if info.get("winner") == "AI1" else 0)

            # 진행 상황 출력 (20 에피소드마다)
            if ep % 20 == 0:
                recent_wins = sum(win_history[-100:]) if len(win_history) >= 100 else sum(win_history)
                recent_count = min(100, len(win_history))
                win_rate = (recent_wins / recent_count * 100) if recent_count > 0 else 0
                
                avg_reward = np.mean(rewards_history[-100:]) if len(rewards_history) >= 100 else np.mean(rewards_history)
                avg_steps = np.mean(steps_history[-100:]) if len(steps_history) >= 100 else np.mean(steps_history)
                
                elapsed = time.time() - start_time
                
                print(f"Ep {ep:5d} | "
                      f"Reward: {total_reward:7.2f} (avg: {avg_reward:7.2f}) | "
                      f"Win Rate: {win_rate:5.1f}% | "
                      f"Steps: {steps:3d} (avg: {avg_steps:.1f}) | "
                      f"ε: {epsilon:.4f} | "
                      f"Q-size: {len(Q):6d} | "
                      f"Time: {elapsed:.0f}s")
            
            # 중간 저장
            if ep % save_interval == 0:
                print(f"\n{'='*60}")
                print(f"[CHECKPOINT] {ep} 에피소드 완료 - 중간 저장 중...")
                save_q_table_to_supabase(Q, "q_table_maze_v4")
                print(f"{'='*60}\n")

    except KeyboardInterrupt:
        print("\n" + "="*60)
        print("[STOP] 학습이 중단되었습니다.")
        print("="*60)
        
    finally:
        # 최종 저장
        if len(Q) > 0:
            print("\n" + "="*60)
            print("[FINAL SAVE] 최종 Q-table 저장 중...")
            save_q_table_to_supabase(Q, "q_table_maze_v4")
            
            # 학습 통계 출력
            total_time = time.time() - start_time
            print(f"\n{'='*60}")
            print("[학습 완료 통계]")
            print(f"{'='*60}")
            print(f"총 에피소드: {len(rewards_history)}")
            print(f"총 학습 시간: {total_time:.1f}초 ({total_time/60:.1f}분)")
            print(f"최종 승률: {sum(win_history[-100:])/min(100, len(win_history))*100:.1f}% (최근 100게임)")
            print(f"평균 보상: {np.mean(rewards_history[-100:]):.2f} (최근 100게임)")
            print(f"Q-table 크기: {len(Q)} 상태-행동 쌍")
            print(f"{'='*60}")
        else:
            print("[WARN] 저장할 데이터가 없습니다.")

if __name__ == "__main__":
    main()