# run_random_demo.py
"""
환경이 제대로 동작하는지 확인하기 위한 랜덤 데모 (u)
- AI1, AI2 둘 다 랜덤으로 움직인다.
- 나중에 이 파일을 Q-learning / DQN 학습 스크립트로 교체해도 된다.
"""

import random
from env_maze import (
    MazeEnv,
    ACTION_UP,
    ACTION_DOWN,
    ACTION_LEFT,
    ACTION_RIGHT,
    NUM_ACTIONS,
)


def sample_random_action() -> int:
    """임의의 행동 하나 뽑기 (u)"""
    return random.randint(0, NUM_ACTIONS - 1)


def main():  # (u)
    # pygame 설치 안 되어 있으면 자동으로 텍스트 모드
    env = MazeEnv(cell_size=60, use_pygame=True)
    state = env.reset()

    print("초기 상태:", state)

    done = False
    episode = 1

    while episode <= 3:  # 에피소드 3판만 테스트 (u)
        done = False
        state = env.reset()
        print(f"\n=== Episode {episode} 시작 ===")

        while not done:
            # 여기서 나중에 Q-learning / DQN 코드를 넣으면 된다.
            action1 = sample_random_action()
            action2 = sample_random_action()

            next_state, (reward1, reward2), done, info = env.step(
                action1, action2)

            print(
                f"Step {info['steps']}: "
                f"a1_action={action1}, r1={reward1:.2f} | "
                f"a2_action={action2}, r2={reward2:.2f}"
            )

            env.render()

            state = next_state

            if done:
                print("Episode 종료:", info)
                print(
                    f"최종 스코어 => "
                    f"Agent1: {state['agent1_score']:.2f}, "
                    f"Agent2: {state['agent2_score']:.2f}"
                )

        episode += 1


if __name__ == "__main__":
    main()
