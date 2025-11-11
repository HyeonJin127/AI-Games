# plot_rewards.py
"""
qlearning_rewards.csv 파일을 읽어서 리워드 그래프를 그리는 코드
사용 전: pip install matplotlib
"""

import csv
import matplotlib.pyplot as plt

def main():
    episodes = []
    rewards = []

    with open("qlearning_rewards.csv", "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            episodes.append(int(row["episode"]))
            rewards.append(float(row["total_reward_agent1"]))

    plt.figure()
    plt.plot(episodes, rewards)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward (Agent1)")
    plt.title("Q-learning Training Rewards")
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()
