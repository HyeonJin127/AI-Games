"""
Q-Learning 미로 탈출 AI - Pygame 버전
모듈화된 구조로 각 컴포넌트를 쉽게 수정 가능
"""

import pygame
import numpy as np
import sys
from collections import deque

# ==================== 설정 컨테이너 ====================
class Config:
    """게임 및 학습 설정"""
    # 화면 설정
    SCREEN_WIDTH = 1200
    SCREEN_HEIGHT = 700
    CELL_SIZE = 80
    FPS = 60
    
    # 색상
    COLOR_BG = (240, 240, 245)
    COLOR_WALL = (40, 40, 40)
    COLOR_PATH = (255, 255, 255)
    COLOR_START = (100, 200, 100)
    COLOR_GOAL = (255, 215, 0)
    COLOR_AGENT = (220, 50, 50)
    COLOR_TRAIL = (255, 150, 150)
    COLOR_TEXT = (50, 50, 50)
    COLOR_ARROW = (100, 150, 255)
    
    # 미로 설정
    MAZE_SIZE = 8
    MAZE = np.array([
        [2, 0, 1, 0, 0, 0, 1, 0],
        [0, 0, 1, 0, 1, 0, 1, 0],
        [0, 0, 0, 0, 1, 0, 0, 0],
        [1, 1, 1, 0, 1, 0, 1, 1],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 1, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 1, 0, 0, 3]
    ])
    
    START_POS = (0, 0)
    GOAL_POS = (7, 7)
    
    # Q-Learning 하이퍼파라미터
    LEARNING_RATE = 0.1
    DISCOUNT_FACTOR = 0.95
    EPSILON_START = 1.0
    EPSILON_MIN = 0.01
    EPSILON_DECAY = 0.995
    
    # 보상
    REWARD_GOAL = 100
    REWARD_STEP = -0.1
    REWARD_WALL = -1
    
    # 학습 설정
    MAX_STEPS = 100
    ANIMATION_SPEED = 10  # ms


# ==================== 환경 컨테이너 ====================
class MazeEnvironment:
    """미로 환경 - 상태 전이 및 보상 관리"""
    
    def __init__(self, config):
        self.config = config
        self.maze = config.MAZE.copy()
        self.size = config.MAZE_SIZE
        self.start = config.START_POS
        self.goal = config.GOAL_POS
        self.agent_pos = list(self.start)
        self.actions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # 상하좌우
        
    def reset(self):
        """환경 초기화"""
        self.agent_pos = list(self.start)
        return tuple(self.agent_pos)
    
    def step(self, action):
        """
        행동 실행
        Returns: (next_state, reward, done, info)
        """
        dx, dy = self.actions[action]
        new_pos = [self.agent_pos[0] + dx, self.agent_pos[1] + dy]
        
        # 경계 체크
        if not (0 <= new_pos[0] < self.size and 0 <= new_pos[1] < self.size):
            return tuple(self.agent_pos), self.config.REWARD_WALL, False, {'hit': 'boundary'}
        
        # 벽 체크
        if self.maze[new_pos[0], new_pos[1]] == 1:
            return tuple(self.agent_pos), self.config.REWARD_WALL, False, {'hit': 'wall'}
        
        # 이동
        self.agent_pos = new_pos
        
        # 목표 도달
        if tuple(self.agent_pos) == self.goal:
            return tuple(self.agent_pos), self.config.REWARD_GOAL, True, {'result': 'goal'}
        
        return tuple(self.agent_pos), self.config.REWARD_STEP, False, {'result': 'move'}
    
    def get_valid_actions(self, state):
        """해당 상태에서 가능한 행동 리스트"""
        valid = []
        for i, (dx, dy) in enumerate(self.actions):
            new_x, new_y = state[0] + dx, state[1] + dy
            if (0 <= new_x < self.size and 0 <= new_y < self.size and 
                self.maze[new_x, new_y] != 1):
                valid.append(i)
        return valid


# ==================== 에이전트 컨테이너 ====================
class QLearningAgent:
    """Q-Learning 에이전트 - 학습 및 행동 선택"""
    
    def __init__(self, config, n_actions=4):
        self.config = config
        self.n_actions = n_actions
        self.q_table = {}
        self.epsilon = config.EPSILON_START
        
    def get_q_values(self, state):
        """상태의 Q-value 배열 반환"""
        if state not in self.q_table:
            self.q_table[state] = np.zeros(self.n_actions)
        return self.q_table[state]
    
    def choose_action(self, state, valid_actions=None):
        """Epsilon-greedy 정책으로 행동 선택"""
        if np.random.random() < self.epsilon:
            # 탐험: 랜덤 행동
            if valid_actions:
                return np.random.choice(valid_actions)
            return np.random.randint(self.n_actions)
        else:
            # 활용: 최선의 행동
            q_values = self.get_q_values(state)
            if valid_actions:
                valid_q = [(a, q_values[a]) for a in valid_actions]
                return max(valid_q, key=lambda x: x[1])[0]
            return np.argmax(q_values)
    
    def learn(self, state, action, reward, next_state, done):
        """Q-Learning 업데이트"""
        current_q = self.get_q_values(state)[action]
        
        if done:
            target = reward
        else:
            next_q = np.max(self.get_q_values(next_state))
            target = reward + self.config.DISCOUNT_FACTOR * next_q
        
        # Q-value 업데이트
        self.q_table[state][action] += self.config.LEARNING_RATE * (target - current_q)
    
    def decay_epsilon(self):
        """Epsilon 감소"""
        self.epsilon = max(self.config.EPSILON_MIN, 
                          self.epsilon * self.config.EPSILON_DECAY)
    
    def get_best_action(self, state):
        """학습된 최선의 행동 반환"""
        if state not in self.q_table:
            return 0
        return np.argmax(self.q_table[state])


# ==================== 통계 컨테이너 ====================
class Statistics:
    """학습 통계 및 기록"""
    
    def __init__(self, window_size=100):
        self.episode = 0
        self.rewards = deque(maxlen=window_size)
        self.steps = deque(maxlen=window_size)
        self.success = deque(maxlen=window_size)
        
    def add_episode(self, reward, step, is_success):
        """에피소드 결과 추가"""
        self.episode += 1
        self.rewards.append(reward)
        self.steps.append(step)
        self.success.append(1 if is_success else 0)
    
    def get_avg_reward(self):
        """평균 보상"""
        return np.mean(self.rewards) if self.rewards else 0
    
    def get_avg_steps(self):
        """평균 스텝"""
        return np.mean(self.steps) if self.steps else 0
    
    def get_success_rate(self):
        """성공률"""
        return np.mean(self.success) * 100 if self.success else 0


# ==================== 렌더러 컨테이너 ====================
class Renderer:
    """Pygame 렌더링 담당"""
    
    def __init__(self, config):
        self.config = config
        pygame.init()
        self.screen = pygame.display.set_mode((config.SCREEN_WIDTH, config.SCREEN_HEIGHT))
        pygame.display.set_caption("Q-Learning 미로 탈출 AI")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 36)
        self.font_small = pygame.font.Font(None, 24)
        
        # 화살표 문자
        self.arrows = ['↑', '↓', '←', '→']
        
    def draw_maze(self, env, agent_pos, trail=None, show_q_values=False, agent=None):
        """미로 그리기"""
        self.screen.fill(self.config.COLOR_BG)
        
        offset_x = 50
        offset_y = 50
        cell_size = self.config.CELL_SIZE
        
        # 셀 그리기
        for i in range(env.size):
            for j in range(env.size):
                x = offset_x + j * cell_size
                y = offset_y + i * cell_size
                
                # 셀 타입별 색상
                if env.maze[i, j] == 1:  # 벽
                    color = self.config.COLOR_WALL
                elif (i, j) == env.goal:  # 목표
                    color = self.config.COLOR_GOAL
                elif (i, j) == env.start:  # 시작
                    color = self.config.COLOR_START
                else:  # 길
                    color = self.config.COLOR_PATH
                
                pygame.draw.rect(self.screen, color, (x, y, cell_size, cell_size))
                pygame.draw.rect(self.screen, (200, 200, 200), (x, y, cell_size, cell_size), 2)
                
                # Q-value 화살표 표시
                if show_q_values and agent and env.maze[i, j] != 1:
                    state = (i, j)
                    if state in agent.q_table:
                        best_action = agent.get_best_action(state)
                        arrow_text = self.font.render(self.arrows[best_action], True, 
                                                     self.config.COLOR_ARROW)
                        arrow_text.set_alpha(100)
                        text_rect = arrow_text.get_rect(center=(x + cell_size//2, y + cell_size//2))
                        self.screen.blit(arrow_text, text_rect)
        
        # 이동 궤적
        if trail:
            for pos in trail:
                x = offset_x + pos[1] * cell_size + cell_size // 2
                y = offset_y + pos[0] * cell_size + cell_size // 2
                pygame.draw.circle(self.screen, self.config.COLOR_TRAIL, (x, y), 8)
        
        # 에이전트
        x = offset_x + agent_pos[1] * cell_size + cell_size // 2
        y = offset_y + agent_pos[0] * cell_size + cell_size // 2
        pygame.draw.circle(self.screen, self.config.COLOR_AGENT, (x, y), cell_size // 3)
    
    def draw_stats(self, stats, agent, is_training):
        """통계 정보 그리기"""
        x_start = 700
        y_start = 50
        
        texts = [
            f"Episode: {stats.episode}",
            f"Epsilon: {agent.epsilon:.3f}",
            f"Avg Reward: {stats.get_avg_reward():.1f}",
            f"Avg Steps: {stats.get_avg_steps():.1f}",
            f"Success Rate: {stats.get_success_rate():.1f}%",
            "",
            "Controls:",
            "SPACE: Start/Stop Training",
            "R: Reset",
            "D: Demo Best Path",
            "Q: Toggle Q-values",
        ]
        
        for i, text in enumerate(texts):
            if text == "":
                continue
            color = self.config.COLOR_TEXT if i < 5 else (100, 100, 100)
            font = self.font if i < 5 else self.font_small
            rendered = font.render(text, True, color)
            self.screen.blit(rendered, (x_start, y_start + i * 40))
        
        # 학습 상태 표시
        status = "TRAINING..." if is_training else "PAUSED"
        status_color = (50, 200, 50) if is_training else (200, 50, 50)
        status_text = self.font.render(status, True, status_color)
        self.screen.blit(status_text, (x_start, y_start + 450))
    
    def update(self):
        """화면 업데이트"""
        pygame.display.flip()
        self.clock.tick(self.config.FPS)


# ==================== 메인 게임 컨테이너 ====================
class QLearningGame:
    """메인 게임 루프 및 제어"""
    
    def __init__(self):
        self.config = Config()
        self.env = MazeEnvironment(self.config)
        self.agent = QLearningAgent(self.config)
        self.stats = Statistics()
        self.renderer = Renderer(self.config)
        
        self.is_training = False
        self.show_q_values = True
        self.trail = []
        self.training_step = 0
        
    def run_episode(self):
        """한 에피소드 실행"""
        state = self.env.reset()
        total_reward = 0
        steps = 0
        done = False
        self.trail = [state]
        
        while not done and steps < self.config.MAX_STEPS:
            # 행동 선택
            valid_actions = self.env.get_valid_actions(state)
            action = self.agent.choose_action(state, valid_actions)
            
            # 환경 스텝
            next_state, reward, done, info = self.env.step(action)
            
            # 학습
            self.agent.learn(state, action, reward, next_state, done)
            
            # 기록
            state = next_state
            total_reward += reward
            steps += 1
            self.trail.append(state)
            
            # 렌더링 (느리게)
            if self.training_step % 5 == 0:
                self.renderer.draw_maze(self.env, self.env.agent_pos, self.trail, 
                                       self.show_q_values, self.agent)
                self.renderer.draw_stats(self.stats, self.agent, self.is_training)
                self.renderer.update()
                pygame.time.wait(self.config.ANIMATION_SPEED)
        
        # 통계 업데이트
        self.stats.add_episode(total_reward, steps, done)
        self.agent.decay_epsilon()
        self.training_step += 1
    
    def demo_best_path(self):
        """학습된 최적 경로 시연"""
        state = self.env.reset()
        self.trail = [state]
        steps = 0
        
        while steps < 50:
            if state not in self.agent.q_table:
                print("아직 충분히 학습되지 않았습니다!")
                break
            
            action = self.agent.get_best_action(state)
            next_state, reward, done, info = self.env.step(action)
            
            state = next_state
            steps += 1
            self.trail.append(state)
            
            # 렌더링
            self.renderer.draw_maze(self.env, self.env.agent_pos, self.trail, 
                                   self.show_q_values, self.agent)
            self.renderer.draw_stats(self.stats, self.agent, False)
            self.renderer.update()
            pygame.time.wait(200)
            
            if done:
                print(f"✅ 목표 도달! {steps}걸음")
                break
    
    def reset(self):
        """전체 리셋"""
        self.agent = QLearningAgent(self.config)
        self.stats = Statistics()
        self.env.reset()
        self.trail = []
        self.training_step = 0
        print("리셋 완료!")
    
    def run(self):
        """메인 루프"""
        running = True
        
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        self.is_training = not self.is_training
                        print(f"Training: {self.is_training}")
                    elif event.key == pygame.K_r:
                        self.reset()
                    elif event.key == pygame.K_d:
                        self.is_training = False
                        self.demo_best_path()
                    elif event.key == pygame.K_q:
                        self.show_q_values = not self.show_q_values
            
            # 학습 진행
            if self.is_training:
                self.run_episode()
            else:
                # 일시정지 상태에서도 화면 유지
                self.renderer.draw_maze(self.env, self.env.agent_pos, self.trail, 
                                       self.show_q_values, self.agent)
                self.renderer.draw_stats(self.stats, self.agent, self.is_training)
                self.renderer.update()
        
        pygame.quit()
        sys.exit()


# ==================== 실행 ====================
if __name__ == "__main__":
    game = QLearningGame()
    game.run()