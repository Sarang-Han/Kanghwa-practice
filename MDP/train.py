import os
import numpy as np
import matplotlib.pyplot as plt
from environment import GridWorld
from agent import QLearningAgent
from visualization import visualize_grid, plot_rewards, visualize_q_values

def train_agent(num_episodes=30000):
    # 환경 및 에이전트 생성
    env = GridWorld(size=5)
    agent = QLearningAgent(env, learning_rate=0.1, discount_factor=0.9, epsilon=0.1)
    
    # 학습 로그 저장
    episode_rewards = []
    
    for episode in range(num_episodes):
        # 환경 초기화
        state = env.reset()
        done = False
        total_reward = 0
        
        # 에피소드 실행
        while not done:
            # 행동 선택
            action = agent.select_action(state)
            
            # 환경에서 한 스텝 진행
            next_state, reward, done = env.step(action)
            
            # 에이전트 학습
            agent.learn(state, action, reward, next_state, done)
            
            # 상태 업데이트
            state = next_state
            total_reward += reward
        
        # 에피소드 로그 저장
        episode_rewards.append(total_reward)
        
        # 학습 진행 출력
        if (episode + 1) % 100 == 0:
            print(f"에피소드 {episode + 1}/{num_episodes}, 평균 보상: {np.mean(episode_rewards[-100:]):.2f}")
            # 중간 학습 결과 시각화를 위해 엡실론 감소
            agent.epsilon *= 0.9
    
    return env, agent, episode_rewards

def visualize_path(env, agent):
    """
    최적 경로 추적 및 시각화
    """
    state = env.reset()
    path = [state]
    done = False
    
    while not done:
        action = np.argmax(agent.q_table[env.get_state_index(state)])  # 최적 행동 선택
        next_state, _, done = env.step(action)
        path.append(next_state)
        state = next_state
    
    return path

def main():
    # 학습 실행
    print("Q-Learning 학습 시작...")
    env, agent, rewards = train_agent(num_episodes=500)
    
    # 학습 결과 출력
    print("학습 완료!")
    print(f"최종 Q-테이블:\n{agent.q_table.reshape(env.size, env.size, env.action_space)}")
    
    # 최적 정책 추출
    optimal_policy = agent.get_optimal_policy()
    print("최적 정책:")
    for state, action in optimal_policy.items():
        action_name = ['위', '오른쪽', '아래', '왼쪽'][action]
        print(f"상태 {state}: {action_name}")
    
    # 보상 그래프 시각화
    plot_rewards(rewards)
    
    # Q-값 히트맵 시각화
    visualize_q_values(env, agent.q_table)
    
    # 격자와 최적 정책 시각화
    fig, ax = visualize_grid(env, policy=optimal_policy)
    plt.savefig('optimal_policy.png')
    
    # 최적 경로 추적
    path = visualize_path(env, agent)
    print(f"최적 경로: {path}")
    print(f"최단 경로 길이: {len(path) - 1} 스텝")
    
    plt.show()

if __name__ == "__main__":
    main()