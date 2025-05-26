import numpy as np

from jump_game_env import JumpGameEnv
from monte_carlo_prediction import MonteCarloPredictor
from visualization import (
    plot_value_function, 
    visualize_value_convergence,
    plot_visit_counts
)

# 시드 설정
np.random.seed(42)

# 환경 및 에이전트 초기화
env = JumpGameEnv()
mc_predictor = MonteCarloPredictor(env)

def random_policy(state):
    """무작위 정책: 가능한 행동 중 무작위로 선택"""
    position, energy = state
    
    # 에너지가 없으면 걷기만 가능
    if energy == 0:
        return 0
    
    # 장애물이 있고 점프할 수 있으면 점프
    if position + 1 in env.obstacles and energy > 0:
        return 1
    
    # 그 외에는 무작위 선택
    return np.random.choice([0, 1])

def better_policy(state):
    """더 나은 정책: 가능한 행동 중 현명한 선택"""
    position, energy = state
    
    # 에너지가 없으면 걷기만 가능
    if energy == 0:
        return 0
    
    # 목표까지 남은 거리 계산
    distance_to_goal = env.goal - position
    
    # 남은 장애물 확인
    obstacles_ahead = [obs for obs in env.obstacles if position < obs < env.goal]
    
    # 장애물이 바로 앞에 있는 경우 (절대 점프)
    if position + 1 in env.obstacles:
        return 1
    
    # 목표가 2칸 이내에 있으면 걷기 
    if distance_to_goal <= 2:
        return 0
    
    # 중요: 목표가 가까운데 앞에 장애물이 있으면 점프로 넘기
    if distance_to_goal <= 4 and obstacles_ahead:
        return 1
    
    # 에너지 관리: 남은 장애물 수에 따라 점프 사용
    if len(obstacles_ahead) >= energy:
        # 장애물이 많으면 에너지 아껴서 걷기
        return 0
    
    # 그 외에는 적절히 선택 (이전보다 점프 확률 증가)
    if energy >= 2:  # 에너지가 충분하면 점프 확률 높임
        return np.random.choice([0, 1], p=[0.5, 0.5])
    else:  # 에너지가 1이면 걷기를 선호하되 가끔 점프
        return np.random.choice([0, 1], p=[0.8, 0.2])

def test_policy_success_rate(policy, num_tests=100):
    """정책의 성공률 테스트"""
    successes = 0
    # 목표 위치까지 포함할 수 있도록 크기 1 증가
    final_positions = [0] * (env.goal + 1)
    
    for _ in range(num_tests):
        state = env.reset()
        done = False
        
        while not done:
            action = policy(state)
            state, reward, done = env.step(action)
        
        # 최종 위치 기록
        final_pos = state[0]
        
        # 안전 검사 추가
        if 0 <= final_pos <= env.goal:
            final_positions[final_pos] += 1
            
            # 목표 도달 확인
            if final_pos == env.goal:
                successes += 1
    
    # 결과 출력
    print(f"Policy success rate: {successes/num_tests:.2%}")
    print(f"Final positions distribution:")
    for pos, count in enumerate(final_positions):
        if count > 0:
            print(f"  Position {pos}: {count} times ({count/num_tests:.1%})")
    
    return successes/num_tests

def main():
    # 에피소드 수
    num_episodes = 5000
    
    print("=== Monte Carlo Prediction in Jump Game ===")
    print(f"Number of episodes: {num_episodes}")
    print("Game Rules:")
    print("- Starting position: 0, Goal position: 9")
    print("- Obstacles at positions: 6, 8")
    print("- Actions: 0 (walk), 1 (jump)")
    print("- Walk: move 1 step forward, stay if obstacle ahead")
    print("- Jump: move 2 steps forward, consume 1 energy")
    print("- Starting energy: 2")
    print("\nSimulating...\n")
    
    # 정책 테스트
    print("\nTesting policy success rate...")
    policy = random_policy  # 사용할 정책 선택
    success_rate = test_policy_success_rate(policy, 1000)
    
    # 몬테카를로 예측 실행 (정책은 변경 가능)
    values, value_history, visit_counts = mc_predictor.predict(num_episodes, random_policy)
    
    # 최종 가치 함수 테이블
    value_table = mc_predictor.get_value_table()
    
    # 결과 시각화
    print("Generating final visualizations...")
    
    # 1. 상태 가치 함수 시각화
    value_plot = plot_value_function(value_table, env)
    value_plot.savefig("value_function.png")
    
    # 2. 가치 함수 수렴 과정 시각화
    convergence_plot = visualize_value_convergence(value_history, env)
    convergence_plot.savefig("value_convergence.png")
    
    # 3. 상태 방문 횟수 시각화
    visit_plot = plot_visit_counts(visit_counts, env)
    visit_plot.savefig("visit_counts.png")
    
    print("\nAll visualizations completed successfully...")

if __name__ == "__main__":
    main()