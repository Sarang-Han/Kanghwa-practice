import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def visualize_grid(env, q_table=None, policy=None):
    """
    GridWorld와 학습된 Q-값 또는 정책을 시각화
    """
    fig, ax = plt.subplots(figsize=(5, 5))
    
    # 격자 생성
    for i in range(env.size + 1):
        ax.axhline(i, color='black', lw=1)
        ax.axvline(i, color='black', lw=1)
    
    # 장애물 표시
    for obstacle in env.obstacles:
        x, y = obstacle
        ax.add_patch(plt.Rectangle((y, env.size - x - 1), 1, 1, fill=True, color='gray'))
    
    # 시작점과 목표점 표시
    start_x, start_y = env.start
    goal_x, goal_y = env.goal
    ax.add_patch(plt.Rectangle((start_y, env.size - start_x - 1), 1, 1, fill=True, color='lightblue', alpha=0.5))
    ax.add_patch(plt.Rectangle((goal_y, env.size - goal_x - 1), 1, 1, fill=True, color='green', alpha=0.5))
    
    ax.text(start_y + 0.5, env.size - start_x - 0.5, 'S', fontsize=20, ha='center', va='center')
    ax.text(goal_y + 0.5, env.size - goal_x - 0.5, 'G', fontsize=20, ha='center', va='center')
    
    # 정책 표시 (화살표로)
    if policy:
        for state, action in policy.items():
            if state != env.goal:
                x, y = state
                # 플롯 좌표로 변환
                plot_x = y + 0.5
                plot_y = env.size - x - 0.5
                
                # 화살표 방향 (그리드월드의 좌표계에 맞춰 수정)
                if action == 0:  # 위
                    dx, dy = 0, 0.3  # 그래프에서는 위쪽으로 이동
                elif action == 1:  # 오른쪽
                    dx, dy = 0.3, 0  # 그래프에서는 오른쪽으로 이동
                elif action == 2:  # 아래
                    dx, dy = 0, -0.3  # 그래프에서는 아래쪽으로 이동
                else:  # 왼쪽
                    dx, dy = -0.3, 0  # 그래프에서는 왼쪽으로 이동
                
                ax.arrow(plot_x, plot_y, dx, dy, head_width=0.1, head_length=0.1, fc='black', ec='black')
    
    plt.xlim(0, env.size)
    plt.ylim(0, env.size)
    plt.xticks(np.arange(0.5, env.size, 1), range(env.size))
    plt.yticks(np.arange(0.5, env.size, 1), range(env.size-1, -1, -1))
    plt.grid(False)
    plt.title('GridWorld with Optimal Policy')
    
    return fig, ax

def plot_rewards(rewards):
    """
    각 에피소드의 보상을 그래프로 시각화
    """
    plt.figure(figsize=(10, 6))
    plt.plot(rewards)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Rewards per Episode')
    plt.savefig('rewards.png')
    plt.close()

def visualize_q_values(env, q_table):
    """
    학습된 Q-값을 히트맵으로 시각화
    """
    # 각 상태의 최대 Q값을 추출
    max_q_values = np.zeros((env.size, env.size))
    
    for x in range(env.size):
        for y in range(env.size):
            state_idx = env.get_state_index((x, y))
            max_q = np.max(q_table[state_idx])
            max_q_values[x, y] = max_q
    
    # 장애물에는 낮은 값 할당
    for obstacle in env.obstacles:
        x, y = obstacle
        max_q_values[x, y] = np.min(max_q_values) - 1
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(max_q_values, annot=True, cmap='viridis', fmt='.2f')
    plt.title('Max Q-Values for Each State')
    plt.savefig('q_values.png')
    plt.close()