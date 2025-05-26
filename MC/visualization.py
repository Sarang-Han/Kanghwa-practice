import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.animation import FuncAnimation

def plot_value_function(value_table, env):
    """상태 가치 함수 히트맵 시각화"""
    # 데이터 준비
    positions = range(env.board_size)
    max_energy = 4  # 에너지 최대값 조정 (0, 1, 2, 3)
    
    # 2D 배열로 변환
    value_grid = np.zeros((max_energy, env.board_size))
    for pos in positions:
        for energy in range(max_energy):
            value_grid[energy, pos] = value_table.get((pos, energy), 0)
    
    plt.figure(figsize=(12, 5))
    sns.heatmap(value_grid, annot=True, fmt=".2f", cmap="YlGnBu")
    
    # 장애물 위치 표시
    for obs in env.obstacles:
        plt.axvline(x=obs + 0.5, color='r', linestyle='--', alpha=0.5)
    
    plt.xlabel("Position")
    plt.ylabel("Energy")
    plt.title("State Value Function V(s)")
    plt.tight_layout()
    
    return plt

def visualize_value_convergence(value_history, env, key_states=None):
    """에피소드에 따른 가치 함수 수렴 과정 시각화"""
    if key_states is None:
        # 기본 관찰 상태
        key_states = [(0, 2), (2, 2), (4, 1), (7, 0)]
    
    episodes = [h[0] for h in value_history]
    
    plt.figure(figsize=(10, 6))
    
    for state in key_states:
        values = [h[1].get(state, 0) for h in value_history]
        plt.plot(episodes, values, marker='o', label=f"State {state}")
    
    plt.xlabel("Episodes")
    plt.ylabel("State Value V(s)")
    plt.title("Monte Carlo Prediction: Value Function Convergence")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    return plt

def plot_visit_counts(visit_counts, env):
    """상태 방문 횟수 시각화"""
    # 데이터 준비
    positions = range(env.board_size)
    max_energy = 4  # 에너지 최대값 조정 (0, 1, 2, 3)
    
    # 2D 배열로 변환
    count_grid = np.zeros((max_energy, env.board_size))
    for pos in positions:
        for energy in range(max_energy):
            count_grid[energy, pos] = visit_counts.get((pos, energy), 0)
    
    plt.figure(figsize=(12, 5))
    # fmt 형식 변경: "d" -> ".0f"
    sns.heatmap(count_grid, annot=True, fmt=".0f", cmap="Greens")
    
    # 장애물 위치 표시
    for obs in env.obstacles:
        plt.axvline(x=obs + 0.5, color='r', linestyle='--', alpha=0.5)
    
    plt.xlabel("Position")
    plt.ylabel("Energy")
    plt.title("State Visit Counts")
    plt.tight_layout()
    
    return plt