import numpy as np
from collections import defaultdict

class MonteCarloPredictor:
    """
    몬테카를로 예측 알고리즘 구현
    """
    def __init__(self, env, gamma=0.9):
        self.env = env
        self.gamma = gamma  # 할인율
        self.reset()
    
    def reset(self):
        # 가치 함수 및 방문 횟수 초기화
        self.values = defaultdict(float)  # 상태 가치 함수
        self.returns = defaultdict(list)  # 각 상태의 반환값 목록
        self.visit_counts = defaultdict(int)  # 상태 방문 횟수
        
        # 각 에피소드의 가치 함수 변화 추적
        self.value_history = []
    
    def generate_episode(self, policy):
        """무작위 정책을 따라 에피소드 생성"""
        episode = []
        state = self.env.reset()
        done = False
        
        while not done:
            action = policy(state)
            next_state, reward, done = self.env.step(action)
            episode.append((state, action, reward))
            state = next_state
        
        return episode
    
    def predict(self, num_episodes, policy):
        """몬테카를로 예측 수행"""
        self.reset()
        
        for i in range(num_episodes):
            # 에피소드 생성
            episode = self.generate_episode(policy)
            
            # 에피소드에서 방문한 상태들
            states_in_episode = set([step[0] for step in episode])
            
            # 각 상태에 대해 리턴(return) 계산
            G = 0
            for t in range(len(episode)-1, -1, -1):
                state, action, reward = episode[t]
                G = self.gamma * G + reward
                
                # 첫 방문 몬테카를로 방식 - 이미 계산한 상태는 건너뜀
                if state in states_in_episode:
                    self.returns[state].append(G)
                    self.values[state] = np.mean(self.returns[state])
                    self.visit_counts[state] += 1
                    states_in_episode.remove(state)
            
            # 현재 가치 함수 저장
            if i % 10 == 0 or i == num_episodes - 1:  # 10 에피소드마다 또는 마지막에 저장
                value_snapshot = dict(self.values)
                self.value_history.append((i, value_snapshot))
        
        return self.values, self.value_history, self.visit_counts
    
    def get_value_table(self):
        """모든 상태에 대한 가치 함수 테이블 반환"""
        all_states = self.env.get_all_states()
        value_table = {}
        
        for state in all_states:
            value_table[state] = self.values.get(state, 0.0)
            
        return value_table