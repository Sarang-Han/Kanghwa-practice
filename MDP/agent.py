import numpy as np
import random

class QLearningAgent:
    def __init__(self, env, learning_rate=0.1, discount_factor=0.9, epsilon=0.1):
        self.env = env
        self.learning_rate = learning_rate  # 학습률
        self.discount_factor = discount_factor  # 할인계수
        self.epsilon = epsilon  # 탐험률
        
        # Q-테이블 초기화 (상태 x 액션)
        self.q_table = np.zeros((env.size * env.size, env.action_space))
        
    def select_action(self, state):
        """
        입실론-탐욕 정책에 따라 행동 선택
        """
        state_idx = self.env.get_state_index(state)
        
        # 탐험: 랜덤 액션
        if random.random() < self.epsilon:
            return random.randint(0, self.env.action_space - 1)
        # 활용: 최대 Q값을 갖는 액션
        else:
            return np.argmax(self.q_table[state_idx])
    
    def learn(self, state, action, reward, next_state, done):
        """
        Q-러닝 업데이트 규칙을 사용하여 Q 테이블 업데이트
        """
        state_idx = self.env.get_state_index(state)
        next_state_idx = self.env.get_state_index(next_state)
        
        # 현재 Q값
        current_q = self.q_table[state_idx, action]
        
        # 다음 상태의 최대 Q값
        max_next_q = np.max(self.q_table[next_state_idx])
        
        # Q-러닝 업데이트 규칙
        if done:
            target_q = reward
        else:
            target_q = reward + self.discount_factor * max_next_q
        
        # Q-테이블 업데이트
        self.q_table[state_idx, action] += self.learning_rate * (target_q - current_q)
    
    def get_optimal_policy(self):
        """
        학습된 Q-테이블을 바탕으로 최적 정책 반환
        """
        policy = {}
        for state in self.env.get_all_states():
            state_idx = self.env.get_state_index(state)
            policy[state] = np.argmax(self.q_table[state_idx])
        return policy