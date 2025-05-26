import numpy as np

class GridWorld:
    def __init__(self, size=5):
        self.size = size
        self.obstacles = [(1, 1), (2, 2), (3, 1)]  # 장애물 위치
        self.goal = (4, 4)  # 목표 위치
        self.start = (0, 0)  # 시작 위치
        
        # 액션: 0: 위, 1: 오른쪽, 2: 아래, 3: 왼쪽
        self.action_space = 4
        self.current_state = self.start
        
    def reset(self):
        """환경 초기화 및 초기 상태 반환"""
        self.current_state = self.start
        return self.current_state
    
    def step(self, action):
        """
        액션을 받아 다음 상태, 보상, 종료 여부 반환
        action: 0(위), 1(오른쪽), 2(아래), 3(왼쪽)
        """
        x, y = self.current_state
        
        # 액션에 따른 다음 상태 계산
        if action == 0:  # 위
            next_state = (max(0, x-1), y)
        elif action == 1:  # 오른쪽
            next_state = (x, min(self.size-1, y+1))
        elif action == 2:  # 아래
            next_state = (min(self.size-1, x+1), y)
        elif action == 3:  # 왼쪽
            next_state = (x, max(0, y-1))
        else:
            raise ValueError("유효하지 않은 액션입니다.")
        
        # 장애물 확인
        if next_state in self.obstacles:
            next_state = self.current_state  # 장애물이면 이동 불가
            
        # 보상 계산
        if next_state == self.goal:
            reward = 1.0  # 목표 도달
            done = True
        else:
            reward = -0.01  # 이동 페널티
            done = False
            
        self.current_state = next_state
        return next_state, reward, done
    
    #-- q 테이블을 관리하기 위한 메소드들 --#
    
    def get_state_index(self, state):
        """상태(x, y)를 인덱스로 변환"""
        x, y = state
        return x * self.size + y
    
    def get_state_from_index(self, index):
        """인덱스를 상태(x, y)로 변환"""
        x = index // self.size
        y = index % self.size
        return (x, y)
    
    def get_all_states(self):
        """가능한 모든 상태 반환"""
        states = []
        for i in range(self.size):
            for j in range(self.size):
                if (i, j) not in self.obstacles:
                    states.append((i, j))
        return states