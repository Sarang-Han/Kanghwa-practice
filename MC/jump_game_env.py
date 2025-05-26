import numpy as np

class JumpGameEnv:
    """
    간단한 점프 게임 환경
    
    상태: 0부터 10까지의 위치 (0: 시작, 10: 목표)
    행동: 0 (걷기), 1 (점프)
    
    규칙:
    - 걷기: 1칸 전진, 장애물이 있으면 제자리
    - 점프: 2칸 전진, 에너지 1 소모
    - 장애물 위치: 3, 6, 8
    - 시작 시 에너지: 3
    - 목표(10)에 도달하거나 더 이상 움직일 수 없으면 종료
    """
    
    def __init__(self):
        # 게임판 크기 (0부터 9까지)
        self.board_size = 10
        # 장애물 위치
        self.obstacles = [3, 6, 8]
        # 목표 위치
        self.goal = 10
        # 에이전트 초기 위치와 에너지
        self.reset()
    
    def reset(self):
        # 에이전트 위치 초기화
        self.position = 0
        # 에너지 초기화
        self.energy = 3
        # 종료 여부
        self.done = False
        # 현재 상태 반환
        return self._get_state()
    
    def step(self, action):
        # 이미 종료된 경우
        if self.done:
            return self._get_state(), 0, True
        
        # 이전 위치 저장
        prev_position = self.position
        
        # 행동에 따른 상태 변화
        if action == 0:  # 걷기
            if self.position + 1 in self.obstacles:
                # 장애물이 있으면 제자리
                pass
            else:
                # 없으면 한 칸 전진
                self.position += 1
        elif action == 1:  # 점프
            if self.energy > 0:
                # 에너지가 있으면 2칸 전진하고 에너지 소모
                target_pos = self.position + 2
                
                # 목표가 장애물인 경우, 장애물 바로 앞에 머무름
                if target_pos in self.obstacles:
                    self.position = target_pos - 1  # 장애물 바로 앞에 위치
                else:
                    self.position = target_pos  # 일반적인 점프
                
                self.energy -= 1
            else:
                # 에너지가 없으면 제자리
                pass
        
        # 보상
        reward = 0
        
        # 전진했다면 약간의 보상
        if self.position > prev_position:
            # 거리에 비례한 보상 - 더 멀리 갈수록 더 높은 보상
            reward = 0.2 * (self.position - prev_position)
        
        if self.position >= self.goal:
            # 목표 도달 시 큰 보상
            reward = 10
            self.position = self.goal
            self.done = True
        
        # 더 이상 움직일 수 없는 경우 게임 종료 (약간의 페널티)
        if not self.done:
            stuck = False
            if self.position + 1 in self.obstacles and (self.energy == 0 or self.position + 2 >= self.board_size):
                stuck = True
            
            if stuck:
                reward = -1  # 막다른 길에 갇히면 페널티
                self.done = True
        
        return self._get_state(), reward, self.done
    
    def _get_state(self):
        # 상태는 (위치, 에너지)의 조합
        return (self.position, self.energy)
    
    def get_all_states(self):
        # 가능한 모든 상태 조합
        states = []
        for pos in range(self.board_size):
            for energy in range(4):
                # 목표 도달 후의 에너지는 중요하지 않음
                if pos == self.goal:
                    states.append((pos, 0))
                    break
                else:
                    states.append((pos, energy))
        return states

    def render(self):
        # 현재 게임 상태를 텍스트로 출력
        board = ['-'] * self.board_size
        for obs in self.obstacles:
            board[obs] = 'X'
        board[self.goal] = 'G'
        board[self.position] = 'P'
        
        print(''.join(board))
        print(f"위치: {self.position}, 에너지: {self.energy}")