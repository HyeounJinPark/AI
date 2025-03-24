import copy
import heapq
import time
import random

def generate_random_initial_state():
    """랜덤한 초기 상태를 생성하는 함수"""
    # 0부터 9까지의 숫자를 포함하는 리스트 생성
    numbers = list(range(10))
    # 리스트를 무작위로 섞음
    random.shuffle(numbers)
    
    # 2x5 배열로 변환
    state = [numbers[:5], numbers[5:]]
    return state

def is_solvable(state, goal):
    """
    퍼즐이 해결 가능한지 확인하는 함수
    8 퍼즐에서는 반전 수(inversion count)가 짝수일 때만 해결 가능
    """
    # 1차원 배열로 변환 (빈 칸 제외)
    flat_state = [num for row in state for num in row if num != 0]
    flat_goal = [num for row in goal for num in row if num != 0]
    
    # 반전 수 계산
    inversions = 0
    for i in range(len(flat_state)):
        for j in range(i + 1, len(flat_state)):
            # 목표 상태에서의 위치를 기준으로 반전 수 계산
            idx_i = flat_goal.index(flat_state[i])
            idx_j = flat_goal.index(flat_state[j])
            if idx_i > idx_j:
                inversions += 1
    
    # 빈 칸의 행 위치 차이 계산
    blank_row_state = next(i for i, row in enumerate(state) if 0 in row)
    blank_row_goal = next(i for i, row in enumerate(goal) if 0 in row)
    row_diff = abs(blank_row_state - blank_row_goal)
    
    # 반전 수가 짝수이고 행 차이가 짝수이거나,
    # 반전 수가 홀수이고 행 차이가 홀수일 때 해결 가능
    return (inversions % 2 == 0 and row_diff % 2 == 0) or (inversions % 2 == 1 and row_diff % 2 == 1)

# 목표 상태 정의 (2x5 배열)
goal_state = [[1, 2, 3, 4, 5],
              [6, 7, 8, 9, 0]]

def get_blank_position(state):
    """빈 칸(0)의 위치를 찾는 함수"""
    for i in range(len(state)):
        for j in range(len(state[0])):
            if state[i][j] == 0:
                return i, j
    return None

def get_possible_moves(i, j, rows, cols):
    """가능한 이동 방향을 반환하는 함수"""
    moves = []
    # 상하좌우 이동 가능한 위치 확인
    if i > 0: moves.append((-1, 0))  # 위
    if i < rows-1: moves.append((1, 0))   # 아래
    if j > 0: moves.append((0, -1))  # 왼쪽
    if j < cols-1: moves.append((0, 1))   # 오른쪽
    return moves

def make_move(state, move, blank_pos):
    """이동을 수행하여 새로운 상태를 생성하는 함수"""
    new_state = copy.deepcopy(state)
    i, j = blank_pos
    di, dj = move
    new_i, new_j = i + di, j + dj
    
    # 빈 칸과 이동할 위치의 숫자를 교환
    new_state[i][j], new_state[new_i][new_j] = new_state[new_i][new_j], new_state[i][j]
    return new_state

def state_to_tuple(state):
    """2D 배열을 1D 튜플로 변환 (해시 가능하게)"""
    return tuple(sum(state, []))

def manhattan_distance(state, goal_state):
    """
    맨해튼 거리 휴리스틱: 각 타일이 목표 위치까지 이동해야 하는 최소 거리의 합
    """
    distance = 0
    rows = len(state)
    cols = len(state[0])
    
    for i in range(rows):
        for j in range(cols):
            if state[i][j] != 0:  # 빈 칸은 제외
                # 목표 상태에서의 위치 찾기
                for gi in range(rows):
                    for gj in range(cols):
                        if goal_state[gi][gj] == state[i][j]:
                            distance += abs(i - gi) + abs(j - gj)
                            break
    return distance

def misplaced_tiles(state, goal_state):
    """
    잘못된 위치에 있는 타일 수 휴리스틱
    """
    count = 0
    rows = len(state)
    cols = len(state[0])
    
    for i in range(rows):
        for j in range(cols):
            if state[i][j] != 0 and state[i][j] != goal_state[i][j]:
                count += 1
    return count

def a_star_search(initial_state, goal_state, time_limit=30):
    """
    A* 알고리즘을 사용한 퍼즐 해결
    """
    print("A* 알고리즘으로 퍼즐 해결 중...")
    start_time = time.time()
    nodes_expanded = 0
    max_queue_size = 0
    
    rows = len(initial_state)
    cols = len(initial_state[0])
    
    # (f값, 이동 횟수, 상태, 경로)
    open_set = [(manhattan_distance(initial_state, goal_state), 0, initial_state, [initial_state])]
    heapq.heapify(open_set)
    
    # 방문한 상태와 그 상태에 도달하는 최소 이동 횟수
    closed_set = {state_to_tuple(initial_state): 0}
    
    while open_set:
        # 시간 제한 확인
        if time.time() - start_time > time_limit:
            end_time = time.time()
            print(f"시간 제한 초과 ({time_limit}초)")
            print(f"확장된 노드 수: {nodes_expanded}")
            print(f"최대 큐 크기: {max_queue_size}")
            print(f"소요 시간: {end_time - start_time:.6f}초")
            return None
            
        max_queue_size = max(max_queue_size, len(open_set))
        
        # f값이 가장 작은 노드 선택
        f, g, state, path = heapq.heappop(open_set)
        nodes_expanded += 1
        
        # 진행 상황 주기적으로 출력
        if nodes_expanded % 1000 == 0:
            print(f"노드 {nodes_expanded}개 확장, 현재 f값: {f}, 경로 길이: {g}, 경과 시간: {time.time() - start_time:.2f}초")
        
        # 목표 상태에 도달했는지 확인
        if state == goal_state:
            end_time = time.time()
            print("\n해결책을 찾았습니다!")
            print(f"이동 횟수: {len(path) - 1}")
            print(f"확장된 노드 수: {nodes_expanded}")
            print(f"최대 큐 크기: {max_queue_size}")
            print(f"소요 시간: {end_time - start_time:.6f}초")
            
            # 해결 경로 출력
            print("\n해결 경로:")
            for step, s in enumerate(path):
                print(f"\n단계 {step}:")
                for row in s:
                    print(row)
            
            return path
        
        # 가능한 모든 이동 탐색
        blank_pos = get_blank_position(state)
        possible_moves = get_possible_moves(*blank_pos, rows, cols)
        
        for move in possible_moves:
            new_state = make_move(state, move, blank_pos)
            new_g = g + 1  # 이동 횟수 증가
            new_f = new_g + manhattan_distance(new_state, goal_state)  # f = g + h
            
            state_tuple = state_to_tuple(new_state)
            
            # 이전에 방문하지 않았거나, 더 적은 비용으로 방문 가능한 경우
            if state_tuple not in closed_set or new_g < closed_set[state_tuple]:
                closed_set[state_tuple] = new_g
                heapq.heappush(open_set, (new_f, new_g, new_state, path + [new_state]))
    
    # 해결책을 찾지 못한 경우
    end_time = time.time()
    print("\n해결책을 찾을 수 없습니다.")
    print(f"확장된 노드 수: {nodes_expanded}")
    print(f"최대 큐 크기: {max_queue_size}")
    print(f"소요 시간: {end_time - start_time:.6f}초")
    return None

def print_puzzle_state(state):
    """퍼즐 상태를 출력하는 함수"""
    print("\n현재 퍼즐 상태:")
    for row in state:
        print(row)
    print()

def main():
    print("2x5 8 퍼즐 게임 - A* 알고리즘 (랜덤 초기 상태)")
    
    # 해결 가능한 랜덤 초기 상태 생성
    while True:
        initial_state = generate_random_initial_state()
        if is_solvable(initial_state, goal_state):
            break
        print("생성된 초기 상태가 해결 불가능합니다. 다시 생성합니다...")
    
    print("초기 상태:")
    for row in initial_state:
        print(row)
    print("\n목표 상태:")
    for row in goal_state:
        print(row)
    
    # 난이도 선택 (맨해튼 거리로 추정)
    manhattan_dist = manhattan_distance(initial_state, goal_state)
    
    if manhattan_dist <= 10:
        difficulty = "쉬움"
    elif manhattan_dist <= 20:
        difficulty = "중간"
    else:
        difficulty = "어려움"
    
    print(f"\n추정 난이도: {difficulty} (맨해튼 거리: {manhattan_dist})")
    
    print("\n퍼즐 해결을 시작합니다...")
    solution = a_star_search(initial_state, goal_state)
    
    if solution is None:
        print("\nA* 알고리즘으로 해결책을 찾지 못했습니다.")
        print("이는 시간 제한 내에 해결책을 찾지 못했기 때문일 수 있습니다.")
        print("시간 제한을 늘리거나 다른 휴리스틱 함수를 시도해보세요.")

if __name__ == "__main__":
    # 랜덤 시드 설정 (재현 가능한 결과를 위해)
    random.seed()
    main()
