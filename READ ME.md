# DFS와 A* 알고리즘을 활용한 8 퍼즐 게임 비교 분석

## 1. 서론

8 퍼즐 게임은 인공지능과 알고리즘 연구에서 중요한 테스트베드로 활용되는 문제입니다. 이 분석에서는 깊이 우선 탐색(Depth-First Search, DFS)과 A* 알고리즘을 사용하여 8 퍼즐 문제를 해결하는 방법을 비교 분석하겠습니다. 특히 두 알고리즘의 성능, 효율성, 그리고 최적해 찾기 능력에 초점을 맞추어 살펴보겠습니다.

## 2. 알고리즘 개요

### 2.1 깊이 우선 탐색(DFS)

깊이 우선 탐색은 그래프나 트리 구조에서 가능한 한 깊이 들어가면서 탐색을 진행하는 알고리즘입니다. 더 이상 탐색할 수 없는 노드에 도달하면 백트래킹(backtracking)하여 다른 경로를 탐색합니다.

**DFS의 주요 특징:**
- 메모리 효율성: 현재 경로상의 노드만 저장하므로 메모리 사용이 효율적입니다.
- 완전성(Completeness): 유한한 상태 공간에서는 모든 노드를 방문할 수 있습니다.
- 최적성(Optimality): 최단 경로를 보장하지 않습니다.
- 시간 복잡도: 최악의 경우 O(b^m)입니다. 여기서 b는 분기 계수(branching factor), m은 최대 깊이입니다.

### 2.2 A* 알고리즘

A* 알고리즘은 정보 기반 탐색 알고리즘으로, 시작 노드에서 목표 노드까지의 최단 경로를 찾기 위해 휴리스틱 함수를 사용합니다. 각 노드에 대해 f(n) = g(n) + h(n) 값을 계산하여 탐색 순서를 결정합니다. 여기서 g(n)은 시작 노드에서 현재 노드까지의 비용, h(n)은 현재 노드에서 목표 노드까지의 추정 비용입니다.

**A*의 주요 특징:**
- 완전성: 모든 노드를 방문하기 전에 해결책을 찾을 수 있습니다.
- 최적성: 휴리스틱 함수가 허용적(admissible)이면 최적해를 보장합니다.
- 효율성: 휴리스틱 함수를 통해 불필요한 탐색을 줄일 수 있습니다.
- 시간 및 공간 복잡도: 휴리스틱 함수의 품질에 크게 의존합니다.

## 3. 구현 세부 사항

### 3.1 문제 정의

본 분석에서는 두 가지 크기의 퍼즐을 고려했습니다:
1. **3x3 그리드**: 전통적인 8 퍼즐 문제
2. **2x5 그리드**: 변형된 형태의 퍼즐 문제

각 퍼즐에서 빈 칸(0)을 이동시켜 초기 상태에서 목표 상태로 변환하는 것이 목표입니다.

### 3.2 DFS 구현

DFS 구현은 반복적(iterative) 방식을 사용하여 재귀 제한 문제를 해결했습니다. 스택 자료구조를 사용하여 탐색할 노드를 관리하고, 방문한 상태를 집합(set)에 저장하여 중복 방문을 방지했습니다. 또한 깊이 제한과 시간 제한을 설정하여 무한 탐색을 방지했습니다.

```python
def dfs_search(time_limit=30):
    start_time = time.time()
    nodes_expanded = 0
    max_stack_size = 0
    
    stack = [(initial_state, [initial_state])]
    visited = {state_to_tuple(initial_state)}
    depth_limit = 50 if USE_3X3 else 1000
    
    while stack:
        # 시간 제한 확인
        if time.time() - start_time > time_limit:
            return {"error": "Time limit exceeded", ...}
            
        max_stack_size = max(max_stack_size, len(stack))
        state, path = stack.pop()
        nodes_expanded += 1
        
        if len(path) > depth_limit:
            continue
        
        if state == goal_state:
            return {"solution": path, ...}
        
        blank_pos = get_blank_position(state)
        possible_moves = get_possible_moves(*blank_pos)
        
        for move in reversed(possible_moves):
            new_state = make_move(state, move, blank_pos)
            state_tuple = state_to_tuple(new_state)
            
            if state_tuple not in visited:
                visited.add(state_tuple)
                stack.append((new_state, path + [new_state]))
    
    return {"solution": None, "error": "No solution found", ...}
```

### 3.3 A* 구현

A* 구현은 우선순위 큐(priority queue)를 사용하여 f값이 가장 작은 노드를 선택하도록 했습니다. 휴리스틱 함수로는 맨해튼 거리(Manhattan distance)를 사용했으며, 이는 각 타일이 목표 위치까지 이동해야 하는 최소 거리의 합입니다.

```python
def manhattan_distance(state):
    distance = 0
    for i in range(ROWS):
        for j in range(COLS):
            if state[i][j] != 0:  # 빈 칸은 제외
                for gi in range(ROWS):
                    for gj in range(COLS):
                        if goal_state[gi][gj] == state[i][j]:
                            distance += abs(i - gi) + abs(j - gj)
                            break
    return distance

def a_star_search(time_limit=30):
    start_time = time.time()
    nodes_expanded = 0
    max_queue_size = 0
    
    open_set = [(manhattan_distance(initial_state), 0, initial_state, [initial_state])]
    heapq.heapify(open_set)
    closed_set = {state_to_tuple(initial_state): 0}
    
    while open_set:
        # 시간 제한 확인
        if time.time() - start_time > time_limit:
            return {"error": "Time limit exceeded", ...}
            
        max_queue_size = max(max_queue_size, len(open_set))
        f, g, state, path = heapq.heappop(open_set)
        nodes_expanded += 1
        
        if state == goal_state:
            return {"solution": path, ...}
        
        blank_pos = get_blank_position(state)
        possible_moves = get_possible_moves(*blank_pos)
        
        for move in possible_moves:
            new_state = make_move(state, move, blank_pos)
            new_g = g + 1
            new_f = new_g + manhattan_distance(new_state)
            state_tuple = state_to_tuple(new_state)
            
            if state_tuple not in closed_set or new_g < closed_set[state_tuple]:
                closed_set[state_tuple] = new_g
                heapq.heappush(open_set, (new_f, new_g, new_state, path + [new_state]))
    
    return {"solution": None, "error": "No solution found", ...}
```

## 4. 성능 비교 분석

### 4.1 3x3 그리드에서의 성능 비교

3x3 그리드에서 두 알고리즘의 성능을 비교한 결과는 다음과 같습니다:

| 지표 | A* 알고리즘 | DFS 알고리즘 | 우수한 알고리즘 |
|------|------------|-------------|--------------|
| 해결 여부 | 성공 | 실패 | A* |
| 경로 길이 | 14 | 해결 실패 | A* |
| 확장된 노드 수 | 128 | 140,145 | A* |
| 소요 시간 | 0.012초 | 3.332초 | A* |
| 메모리 사용 (큐/스택 크기) | 86 | 43 | DFS |

**분석:**
1. **해결 능력**: A* 알고리즘은 14단계만에 해결책을 찾았지만, DFS 알고리즘은 제한된 시간 내에 해결책을 찾지 못했습니다.
2. **효율성**: A* 알고리즘은 DFS보다 훨씬 적은 수의 노드를 확장했습니다(128 vs 140,145).
3. **실행 시간**: A* 알고리즘은 DFS보다 약 277배 빠르게 실행되었습니다(0.012초 vs 3.332초).
4. **메모리 사용**: DFS 알고리즘은 A*보다 메모리 사용이 적었습니다(43 vs 86). 이는 DFS가 현재 경로상의 노드만 저장하는 반면, A*는 우선순위 큐에 많은 노드를 저장하기 때문입니다.

### 4.2 2x5 그리드에서의 성능 예측

2x5 그리드는 상태 공간이 더 크기 때문에 3x3 그리드보다 해결이 더 어렵습니다. 3x3 그리드에서의 결과를 바탕으로 2x5 그리드에서의 성능을 예측해보면:

1. **A* 알고리즘**: 상태 공간이 커지면 확장되는 노드 수와 메모리 사용량이 증가하지만, 휴리스틱 함수의 안내로 인해 여전히 효율적으로 해결책을 찾을 가능성이 높습니다.

2. **DFS 알고리즘**: 상태 공간이 커지면 탐색해야 할 경로가 기하급수적으로 증가하므로, 3x3 그리드에서도 해결책을 찾지 못한 DFS는 2x5 그리드에서는 더욱 어려움을 겪을 것으로 예상됩니다.

## 5. 알고리즘 특성 비교

### 5.1 최적해 보장

- **A* 알고리즘**: 맨해튼 거리와 같은 허용적 휴리스틱을 사용할 경우, 최적해(최단 경로)를 보장합니다.
- **DFS 알고리즘**: 최적해를 보장하지 않습니다. 첫 번째로 발견한 해결책을 반환하므로, 이것이 최단 경로가 아닐 수 있습니다.

### 5.2 시간 효율성

- **A* 알고리즘**: 휴리스틱 함수를 통해 유망한 경로를 우선적으로 탐색하므로, 불필요한 탐색을 줄이고 효율적으로 해결책을 찾을 수 있습니다.
- **DFS 알고리즘**: 깊이 우선으로 탐색하기 때문에, 해결책이 깊은 곳에 있거나 상태 공간이 큰 경우 비효율적입니다.

### 5.3 메모리 효율성

- **A* 알고리즘**: 우선순위 큐에 많은 노드를 저장해야 하므로, 메모리 사용량이 많을 수 있습니다.
- **DFS 알고리즘**: 현재 경로상의 노드만 저장하므로, 메모리 사용이 효율적입니다.

### 5.4 구현 복잡성

- **A* 알고리즘**: 우선순위 큐와 휴리스틱 함수를 구현해야 하므로, DFS보다 구현이 복잡합니다.
- **DFS 알고리즘**: 스택을 사용한 간단한 구현이 가능합니다.

## 6. 결론

8 퍼즐 게임과 같은 상태 공간 탐색 문제에서 A* 알고리즘은 DFS 알고리즘보다 훨씬 효율적인 성능을 보여주었습니다. 특히 다음과 같은 장점이 있습니다:

1. **해결 능력**: A*는 DFS가 해결하지 못한 문제를 성공적으로 해결했습니다.
2. **최적해 보장**: A*는 최단 경로를 찾을 수 있습니다.
3. **효율성**: A*는 DFS보다 훨씬 적은 수의 노드를 확장하고, 빠르게 해결책을 찾았습니다.

반면, DFS는 메모리 사용 측면에서 더 효율적이었지만, 이 장점은 해결 능력과 시간 효율성의 단점을 상쇄하기에는 충분하지 않았습니다.

따라서, 8 퍼즐과 같은 문제에서는 A* 알고리즘이 DFS보다 훨씬 적합한 선택입니다. 특히 상태 공간이 큰 2x5 그리드와 같은 변형된 퍼즐에서는 A*의 장점이 더욱 두드러질 것으로 예상됩니다.

## 7. 개선 방안

### 7.1 DFS 개선 방안

1. **반복 깊이 제한 탐색(IDDFS)**: DFS의 변형으로, 깊이 제한을 점진적으로 증가시키며 탐색합니다. 이를 통해 최적해를 보장하면서도 메모리 효율성을 유지할 수 있습니다.
2. **양방향 탐색**: 시작 상태와 목표 상태에서 동시에 탐색을 진행하여 효율성을 높일 수 있습니다.

### 7.2 A* 개선 방안

1. **더 나은 휴리스틱 함수**: 맨해튼 거리보다 더 정확한 휴리스틱 함수를 사용하면 효율성을 더욱 높일 수 있습니다. 예를 들어, 패턴 데이터베이스(pattern database)를 사용할 수 있습니다.
2. **메모리 최적화**: IDA*(Iterative Deepening A*) 알고리즘을 사용하여 메모리 사용량을 줄일 수 있습니다.

### 7.3 일반적인 개선 방안

1. **상태 표현 최적화**: 2차원 배열 대신 1차원 배열이나 정수로 상태를 표현하면 메모리 사용량과 비교 연산을 최적화할 수 있습니다.
2. **병렬 처리**: 여러 개의 스레드나 프로세스를 사용하여 병렬로 탐색을 수행하면 성능을 향상시킬 수 있습니다.

이러한 개선 방안을 적용하면 더 큰 퍼즐 문제(예: 15 퍼즐)에서도 효율적으로 해결책을 찾을 수 있을 것입니다.
