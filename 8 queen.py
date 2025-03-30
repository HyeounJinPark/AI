import random
import matplotlib.pyplot as plt
import numpy as np

# 유전자 알고리즘 상수 정의
POPULATION_SIZE = 100  # 개체군 크기
MUTATION_RATE = 0.1    # 돌연변이 확률
MAX_GENERATIONS = 1000  # 최대 세대 수
TOURNAMENT_SIZE = 5    # 토너먼트 선택 크기

class Chromosome:
    """8-퀸 문제를 위한 염색체 클래스"""
    
    def __init__(self, genes=None):
        # 유전자가 주어지지 않으면 랜덤하게 생성
        if genes is None:
            self.genes = [random.randint(0, 7) for _ in range(8)]
        else:
            self.genes = genes.copy()
        self.fitness = 0
        self.calculate_fitness()
    
    def calculate_fitness(self):
        """적합도 계산 - 충돌하지 않는 퀸 쌍의 수"""
        conflicts = 0
        for i in range(8):
            for j in range(i + 1, 8):
                # 같은 행에 있는 경우 (이미 다른 열에 있으므로 확인 불필요)
                
                # 같은 대각선에 있는 경우
                if abs(i - j) == abs(self.genes[i] - self.genes[j]):
                    conflicts += 1
        
        # 최대 충돌 가능 쌍의 수는 28 (8C2)
        # 적합도는 충돌이 없는 쌍의 수
        self.fitness = 28 - conflicts
        return self.fitness
    
    def __str__(self):
        return f"염색체: {self.genes}, 적합도: {self.fitness}"


class GeneticAlgorithm:
    """8-퀸 문제를 위한 유전자 알고리즘 클래스"""
    
    def __init__(self, population_size=POPULATION_SIZE, mutation_rate=MUTATION_RATE):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.population = []
        self.generation = 0
        self.best_fitness_history = []
        self.avg_fitness_history = []
        
        # 초기 개체군 생성
        self.initialize_population()
    
    def initialize_population(self):
        """초기 개체군 생성"""
        self.population = [Chromosome() for _ in range(self.population_size)]
    
    def tournament_selection(self):
        """토너먼트 선택 방법으로 부모 선택"""
        tournament = random.sample(self.population, TOURNAMENT_SIZE)
        return max(tournament, key=lambda chromosome: chromosome.fitness)
    
    def crossover(self, parent1, parent2):
        """교차 연산 - 단일점 교차"""
        crossover_point = random.randint(1, 6)  # 1부터 6 사이의 교차점 선택
        
        # 자식 염색체 생성
        child_genes = parent1.genes[:crossover_point] + parent2.genes[crossover_point:]
        return Chromosome(child_genes)
    
    def mutate(self, chromosome):
        """돌연변이 연산"""
        for i in range(8):
            if random.random() < self.mutation_rate:
                chromosome.genes[i] = random.randint(0, 7)  # 랜덤한 위치로 변경
        chromosome.calculate_fitness()
        return chromosome
    
    def evolve(self):
        """한 세대 진화"""
        new_population = []
        
        # 엘리트 전략: 가장 적합도가 높은 개체는 그대로 다음 세대로 전달
        elite = max(self.population, key=lambda chromosome: chromosome.fitness)
        new_population.append(Chromosome(elite.genes))
        
        # 나머지 개체 생성
        while len(new_population) < self.population_size:
            parent1 = self.tournament_selection()
            parent2 = self.tournament_selection()
            
            # 교차
            child = self.crossover(parent1, parent2)
            
            # 돌연변이
            child = self.mutate(child)
            
            new_population.append(child)
        
        self.population = new_population
        self.generation += 1
        
        # 현재 세대의 통계 기록
        fitnesses = [chromosome.fitness for chromosome in self.population]
        self.best_fitness_history.append(max(fitnesses))
        self.avg_fitness_history.append(sum(fitnesses) / len(fitnesses))
    
    def run(self, max_generations=MAX_GENERATIONS):
        """유전자 알고리즘 실행"""
        for _ in range(max_generations):
            best_chromosome = max(self.population, key=lambda chromosome: chromosome.fitness)
            
            # 최적해를 찾았으면 종료
            if best_chromosome.fitness == 28:
                break
            
            self.evolve()
        
        return max(self.population, key=lambda chromosome: chromosome.fitness)
    
    def plot_fitness_history(self):
        """적합도 변화 그래프 그리기"""
        plt.figure(figsize=(10, 6))
        plt.plot(self.best_fitness_history, label='최고 적합도')
        plt.plot(self.avg_fitness_history, label='평균 적합도')
        plt.xlabel('세대')
        plt.ylabel('적합도')
        plt.title('세대별 적합도 변화')
        plt.legend()
        plt.grid(True)
        plt.savefig('fitness_history.png')
        plt.close()


def visualize_solution(solution):
    """8-퀸 문제의 해를 시각화"""
    board = np.zeros((8, 8))
    
    # 퀸 위치 표시
    for col, row in enumerate(solution.genes):
        board[row, col] = 1
    
    # 체스판 그리기
    plt.figure(figsize=(8, 8))
    plt.imshow(board, cmap='binary')
    
    # 격자 그리기
    for i in range(9):
        plt.axhline(i - 0.5, color='black', linewidth=1)
        plt.axvline(i - 0.5, color='black', linewidth=1)
    
    # 퀸 위치에 'Q' 표시
    for col, row in enumerate(solution.genes):
        plt.text(col, row, 'Q', ha='center', va='center', color='red', fontsize=20)
    
    plt.title('8-퀸 문제 해결책')
    plt.xticks(range(8))
    plt.yticks(range(8))
    plt.savefig('eight_queens_solution.png')
    plt.close()


def main():
    print("8-퀸 문제를 유전자 알고리즘으로 해결합니다.")
    
    # 유전자 알고리즘 객체 생성 및 실행
    ga = GeneticAlgorithm()
    solution = ga.run()
    
    print(f"\n{ga.generation}세대 후 찾은 최적해:")
    print(solution)
    
    # 적합도 변화 그래프 그리기
    ga.plot_fitness_history()
    
    # 해결책 시각화
    visualize_solution(solution)
    
    # 체스판에 퀸 배치 출력
    print("\n체스판 표현:")
    for row in range(8):
        line = ""
        for col in range(8):
            if solution.genes[col] == row:
                line += "Q "
            else:
                line += ". "
        print(line)


if __name__ == "__main__":
    main()
