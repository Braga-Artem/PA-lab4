import numpy as np
from numba import njit
import matplotlib.pyplot as plt
import networkx as nx

@njit
def weighted_random_choice(probabilities):

    cumulative_sum = np.cumsum(probabilities)
    random_value = np.random.rand() * cumulative_sum[-1]
    for i in range(len(cumulative_sum)):
        if random_value <= cumulative_sum[i]:
            return i
    return len(cumulative_sum) - 1

@njit
def ant_colony_optimization(distance_matrix, num_ants, num_elite_ants, alpha, beta, rho, iterations, step):
    num_cities = len(distance_matrix)
    pheromone = np.ones((num_cities, num_cities))
    best_distance = np.inf
    best_path = None
    quality_over_time = []

    for iteration in range(iterations):
        paths = np.zeros((num_ants, num_cities), dtype=np.int32)
        distances = np.zeros(num_ants)

        for ant in range(num_ants):
            visited = np.zeros(num_cities, dtype=np.bool_)
            current_city = np.random.randint(0, num_cities)
            visited[current_city] = True
            path = [current_city]

            for _ in range(num_cities - 1):
                probabilities = np.zeros(num_cities)
                for next_city in range(num_cities):
                    if not visited[next_city]:
                        tau = pheromone[current_city][next_city] ** alpha
                        eta = (1.0 / distance_matrix[current_city][next_city]) ** beta
                        probabilities[next_city] = tau * eta

                probabilities_sum = probabilities.sum()
                if probabilities_sum == 0:
                    probabilities += 1
                    probabilities_sum = probabilities.sum()
                probabilities /= probabilities_sum

                next_city = weighted_random_choice(probabilities)
                path.append(next_city)
                visited[next_city] = True
                current_city = next_city

            paths[ant] = path
            distance = 0
            for i in range(num_cities):
                distance += distance_matrix[path[i - 1], path[i]]
            distances[ant] = distance

            if distances[ant] < best_distance:
                best_distance = distances[ant]
                best_path = path


        pheromone *= (1 - rho)
        for ant in range(num_ants):
            contribution = 1.0 / distances[ant]
            for i in range(num_cities):
                pheromone[paths[ant, i - 1], paths[ant, i]] += contribution

        for elite in range(num_elite_ants):
            contribution = 2.0 / distances[elite]
            for i in range(num_cities):
                pheromone[paths[elite, i - 1], paths[elite, i]] += contribution

        if iteration % step == 0:
            quality_over_time.append((iteration, best_distance))

    return best_distance, best_path, quality_over_time

def generate_distance_matrix(num_cities, min_distance=1, max_distance=60):

    distance_matrix = np.random.randint(min_distance, max_distance + 1, size=(num_cities, num_cities))
    np.fill_diagonal(distance_matrix, 0)
    return (distance_matrix + distance_matrix.T) // 2

def plot_quality(quality_over_time):

    iterations, distances = zip(*quality_over_time)
    plt.plot(iterations, distances, marker='o')
    plt.xlabel("Ітерації")
    plt.ylabel("Якість розв'язку (довжина шляху)")
    plt.title("Залежність якості розв'язку від числа ітерацій")
    plt.grid()
    plt.show()

def plot_graph(distance_matrix, best_path=None):

    num_cities = len(distance_matrix)
    G = nx.Graph()


    for i in range(num_cities):
        G.add_node(i, pos=(np.cos(2 * np.pi * i / num_cities), np.sin(2 * np.pi * i / num_cities)))


    for i in range(num_cities):
        for j in range(i + 1, num_cities):
            G.add_edge(i, j, weight=distance_matrix[i, j])


    pos = nx.get_node_attributes(G, 'pos')


    plt.figure(figsize=(10, 10))
    nx.draw_networkx_nodes(G, pos, node_size=50, node_color='blue', alpha=0.7)
    nx.draw_networkx_edges(G, pos, alpha=0.5, edge_color='gray')


    if best_path:
        path_edges = [(best_path[i - 1], best_path[i]) for i in range(len(best_path))]
        nx.draw_networkx_edges(G, pos, edgelist=path_edges, edge_color='red', width=2)

    plt.title("Граф міст на основі матриці відстаней")
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    num_cities = 300
    num_ants = 45
    num_elite_ants = 15
    alpha = 3
    beta = 2
    rho = 0.6
    iterations = 1000
    step = 20

    distance_matrix = generate_distance_matrix(num_cities)

    best_distance, best_path, quality_over_time = ant_colony_optimization(
        distance_matrix, num_ants, num_elite_ants, alpha, beta, rho, iterations, step
    )

    print(f"Найкраща довжина шляху: {best_distance}")
    print(f"Найкращий шлях: {best_path}")

    plot_quality(quality_over_time)
    plot_graph(distance_matrix, best_path)
