

import random
import time
from collections import defaultdict
import networkx as nx
import matplotlib.pyplot as plt

def load_col_file(filename):
    adj_list = {}
    with open(filename, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("c"):
                continue
            if line.startswith("e"):
                _, u, v = line.split()
                u = int(u) - 1
                v = int(v) - 1
                adj_list.setdefault(u, []).append(v)
                adj_list.setdefault(v, []).append(u)
    return adj_list


def load_basic_file(filename):
    adj_list = {}
    with open(filename, "r") as f:
        lines = f.readlines()
    num_vertices = int(lines[0].strip())
    for i in range(num_vertices):
        adj_list[i] = []
    for line in lines[1:]:
        u, v = map(int, line.strip().split())
        u -= 1
        v -= 1
        adj_list[u].append(v)
        adj_list[v].append(u)
    return adj_list


class TabuColoring:
    def __init__(self, graph, max_time=30):
        self.graph = graph
        self.n = len(graph)
        self.max_time = max_time
        self.best_coloring = None
        self.best_colors_used = self.n

    def greedy_initial_coloring(self, k):
        colors = [-1] * self.n
        for node in sorted(range(self.n), key=lambda x: -len(self.graph[x])):
            used = {colors[nei] for nei in self.graph[node] if colors[nei] != -1}
            for color in range(k):
                if color not in used:
                    colors[node] = color
                    break
            if colors[node] == -1:
                # Awaryjne przypisanie koloru
                colors[node] = random.randint(0, k - 1)
        return colors

    def reduce_coloring(self, coloring, k_new):
        # Grupowanie wierzchołków według kolorów
        color_classes = defaultdict(list)
        for node, color in enumerate(coloring):
            color_classes[color].append(node)

        # Sortujemy kolory po liczbie przypisanych wierzchołków (rosnąco)
        sorted_colors = sorted(color_classes.items(), key=lambda x: len(x[1]))

        # Wybieramy dwie najmniejsze klasy do scalenia
        (c1, nodes1), (c2, nodes2) = sorted_colors[:2]

        # Mapujemy stare kolory na nowe
        color_mapping = {}
        new_color = 0
        for old_color in range(k_new + 1):  # z k -> k_new (czyli -1)
            if old_color in (c1, c2):
                continue
            color_mapping[old_color] = new_color
            new_color += 1
        merged_color = new_color
        color_mapping[c1] = merged_color
        color_mapping[c2] = merged_color

        # Nowe kolorowanie
        new_coloring = [color_mapping[old] for old in coloring]
        return new_coloring

    def count_conflicts(self, coloring):
        conflicts = 0
        for u in range(self.n):
            for v in self.graph[u]:
                if u < v and coloring[u] == coloring[v]:
                    conflicts += 1
        return conflicts

    def tabu_search(self, k):
        no_improve = 0
        max_no_improve = 25000
        print(f"\nStart tabu_search dla k={k}")
        if self.best_coloring and self.best_colors_used == k + 1:
            print("🔄 Startujemy od zredukowanego poprzedniego kolorowania.")
            current = self.reduce_coloring(self.best_coloring, k)
        else:
            current = self.greedy_initial_coloring(k)
        current_conflicts = self.count_conflicts(current)
        print(f"Greedy dał {len(set(current))} kolorów, konflikty: {current_conflicts}")

        best = current[:]
        best_conflicts = current_conflicts

        tabu_list = defaultdict(int)
        tabu_tenure = max(5, int(self.n / 10 + current_conflicts / 5))
        max_iterations = 100000
        iteration = 0
        start_time = time.time()

        while iteration < max_iterations and time.time() - start_time < self.max_time:

            neighborhood = []
            sample_size = max(10, self.n // 20)
            nodes_to_check = random.sample(range(self.n), sample_size)

            for node in nodes_to_check:
                original_color = current[node]
                for color in range(k):
                    if color != original_color:
                        delta = 0
                        for neighbor in self.graph[node]:
                            if current[neighbor] == original_color:
                                delta -= 1
                            if current[neighbor] == color:
                                delta += 1

                        move = (node, color)
                        if tabu_list[move] <= iteration or (current_conflicts + delta) < best_conflicts:
                            neighborhood.append((current_conflicts + delta, delta, node, color))

            if not neighborhood:
                break

            neighborhood.sort()
            new_conflicts, delta, node, color = neighborhood[0]
            original_color = current[node]
            current[node] = color
            current_conflicts += delta
            tabu_list[(node, original_color)] = iteration + tabu_tenure

            if current_conflicts < best_conflicts:
                best = current[:]
                best_conflicts = current_conflicts
                no_improve = 0
            else:
                no_improve += 1

            if no_improve >= max_no_improve:
                # dywersyfikacja: np. wstrząs kolorami
                for i in range(self.n):
                    current[i] = random.randint(0, k - 1)
                current_conflicts = self.count_conflicts(current)
                no_improve = 0
                print(f"🔀 Dywersyfikacja po {max_no_improve} krokach bez poprawy.")

            if iteration % 100 == 0 or current_conflicts == 0:
                elapsed = time.time() - start_time
                print(f"Iteracja {iteration}, konflikty: {current_conflicts}, czas: {elapsed:.2f}s")

            if current_conflicts == 0:
                break

            iteration += 1

        final_conflicts = self.count_conflicts(best)
        if final_conflicts == 0 and -1 not in best:
            print("✅ Znaleziono poprawne kolorowanie.")
            return best
        else:
            print(f"❌ Konflikty pozostałe: {final_conflicts}")
            return None

    def solve(self, total_time_limit=180):
        start_time = time.time()
        initial_coloring = self.greedy_initial_coloring(self.n)
        initial_k = max(initial_coloring) + 1
        print(f"Algorytm zachłanny znalazł {initial_k} kolorów.")
        k = initial_k

        while k > 0:
            elapsed = time.time() - start_time
            remaining_time = total_time_limit - elapsed
            if remaining_time <= 0:
                print("⏳ Przekroczono limit czasu globalnego.")
                break

            print(f"\n🟡 Próbujemy z k = {k} (pozostało {remaining_time:.1f}s)")
            # ustaw czas tylko na tyle, ile zostało
            self.max_time = remaining_time
            result = self.tabu_search(k)
            if result and -1 not in result:
                print(f"✅ Znaleziono kolorowanie z {k} kolorami.")
                self.best_coloring = result
                self.best_colors_used = k
                k -= 1
            else:
                print(f"❌ Nie udało się dla k = {k}")
                break

        return self.best_coloring, self.best_colors_used, initial_k

def draw_colored_graph(graph, coloring, title="Pokolorowany graf"):
    G = nx.Graph()
    for u in range(len(graph)):
        for v in graph[u]:
            if u < v:
                G.add_edge(u, v)
    colors = [coloring[node] for node in G.nodes()]
    pos = nx.spring_layout(G, seed=42)
    plt.figure(figsize=(8, 6))
    nx.draw(G, pos, with_labels=True, node_color=colors, cmap=plt.colormaps["tab10"], node_size=500)
    plt.title(title)
    plt.show()



adj_list = load_col_file("queen8_12.txt")
# adj_list = load_basic_file("gc500.txt")
solver = TabuColoring(adj_list, max_time=60)
coloring, num_colors, greedy_num_colors = solver.solve(total_time_limit=180)

print("\n--- Wynik końcowy ---")
print("Liczba użytych kolorów algorytmu zachłannego:", greedy_num_colors)
print("Liczba użytych kolorów Tabu Search:", num_colors)
print("Najlepsze kolorowanie Tabu Search:", coloring)


if coloring and len(adj_list) < 50:
    draw_colored_graph(adj_list, coloring, title=f"Pokolorowany graf - {num_colors} kolorów")
else:
    print("Nie można narysować grafu, ponieważ jest zbyt duży lub nie znaleziono poprawnego kolorowania.")
