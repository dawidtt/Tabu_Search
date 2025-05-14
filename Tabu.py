import random
import matplotlib.pyplot as plt
import networkx as nx
import math
from collections import deque
import time

def compute_tabu_tenure(num_vertices, alpha=1.5):
    return max(5, int(alpha * math.sqrt(num_vertices)))

def draw_colored_graph(graph, coloring, title="Pokolorowany graf"):
    G = nx.Graph()
    for u in graph:
        for v in graph[u]:
            if u < v:
                G.add_edge(u, v)
    colors = [coloring[node] for node in G.nodes()]
    pos = nx.spring_layout(G, seed=42)
    plt.figure(figsize=(8, 6))
    nx.draw(G, pos, with_labels=True, node_color=colors, cmap=plt.colormaps["tab10"], node_size=500)
    plt.title(title)
    plt.show()

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

def largest_first_coloring(adj_list):
    degrees = {node: len(neighbors) for node, neighbors in adj_list.items()}
    sorted_vertices = sorted(degrees, key=lambda x: -degrees[x])
    coloring = [-1] * len(adj_list)
    for vertex in sorted_vertices:
        neighbor_colors = set(coloring[neigh] for neigh in adj_list[vertex] if coloring[neigh] != -1)
        color = 0
        while color in neighbor_colors:
            color += 1
        coloring[vertex] = color
    return coloring

def count_conflicts(graph, coloring):
    visited = set()
    conflicts = 0
    for u in graph:
        for v in graph[u]:
            if u < v and coloring[u] == coloring[v]:
                conflicts += 1
    return conflicts

def get_conflicts_per_vertex(graph, coloring):
    conflicts = [0] * len(graph)
    for u in graph:
        for v in graph[u]:
            if coloring[u] == coloring[v]:
                conflicts[u] += 1
    return conflicts

def initialize_neighbor_color_count(graph, coloring):
    neighbor_color_count = [{} for _ in range(len(graph))]
    for u in graph:
        for v in graph[u]:
            color_u = coloring[u]
            color_v = coloring[v]
            if color_u != -1:
                neighbor_color_count[v][color_u] = neighbor_color_count[v].get(color_u, 0) + 1
            if color_v != -1:
                neighbor_color_count[u][color_v] = neighbor_color_count[u].get(color_v, 0) + 1
    return neighbor_color_count

def fast_delta(vertex, new_color, neighbor_color_count, coloring):
    old_color = coloring[vertex]
    return neighbor_color_count[vertex].get(new_color, 0) - neighbor_color_count[vertex].get(old_color, 0)

def tabu_search_dynamic(graph, initial_num_colors, max_iter=1000, tabu_tenure=7):
    n = len(graph)
    best_valid_coloring = None
    best_valid_color_count = None
    best_valid_conflicts = float('inf')
    best_invalid_coloring = None
    best_invalid_conflicts = float('inf')
    best_invalid_color_count = None
    found = True
    for num_colors in range(initial_num_colors, 0, -1):
        print(f"\nPróba z {num_colors} kolorami...")

        coloring = [random.randint(0, num_colors - 1) for _ in range(n)]
        tabu_list = deque()
        tabu_set = set()
        global_best = coloring[:]
        global_best_conflicts = count_conflicts(graph, coloring)
        neighbor_color_count = initialize_neighbor_color_count(graph, coloring)

        mode = "intensification"
        switch_interval = 500
        no_improvement_count = 0

        if found == False:
            break
        for iteration in range(max_iter):
            found = False
            if iteration % switch_interval == 0 and iteration != 0:
                if mode == "intensification":
                    mode = "diversification"
                    tabu_tenure *= 2
                else:
                    mode = "intensification"
                    tabu_tenure = max(5, tabu_tenure // 2)
                print(f"🔄 Zmiana trybu na: {mode.upper()} (iteracja {iteration}, tabu_tenure = {tabu_tenure})")

            total_conflicts = count_conflicts(graph, coloring)
            if total_conflicts == 0 and (best_valid_color_count is None or num_colors < best_valid_color_count):
                found = True
                print(f"✅ Poprawne kolorowanie ({num_colors} kolorów) znalezione po {iteration} iteracjach.")
                best_valid_coloring = coloring[:]
                best_valid_color_count = num_colors
                best_valid_conflicts = 0
                break

            conflicts_per_vertex = get_conflicts_per_vertex(graph, coloring)
            best_move = None
            best_move_conflicts = total_conflicts

            for vertex in range(n):
                if conflicts_per_vertex[vertex] == 0:
                    continue
                current_color = coloring[vertex]
                for color in range(num_colors):
                    if color == current_color:
                        continue
                    delta = fast_delta(vertex, color, neighbor_color_count, coloring)
                    new_conflicts = total_conflicts + delta
                    is_tabu = (vertex, color) in tabu_set
                    if new_conflicts < best_move_conflicts or (is_tabu and new_conflicts < global_best_conflicts):
                        best_move = (vertex, color)
                        best_move_conflicts = new_conflicts

            if best_move is None:
                vertex = random.randint(0, n - 1)
                color = random.randint(0, num_colors - 1)
                best_move = (vertex, color)

            vertex, new_color = best_move
            coloring[vertex] = new_color
            neighbor_color_count = initialize_neighbor_color_count(graph, coloring)

            if best_move_conflicts < global_best_conflicts:
                global_best_conflicts = best_move_conflicts
                global_best = coloring[:]
                no_improvement_count = 0
            else:
                no_improvement_count += 1

            tabu_list.append((vertex, new_color))
            tabu_set.add((vertex, new_color))
            if len(tabu_list) > tabu_tenure:
                expired = tabu_list.popleft()
                tabu_set.discard(expired)

        if global_best_conflicts < best_invalid_conflicts:
            best_invalid_coloring = global_best[:]
            best_invalid_conflicts = global_best_conflicts
            best_invalid_color_count = num_colors

        print(f"🔸 Najlepsze dla {num_colors} kolorów: {global_best_conflicts} konfliktów.")
        if global_best_conflicts > 0:
            break

    if best_valid_coloring:
        return best_valid_coloring, best_valid_conflicts, best_valid_color_count
    else:
        return best_invalid_coloring, best_invalid_conflicts, best_invalid_color_count

# ---------------------- MAIN ----------------------

if __name__ == "__main__":
    adj_list = load_basic_file("gc500.txt")

    # Largest First
    lf_coloring = largest_first_coloring(adj_list)
    used_colors = set(lf_coloring)
    print(f"Liczba kolorów LF: {len(set(lf_coloring))}")
    if len(used_colors) < 50:
        draw_colored_graph(adj_list, lf_coloring, title="Largest First Coloring")

    # Tabu Search
    best_coloring, best_score, best_num_colors = tabu_search_dynamic(
        adj_list,
        initial_num_colors=len(set(lf_coloring)),
        max_iter=5000,
        tabu_tenure=compute_tabu_tenure(len(adj_list), 1.8)
    )

    print(f"Liczba konfliktów: {best_score}")
    print(f"Użyte kolory: {set(best_coloring)}")
    print(f"Liczba kolorów: {len(set(best_coloring))}")

    if best_coloring and len(set(best_coloring)) < 50:
        draw_colored_graph(adj_list, best_coloring, title=f"Tabu Search Coloring ({best_num_colors} colors)")
