import random

def generate_random_graph_with_size_range(
    min_vertices=30, max_vertices=150,
    edge_prob_min=0.02, edge_prob_max=0.15,
    filename="generated.col"
):
    num_vertices = random.randint(min_vertices, max_vertices)

    # Przykładowa funkcja: gęstość maleje z rozmiarem grafu
    edge_probability = random.uniform(edge_prob_min, edge_prob_max)
    edges = []

    for u in range(num_vertices):
        for v in range(u + 1, num_vertices):
            if random.random() < edge_probability:
                edges.append((u + 1, v + 1))  # DIMACS format: 1-based

    with open(filename, "w") as f:
        f.write(f"p edge {num_vertices} {len(edges)}\n")
        for u, v in edges:
            f.write(f"e {u} {v}\n")

    print(f"Wygenerowano graf z {num_vertices} wierzchołkami i {len(edges)} krawędziami do pliku '{filename}'.")

# Przykład użycia:
for i in range(10):
    generate_random_graph_with_size_range(
        min_vertices=30,
        max_vertices=350,
        edge_prob_min=0.02,
        edge_prob_max=0.1,
        filename=f"test_graph-{i}.txt"
    )
