import networkx as nx
from search import shortest_paths, all_shortest_paths


def degree_centrality(v, G):
    return G.degree[v]


def closeness_centrality(v, G):
    paths = shortest_paths(v, G)
    return 1 / sum(len(path)
           for path in paths.values())


def betweenness_centrality(G):
    all_paths = {v: all_shortest_paths(v, G)
                          for v in G.nodes()}
    scores = {v: 0.0 for v in G.nodes()}
    vertices = list(G.nodes());
    for i in range(len(vertices)):
        for j in range(len(vertices)):
            if (i != j):
                paths = all_paths[vertices[i]][vertices[j]]
                c = 1 / len(paths)  # contribution to betweenness
                for path in paths:
                    for v in path:
                        if v not in [vertices[i], vertices[j]]:
                            scores[v] += c
    return scores


def pagerank(G, damping=0.85, iterations=30):
    n = len(G.nodes())
    scores = { vertex : 1 / n for vertex in G.nodes() }  # initialization
    for __ in range(1, iterations):
        new_scores = { vertex : (1 - damping) / n for vertex in G.nodes() }
        for vertex in G.nodes():
            score = scores[vertex] * damping  # distribute score to neighbors
            for edge in G.out_edges(vertex):
                neighbor = edge[1]
                new_scores[neighbor] += score / len(G.out_edges(vertex))
        scores = new_scores
    return scores


G = nx.DiGraph()
G.add_edges_from([("Bill", "Mark"), ("Bill", "Larry E"), ("Bill", "Sergey"),
    ("Mark", "Larry E"), ("Mark", "Larry P"),
    ("Larry E", "Larry P"),
    ("Sergey", "Larry P"),
    ("Larry P", "Sergey"), ("Larry P", "Jeff"),
    ("Jeff", "Bill")
    ])

print("Degree Centrality")
for vertex in G.nodes():
    print(vertex, degree_centrality(vertex, G))
print()
    
print("Closeness Centrality")
for vertex in G.nodes():
    print(vertex, closeness_centrality(vertex, G))
print()
    
print("Betweenness Centrality")
for vertex, score in betweenness_centrality(G).items():
    print(vertex, score)
print()

print("PageRank")
for vertex, score in pagerank(G).items():
    print(vertex, score)
print()