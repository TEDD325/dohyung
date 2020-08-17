import networkx as nx
from collections import defaultdict
from centrality import pagerank


def compute_pagerank(v, msgs, i, G, damping=0.85):
    n = len(G.nodes())
    if i == 0:
        G.node[v]["pagerank"] = 1 / n
    else:
        G.node[v]["pagerank"] = (1 - damping) / n + damping * sum(msgs)
    for edge in G.out_edges(v):  # distribute score to neighbors
        yield (edge[1], G.node[v]["pagerank"] / len(G.out_edges(v)))


def pregel(compute, G, num_iters=30):
    msgs = defaultdict(list)
    for v in G.nodes():
        for n, m in compute(v, {}, 0, G):
            msgs[n].append(m)
    for i in range(1, num_iters):
        new_msgs = defaultdict(list)
        for v, m in msgs.items():
            for n, m in compute(v, m, i, G):
                new_msgs[n].append(m)
        msgs = new_msgs;


G = nx.DiGraph()
G.add_edges_from([("Bill", "Mark"), ("Bill", "Larry E"), ("Bill", "Sergey"),
    ("Mark", "Larry E"), ("Mark", "Larry P"),
    ("Larry E", "Larry P"),
    ("Sergey", "Larry P"),
    ("Larry P", "Sergey"), ("Larry P", "Jeff"),
    ("Jeff", "Bill")
    ])

print("PageRank - Custom")
for vertex, score in pagerank(G).items():
    print(vertex, score)
print()
    
print("PageRank - Pregel")
pregel(compute_pagerank, G)
for vertex in G.nodes():
    print(vertex, G.node[vertex]["pagerank"])