import matplotlib.pyplot as plt
import networkx as nx
from search import shortest_paths

# TODO: please add your code here
G = nx.DiGraph()

vertices = ["Bill", "Mark", "Larry E", "Sergey", "Larry P", "Jeff"]
G.add_nodes_from(vertices)

print("vertices: ", G.nodes())

nx.draw(G, with_labels=True)
plt.show() 

nx.draw(G, node_size=3000, with_labels=True)
plt.show() 

G.add_edges_from([("Bill", "Mark"), ("Bill", "Larry E"), ("Bill", "Sergey"),
                  ("Mark", "Larry E"), ("Mark", "Larry P"),
                  ("Larry E", "Larry P"),
                  ("Sergey", "Larry P"),
                  ("Larry P", "Sergey"), ("Larry P", "Jeff"),
                  ("Jeff", "Bill")
                  ])
print("edges:")
for edge in G.edges():
    print(" ", str(edge))

nx.draw(G, node_size=3000, with_labels=True)
plt.show() 

print("shortest path from Bill to Jeff:", end=" ")
print(nx.astar_path(G, "Bill", "Jeff"))

for item in shortest_paths("Bill", G).items():
    print(item)