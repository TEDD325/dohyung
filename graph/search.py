from collections import deque


def shortest_paths(src, G):
    # a dictionary containing (each reachable vertex, a shortest path from src)
    reachable = { src : [src] }
    queue = deque((src, edge[1])  # a queue of (visited vertex, neighbor)
                     for edge in G.out_edges(src))
    while queue:  # until the queue is empty
        prev, v = queue.popleft()  # remove the top entry
        path_to_prev = reachable[prev]  # path to the previous vertex
        reachable[v] = path_to_prev + [v]  # paths to v via prev
        queue.extend((v, edge[1])  # enqueue neighbors of v
            for edge in G.out_edges(v)
            if edge[1] not in reachable)
    return reachable


def all_shortest_paths(src, G):
    # a dictionary containing (each reachable vertex, all shortest paths from src)
    reachable = { src : [[]] }
    queue = deque((src, edge[1])  # a queue of (visited vertex, neighbor)
                     for edge in G.out_edges(src))
    while queue:  # until the queue is empty
        prev, v = queue.popleft()  # remove the top entry
        paths_to_prev = reachable[prev]  # path to the previous vertex
        paths_via_prev = [path + [v] for path in paths_to_prev]  # paths to v via prev
        old_paths = reachable.get(v, [])  # other possible paths to v
        if old_paths:  # set the shortest distance
            min_path_length = len(old_paths[0])
        else:
            min_path_length = float('inf')
        new_paths = [path_via_prev
             for path_via_prev in paths_via_prev
             if len(path_via_prev) <= min_path_length
             and path_via_prev not in old_paths]
        reachable[v] = old_paths + new_paths  # all paths from src to v
        queue.extend((v, edge[1])  # enqueue neighbors of v
            for edge in G.out_edges(v)
            if edge[1] not in reachable)
    return reachable
