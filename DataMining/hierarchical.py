import random, math
from KMeans import distance


def construct_hierarchy(observations, aggregation_ftn=min):
    clusters = [(observation,) 
                for observation in observations]  # a leaf cluster for each observation
    while len(clusters) > 1:  # until one cluster is left
        c1, c2 = min([(c1, c2)  # two closest clusters
                     for i, c1 in enumerate(clusters)
                     for c2 in clusters[:i]],
                     key=lambda p: cluster_distance(p[0], p[1], aggregation_ftn))
        clusters = [c for c in clusters if c != c1 and c != c2]  # remove the clusters
        merged_cluster = (len(clusters), [c1, c2])  # merge the clusters
        clusters.append(merged_cluster)  # register the resulting cluster
    return clusters[0]


def get_clusters(hierarchy, num_clusters):
    clusters = [hierarchy]
    while len(clusters) < num_clusters:
        next_cluster = min(clusters, key=get_merge_order)
        clusters = [c for c in clusters if c != next_cluster]
        clusters.extend(next_cluster[1])
    return clusters


def get_merge_order(cluster):
    if len(cluster) == 1:  # leaf node
        return float('inf')
    else:
        return cluster[0]


def get_members(cluster):
    if len(cluster) == 1:  # leaf node
        return cluster
    else:
        return [value
                for child in cluster[1]
                for value in get_members(child)]


def cluster_distance(c1, c2, aggregation_ftn=min):
    return aggregation_ftn([distance(m1, m2)
                        for m1 in get_members(c1)
                        for m2 in get_members(c2)])


random.seed(0) 
observations = []
for i in range(3):
    cx = random.randint(10, 20)
    cy = random.randint(10, 20)
    for __ in range(10):
        observations.append([cx + math.floor(10 * random.random()) / 10,
                             cy + math.floor(10 * random.random()) / 10])

hierarchy = construct_hierarchy(observations)
print(hierarchy)

print()
print("three clusters (min-based)")
for cluster in get_clusters(hierarchy, 3):
    print(get_members(cluster))

print()
print("three clusters (max-based)")
hierarchy = construct_hierarchy(observations, max)
for cluster in get_clusters(hierarchy, 3):
    print(get_members(cluster))
