import random, math
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np


class KMeans:

    def __init__(self, observations, k, iterations=10):
        self.k = k  # number of clusters
        self.means = random.sample(observations, k)
        clusters = None
        for __ in range(iterations):
            new_clusters = list(map(self.classify, observations))
            if clusters == new_clusters:  # if no change in assignment
                return
            clusters = new_clusters
            for c in range(k):  # for each cluster c
                observations_in_c = [p for p, a in zip(observations, clusters) 
                                    if a == c]
                if observations_in_c:  # if observations in cluster c
                    self.means[c] = np.mean(observations_in_c, 0).tolist()

    def classify(self, observation):
        return min(range(self.k),
                   key=lambda i: distance(observation, self.means[i]))


def distance(o1, o2):
    return np.linalg.norm(np.array(o1) - np.array(o2))

    
def clustering_error(observations, k):
    clusterer = KMeans(observations, k)
    means = clusterer.means
    clusters = list(map(clusterer.classify, observations))
    return sum(distance(observation, means[cluster])
               for observation, cluster in zip(observations, clusters))


def plot_clustering_error(observations):
    ks = range(1, len(observations) + 1)
    errors = [clustering_error(observations, k) for k in ks]
    plt.plot(ks, errors)
    plt.xticks(ks)
    plt.xlabel("k")
    plt.ylabel("error")
    plt.show()

if __name__ == "__main__":
    cluster_num  = 3
    random.seed(0) 
    observations = []
    for i in range(cluster_num):
        cx = random.randint(10, 20)
        cy = random.randint(10, 20)
        for __ in range(10):
            observations.append([cx + math.floor(10 * random.random()) / 10,
                           cy + math.floor(10 * random.random()) / 10])
    plt.scatter([x for x, __ in observations], [y for __, y in observations])
    plt.show()
    
    clusterer = KMeans(observations, cluster_num)
    print("3-means:", clusterer.means)
    
    colors = ["red", "green", "blue"]
    if len(colors) != cluster_num:
        print("colors variable's length is less than cluster_num variable. [cluster_num: ", cluster_num, "]")
    plt.scatter([x for x, __ in observations], [y for __, y in observations],
                color=[colors[clusterer.classify(observation)] 
                       for observation in observations])
    plt.show()
    
    print()
    
    random.seed(0)
    clusterer = KMeans(observations, cluster_num)
    plt.scatter([x for x, __ in observations], [y for __, y in observations],
                color=[colors[clusterer.classify(observation)] 
                       for observation in observations])
    plt.show()
    
    print("2-means:", clusterer.means)
    
    print()
    
    for k in range(1, len(observations) + 1):
        print(k, clustering_error(observations, k))
    print()
    
    k = range(1, len(observations) + 1)
    plt.plot(k, [clustering_error(observations, k_i) for k_i in k])
    plt.xlabel("k")
    plt.ylabel("Error")
    plt.show()
    
    img = mpimg.imread("butterfly.png")
    pixels = [pixel for row in img for pixel in row]
    clusterer = KMeans(pixels, 5)
    new_img = [[clusterer.means[clusterer.classify(pixel)] for pixel in row]
               for row in img]
    plt.imshow(new_img)
    plt.show()