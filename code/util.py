from collections import defaultdict
import numpy as np
from scipy.stats import entropy
import math


def count_clusters(clustering):
    if isinstance(clustering[0], int):
        clusters = set()
        for c in clustering:
            clusters.add(c)
        return len(clusters)
    else:
        assert isinstance(clustering[0], list)
        sum_size = 0.0
        for cl in clustering:
            clusters = set()
            for c in cl:
                clusters.add(c)
            sum_size += len(clusters)
        return sum_size / len(clustering)

def size_clusters(clustering):
    if isinstance(clustering[0], int):
        clusters = defaultdict(int)
        for c in clustering:
            clusters[c] += 1
        sum_size = 0.0
        for _,v in clusters.items():
            sum_size += v
        return sum_size/len(clusters.keys())
    else:
        # clustering is a list of clusterings
        assert isinstance(clustering[0], list)
        sum_avg_sizes = 0.0
        for clust in clustering:
            clusters = defaultdict(int)
            for c in clust:
                clusters[c] += 1
            sum_size = 0.0
            for _,v in clusters.items():
                sum_size += v
            sum_avg_sizes +=  (sum_size/len(clusters.keys()))
        return sum_avg_sizes/len(clustering)

def max_size_cluster(clustering):
    if isinstance(clustering[0], int):
        clusters = defaultdict(int)
        for c in clustering:
            clusters[c] += 1
        values = clusters.values()
        return max(values)
    else:
        # clustering is a list of clusterings
        assert isinstance(clustering[0], list)
        sum_max_size = 0.0
        for clust in clustering:
            clusters = defaultdict(int)
            for c in clust:
                clusters[c] += 1
            values = clusters.values()
            sum_max_size += max(values)
        return sum_max_size/len(clustering)

def compute_variability(objects, numerical_attributes, categorical_attributes):
    var_dict = {}
    for attribute in numerical_attributes:
        variation_coeff = np.std(objects[attribute].values)/abs(np.mean(objects[attribute].values))
        # clip variation coefficent to 1 in order to compare to normalized entropy
        if variation_coeff >= 1.0:
            variation_coeff = 1.0
        var_dict[attribute] = variation_coeff
    for attribute in categorical_attributes:
        entropy_value = compute_entropy(objects[attribute].values)
        var_dict[attribute] = entropy_value
    return var_dict

def compute_entropy(labels, base=None):
    # use base e logaritm 
    value, counts = np.unique(labels, return_counts=True)
    return entropy(counts, base=base)/math.log(len(value))