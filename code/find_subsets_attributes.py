from sklearn.metrics import pairwise_distances
from sklearn.metrics import silhouette_score
from statistics import mean 
from collections import defaultdict
from scipy.spatial.distance import euclidean
from scipy.stats import wasserstein_distance
from collections import Counter
from collections import defaultdict
import gc
import numpy as np
import pandas as pd
import math
import random
import operator
import time

from pivot import pivot
from complete_cc_instance import CompleteCCInstance
from util import count_clusters
from util import size_clusters
from util import max_size_cluster

def heuristic_greedy_find_subsets(objects, cur_not_sensitive_subset, cur_sensitive_subset, var_dict, output_path, reverse_priority=True, balance_sets = True, numeric_metric="euclidean", categorical_metric="jaccard", n_iter=25, balance_condition=False, greedy=False):
    categorical_attributes = get_categorical_attributes(objects)
    # get the full set of sensitive and not sensitive attributes before working with these sets
    sensitive_attributes = set(cur_sensitive_subset)
    not_sensitive_attributes = set(cur_not_sensitive_subset)
    # build rankings according to variability 
    var_dict_sensitive = {}
    for attribute in sensitive_attributes:
        var_dict_sensitive[attribute] = var_dict[attribute]
    var_dict_not_sensitive = {}
    for attribute in not_sensitive_attributes:
        var_dict_not_sensitive[attribute] = var_dict[attribute]
    ranking_sensitive = sorted(var_dict_sensitive.items(), key=operator.itemgetter(1), reverse=reverse_priority)
    ranking_not_sensitive = sorted(var_dict_not_sensitive.items(), key=operator.itemgetter(1), reverse=reverse_priority)
    ranking_global = sorted(var_dict.items(), key=operator.itemgetter(1), reverse=reverse_priority) 
    #print(ranking_sensitive)
    #print(ranking_not_sensitive)
    #print(ranking_global)

    sequence_removed = []
    sequence_avg_plus = []
    sequence_avg_minus = []
    sequence_gap = []
    sequence_diff = []
    sequence_clustering = []
    sequence_size_clustering = []
    sequence_size_clusters = []
    sequence_max_size_cluster = []
    sequence_cost = []
    sequence_ubratio = []
    # perc is the % of pairs s.t. w+>w-
    sequence_perc = []

    ex_time = 0.0

    start = time.time()
    smoothing_plus, smoothing_minus = get_smoothing_factors(cur_not_sensitive_subset, cur_sensitive_subset)
    # build matrix W_plus
    W_plus = smoothing_plus * compute_weights(objects, cur_not_sensitive_subset, numeric_metric, categorical_metric, categorical_attributes)
    # build matrix W_minus
    W_minus = smoothing_minus * compute_weights(objects, cur_sensitive_subset, numeric_metric, categorical_metric, categorical_attributes)
    cc_instance = CompleteCCInstance(W_plus, W_minus)
    iteration = 1
    approx_condition_satisfied, _, avg_plus, avg_minus, gap, avg_diff, perc = cc_instance.compute_global_bound()

    ex_time += time.time() - start

    clustering = compute_average_clustering(cc_instance, n_runs=n_iter)
    cost = cc_instance.evaluate_clustering(clustering)
    sequence_clustering.append(clustering)
    sequence_size_clustering.append(count_clusters(clustering))
    sequence_size_clusters.append(size_clusters(clustering))
    sequence_max_size_cluster.append(max_size_cluster(clustering))
    sequence_cost.append(cost)
    sequence_ubratio.append(cost/cc_instance.get_upper_bound())
    sequence_avg_plus.append(avg_plus)
    sequence_avg_minus.append(avg_minus)
    sequence_gap.append(gap)
    sequence_diff.append(avg_diff)
    sequence_perc.append(perc)

    while not approx_condition_satisfied:

        start = time.time()

        if len(cur_not_sensitive_subset)==1 and len(cur_sensitive_subset)==1:
            print("Both sets have size 1 -- no solution found")
            print(sequence_removed)
            break
        print("Iteration # = " + str(iteration))

        if greedy:
            attribute_removed = remove_attribute_greedy_direct(objects, cur_not_sensitive_subset, cur_sensitive_subset, numeric_metric, categorical_metric, categorical_attributes)
        else:
            attribute_removed = remove_attribute_greedy(cur_not_sensitive_subset, cur_sensitive_subset, ranking_not_sensitive, ranking_sensitive, ranking_global, balance_sets, balance_condition, avg_plus, avg_minus, reverse_priority)
        sequence_removed.append(attribute_removed)

        smoothing_plus, smoothing_minus = get_smoothing_factors(cur_not_sensitive_subset, cur_sensitive_subset)

        # compute new instance weights w.r.t. new current subsets
        cc_instance.set_W_plus(smoothing_plus * compute_weights(objects, cur_not_sensitive_subset, numeric_metric, categorical_metric, categorical_attributes))
        cc_instance.set_W_minus(smoothing_minus * compute_weights(objects, cur_sensitive_subset, numeric_metric, categorical_metric, categorical_attributes))
        approx_condition_satisfied, _, avg_plus, avg_minus, gap, avg_diff, perc = cc_instance.compute_global_bound()
        iteration += 1

        ex_time += time.time() - start
        
        clustering = compute_average_clustering(cc_instance, n_runs=n_iter)
        cost = cc_instance.evaluate_clustering(clustering)
        sequence_clustering.append(clustering)
        sequence_size_clustering.append(count_clusters(clustering))
        sequence_size_clusters.append(size_clusters(clustering))
        sequence_max_size_cluster.append(max_size_cluster(clustering))
        sequence_cost.append(cost)
        sequence_ubratio.append(cost/cc_instance.get_upper_bound())
        
        sequence_avg_plus.append(avg_plus)
        sequence_avg_minus.append(avg_minus)
        sequence_gap.append(gap)
        sequence_diff.append(avg_diff)
        sequence_perc.append(perc)
        
        #print("New NOT sensitive subset " + str(cur_not_sensitive_subset))
        #print("New sensitive subset " + str(cur_sensitive_subset))

    print("########### PRINTING SEQUENCE RESULTS ###########")

    print("Sequence removed")
    print(sequence_removed)
    sequence_removed_file = output_path + "/sequence_removed.txt"
    with open(sequence_removed_file, "w+") as f:
        for att in sequence_removed:
            f.write(att + "\n")
    
    sequence_error = [(z-x-y) for x, y, z in zip(sequence_avg_plus, sequence_avg_minus, sequence_gap)]
    print("ERROR CONSTRAINT")
    print(sequence_error)
    sequence_error_file = output_path + "/sequence_error.txt"
    with open(sequence_error_file, "w+") as f:
        for x in sequence_error:
            f.write(str(x) + "\n")

    print("COST")
    print(sequence_cost)
    sequence_cost_file = output_path + "/sequence_cost.txt"
    with open(sequence_cost_file, "w+") as f:
        for cost in sequence_cost:
            f.write(str(cost) + "\n")
    
    print("COST/UpperBound")
    print(sequence_ubratio)
    sequence_ubratio_file = output_path + "/sequence_ubratio.txt"
    with open(sequence_ubratio_file, "w+") as f:
        for cost in sequence_ubratio:
            f.write(str(cost) + "\n")
    
    print("AVG(W+)")
    print(sequence_avg_plus)
    sequence_avgplus_file = output_path + "/sequence_avgplus.txt"
    with open(sequence_avgplus_file, "w+") as f:
        for avgplus in sequence_avg_plus:
            f.write(str(avgplus) + "\n")

    print("AVG(W-)")
    print(sequence_avg_minus)
    sequence_avgminus_file = output_path + "/sequence_avgminus.txt"
    with open(sequence_avgminus_file, "w+") as f:
        for avgminus in sequence_avg_minus:
            f.write(str(avgminus) + "\n")

    sequence_sum_avf = [(x+y) for x, y in zip(sequence_avg_plus, sequence_avg_minus)]
    print("avg(w+) + avg(w-)")
    print(sequence_sum_avf)
    sequence_sumavg_file = output_path + "/sequence_sumavg.txt"
    with open(sequence_sumavg_file, "w+") as f:
        for sumavg in sequence_sum_avf:
            f.write(str(sumavg) + "\n")


    print("AVG. DIFF |w+-w-|")
    print(sequence_diff)
    sequence_diffavg_file = output_path + "/sequence_diffavg.txt"
    with open(sequence_diffavg_file, "w+") as f:
        for diffavg in sequence_diff:
            f.write(str(diffavg) + "\n")

    print("GAPS")
    print(sequence_gap)
    sequence_gap_file = output_path + "/sequence_gap.txt"
    with open(sequence_gap_file, "w+") as f:
        for gap in sequence_gap:
            f.write(str(gap) + "\n")

    del W_plus
    del W_minus
    del cc_instance
    gc.collect()

    print("Computing fairness scores... ")
    sequence_avgE, sequence_avgW, sequence_maxE, sequence_maxW = compute_fairness_sequences(objects, sensitive_attributes, sequence_clustering)

    print("AVG Euclidean fairness")
    print(sequence_avgE)
    sequence_avgE_file = output_path + "/sequence_fairness_avgE.txt"
    with open(sequence_avgE_file, "w+") as f:
        for fair in sequence_avgE:
            f.write(str(fair) + "\n")

    print("AVG Wasserstein fairness")
    print(sequence_avgW)
    sequence_avgW_file = output_path + "/sequence_fairness_avgW.txt"
    with open(sequence_avgW_file, "w+") as f:
        for fair in sequence_avgW:
            f.write(str(fair) + "\n")

    print("MAX Euclidean fairness")
    print(sequence_maxE)
    sequence_maxE_file = output_path + "/sequence_fairness_maxE.txt"
    with open(sequence_maxE_file, "w+") as f:
        for fair in sequence_maxE:
            f.write(str(fair) + "\n")

    print("MAX Wasserstein fairness")
    print(sequence_maxW)
    sequence_maxW_file = output_path + "/sequence_fairness_maxW.txt"
    with open(sequence_maxW_file, "w+") as f:
        for fair in sequence_maxW:
            f.write(str(fair) + "\n")

    print("% w+ > w-")
    print(sequence_perc)
    sequence_perc_file = output_path + "/sequence_perc.txt"
    with open(sequence_perc_file, "w+") as f:
        for perc in sequence_perc:
            f.write(str(perc) + "\n")

    print("SIZE CLUSTERING")
    print(sequence_size_clustering)
    sequence_sizecl = output_path + "/sequence_size_clustering.txt"
    with open(sequence_sizecl, "w+") as f:
        for size in sequence_size_clustering:
            f.write(str(size) + "\n")

    print("SIZE CLUSTERS")
    print(sequence_size_clusters)
    sequence_sizecl = output_path + "/sequence_size_clusters.txt"
    with open(sequence_sizecl, "w+") as f:
        for size in sequence_size_clusters:
            f.write(str(size) + "\n")

    print("MAX SIZE CLUSTER")
    print(sequence_max_size_cluster)
    sequence_sizecl = output_path + "/sequence_maxsize_cluster.txt"
    with open(sequence_sizecl, "w+") as f:
        for size in sequence_max_size_cluster:
            f.write(str(size) + "\n")

    print("EXEC TIME")
    print(str(ex_time))
    time_file = output_path + "/time.txt"
    with open(time_file, "w+") as f:
        f.write(str(ex_time) + "\n")

    print("Computing pairwise similarities...")
    sequence_intra_plus, sequence_intra_minus, sequence_inter_plus, sequence_inter_minus = compute_inter_intra_clustersim_sequences(sequence_clustering, objects, not_sensitive_attributes, sensitive_attributes, numeric_metric, categorical_metric, categorical_attributes, output_path)
    
    print("Intra-clust not-sensitive")
    print(sequence_intra_plus)
    print("Intra-clust sensitive")
    print(sequence_intra_minus)
    print("Inter-clust not-sensitive")
    print(sequence_inter_plus)
    print("Inter-clust sensitive")
    print(sequence_inter_minus)
    # write_res(sequence_intra_plus, output_path + "/sequence_intra_plus.txt")
    # write_res(sequence_intra_minus, output_path + "/sequence_intra_minus.txt")
    # write_res(sequence_inter_plus, output_path + "/sequence_inter_plus.txt")
    # write_res(sequence_inter_minus, output_path + "/sequence_inter_minus.txt")

    """ sequence_silhouette_in, sequence_silhouette_out = compute_silhouette_sequences(sequence_clustering, objects, not_sensitive_attributes, sensitive_attributes, numeric_metric, categorical_metric, categorical_attributes)
    print("SILHOUETTE IN")
    print(sequence_silhouette_in)
    sequence_silhouettein_file = output_path + "/sequence_silhouettein.txt"
    with open(sequence_silhouettein_file, "w+") as f:
        for sil in sequence_silhouette_in:
            f.write(str(sil) + "\n")

    print("SILHOUETTE OUT")
    print(sequence_silhouette_out)
    sequence_silhouetteout_file = output_path + "/sequence_silhouetteout.txt"
    with open(sequence_silhouetteout_file, "w+") as f:
        for sil in sequence_silhouette_out:
            f.write(str(sil) + "\n")
    
    sequence_silhouette_inout = [x - y for x,y in zip(sequence_silhouette_in, sequence_silhouette_out)]
    print("SILHOUETTE IN/OUT")
    print(sequence_silhouette_inout)
    sequence_silhouetteinout_file = output_path + "/sequence_silhouettein+out.txt"
    with open(sequence_silhouetteinout_file, "w+") as f:
        for sil in sequence_silhouette_inout:
            f.write(str(sil) + "\n") """

    sequence_cc = compute_cc_initial_weights(sequence_clustering, objects, not_sensitive_attributes, sensitive_attributes, numeric_metric, categorical_metric, categorical_attributes)
    print("CC_OBJECTIVES (w.r.t. initial weights)")
    print(sequence_cc)  
    sequence_cc_file = output_path + "/sequence_cc.txt"
    with open(sequence_cc_file, "w+") as f:
        for cc in sequence_cc:
            f.write(str(cc) + "\n")

def write_res(sequence, output_file):
    with open(output_file, "w+") as f:
        for v in sequence:
            f.write(str(v) + "\n")

def remove_attribute_greedy_direct(objects, cur_not_sensitive_subset, cur_sensitive_subset, numeric_metric, categorical_metric, categorical_attributes):
    #cur_not_sensitive_subset = set(cur_not_sensitive_subset)
    #cur_sensitive_subset = set(cur_sensitive_subset)
    
    # compute the smoothing factors regardless the subsets, they depends only on the size of the sets -- remove one attribute
    local_variation_sensitive_subset = cur_sensitive_subset.difference(set([list(cur_sensitive_subset)[0]]))
    smoothing_plus, smoothing_minus = get_smoothing_factors(cur_not_sensitive_subset, local_variation_sensitive_subset)

    # compute positive weights once and iterate over removals of sensitive attributes
    W_plus = smoothing_plus * compute_weights(objects, cur_not_sensitive_subset, numeric_metric, categorical_metric, categorical_attributes)
    
    from_sensitive = False
    best_score = float("inf")
    best_attribute = None
    for sensitive_attribute in cur_sensitive_subset: 
        #print(sensitive_attribute)
        local_variation_sensitive_subset = cur_sensitive_subset.difference(set([sensitive_attribute]))
        #print(local_variation_sensitive_subset)
        W_minus = smoothing_minus * compute_weights(objects, local_variation_sensitive_subset, numeric_metric, categorical_metric, categorical_attributes)
        cc_instance = CompleteCCInstance(W_plus, W_minus)
        _, _, avg_plus, avg_minus, gap, _, _ = cc_instance.compute_global_bound()
        cost = gap - avg_plus - avg_minus
        if cost < best_score:
            best_attribute = sensitive_attribute
            best_score = cost
            from_sensitive = True


    # compute the smoothing factors regardless the subsets, they depends only on the size of the sets -- remove one attribute
    local_variation_not_sensitive_subset = cur_not_sensitive_subset.difference(set([list(cur_not_sensitive_subset)[0]]))
    smoothing_plus, smoothing_minus = get_smoothing_factors(local_variation_not_sensitive_subset, cur_sensitive_subset)

    from_not_sensitive = False
    # compute negative weights once and iterate over removals of not sensitive attributes
    W_minus = smoothing_minus * compute_weights(objects, cur_sensitive_subset, numeric_metric, categorical_metric, categorical_attributes)
    for not_sensitive_attribute in cur_not_sensitive_subset:
        #print(not_sensitive_attribute)
        local_variation_not_sensitive_subset = cur_not_sensitive_subset.difference(set([not_sensitive_attribute]))
        #print(local_variation_not_sensitive_subset)
        W_plus = smoothing_plus * compute_weights(objects, local_variation_not_sensitive_subset, numeric_metric, categorical_metric, categorical_attributes)
        cc_instance = CompleteCCInstance(W_plus, W_minus)
        _, _, avg_plus, avg_minus, gap, _, _ = cc_instance.compute_global_bound()
        cost = gap - avg_plus - avg_minus
        if cost < best_score:
            best_attribute = not_sensitive_attribute
            best_score = cost
            from_not_sensitive = True
            from_sensitive = False
    if from_sensitive:
        #assert best_attribute in cur_sensitive_subset
        cur_sensitive_subset.remove(best_attribute)
    if from_not_sensitive:
        #assert best_attribute in cur_not_sensitive_subset
        cur_not_sensitive_subset.remove(best_attribute)
    return best_attribute


def compute_fairness_sequences(objects, sensitive_attributes, sequence_clustering):
    sequence_avgE = []
    sequence_avgW = []
    sequence_maxE = []
    sequence_maxW = []
    for clusterings in sequence_clustering:
        avgE, avgW, maxE, maxW = compute_fairness(objects, sensitive_attributes, clusterings)
        sequence_avgE.append(avgE)
        sequence_avgW.append(avgW)
        sequence_maxE.append(maxE)
        sequence_maxW.append(maxW)
    return sequence_avgE, sequence_avgW, sequence_maxE, sequence_maxW

def compute_fairness(objects, sensitive_attributes, clusterings):
    # compute for each attribute the distribution of values w.r.t. the whole dataset
    dist_per_attribute = {}
    for attribute in sensitive_attributes:
        counts = objects[attribute].value_counts()
        assert(sum(counts.to_dict().values()) == len(objects))
        dist_values = counts / len(objects)
        dist_values = dist_values.to_dict()
        # ordered_values = [dist_values[key] for key in sorted(dist_values.keys())]
        dist_per_attribute[attribute] = dist_values
    
    sum_fairness_avgE = 0.0
    sum_fairness_avgW = 0.0
    sum_fairness_maxE = 0.0
    sum_fairness_maxW = 0.0
    for attribute in sensitive_attributes:
        fairness_avgE, fairness_avgW, fairness_maxE, fairness_maxW = compute_fairness_for_attribute(objects, attribute, dist_per_attribute[attribute], clusterings)
        sum_fairness_avgE += fairness_avgE
        sum_fairness_avgW += fairness_avgW
        sum_fairness_maxE += fairness_maxE
        sum_fairness_maxW += fairness_maxW
    n_attributes = len(sensitive_attributes)
    return sum_fairness_avgE/n_attributes, sum_fairness_avgW/n_attributes, sum_fairness_maxE/n_attributes, sum_fairness_maxW/n_attributes

def compute_fairness_for_attribute(objects, attribute, dist_per_attribute, clusterings):
    sum_fairness_avgE = 0.0
    sum_fairness_avgW = 0.0
    sum_fairness_maxE = 0.0
    sum_fairness_maxW = 0.0
    ordered_values_global = [dist_per_attribute[key] for key in sorted(dist_per_attribute.keys())]
    for clustering in clusterings:
        # dicts where each element refers to a cluster
        euclidean_distances = {}
        wasserstein_distances = {}
        # obtain id objects for each cluster
        clusters = defaultdict(set)
        for idx, cluster in enumerate(clustering):
            clusters[cluster].add(idx)
        
        for cluster, ids in clusters.items():
            counts = objects.iloc[list(ids)][attribute].value_counts()
            assert(sum(counts.to_dict().values()) == len(ids))
            dist_values = counts / len(ids)
            dist_values = dist_values.to_dict()

            # some values of the attribute may be missing in the cluster -- add fake entries in order to make the same size
            missing_attributes_values = set(dist_per_attribute.keys()).difference(dist_values.keys())
            for missing in missing_attributes_values:
                dist_values[missing] = 0.0

            # distribution of values of attribute in the current cluster
            ordered_values = [dist_values[key] for key in sorted(dist_values.keys())]
            
            euclidean_distances[cluster] = euclidean(ordered_values, ordered_values_global)
            wasserstein_distances[cluster] = wasserstein_distance(ordered_values, ordered_values_global) 
        # normalize by the size of the clusters before adding to initial variables
        sumE = 0.0
        sumW = 0.0
        sumsizes = 0
        for cluster, v in euclidean_distances.items():
            size_c = len(clusters[cluster])
            sumsizes += size_c
            sumE += size_c * v
        for cluster, v in wasserstein_distances.items():
            sumW += len(clusters[cluster]) * v
        sum_fairness_avgE += sumE / sumsizes
        sum_fairness_avgW += sumW / sumsizes
        sum_fairness_maxE += max(euclidean_distances.values())
        sum_fairness_maxW += max(wasserstein_distances.values())
    n_clusterings = len(clusterings)
    return sum_fairness_avgE/n_clusterings, sum_fairness_avgW/n_clusterings, sum_fairness_maxE/n_clusterings, sum_fairness_maxW/n_clusterings

def compute_inter_intra_clustersim_sequences(sequence_clustering, objects, not_sensitive_attributes, sensitive_attributes, numeric_metric, categorical_metric, categorical_attributes, output_path):
    cur_not_sensitive_subset = set(not_sensitive_attributes)
    cur_sensitive_subset = set(sensitive_attributes)
 
    #print(cur_not_sensitive_subset)
    #print(cur_sensitive_subset)

    smoothing_plus, smoothing_minus = get_smoothing_factors(cur_not_sensitive_subset, cur_sensitive_subset)

    # build matrix W_plus
    W_plus = smoothing_plus * compute_weights(objects, cur_not_sensitive_subset, numeric_metric, categorical_metric, categorical_attributes, return_distances=False)
    
    #np.fill_diagonal(W_plus, 1.0)
    # it is required to set diagonal to zero because singletons contributions to intra cluster are considered separately
    np.fill_diagonal(W_plus, 0.0)
    # build matrix W_minus
    W_minus = smoothing_minus * compute_weights(objects, cur_sensitive_subset, numeric_metric, categorical_metric, categorical_attributes, return_distances=False)
    
    #np.fill_diagonal(W_minus, 1.0)
    np.fill_diagonal(W_minus, 0.0)

    #cc_instance = CompleteCCInstance(W_plus, W_minus)

    #assert np.min(W_plus) >= 0 and np.max(W_plus) <= 1.0
    #assert np.min(W_minus) >= 0 and np.max(W_minus) <= 1.0

    sequence_inter_plus = []
    sequence_inter_minus = []
    sequence_intra_plus = []
    sequence_intra_minus = []
    for clustering in sequence_clustering:
        sum_inter_plus = 0.0
        sum_inter_minus = 0.0 
        for cl in clustering:
            n = len(cl)
            if not all(i==0 for i in cl):
                v = np.array(cl)
                same_cluster_mask = v[:, np.newaxis] == v[np.newaxis, :]
                pairs_same = (same_cluster_mask.sum() - n)/2
                different_cluster_mask = ~same_cluster_mask
                pairs_different = n * (n - 1)/2 - pairs_same

                inter_plus_m = different_cluster_mask * W_plus
                np.fill_diagonal(inter_plus_m, 0.0)
                sum_inter_plus += inter_plus_m.sum()/(2 * pairs_different)

                inter_minus_m = different_cluster_mask * W_minus
                np.fill_diagonal(inter_minus_m, 0.0)
                sum_inter_minus += inter_minus_m.sum()/(2 * pairs_different)

            # if single cluster --> default value of inter cluster similarity of 0.0
        sequence_inter_plus.append(sum_inter_plus/len(clustering))
        sequence_inter_minus.append(sum_inter_minus/len(clustering))

    del inter_plus_m
    del inter_minus_m
    gc.collect()

    for clustering in sequence_clustering:
        sum_intra_plus = 0.0
        sum_intra_minus = 0.0
        for cl in clustering:
            n = len(cl)
            if not all(i==0 for i in cl):
                v = np.array(cl)
                same_cluster_mask = v[:, np.newaxis] == v[np.newaxis, :]
                pairs_same = (same_cluster_mask.sum() - n)/2

                intra_plus_m = same_cluster_mask * W_plus
                np.fill_diagonal(intra_plus_m, 2.0)
                sum_intra_plus += intra_plus_m.sum()/(2 * pairs_same)

                intra_minus_m = same_cluster_mask * W_minus
                np.fill_diagonal(intra_minus_m, 2.0)
                sum_intra_minus += intra_minus_m.sum()/(2 * pairs_same)

            # if single cluster --> default value of inter cluster similarity of 0.0
        sequence_intra_plus.append(sum_intra_plus/len(clustering))
        sequence_intra_minus.append(sum_intra_minus/len(clustering))

    del intra_plus_m
    del intra_minus_m
    gc.collect()
    # print to files
    write_res(sequence_intra_plus, output_path + "/sequence_intra_plus_sub.txt")
    write_res(sequence_intra_minus, output_path + "/sequence_intra_minus_sub.txt")
    write_res(sequence_inter_plus, output_path + "/sequence_inter_plus.txt")
    write_res(sequence_inter_minus, output_path + "/sequence_inter_minus.txt")
    sequence_intra_plus = []
    sequence_intra_minus = []

    W_plus = W_plus.tolist()
    W_minus = W_minus.tolist()
    for clustering in sequence_clustering:
        sum_intra_plus = 0.0
        sum_intra_minus = 0.0
        for cl in clustering:
            n = len(cl)

            sizes = Counter(cl)
            singletons = sum(value == 1 for value in sizes.values())
            v = np.array(cl)
            same_cluster_mask = v[:, np.newaxis] == v[np.newaxis, :]
            cs = defaultdict(set)
            for i, v in enumerate(cl):
                cs[v].add(i)

            """  data = []
            rows = []
            cols = []
            for id_community, com in cs.items():
                size = sizes[id_community]
                if size > 1:
                    norm = 1/(size*(size-1)/2)
                else:
                    norm = 1
                data += [norm] * (size * size)
                rows += [node_i for node_i in com for _ in com]
                cols += [node_j for _ in com for node_j in com]

            data = np.array(data, copy=False)
            rows = np.array(rows, copy=False)
            cols = np.array(cols, copy=False)
            norm_M = coo_matrix((data, (rows, cols)), shape=(n, n), copy=False)
            norm_M = norm_M.tocsr(copy=False).multiply(same_cluster_mask)
            
            agg = norm_M.multiply(W_plus)
            sum_intra_plus += ((agg.sum()/2) + singletons)/len(cs.keys())

            agg = norm_M.multiply(W_minus)
            sum_intra_minus += ((agg.sum()/2) + singletons)/len(cs.keys())  """

            res_plus = 0.0
            res_minus = 0.0
            for i, v in cs.items():
                size = sizes[i]
                if size == 1:
                    res_plus += 1
                    res_minus += 1
                else:
                    cl_res_plus = 0.0
                    cl_res_minus = 0.0
                    for n1 in v:
                        for n2 in v:
                            if n1 < n2:
                                cl_res_plus += W_plus[n1][n2]
                                cl_res_minus += W_minus[n1][n2]
                    res_plus += cl_res_plus/(size * (size - 1)/2)
                    res_minus += cl_res_minus/(size * (size - 1)/2)
            res_plus /= len(cs.keys())
            res_minus /= len(cs.keys())
            sum_intra_plus += res_plus
            sum_intra_minus += res_minus

        sequence_intra_plus.append(sum_intra_plus/len(clustering))
        sequence_intra_minus.append(sum_intra_minus/len(clustering))

    write_res(sequence_intra_plus, output_path + "/sequence_intra_plus.txt")
    write_res(sequence_intra_minus, output_path + "/sequence_intra_minus.txt")
        
    return sequence_intra_plus, sequence_intra_minus, sequence_inter_plus, sequence_inter_minus

def compute_cc_initial_weights(sequence_clustering, objects, not_sensitive_attributes, sensitive_attributes, numeric_metric, categorical_metric, categorical_attributes):
    cur_not_sensitive_subset = set(not_sensitive_attributes)
    cur_sensitive_subset = set(sensitive_attributes)
 
    #print(cur_not_sensitive_subset)
    #print(cur_sensitive_subset)

    smoothing_plus, smoothing_minus = get_smoothing_factors(cur_not_sensitive_subset, cur_sensitive_subset)

    # build matrix W_plus
    W_plus = smoothing_plus * compute_weights(objects, cur_not_sensitive_subset, numeric_metric, categorical_metric, categorical_attributes, return_distances=False)
    #print(np.diag(W_plus))
    np.fill_diagonal(W_plus, 1.0)
    # build matrix W_minus
    W_minus = smoothing_minus * compute_weights(objects, cur_sensitive_subset, numeric_metric, categorical_metric, categorical_attributes, return_distances=False)
    #print(np.diag(W_minus))
    np.fill_diagonal(W_minus, 1.0)

    cc_instance = CompleteCCInstance(W_plus, W_minus)
    sequence_cc = []

    for clustering in sequence_clustering:
        sequence_cc.append(cc_instance.evaluate_clustering(clustering))
    
    return sequence_cc

def compute_silhouette_sequences(sequence_clustering, objects, not_sensitive_attributes, sensitive_attributes, numeric_metric, categorical_metric, categorical_attributes):
    cur_not_sensitive_subset = set(not_sensitive_attributes)
    cur_sensitive_subset = set(sensitive_attributes)
 
    #print(cur_not_sensitive_subset)
    #print(cur_sensitive_subset)

    smoothing_plus, smoothing_minus = get_smoothing_factors(cur_not_sensitive_subset, cur_sensitive_subset)

    # build matrix W_plus
    W_plus = smoothing_plus * compute_weights(objects, cur_not_sensitive_subset, numeric_metric, categorical_metric, categorical_attributes, return_distances=False)
    #print(np.diag(W_plus))
    np.fill_diagonal(W_plus, 1.0)
    # build matrix W_minus
    W_minus = smoothing_minus * compute_weights(objects, cur_sensitive_subset, numeric_metric, categorical_metric, categorical_attributes, return_distances=False)
    #print(np.diag(W_minus))
    np.fill_diagonal(W_minus, 1.0)
    
    dist_plus = 1 - W_plus
    dist_minus = 1 - W_minus
    sequence_silhouette_in = []
    sequence_silhouette_out = []
    for clustering in sequence_clustering:
        silhouette_in = 0.0
        silhouette_out = 0.0
        for cl in clustering:
            # if single cluster or all singleton clusters silhouette raise an error, so set it to 0 in these cases
            if not (all(i==0 for i in cl) or max(cl) == len(cl)-1) :
                silhouette_in += silhouette_score(X=dist_plus, labels=cl, metric="precomputed")
                silhouette_out += silhouette_score(X=dist_minus, labels=cl, metric="precomputed")
        sequence_silhouette_in.append(silhouette_in/len(clustering))
        sequence_silhouette_out.append(silhouette_out/len(clustering))
      
    return sequence_silhouette_in, sequence_silhouette_out


def compute_average_clustering(cc_instance, n_runs=25):
    clustering_list = []
    for _ in range(n_runs):
        clustering = pivot(cc_instance)
        clustering_list.append(clustering)
    return clustering_list


def remove_attribute_greedy(cur_not_sensitive_subset, cur_sensitive_subset, ranking_not_sensitive, ranking_sensitive, ranking_global, balance_sets, balance_condition, avg_plus, avg_minus, reverse_priority):
    if not is_remove_possible(cur_not_sensitive_subset):
        return remove_attribute_alt(cur_sensitive_subset, ranking_sensitive)
    if not is_remove_possible(cur_sensitive_subset):
        return remove_attribute_alt(cur_not_sensitive_subset, ranking_not_sensitive)
    if balance_condition:

        if avg_plus >= avg_minus:
            if reverse_priority:
                return remove_attribute_alt(cur_sensitive_subset, ranking_sensitive)
            else:
                return remove_attribute_alt(cur_not_sensitive_subset, ranking_not_sensitive)
        else:
            if reverse_priority:
                return remove_attribute_alt(cur_not_sensitive_subset, ranking_not_sensitive)
            else:
                return remove_attribute_alt(cur_sensitive_subset, ranking_sensitive)
    elif balance_sets:

        if len(cur_not_sensitive_subset) > len(cur_sensitive_subset):
            return remove_attribute_alt(cur_not_sensitive_subset, ranking_not_sensitive)
        elif len(cur_not_sensitive_subset) < len(cur_sensitive_subset):
            return remove_attribute_alt(cur_sensitive_subset, ranking_sensitive)
        else:
            choices_list = ["remove_sensitive", "remove_not_sensitive"]
            selected_choice = random.choice(choices_list)
            if selected_choice == "remove_sensitive":
                return remove_attribute_alt(cur_sensitive_subset, ranking_sensitive)
            else:
                return remove_attribute_alt(cur_not_sensitive_subset, ranking_not_sensitive)
    else:
        return remove_attribute_global(cur_not_sensitive_subset, cur_sensitive_subset, ranking_global)

def remove_attribute_alt(cur_subset, ranking_attributes):
    for attribute, _ in ranking_attributes:
        if attribute in cur_subset:
            cur_subset.remove(attribute)
            print("REMOVED attribute = " + str(attribute))
            return attribute

def remove_attribute_global(cur_not_sensitive_subset, cur_sensitive_subset, ranking_global):
    for attribute, _ in ranking_global:
        if attribute in cur_not_sensitive_subset and len(cur_not_sensitive_subset)>1:
            cur_not_sensitive_subset.remove(attribute)
            print("REMOVED attribute = " + str(attribute))
            return attribute
        if attribute in cur_sensitive_subset and len(cur_sensitive_subset)>1:
            cur_sensitive_subset.remove(attribute)
            print("REMOVED attribute = " + str(attribute))
            return attribute

def get_smoothing_factors(cur_not_sensitive_subset, cur_sensitive_subset, mode="exp"):

    if mode == "no":
        # no smooth 
        smoothing_plus = 1.0
        smoothing_minus = 1.0
        return smoothing_plus, smoothing_minus

    # compute smoothing factors
    fraction_not_sensitive = float(len(cur_not_sensitive_subset))/(len(cur_not_sensitive_subset) + len(cur_sensitive_subset))
    fraction_sensitive = 1 - fraction_not_sensitive

    smoothing_plus = fraction_not_sensitive
    smoothing_minus = fraction_sensitive

    if mode == "exp":
        # exponential smooth 
        smoothing_plus = math.exp(fraction_not_sensitive - 1)
        smoothing_minus = math.exp(fraction_sensitive - 1)

    return smoothing_plus, smoothing_minus

def compute_weights(objects, subset_attributes, numeric_metric, categorical_metric, categorical_attributes, return_distances=False):
    subset_numerical = set()
    for attribute in subset_attributes:
        if attribute not in categorical_attributes:
            subset_numerical.add(attribute)
    subset_categorical = subset_attributes.difference(subset_numerical)
    numerical_weight = len(subset_numerical)/(len(subset_numerical) + len(subset_categorical))
    categorical_weight = 1 - numerical_weight
    if categorical_weight == 0:
        return compute_pair_similarities(objects, subset_numerical, numeric_metric, return_distances=return_distances)
    if numerical_weight == 0:
        return compute_pair_similarities(objects, subset_categorical, categorical_metric, return_distances=return_distances)
    return numerical_weight * compute_pair_similarities(objects, subset_numerical, numeric_metric, return_distances=return_distances) + categorical_weight * compute_pair_similarities(objects, subset_categorical, categorical_metric, return_distances=return_distances)

def compute_pair_similarities(objects, subset, metric, return_distances):
    if metric == "jaccard":
        if return_distances:
            return pairwise_distances(encode(objects, subset).to_numpy(), metric=metric, n_jobs=-1)
        else:
            X = 1 - pairwise_distances(encode(objects, subset).to_numpy(), metric=metric, n_jobs=-1)
            #print("jaccard")
            #print(X)
            return X
    if metric == "euclidean":
        if return_distances:
            return 1 - (1/(1 + pairwise_distances(objects[subset], metric=metric, n_jobs=-1)))
        else:
            X = 1/(1 + pairwise_distances(objects[subset], metric=metric, n_jobs=-1))
            #print("euclidean")
            #print(X)
            return X
    if metric == "cosine":
        if return_distances:
            return pairwise_distances(objects[subset], metric=metric, n_jobs=-1)
        else:
            return 1 - pairwise_distances(objects[subset], metric=metric, n_jobs=-1)

def get_categorical_attributes(objects):
    # Get  columns whose data type is object i.e. string
    filteredColumns = objects.dtypes[objects.dtypes == np.object]
    # list of columns whose data type is object i.e. string
    listOfColumnNames = list(filteredColumns.index)
    return listOfColumnNames

def encode(original_dataframe, feature_to_encode):
    dummies = pd.get_dummies(original_dataframe[feature_to_encode], dtype=bool)
    #res = pd.concat([original_dataframe, dummies], axis=1)
    #res = res.drop([feature_to_encode], axis=1)
    # return res
    return dummies 

def remove_attribute(cur_subset, ranking_attributes):
    for attribute, _ in reversed(ranking_attributes):
        if attribute in cur_subset:
            cur_subset.remove(attribute)
            print("REMOVED attribute = " + str(attribute))
            return

def is_remove_possible(current_attribute_set):
    return len(current_attribute_set) > 1