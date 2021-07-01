from complete_cc_instance import CompleteCCInstance
import time
import numpy as np
def pivot(cc_instance):
    start = time.time()
    num_objects = cc_instance.get_n_objects()
    permutation = np.arange(num_objects)
    marked_objects = [False] * num_objects
    cluster_membership = [-1] * num_objects
    # generate random permutation of objects (Fisher-Yates algorithm O(n))
    np.random.shuffle(permutation)
    permutation = permutation.tolist()
    cluster_id = 0
    W_plus = cc_instance.get_W_plus()
    W_minus = cc_instance.get_W_minus()
    for cur_object in permutation:
        if not marked_objects[cur_object]:
            cluster_membership[cur_object] = cluster_id
            marked_objects[cur_object] = True
            # get neighbors of current object
            w_cur_plus = W_plus[cur_object,:].tolist()
            w_cur_minus = W_minus[cur_object,:].tolist()
            for neigh, (w_neigh_plus, w_neigh_minus) in enumerate(zip(w_cur_plus, w_cur_minus)):
                if not marked_objects[neigh] and w_neigh_plus >= w_neigh_minus:
                    cluster_membership[neigh] = cluster_id
                    marked_objects[neigh] = True
            cluster_id += 1
    return cluster_membership



