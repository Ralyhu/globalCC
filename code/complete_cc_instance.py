import numpy as np
class CompleteCCInstance:
    def __init__(self, W_plus, W_minus):
        self.W_plus = W_plus
        self.W_minus = W_minus
        self.n_objects = W_plus.shape[0]

    def get_n_objects(self):
        return self.n_objects

    def get_W_plus(self):
        return self.W_plus

    def get_W_minus(self):
        return self.W_minus

    def set_W_plus(self, W_plus):
        self.W_plus = W_plus

    def set_W_minus(self, W_minus):
        self.W_minus = W_minus

    def get_upper_bound(self):
        max_W = np.maximum(self.W_plus, self.W_minus)
        np.fill_diagonal(max_W, 0.0)
        return max_W.sum()/2

    def compute_global_bound(self):
        n_pairs = self.get_n_objects() * (self.get_n_objects() - 1)/2
        sum_plus = (self.get_W_plus().sum() - np.diag(self.get_W_plus()).sum())/2
        avg_plus = sum_plus / n_pairs
        sum_minus = (self.get_W_minus().sum() - np.diag(self.get_W_minus()).sum())/2
        avg_minus = sum_minus / n_pairs
        tmp = self.get_W_plus() - self.get_W_minus()
        # fill diagonal with 0 thus avoiding considering them while computing max diff
        np.fill_diagonal(tmp, 0.0)
        count = np.count_nonzero(tmp > 0.0)/2
        diff = np.abs(tmp)
        avg_diff = diff.sum()/(2 * n_pairs)
        perc = count/(n_pairs)
        max_diff = np.max(diff)
        # indices where the max_diff is realized
        max_diff_ind = np.unravel_index(np.argmax(diff, axis=None), diff.shape)  # returns a tuple
        condition_satisfied = avg_plus + avg_minus >= max_diff
        return condition_satisfied, max_diff_ind, avg_plus, avg_minus, max_diff, avg_diff, perc

    def evaluate_clustering(self, clustering):
        if isinstance(clustering[0], int):
            v = np.array(clustering)
            same_cluster_mask = v[:, np.newaxis] == v[np.newaxis, :]
            different_cluster_mask = ~same_cluster_mask
            min_cc_objective = same_cluster_mask * self.W_minus + different_cluster_mask * self.W_plus
            np.fill_diagonal(min_cc_objective, 0.0)
            value = min_cc_objective.sum()/2
            #print("VALUE = " + str(value))
            return value
        else:
            assert isinstance(clustering[0], list)
            sum_loss = 0.0
            for cl in clustering:
                v = np.array(cl)
                same_cluster_mask = v[:, np.newaxis] == v[np.newaxis, :]
                different_cluster_mask = ~same_cluster_mask
                min_cc_objective = same_cluster_mask * self.W_minus + different_cluster_mask * self.W_plus
                np.fill_diagonal(min_cc_objective, 0.0)
                value = min_cc_objective.sum()/2
                sum_loss += value
                #print("VALUE = " + str(value))
            return sum_loss/len(clustering)

        
    