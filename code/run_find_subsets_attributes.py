import argparse as args
import os
import sys
import traceback
import random
import numpy as np
import pandas as pd

import constants
from imputer import DataFrameImputer
from find_subsets_attributes import heuristic_greedy_find_subsets
from find_subsets_attributes import get_categorical_attributes
from util import compute_entropy
from util import compute_variability

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), os.pardir)))

basepath = os.path.dirname(os.path.abspath(__file__)) + "/../"
basePath_data = basepath + "data_fairness/"

def str2bool(v):
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false','f','n','0'):
            return False
        else:
            raise args.ArgumentTypeError('Boolean value expected.')

def create_parser():
    parser = args.ArgumentParser()
    parser.add_argument('-d', '--dataset_name', help="Input dataset, whose name identifies a particular subfolder in 'data/'", type=str, required=True)
    parser.add_argument("-g", "--greedy",type=str2bool, nargs='?', const=True, default='f', help="Direct optimization of global weight bound")
    parser.add_argument("-r", "--reverse_priority",type=str2bool, nargs='?', const=True, default='t', help="Remove most variable attribute first")
    parser.add_argument("-b", "--balance_sets",type=str2bool, nargs='?', const=True, default='t', help="Keep sensitive and not-sensitive subsets balanced")
    parser.add_argument("-bc", "--balance_condition",type=str2bool, nargs='?', const=True, default='f', help="Remove attributes by trying to balance avg(w^+) and avg(w^-)")
    parser.add_argument('-s', '--seed', help="Seed for sampling", type=int, default=100)
    parser.add_argument('-i', '--iterations', help="Number of iterations of Pivot", type=int, default=25)
    return parser

def main(parsed):
    dataset = parsed.dataset
    dataset_path = basePath_data + dataset + "/" + dataset + ".csv"
    # set the random seed for reproducibility, in algorithms.py the various algorithms use numpy library to sample integers
    seed = parsed.seed
    constants.seed = seed
    np.random.seed(seed)
    random.seed(seed)
    reverse_priority = parsed.reverse_priority
    balance_sets = parsed.balance_sets
    balance_condition = parsed.balance_condition
    greedy = parsed.greedy
    if greedy:
        if reverse_priority:
            print("Incompatible value with option --greedy, it will be ignored")
            reverse_priority = False
        if balance_condition:
            print("Incompatible value with option --greedy, it will be ignored")
            balance_condition = False
        if balance_sets:
            print("Incompatible value with option --greedy, it will be ignored")
            balance_sets = False
        algstr = "greedy_direct"
    else:
        if reverse_priority:
            algstr = "greedy_r"
        else:
            algstr = "greedy"
        if balance_condition:
            algstr += "_bc"
            if balance_sets:
                print("Incompatible value with option --balance_condition, it will be ignored")
                balance_sets = False
        if balance_sets:
            algstr += "_b"

    n_iter = parsed.iterations
    basepath_output = basepath + "/output_fairness"
    if not os.path.exists(basepath_output):
                os.makedirs(basepath_output)
    output_path = basepath_output + "/" + dataset
    if not os.path.exists(output_path):
                os.makedirs(output_path)
    output_path = output_path + "/" + algstr
    if not os.path.exists(output_path):
                os.makedirs(output_path)
    log_path = output_path
    try:
        # - 1 to to run over the complete dataset, otherwise specify the number of objects
        test_size = -1
        # read csv data e save it to dataframe object 
        attributes_names = constants.headers[dataset]
        df = pd.read_csv(dataset_path, sep=",", header=None, names=attributes_names, na_values=constants.nulls[dataset])
        #shuffle rows
        df = df.sample(frac=1)
        #print(df.head(test_size))
        #print(df.describe())
        categorical_attributes = set(get_categorical_attributes(df))
        numerical_attributes = list(set(attributes_names).difference(categorical_attributes))

        categorical_features = list(categorical_attributes)
        print(categorical_features)
        
        print('Missing attribute values before imputation: %d' % sum(pd.isnull(df.values).flatten()))
        df = DataFrameImputer().fit_transform(df)
        print('Missing attribute values after imputation: %d' % sum(pd.isnull(df.values).flatten()))
        

        # compute variation indeces before normalizing 
        # compute variability for each attributes and store in a dict
        var_dict = compute_variability(df, numerical_attributes, categorical_attributes)
        #print(var_dict)

        # normalize numerical attributes
        df[numerical_attributes] = df[numerical_attributes].apply(lambda x: (x - x.min()) / (x.max() - x.min()))

        not_sensitive_subset = set(constants.initial_subsets[dataset][constants.NOT_SENSITIVE])
        sensitive_subset = set(constants.initial_subsets[dataset][constants.SENSITIVE])
        assert not_sensitive_subset.union(sensitive_subset) == set(attributes_names)
        print("Initial NOT sensitive subset " + str(not_sensitive_subset))
        print("Initial sensitive subset " + str(sensitive_subset))

        heuristic_greedy_find_subsets(df[:test_size], not_sensitive_subset, sensitive_subset, var_dict, output_path, balance_sets=balance_sets, reverse_priority=reverse_priority, n_iter=n_iter, balance_condition=balance_condition, greedy=greedy)        

    except FileNotFoundError as e:
        print(str(e))
    except Exception as e:
        print(str(e))
        trace_str = traceback.format_exc()
        print(trace_str)
        for f in os.listdir(output_path):
            os.remove(os.path.join(output_path, f))
        with open(log_path + "/error_log.txt", "w+") as f:
            f.write(trace_str)

if __name__ == '__main__':
    parsed = create_parser().parse_args()
    main(parsed)