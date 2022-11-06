# Correlation Clustering with Global Weight Bounds

## Overview

This project is developed as part of the following research papers:

- D. Mandaglio, A. Tagarelli, F. Gullo (2021). *Correlation Clustering with Global Weight Bounds.* In Procs. of the European Conference on Machine Learning and Principles and Practice of Knowledge Discovery in Databases, September 13rd - September 17th, 2021, Bilbao, Spain.

- F. Gullo, L. La Cava, D. Mandaglio, A. Tagarelli (2022) . *When Correlation Clustering Meets Fairness Constraints.* In Procs. of the International Conference on Discovery Science, October 10-12, 2022, Montpellier, France.

Please cite the above papers in any research publication you may produce using this code or data/analysis derived from it.


## Folders
- data: contains the network data used in the experiments described in Section 5.1 of the paper
- data_fairness: contains the relational data used in the fairness clustering application (Section 5.2 of the paper)
- code: it contains this project code and links to the competing methods 
- output: it stores all results produced by the algorithms when runned on the data contained in the "data" folder
- output_fairness: it stores all results produced by the algorithms when runned on the data contained in the "data_fairness" folder

## Dependencies
- scipy==1.6.3
- numpy==1.20.3
- pandas==1.2.4
- pulp==2.4
- scikit_learn==0.24.2

## Usage

### Reproducing global-weight-bounds results (Section 5.1 of the paper)

``` python src/pyccalg.py -d <DATASET_FILE> [-r <LB,UB>] [-a <PROB>] [-s {'pulp','scipy'}] [-m {'charikar','demaine','kwik'}] [-t <targetratio>]```

* Optional arguments: 
   * `-r <LB,UB>`, if you want to generate random edge weights from `[LB,UB]` range
   * `-a <PROB>`, if you want to randomly add edges with probability `PROB`
   * `-m {'charikar','demaine','kwik'}`, to choose the algorithm (default: `'charikar'`). NOTE: `kwikcluster` is always run too
   * `-s {'pulp','scipy'}`, to select the solver to be used (default: `'scipy'` (it seems faster))
   * `-t <targetratio>`, randomly generate (and assign to edges) correlation clustering weights in range `<LB,UB>` with specified target ratio
* Dataset-file format:
   * First line: `#VERTICES \t #EDGES`
   * One line per edge; every line is a quadruple: `NODE1 \t NODE2 \t POSITIVE_WEIGHT \t NEGATIVE_WEIGHT` (`POSITIVE_WEIGHT` and `NEGATIVE_WEIGHT` are ignored if code is run with `-r` option)
   * Look at `data` folder for some examples
  
The above command will run both Pivot algorithm and the selected O(logn)-approximation algorithm specified by the -m parameter. The output is stored in a text file in a subdirectory in the 'output/' folder.

The original implementation of the 'pyccalg.py' can be found at https://github.com/fgullo/pyccalg

### Reproducing fairness clustering results (Section 5.2 of the paper)

From the folder 'global-CC/code', run the following command:

``` python run_find_subsets_attributes.py [-h] -d DATASET [-g [GREEDY]] [-r [REVERSE_PRIORITY]] [-b [BALANCE_SETS]] [-bc [BALANCE_CONDITION]] [-s SEED] [-i ITERATIONS] ```

```
arguments:
  -d DATASET, --dataset_name DATASET
                        Input dataset, whose name identifies a particular subfolder in 'data/'
  -g [GREEDY], --greedy [GREEDY]
                        Heuristic which removes the attribute which leads to Min-CC weights with the best improvement in terms of the global weight bound (default value False)
  -r [REVERSE_PRIORITY], --reverse_priority [REVERSE_PRIORITY]
                        Remove most variable attribute first (default value True)
  -b [BALANCE_SETS], --balance_sets [BALANCE_SETS]
                        Keep sensitive and not-sensitive subsets balanced (default value True)
  -bc [BALANCE_CONDITION], --balance_condition [BALANCE_CONDITION]
                        Remove attributes by trying to balance avg(w^+) and avg(w^-) (default value False)
  -s SEED, --seed SEED  Random generation seed -- for reproducibility (default value 100)
  -i ITERATIONS, --iterations ITERATIONS
                        Number of runs of Pivot (default value 25)
```
Default values for -s and -i correspond to the settings relating to the results presented in the paper.

The above command will run the select heuristic and, at each iteration, also the Pivot algorithm with the current subsets of attributes. The output is stored in text files in a subdirectory in the 'output_fairness/' folder. For each of these files, row i contains the results corresponding to the clustering yielded by the Pivot algorithm at the i-th iteration.

Mapping between heuristic name (used in the paper) and parameter choice:
- Hlv: ``` python run_find_subsets_attributes.py -d 'dataset' -r False -b False ```
- Hlv_B: ``` python run_find_subsets_attributes.py -d 'dataset' -r False``` 
- Hmv: ``` python run_find_subsets_attributes.py -d 'dataset' -b False``` 
- Hmv_B: ``` python run_find_subsets_attributes.py -d 'dataset' ``` 
- Hlv_BW: ``` python run_find_subsets_attributes.py -d 'dataset' -r False -bc``` 
- Hmv_SW: ``` python run_find_subsets_attributes.py -d 'dataset' -bc``` 
- Greedy: ``` python run_find_subsets_attributes.py -g``` 

Important note: customizing the set of non-sensitive and sensitive attributes can be done by modifying the variable 'initial_subset' in 'global-CC/code/constants.py'. If you are interested in performing just a fair clustering task (with no interest in the subsets discovery part) just run the above command (with any heuristic) and consider the results corresponding to the first iteration, corresponding to the initial attributes subsets specified in the 'global-CC/code/constants.py' file.
