# Agglomerative and Divisive Hierarchical CLustering

Course Assignment for CS F415- Data Mining @ BITS Pilani, Hyderabad Campus.

**Done under the guidance of Dr. Aruna Malapati, Assistant Professor, BITS Pilani, Hyderabad Campus.**

## Table of contents


## Introduction
Hierarchical clustering is a method of cluster analysis which seeks to build a hierarchy of clusters. Strategies for hierarchical clustering generally fall into two types:

1. Agglomerative: This is a "bottom up" approach: each observation starts in its own cluster, and pairs of clusters are merged as one moves up the hierarchy.

2. Divisive: This is a "top down" approach: all observations start in one cluster, and splits are performed recursively as one moves down the hierarchy.

In general, the merges and splits are determined in a greedy manner. The results of hierarchical clustering are usually presented in a dendrogram.


**The main purpose of this project is to get an in depth understanding of how the Divisive and Agglomerative hierarchical clustering algorithms work.**

*More on [Hierarchical clustering](https://en.wikipedia.org/wiki/Hierarchical_clustering)*

## Data
We used the **Human Gene DNA Sequence** dataset, which can be found [here](http://genome.crg.es/datasets/ggalhsapgenes2005/hg16.311.putative.cds.fa). The dataset contains **311 gene sequences**. The data can be found in the folder **'data'**.

## Instructions to run the scripts
Run the following command:

##### Divisive clustering
```python
python divisive.py
```

##### Agglomerative clustering
```python
python agglomerative.py
```


## Equations used
```
confidence(X->Y) = support(X U Y) / support(X)
support(X, Y) = support count(X, Y) / total dataset size
```


## Pre-processing done
The csv file was read sequence by sequence and was saved in the form of a dictionary, where the key is the gene sequence's name and the value contains the enyire gene string.

A mapping was created from the unique gene sequences in the dataset to integers so that each sequence corresponded to a unique integer.

The entire data was mapped to integers to reduce the storage and computational requirement.

## Directory Structure
```
association-rule-mining-apriori/
+-- data
|   +-- groceries.csv (original data file containing transactions)
+--  arm.py(python script to read the data, mine frequent itemsets and interesting rules)
+--  hash_tree.py(python file containing the Tree and Node classes used to build the hash tree for support counting)
+--  timing_wrapper.py(python decorator used to measure execution time of functions)
+--  l_final.pkl(all the frequent itemsets in pickled format)
+--  outputs(destination to save the outputs generated)
|   +-- frequent_itemsets.txt(all the frequent itemsets presented in the prescribed format)
|   +-- association_rules.txt(all the interesting association rules mined and presented in the prescribed format)
+--  results(folder containing the results of this project)
+--  reverse_map.pkl(mapping from items to index in pickled format)
+--  requirements.txt
```

## Machine specs
Processor: i7-7500U

Ram: 16 GB DDR4

OS: Ubuntu 16.04 LTS

## Results

| Confidence/Support | No. of itemsets | No of rules |
|---------------------|-------|--------|
| High confidence(MIN_CONF=0.5) High support count(MINSUP=60)               | 725  |  60      |
| Low confidence(MIN_CONF=0.1) High support count(MINSUP=60)              | 725   |    1189    |
| High confidence(MIN_CONF=0.5) Low support count(MINSUP=10)              | 11390   |    4187    |
| Low confidence(MIN_CONF=0.1) Low support count(MINSUP=10)              | 11390   |    35196    |

All the frequent itemsets and rules generated using the above mentioned configurations can be found in the 'results' folder.

## Members
[Shubham Jha](http://github.com/shubhamjha97)

[Praneet Mehta](http://github.com/praneetmehta)

[Abhinav Jain](http://github.com/abhinav1112)