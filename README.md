# Lightweight-Yet-Efficient: Revitalizing Ball-Tree for Point-to-Hyperplane Nearest Neighbor Search

Welcome to the **Ball-Tree and BC-Tree** GitHub!

In this repository, we target at solving the problem of **Point-to-Hyperplane Nearest Neighbor Search (P2HNNS)**. Given a set of data points in a ($d-1$)-dimensional Euclidean space and a hyperplane query (represented as a vector in $d$-dimensional Euclidean space), the problem of P2HNNS aims to find the point whose distance to the hyperplane query is minimized among all the data.

This problem plays a vital role in many research domains. For example, in the applications of pool-based active learning with SVMs, the goal is to request labels for the data points closest (with minimum margin) to the SVM's decision hyperplane to reduce human efforts for annotation. Moreover, motivated by the success of SVM for classification, the maximum margin clustering aims at finding the hyperplane maximizing the minimum margin to the data, which can separate the data from different classes. Such applications require finding the data points that are closest to the hyperplane.

This repository provides the implementations and experiments of our work entitled [Lightweight-Yet-Efficient: Revitalizing Ball-Tree for Point-to-Hyperplane Nearest Neighbor Search](https://arxiv.org/abs/2302.10626) that has been accepted (1st cycle) and will be published in [IEEE ICDE 2023](https://icde2023.ics.uci.edu/). We implement **Ball-Tree** and **BC-Tree** for performing P2HNNS in high-dimensional spaces. To make a systematic comparison, we also include two state-of-the-art hyperplane hashing schemes [NH and FH](https://github.com/HuangQiang/P2HNNS) as baselines for evaluations.

## Data Sets

### Data Sets Details

We study the performance of Ball-Tree and BC-Tree on 16 real-world data sets, i.e., Music, GloVe, Sift, UKBench, Tiny, Msong, NUSW, Cifar-10, Sun, LabelMe, Gist, Enron, Trevi, P53, Deep100M, and Sift100M. Their statistics are summarized as follows.

| Data Sets | # Data Points | # Dim  | # Queries | Data Size | Data Type |
| --------- | -----------   | ------ | --------- | --------- | --------- |
| Music     | 1,000,000     | 100    | 100       | 386 MB    | Rating    |
| GloVe     | 1,183,514     | 100    | 100       | 460 MB    | Text      |
| Sift      | 985,462       | 128    | 100       | 485 MB    | Image     |
| UKBench   | 1,097,907     | 128    | 100       | 541 MB    | Image     |
| Tiny      | 1,000,000     | 384    | 100       | 1.5 GB    | Image     |
| Msong     | 992,272       | 420    | 100       | 1.6 GB    | Audio     |
| NUSW      | 268,643       | 500    | 100       | 514 MB    | Image     |
| Cifar-10  | 50,000        | 512    | 100       |  98 MB    | Image     |
| Sun       | 79,106        | 512    | 100       | 155 MB    | Image     |
| LabelMe   | 181,093       | 512    | 100       | 355 MB    | Image     |
| Gist      | 982,694       | 960    | 100       | 3.6 GB    | Image     |
| Enron     | 94,987        | 1,369  | 100       | 497 MB    | Text      |
| Trevi     | 100,900       | 4,096  | 100       | 1.6 GB    | Image     |
| P53       | 31,153        | 5,408  | 100       | 643 MB    | Biology   |
| Deep100M  | 100,000,000   | 96     | 100       | 36.1 GB   | Image     |
| Sift100M  | 99,986,452    | 128    | 100       | 48.0 GB   | Image     |

### Hyperplane Query Generation

Different from [NH and FH](https://dl.acm.org/doi/abs/10.1145/3448016.3457240) that generate the hyperplane queries uniformly at random from a hypercube, we generate the hyperplane queries uniformly at random from a $d$-ball. We prefer this new way because it is more natural to simulate the hyperplane queries appeared in real-world applications. Please click [here](http://extremelearning.com.au/how-to-generate-uniformly-random-points-on-n-spheres-and-n-balls/) for more details about the $d$-ball and its generation algorithm. The data sets and hyperplane queries used in our experiments can be donwload [here](https://drive.google.com/drive/folders/1C9JWcMyTAUYYxM55FuMrPQ1dPJQ5vhsB?usp=sharing).

Except for the link to download the data sets and hyperplane queries, we have also enclosed the sourece codes to generate the hyperplane query and the source codes for data sets deduplication, so that users can conduct experiemnts on any data sets of interests. Please refer to the fold `pre_process/` for more details.

### Data Format

To reduce the I/O cost, all the datasets and queries are stored in a binary format. For a dataset (or query set) with *n* vectors in *d* dimensions, the binary file comparises with a `float` array of `n*d` length. Please specify the cardinality *n* and data dimension *d* in the bash scripts in advance.

## Requirements

- Ubuntu 18.04 (or higher version)
- g++ 8.3.1 (or higher version).
- Python 3.7 (or higher version)

## Compilation

The source codes are implemented by C++, which requires `g++-8` with `c++17` (or higher version) for compilation. Thus, please check whether the `g++-8` is installed before the compilation. If not, we provide a way to install `g++-8` in `Ubuntu 18.04` as follows.

```bash
sudo add-apt-repository ppa:ubuntu-toolchain-r/test
sudo apt-get update
sudo apt-get install g++-8
sudo apt-get install gcc-8 (optional)
```

Users can use the following commands to compile the C++ source codes:

```bash
cd methods/
make -j
```

## Experiments

We provide the bash scripts to run all experiments. Once you have downloaded the data sets and queries and completed the compilation, you can reproduce the experiments by simply running the following commands:

```bash
cd methods/
bash run.sh
```

## Visualization

Finally, we privode python scripts for visualization. These scripts require `python 3.7` (or higher versions) with **numpy, scipy, and matplotlib** installed. If not, you might need to use `anaconda` to create a new virtual environment and use `pip` to install those packages.

After you have conducted all experiments, you can plot all the figures in our paper with the following commands.

```bash
cd scripts/
python3 plot.py
```

or

```bash
cd scripts/
python plot.py
```

## Reference

Thank you for being patient in reading the user manual. We will appreciate using the following BibTeX to cite this work when you use the source codes in your paper.

```tex
@inproceedings{huang2023lightweight,
  title={Lightweight-Yet-Efficient: Revitalizing Ball-Tree for Point-to-Hyperplane Nearest Neighbor Search},
  author={Huang, Qiang and Tung, Anthony K. H.},
  booktitle={2023 IEEE 39th International Conference on Data Engineering (ICDE)},
  year={2023}
}
```

It is welcome to contact me (huangq@comp.nus.edu.sg) if you meet any issue. Thank you for your interest!
