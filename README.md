# Lightweight-Yet-Efficient: Revitalizing Ball-Tree for Point-to-Hyperplane Nearest Neighbor Search

In this github, we implement **Ball-Tree** and **BC-Tree** for Point-to-Hyperplane Nearest Neighbor Search. We choose two state-of-the-art hyperplane hashing schemes [NH and FH](https://github.com/HuangQiang/P2HNNS) as baselines for comparisons.

## Data Sets and Queries

We study the performance of Ball-Tree and BC-Tree over 12 real-world data sets, i.e., Sift, UKBench, Tiny, Msong, NUSW, Cifar-10, Sun, LabelMe, Gist, Enron, Trevi, and P53. We follow [NH and FH](https://dl.acm.org/doi/abs/10.1145/3448016.3457240) and randomly generate 100 queries for each data set. The data sets and queries used in our experiments can be donwload [here](https://drive.google.com/drive/folders/1C9JWcMyTAUYYxM55FuMrPQ1dPJQ5vhsB?usp=sharing). Their statistics are summarized as follows.

| Data Sets | # Data      | # Dim  | # Queries | Data Size | Data Type |
| --------- | ----------- | ------ | --------- | --------- | --------- |
| Sift      | 985,462     | 128    | 100       | 485 MB    | Image     |
| UKBench   | 1,097,907   | 128    | 100       | 541 MB    | Image     |
| Tiny      | 1,000,000   | 384    | 100       | 1.5 GB    | Image     |
| Msong     | 992,272     | 420    | 100       | 1.6 GB    | Audio     |
| NUSW      | 268,643     | 500    | 100       | 514 MB    | Image     |
| Cifar-10  | 50,000      | 512    | 100       |  98 MB    | Image     |
| Sun       | 79,106      | 512    | 100       | 155 MB    | Image     |
| LabelMe   | 181,093     | 512    | 100       | 355 MB    | Image     |
| Gist      | 982,694     | 960    | 100       | 3.6 GB    | Image     |
| Enron     | 94,987      | 1,369  | 100       | 497 MB    | Text      |
| Trevi     | 100,900     | 4,096  | 100       | 1.6 GB    | Image     |
| P53       | 31,153      | 5,408  | 100       | 643 MB    | Biology   |

## Compilation

The source codes are implemented by C++, which requires ```g++-8``` with ```c++17``` for compilation. Thus, please check whether the `g++-8` is installed before the compilation. If not, here is a way to install `g++-8` in `Ubuntu 18.04` (or higher versions) as follows.

```bash
sudo add-apt-repository ppa:ubuntu-toolchain-r/test
sudo apt-get update
sudo apt-get install g++-8
sudo apt-get install gcc-8 (optional)
```

Users can use the following commands to compile the C++ source codes:

```bash
make -j
```

## Experiments

We provide the bash scripts to run all experiments. Once you have downloaded the data sets and queries and completed the compilation, you can reproduce the experiments by simply running the following commands:

```bash
bash run.sh
```

## Visualization

Finally, we privode python scripts for visualization. These scripts require `python 3.7` (or higher versions) with **numpy, scipy, and matplotlib** installed. If not, you might need to use `anaconda` to create a new virtual environment and use `pip` to install those packages.

After you have conducted all experiments, you can plot all the figures in our paper with the following commands.

```bash
python3 plot.py
```

or

```bash
python plot.py
```

Thank you for your interest!
