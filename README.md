# Feature-based Structural Augmentation in Graph Convolution Network 

## Our Contrubution
We added Similarity Metrix calculation to the optuna method and make modification on GAUG-M source code. 
We also put the output of our model in the folder Ouput.

Contribution:
Improved accuracy in node classification using graph convolution network 
Proposed a framework that significantly improve the applicability of graph convolution network in High-dimensional and Low-Homophily graph.

## Fork Repo 
====
Source code for the AAAI'2021 paper:
[Data Augmentation for Graph Neural Networks](https://arxiv.org/pdf/2006.06830.pdf)
by [Tong Zhao](https://tzhao.io/) (tzhao2@nd.edu), [Yozen Liu](https://research.snap.com/team/yozen-liu),  [Leonardo Neves](https://research.snap.com/team/leonardo-neves), [Oliver Woodford](https://ojwoodford.github.io/), [Meng Jiang](http://www.meng-jiang.com/), and [Neil Shah](http://nshah.net/).

## Requirements

Python 3.7.6. 
Make sure all dependencies specified in the ```requirements.txt``` file are satisfied before running the model. This can be achieved by
```
pip install -r requirements.txt
```

## Usage
### Data
To generate similarity matrix, create the folder, GAug/data/edge_distance, then run:
```
 python pre.py
```

To hyperparameter search, run:
```
# --i: the index of output name, can be "1_manhattan", "1_euclidean", "1_cosine" if runs Optuna.py. Can be "1" if runs Theta.py or Theta_2hop.py
# --gpu: -1, 0, 1, ...
# --dataset: cora, blogcatalog, airport, citeseer, flickr

# example:
python Optuna.py --i 1_manhattan --dataset citeseer
```

For cosine similarity can also directly run Theta.py, an input example could be:
```
python Theta.py --dataset cora --gpu 0 --i 1
```

For cosine Similarity with two hop can directly run Theta_2hop.py, , an input example could be:
```
python Theta_2hop.py --dataset cora --gpu 0 --i 1
```

The Homophily Ratio for each dataset is calculated through Homophily_Ratio.py, an input example could be:
```
python Homophily_Ratio.py --dataset cora
```
### Plotting
On the same directory with the generated output file, change the file name corresponding to the named one. Then run the vis.ipynb. 


## Data
The format of data files are described in detail in the file ```data/README```.
Due to file size limit, for GAugM, only the edge_probabilities of Cora is provided.
Please find the all edge_probabilities files at https://tinyurl.com/gaug-data. The VGAE implementation I used for generating these edge_probabilities are also provided under the folder ```vgae/```.

## Cite
If you find this repository useful in your research, please cite our paper:


