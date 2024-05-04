import os
import time
import copy
import torch
import pickle
import logging
import argparse
import numpy as np
import scipy.sparse as sp
from collections import Counter
import optuna
import torch

from models.GCN_dgl import GCN
from models.GAT_dgl import GAT
from models.GSAGE_dgl import GraphSAGE
from models.JKNet_dgl import JKNet
from sklearn.metrics.pairwise import cosine_similarity

parser = argparse.ArgumentParser(description='single')

parser.add_argument('--dataset', type=str, default='cora')
args = parser.parse_args()

ds = args.dataset
adj_orig = pickle.load(open(f'data/graphs/{ds}_adj.pkl', 'rb'))
labels = pickle.load(open(f'data/graphs/{ds}_labels.pkl', 'rb'))
try:
    labels_np = labels.numpy()
except:
    labels_np = labels

# 定义计算邻域同质性比率的函数
def calculate_homophily(adj_matrix, labels):
    num_nodes = adj_matrix.shape[0]
    homophily_ratios = []

    for node in range(num_nodes):
        neighbors = adj_matrix[node].nonzero()[1]  # 获取节点的邻居
        num_same_label_neighbors = sum(labels[neighbors] == labels[node])  # 统计具有相同标签的邻居数量
        homophily_ratio = num_same_label_neighbors / len(neighbors) if len(neighbors) > 0 else 0  # 计算同质性比率
        homophily_ratios.append(homophily_ratio)

    return homophily_ratios

# 计算每个节点的邻域同质性比率
homophily_ratios = calculate_homophily(adj_orig, labels_np)


print("dataset:",ds,sum(homophily_ratios) / len(homophily_ratios))