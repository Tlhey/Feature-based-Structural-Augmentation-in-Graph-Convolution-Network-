import numpy as np
import scipy.sparse as sp
import pickle
from scipy.spatial import distance
from sklearn.preprocessing import normalize
import torch

def compute_manhattan_distances(features):
    return distance.cdist(features, features, 'cityblock')

def compute_euclidean_distances(features):
    return distance.cdist(features, features, 'euclidean')

def compute_cosine_similarity(features):
    normalized_features = normalize(features)
    return np.dot(normalized_features, normalized_features.T)

def save_matrices(features, dataset_name, index):
    manhattan_distances = compute_manhattan_distances(features)
    euclidean_distances = compute_euclidean_distances(features)
    cosine_similarity = compute_cosine_similarity(features)
    

    manhattan_logits = -manhattan_distances
    euclidean_logits = -euclidean_distances
    cosine_logits = cosine_similarity * features.shape[0]
    print(cosine_logits)

    with open(f'data/edge_distance/{dataset_name}_graph_{index}_manhattan_logits.pkl', 'wb') as f:
        pickle.dump(manhattan_logits, f)
    with open(f'data/edge_distance/{dataset_name}_graph_{index}_euclidean_logits.pkl', 'wb') as f:
        pickle.dump(euclidean_logits, f)
    with open(f'data/edge_distance/{dataset_name}_graph_{index}_cosine_logits.pkl', 'wb') as f:
        pickle.dump(cosine_logits, f)

def main():
    dataset_name = 'flickr' 
    index = '1' 

    with open(f'data/graphs/{dataset_name}_features.pkl', 'rb') as f:
        features = pickle.load(f)
        if isinstance(features, torch.FloatTensor): 
            features = features.numpy()
        if sp.issparse(features):
            features = torch.FloatTensor(features.toarray())

    #print(type(features), features)
    save_matrices(features, dataset_name, index)

if __name__ == "__main__":
    main()