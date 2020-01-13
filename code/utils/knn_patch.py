from sklearn.neighbors import NearestNeighbors
import numpy as np

def extract_knn_patch(queries, pc, k):
    """
    queries [M, C]
    pc [P, C]
    """
    #print(queries.shape)
    #print(pc.shape)
    knn_search = NearestNeighbors(n_neighbors=k, algorithm='auto')
    knn_search.fit(pc[:, 0:3])
    knn_idx = knn_search.kneighbors(queries, return_distance=False)
    k_patches = np.take(pc, knn_idx, axis=0)  # M, K, C
    return k_patches
