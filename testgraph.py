import numpy as np

def get_hop_distance(num_node, edge, max_hop=1):
    A = np.zeros((num_node, num_node))
    for i, j in edge:
        A[j, i] = 1
        A[i, j] = 1
    
    #print(A)
    # compute hop steps
    hop_dis = np.zeros((num_node, num_node)) + np.inf
    transfer_mat = [np.linalg.matrix_power(A, d) for d in range(max_hop + 1)]
    #print(transfer_mat)
    arrive_mat = (np.stack(transfer_mat) > 0)
    for d in range(max_hop, -1, -1):
        hop_dis[arrive_mat[d]] = d
    return hop_dis

def get_adjacency(hop_dis, strategy):
    valid_hop = range(0, 2, 1)
    adjacency = np.zeros((18, 18))
    for hop in valid_hop:
        adjacency[hop_dis == hop] = 1
    normalize_adjacency = normalize_digraph(adjacency)

    if strategy == 'uniform':
        A = np.zeros((1, num_node, num_node))
        A[0] = normalize_adjacency
    elif strategy == 'distance':
        A = np.zeros((len(valid_hop), num_node, num_node))
        for i, hop in enumerate(valid_hop):
            A[i][hop_dis == hop] = normalize_adjacency[hop_dis == hop]
    return A

def normalize_digraph(A):
    Dl = np.sum(A, 0)
    num_node = A.shape[0]
    Dn = np.zeros((num_node, num_node))
    for i in range(num_node):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i]**(-1)
    AD = np.dot(A, Dn)
    return AD

num_node = 18
self_link = [(i, i) for i in range(num_node)]
neighbor_link = [(4, 3), (3, 2), (7, 6), (6, 5), (13, 12), (12,11),
                 (10, 9), (9, 8), (11, 5), (8, 2), (5, 1), (2, 1),
                 (0, 1), (15, 0), (14, 0), (17, 15), (16, 14)]

edge = self_link + neighbor_link
hop_dis = get_hop_distance(18, edge)
print(hop_dis)
print(get_adjacency(hop_dis, 'distance'))

