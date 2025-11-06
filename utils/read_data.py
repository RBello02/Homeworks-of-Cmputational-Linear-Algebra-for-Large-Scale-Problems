import numpy as np
from scipy.sparse import lil_matrix, csr_matrix

def read_pages(file_path):       

    # this function reads the dataset and create the link matrix A

    with open(file_path, "r") as f:
        lines = f.readlines()

    n_nodes, n_edges = map(int, lines[0].split())

    # the first part of the code contains the "nodes" so we'll create a dic that maps the urls into numbers

    node_lines = []
    i = 1
    while i < len(lines) and lines[i].strip():
        parts = lines[i].strip().split(maxsplit=1)
        # at the end of the file there is only numbers, so it will be skipped
        saveit = True
        if parts[1].isdigit():
            saveit = False
            break
        if saveit:
            node_lines.append(parts)
        i += 1

    id_to_url = {int(id_): url for id_, url in node_lines}

    # the second part of the file contains the connections between the pages

    edge_lines = lines[i:]
    edges = [tuple(map(int, line.split())) for line in edge_lines if line.strip()]

    A = lil_matrix((n_nodes, n_nodes), dtype=float)    # sparse mode
    for src, dst in edges:
        A[src - 1, dst - 1] = 1.0   # -1 because the vector in python starts from 0 but the indices starts from 1

    # normalization of the link matrix

    row_sums = np.array(A.sum(axis=1)).flatten()
    row_sums[row_sums == 0] = 1.0  # dont devide by zero

    D_inv = csr_matrix(np.diag(1 / row_sums))

    A = D_inv @ A.tocsr()
    
    return A,id_to_url,edges