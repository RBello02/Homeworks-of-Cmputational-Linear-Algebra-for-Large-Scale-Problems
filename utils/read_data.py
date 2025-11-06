import numpy as np

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
        print(parts)
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

    A = np.zeros((n_nodes, n_nodes), dtype=int)
    for src, dst in edges:
        A[src - 1, dst - 1] = 1   # -1 because the vector in python starts from 0 but the indices starts from 1

    # normalization of the link matrix

    row_sums = A.sum(axis=1)
    row_sums[row_sums == 0] = 1  # is possible that the sum == 0 so in that case we put the sum equal to 1 (to avoid dividing by 0)
    A = A / row_sums.reshape(-1,1)

    return A,id_to_url,edges