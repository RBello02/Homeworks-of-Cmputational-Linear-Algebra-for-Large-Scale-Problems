import numpy as np
from scipy.sparse import lil_matrix, csr_matrix, diags

def read_pages(file_path):       
    with open(file_path, "r") as f:
        lines = f.readlines()

    n_nodes, n_edges = map(int, lines[0].split())

    # --- nodes ---
    node_lines = []
    i = 1
    while i < len(lines) and lines[i].strip():
        parts = lines[i].strip().split(maxsplit=1)
        if parts[1].isdigit():
            break
        node_lines.append(parts)
        i += 1

    id_to_url = {int(id_)-1: url for id_, url in node_lines}  #start from 0

    # --- edges ---
    edge_lines = lines[i:]
    edges = [tuple(map(int, line.split())) for line in edge_lines if line.strip()]

    # --- identify sink nodes ---
    out_deg = {node: 0 for node in range(0, n_nodes)}
    in_deg = {node: 0 for node in range(0,n_nodes)}
    backlinks = {node: [] for node in range(0,n_nodes)}
    for src, dst in edges:
        out_deg[src-1] += 1    # because the pages start from 1
        in_deg[dst-1] += 1
        backlinks[dst-1].append(src-1)

    sink_nodes = [node for node, deg in out_deg.items() if deg == 0]
    no_back_link_node = [node for node, deg in in_deg.items() if deg == 0]

    # --- build sparse matrix ---
    A = lil_matrix((n_nodes, n_nodes), dtype=float)
    for src, dst in edges:
        A[src - 1, dst - 1] = 1.0

    # --- normalize rows ---
    row_sums = np.array(A.sum(axis=1)).flatten()
    row_sums[row_sums == 0] = 1.0
    D_inv = diags(1 / row_sums)
    A = D_inv @ A.tocsr()

    return A, id_to_url, edges, sink_nodes, no_back_link_node, out_deg, in_deg, backlinks
