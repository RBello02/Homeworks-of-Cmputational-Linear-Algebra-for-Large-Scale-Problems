import igraph as ig
import numpy as np
import random
import tempfile
from IPython.display import Image, display

def draw_graph(A, sample_size=1500, seed=None):

    # set the seed
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        ig.set_random_number_generator(random.Random(seed))

    n_nodes = A.shape[0]

    # select a subset of the nodes
    if n_nodes > sample_size:
        sampled_nodes = sorted(random.sample(range(n_nodes), sample_size))
    else:
        sampled_nodes = list(range(n_nodes))

    # extract the submatrix
    A_sub = A[sampled_nodes, :][:, sampled_nodes].tocoo()

    edges = list(zip(A_sub.row, A_sub.col))

    g = ig.Graph(directed=True)
    g.add_vertices(len(sampled_nodes))
    g.add_edges(edges)

    layout = g.layout("fr")

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmpfile:
        ig.plot(g, layout=layout, vertex_size=2, edge_arrow_size=0.2, target=tmpfile.name)
        display(Image(filename=tmpfile.name))
