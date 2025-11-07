import igraph as ig
import numpy as np
import random
import tempfile
from IPython.display import Image, display
import matplotlib.pyplot as plt

def draw_graph(A, sample_size=1500, seed=None, values=None, cmap='viridis', vertex_size=6, fig_size=(10, 10), colorbar_size=(6, 0.6)):

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

    edges = [(int(i), int(j)) for i, j in zip(A_sub.row, A_sub.col)]

    g = ig.Graph(directed=True)
    g.add_vertices(len(sampled_nodes))
    g.add_edges(edges)

    layout = g.layout("fr")

    # --- colors --
    if values is not None:
        vals = np.array(values)[sampled_nodes]
        norm = plt.Normalize(vmin=np.min(vals), vmax=np.max(vals))
        cmap_ = plt.cm.get_cmap(cmap)
        colors = [plt.cm.colors.rgb2hex(cmap_(norm(v))) for v in vals]
        g.vs["color"] = colors
    else:
        g.vs["color"] = "skyblue"

    # --- draw --
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmpfile:
        visual_style = {
            "layout": layout,
            "vertex_size": vertex_size,
            "edge_arrow_size": 0.2,
            "bbox": (fig_size[0] * 100, fig_size[1] * 100),  
            "margin": 30,
            "target": tmpfile.name,
        }
        ig.plot(g, **visual_style)
        display(Image(filename=tmpfile.name))

    # --- colorbar ---
    if values is not None:
        fig, ax = plt.subplots(figsize=colorbar_size)
        fig.subplots_adjust(bottom=0.5)
        cb = plt.colorbar(
            plt.cm.ScalarMappable(norm=norm, cmap=cmap_),
            cax=ax,
            orientation='horizontal'
        )
        cb.set_label('Scores of the nodes')
        cb.ax.tick_params(labelsize=8)  
        cb.ax.xaxis.set_ticks_position('bottom')
        plt.show()