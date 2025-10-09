import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.interpolate import interp1d
from matplotlib import rcParams, rc
from matplotlib.gridspec import GridSpec
import matplotlib.patches as patches
import networkx as nx

def set_np_seed(seed):
    np.random.seed(seed)
    return

set_np_seed(42)

def set_rc_params(fontsize=None):
    '''
    Set figure parameters
    '''

    if fontsize is None:
        fontsize=16
    else:
        fontsize=int(fontsize)

    rc('font',**{'family':'serif'})
    rc('text', usetex=True)

    plt.rcParams.update({'axes.linewidth': 1.3})
    plt.rcParams.update({'xtick.labelsize': fontsize})
    plt.rcParams.update({'ytick.labelsize': fontsize})
    plt.rcParams.update({'xtick.major.size': 8})
    plt.rcParams.update({'xtick.major.width': 1.3})
    plt.rcParams.update({'xtick.minor.visible': True})
    plt.rcParams.update({'xtick.minor.width': 1.})
    plt.rcParams.update({'xtick.minor.size': 6})
    plt.rcParams.update({'xtick.direction': 'out'})
    plt.rcParams.update({'ytick.major.width': 1.3})
    plt.rcParams.update({'ytick.major.size': 8})
    plt.rcParams.update({'ytick.minor.visible': True})
    plt.rcParams.update({'ytick.minor.width': 1.})
    plt.rcParams.update({'ytick.minor.size':6})
    plt.rcParams.update({'ytick.direction':'out'})
    plt.rcParams.update({'axes.labelsize': fontsize})
    plt.rcParams.update({'axes.titlesize': fontsize})
    plt.rcParams.update({'legend.fontsize': int(fontsize-2)})
    plt.rcParams['text.usetex'] = True
    plt.rcParams['text.latex.preamble'] = r'\usepackage{amssymb}'

    return

set_rc_params(fontsize=40)

fig, ax = plt.subplots(3, 3, figsize=(36, 36))

# ===== CH4 (x) layout =====
G = nx.Graph()
positions = {
    (0, 0): (0, 0),
    (1, 0): (1, 0),
    (0, 1): (0, 1),
    (-1, 0): (-1, 0),
    (0, -1): (0, -1)
}
for node, pos in positions.items():
    G.add_node(node, pos=pos)
edges = [((0, 0), (1, 0)), ((0, 0), (0, 1)), ((0, 0), (-1, 0)), ((0, 0), (0, -1))]
G.add_edges_from(edges)

ax[0,0].grid(True, which='both', linestyle='--', linewidth=8)
node_sizes = [5000 for _ in G.nodes()]
node_color = "lightblue"
nx.draw(G, pos=positions, with_labels=False, node_size=node_sizes,
        node_color=node_color, ax=ax[0, 0])
ax[0, 0].set_title(r'$a(\times), p = 0.125$', fontsize=60)

# ===== CH4 (+) layout =====
G = nx.Graph()
sqrt2_over_2 = round(np.sqrt(2) / 2, 2)
positions = {
    (0, 0): (0, 0),
    (sqrt2_over_2, sqrt2_over_2): (sqrt2_over_2, sqrt2_over_2),
    (-sqrt2_over_2, sqrt2_over_2): (-sqrt2_over_2, sqrt2_over_2),
    (-sqrt2_over_2, -sqrt2_over_2): (-sqrt2_over_2, -sqrt2_over_2),
    (sqrt2_over_2, -sqrt2_over_2): (sqrt2_over_2, -sqrt2_over_2)
}
for node, pos in positions.items():
    G.add_node(node, pos=pos)
edges = [
    ((0, 0), (sqrt2_over_2, sqrt2_over_2)),
    ((0, 0), (-sqrt2_over_2, sqrt2_over_2)),
    ((0, 0), (-sqrt2_over_2, -sqrt2_over_2)),
    ((0, 0), (sqrt2_over_2, -sqrt2_over_2))
]
G.add_edges_from(edges)

ax[0,1].grid(True, which='both', linestyle='--', linewidth=3)
node_sizes = [5000 for _ in G.nodes()]
nx.draw(G, pos=positions, with_labels=False, node_size=node_sizes,
        node_color=node_color, ax=ax[0, 1])
ax[0, 1].set_title(r'$a(+), p = 0.125$', fontsize=60)

# ===== H2O layout =====
G = nx.Graph()
positions = {(0, 0): (0, 0), (1, 0): (1, 0), (0, 1): (0, 1)}
for node, pos in positions.items():
    G.add_node(node, pos=pos)
edges = [((0, 0), (1, 0)), ((0, 0), (0, 1))]
G.add_edges_from(edges)
node_sizes = [5000 for _ in G.nodes()]
nx.draw(G, pos=positions, with_labels=False, node_size=node_sizes,
        node_color=node_color, ax=ax[1, 0])
ax[1, 0].set_title(r'$b, p = 0.125$', fontsize=60)

# ===== SO2 layout =====
G = nx.Graph()
positions = {(0, 0): (0, 0), (-1, 0): (-1, 0), (0, 1): (0, 1)}
for node, pos in positions.items():
    G.add_node(node, pos=pos)
edges = [((0, 0), (-1, 0)), ((0, 0), (0, 1))]
G.add_edges_from(edges)
node_sizes = [5000 for _ in G.nodes()]
nx.draw(G, pos=positions, with_labels=False, node_size=node_sizes,
        node_color=node_color, ax=ax[1, 1])
ax[1, 1].set_title(r'$c, p = 0.125$', fontsize=60)

# ===== NH3 layout =====
G = nx.Graph()
positions = {(0, 0): (0, 0), (-1, 0): (-1, 0), (0, 1): (0, 1), (1, 0): (1, 0)}
for node, pos in positions.items():
    G.add_node(node, pos=pos)
edges = [((0, 0), (-1, 0)), ((0, 0), (0, 1)), ((0, 0), (1, 0))]
G.add_edges_from(edges)
node_sizes = [5000 for _ in G.nodes()]
nx.draw(G, pos=positions, with_labels=False, node_size=node_sizes,
        node_color=node_color, ax=ax[2, 1])
ax[2, 1].set_title(r'$d, p = 0.5$', fontsize=60)

# ===== Analytic curves instead of numpy files =====
# Domain for all function plots
x = np.linspace(0, 2*np.pi, 2000)
cos2 = np.cos(x)**2
sin2 = np.sin(x)**2
ones = np.ones_like(x)

std_dev1 = 0.1
std_dev2 = 0.2

# Plot 1 (top-right): f(a_x)=cos^2, f(a_+)=cos^2, h(x)=cos^2
ax[0, 2].plot(x, cos2, label=r'$f(a_{\times})$', color='black')
ax[0, 2].plot(x, cos2, label=r'$f(a_{+})$', color='red')
ax[0, 2].plot(x, cos2, label=r'$h(x)$', color='blue')
ax[0, 2].fill_between(x, cos2 - std_dev1, cos2 + std_dev1,
                    color='blue', alpha=0.2, label=r'$s=0.1 \vec{1}$')
ax[0, 2].legend()
ax[0, 2].set_title(r'$CH_4(+/\times)$ Function Plot')

# Plot 2 (middle-right): f(b)=cos^2, f(c)=\sin^2, h(x)=1
ax[1, 2].plot(x, cos2, label=r'$f(b)$', color='black')
ax[1, 2].plot(x, sin2, label=r'$f(c)$', color='red')
ax[1, 2].plot(x, 0.5*ones, label=r'$h(x)=0.5$', color='blue')
ax[1, 2].fill_between(x, 0.5 - std_dev1, 0.5 + std_dev1,
                    color='blue', alpha=0.2, label=r'$s=0.1 \vec{1}$')
ax[1, 2].legend()
ax[1, 2].set_title(r'$H_2O/SO_2$ Function Plot')

# Plot 3 (bottom-right): f(d)=\sin^2, h(x)=\sin^2 (all three sin^2)
ax[2, 2].plot(x, sin2, label=r'$f(d)$', color='black')
ax[2, 2].plot(x, sin2, label=r'$h(x)$', color='red')
ax[2, 2].fill_between(x, sin2 - std_dev2, sin2 + std_dev2,
                    color='red', alpha=0.2,label=r'$s=0.2 \vec{1}$')
ax[2, 2].legend()
ax[2, 2].set_title(r'$NH_3$ Function Plot')

# Empty bottom-left stays off
ax[2, 0].axis('off')

fig.tight_layout()
plt.subplots_adjust(top=0.92, bottom=0.08, left=0.08, right=0.92,
                    hspace=0.3, wspace=0.3)

plt.savefig('../assets/pc.pdf', dpi=300)

