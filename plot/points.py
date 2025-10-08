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
ax[0, 0].set_title(r'$CH_4 (\times)$')

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

# Titles
ax[0, 0].set_title(r'$a(\times), p = 0.125$', fontsize=60)
ax[0, 1].set_title(r'$a(+), p = 0.125$', fontsize=60)
ax[1, 0].set_title(r'$b, p = 0.125$', fontsize=60)
ax[1, 1].set_title(r'$c, p = 0.125$', fontsize=60)
ax[2, 1].set_title(r'$d, p = 0.5$', fontsize=60)

fig.tight_layout()
plt.subplots_adjust(top=0.92, bottom=0.08, left=0.08, right=0.92,
                    hspace=0.3, wspace=0.3)

# ===== Spectra loading & plots =====
methane_spectra = np.load('../data/methane_spectra.npy')
water_spectra = np.load('../data/water_spectra.npy')
sulfur_dioxide_spectra = np.load('../data/sulfur_dioxide.npy')
ammonia_spectra = np.load('../data/amonia_spectra.npy')

sulfur_dioxide_spectra_x = sulfur_dioxide_spectra[:, 0]
sulfur_dioxide_spectra_y = sulfur_dioxide_spectra[:, 1]

methane_spectra = methane_spectra / np.max(methane_spectra)
water_spectra = water_spectra / np.max(water_spectra)
ammonia_spectra = ammonia_spectra / np.max(ammonia_spectra)

ax[0, 2].plot(methane_spectra, color='black', label=r'f(CH$_4(+))$')
ax[0, 2].plot(methane_spectra, color='red', label=r'$f(CH_4(\times))$')
ax[0, 2].plot(methane_spectra, color='blue', label=r'$h(Orbit)$')
ax[0, 2].legend()
ax[0, 2].set_title(r'$CH_4(+/\times)$ Absorption Spectra')

ax[1, 2].plot(water_spectra, color='black', label=r'f(H$_2O)$')
ax[1, 2].plot(sulfur_dioxide_spectra_x, sulfur_dioxide_spectra_y, color='red', label=r'$f(SO_2)$')

interp_function = interp1d(sulfur_dioxide_spectra_x, sulfur_dioxide_spectra_y,
                           kind='linear', fill_value="extrapolate")
water_x = np.linspace(500, 4000, len(water_spectra))
sulfur_dioxide_y_interpolated = interp_function(water_x)
average_spectra = (water_spectra + sulfur_dioxide_y_interpolated) / 2
ax[1, 2].plot(average_spectra, color='blue', label=r'$h(Orbit)$')
ax[1, 2].legend()
ax[1, 2].set_title(r'$H_2O/SO_2$ Absorption Spectra')

ax[2, 2].plot(ammonia_spectra, color='black', label=r'f(NH$_3)$')
ax[2, 2].plot(ammonia_spectra, color='red', label=r'$h(Orbit)$')
ax[2, 2].legend()
ax[2, 2].set_title(r'$NH_3$ Absorption Spectra')

ax[2, 0].axis('off')

print((np.linalg.norm(water_spectra - average_spectra) +
       np.linalg.norm(water_spectra - sulfur_dioxide_y_interpolated)) / 2)

plt.savefig('../assets/pc.pdf', dpi=300)

