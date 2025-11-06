# Plot the requested 1x1 square with a diagonal, colored regions, axis annotations, and dotted guide lines.
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import numpy as np

from matplotlib import rcParams, rc
from matplotlib.gridspec import GridSpec

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

    #plt.rcParams.update({'figure.facecolor':'w'})
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

set_rc_params(fontsize=28)

fig, ax = plt.subplots(figsize=(6, 6))

# Square boundaries
ax.plot([0, 1, 1, 0, 0], [0, 0, 1, 1, 0], color="black", linewidth=1.5)

# Diagonal segment from (0.5, 1) to (0.75, 0)
x1, y1 = 0.5, 1.0
x2, y2 = 0.75, 0.0
ax.plot([x1, x2], [y1, y2], color="black", linewidth=2)

# Fill left region (blue): polygon around left/top edge, diagonal, and bottom edge
left_poly = Polygon([[0, 0], [0, 1], [x1, y1], [x2, y2], [0, 0]], closed=True, facecolor="blue", alpha=0.25, edgecolor=None)
ax.add_patch(left_poly)

# Fill right region (green): polygon around diagonal, right edge, and top edge
right_poly = Polygon([[x2, 0], [1, 0], [1, 1], [x1, 1], [x2, 0]], closed=True, facecolor="green", alpha=0.25, edgecolor=None)
ax.add_patch(right_poly)

text_size = 48
# X-axis annotations: [0, 0.75] as "c" and [0.75, 1] as "1 - c"
y_anno = -0.06  # slightly below the x-axis
ax.plot([0, 0.75], [y_anno, y_anno], linewidth=3, color="blue")
ax.text(0.375, y_anno - 0.03, "r", ha="center", va="top", fontsize=text_size)

ax.plot([0.75, 1.0], [y_anno, y_anno], linewidth=3, color="green")
ax.text(0.875, y_anno - 0.03, "1 - r", ha="center", va="top", fontsize=text_size)

# Mark the split point at x=0.75 on the axis
ax.plot([0.75, 0.75], [-0.005, 0.005], color="black", linewidth=1.5)

# Dotted guide lines
# Horizontal dotted line (parallel to x-axis) labeled "Gx"
gy = 0.6
ax.plot([0, 1], [gy, gy], linestyle=":", linewidth=2, color="black")
ax.text(1.01, gy, "Gx", ha="left", va="center", fontsize=text_size)

# Vertical dotted line (parallel to y-axis) labeled "F"
fx = 0.6
ax.plot([fx, fx], [0, 1], linestyle=":", linewidth=2, color="black")

ax.text(fx, 1.01, "F", ha="center", va="bottom", fontsize=text_size)

# Ticks and labels
ax.set_xlim(-0.02, 1.05)
ax.set_ylim(-0.12, 1.05)
ax.set_aspect('equal', adjustable='box')

# Clean look
for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)

plt.tight_layout()

ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.set_xticks([])
ax.set_yticks([])

plt.savefig("../assets/correct_ratio_example.pdf", dpi=300)

