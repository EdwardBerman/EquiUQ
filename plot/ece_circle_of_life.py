
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Arc
import numpy as np
from matplotlib.path import Path
from matplotlib.transforms import Affine2D

from matplotlib import rcParams, rc

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
    plt.rcParams['text.usetex'] = False
    plt.rcParams['text.latex.preamble'] = r'\usepackage{amssymb}'

    return

set_rc_params(fontsize=30)

fig, ax = plt.subplots(figsize=(6, 6))

# Radii for annulus and circle placement
inner_radius = 0.5
outer_radius = 1.0
circle_radius = 0.75

# Create full circle
theta_full = np.linspace(0, 2 * np.pi, 500)
x_outer = outer_radius * np.cos(theta_full)
y_outer = outer_radius * np.sin(theta_full)
x_inner = inner_radius * np.cos(theta_full)
y_inner = inner_radius * np.sin(theta_full)

# Fill entire annulus light gray
ax.fill(x_outer, y_outer, color='lightgray')
ax.fill(x_inner, y_inner, color='white', zorder=2)

# Right half with cross-hatching
theta_right = np.linspace(-np.pi/2, np.pi/2, 250)
x_outer_r = outer_radius * np.cos(theta_right)
y_outer_r = outer_radius * np.sin(theta_right)
x_inner_r = inner_radius * np.cos(theta_right)
y_inner_r = inner_radius * np.sin(theta_right)

# Draw hatched annulus on right half
x_right = np.concatenate([x_outer_r, x_inner_r[::-1]])
y_right = np.concatenate([y_outer_r, y_inner_r[::-1]])
ax.fill(x_right, y_right, facecolor='none', edgecolor='gray', hatch='//', linewidth=0.0, zorder=1)

# Hollow circle drawing helper
def draw_hollow_circle(x, y, color):
    circle = plt.Circle((x, y), 0.05, fill=True, edgecolor=color, facecolor=color, linewidth=2)
    ax.add_patch(circle)

# Use angles offset from axis lines to avoid overlap
n = 10
epsilon = np.pi / (2 * n + 1)
angles_upper = np.linspace(epsilon, np.pi - epsilon, n)
angles_lower = np.linspace(-np.pi + epsilon, -epsilon, n)

# Upper half (y > 0)
for i, angle in enumerate(angles_upper):
    x = circle_radius * np.cos(angle)
    y = circle_radius * np.sin(angle)
    
    # Right side: make the 3rd one yellow
    if x > 0:
        draw_hollow_circle(x, y, 'orange' if i == 2 else 'blue')
    else:
        draw_hollow_circle(x, y, 'blue')  # Left side

# Lower half (y < 0): colors per spec
for i, angle in enumerate(angles_lower):
    x = circle_radius * np.cos(angle)
    y = circle_radius * np.sin(angle)

    # Right: 1 yellow, rest blue
    draw_hollow_circle(x, y, 'orange' if i == 0 else 'blue')

    # Left: all yellow
    draw_hollow_circle(-x, y, 'orange')

# Draw axes
ax.plot([-1.2, 1.2], [0, 0], color='black', linewidth=1.2)  # x-axis
ax.plot([0, 0], [-1.2, 1.2], color='black', linewidth=1.2)  # y-axis


def draw_reflection_arrow(x_pos):
    arrow = FancyArrowPatch(
        (x_pos, 0.3), (x_pos, -0.3),
        arrowstyle='<->',
        mutation_scale=15,
        color='black',
        linewidth=1.5,
        zorder=3
    )
    ax.add_patch(arrow)

# Arrows near y-axis on both left and right
draw_reflection_arrow(-0.25)
draw_reflection_arrow(0.25)

# Final formatting
ax.set_aspect('equal')
ax.set_xlim(-1.2, 1.2)
ax.set_ylim(-1.2, 1.2)
ax.set_title('Confidence Fibers of Reflection Invariant Model (a)', fontsize=20, pad=10)
ax.axis('off')
plt.savefig('../assets/ece_circle_of_life_a.pdf', dpi=300, bbox_inches='tight')
plt.close()

fig, ax = plt.subplots(figsize=(6, 6))

# Radii for annulus and circle placement
inner_radius = 0.5
outer_radius = 1.0
circle_radius = 0.75

# Create full circle
theta_full = np.linspace(0, 2 * np.pi, 500)
x_outer = outer_radius * np.cos(theta_full)
y_outer = outer_radius * np.sin(theta_full)
x_inner = inner_radius * np.cos(theta_full)
y_inner = inner_radius * np.sin(theta_full)

# Fill entire annulus light gray
ax.fill(x_outer, y_outer, color='lightgray')
ax.fill(x_inner, y_inner, color='white', zorder=2)

# Right half with cross-hatching
theta_right = np.linspace(-np.pi/2, np.pi/2, 250)
x_outer_r = outer_radius * np.cos(theta_right)
y_outer_r = outer_radius * np.sin(theta_right)
x_inner_r = inner_radius * np.cos(theta_right)
y_inner_r = inner_radius * np.sin(theta_right)

# Draw hatched annulus on right half
x_right = np.concatenate([x_outer_r, x_inner_r[::-1]])
y_right = np.concatenate([y_outer_r, y_inner_r[::-1]])
ax.fill(x_right, y_right, facecolor='none', edgecolor='gray', linewidth=0.0, zorder=1)

# Hollow circle drawing helper
def draw_hollow_circle(x, y, color):
    circle = plt.Circle((x, y), 0.05, fill=True, edgecolor=color, facecolor=color, linewidth=2)
    ax.add_patch(circle)

# Use angles offset from axis lines to avoid overlap
n = 10
epsilon = np.pi / (2 * n + 1)
angles_upper = np.linspace(epsilon, np.pi - epsilon, n)
angles_lower = np.linspace(-np.pi + epsilon, -epsilon, n)

# Upper half (y > 0)
for i, angle in enumerate(angles_upper):
    x = circle_radius * np.cos(angle)
    y = circle_radius * np.sin(angle)
    
    # Right side: make the 3rd one yellow
    if x > 0:
        draw_hollow_circle(x, y, 'orange' if i == 2 else 'blue')
    else:
        draw_hollow_circle(x, y, 'blue')  # Left side

# Lower half (y < 0): colors per spec
for i, angle in enumerate(angles_lower):
    x = circle_radius * np.cos(angle)
    y = circle_radius * np.sin(angle)

    # Right: 1 yellow, rest blue
    draw_hollow_circle(x, y, 'orange' if i == 0 else 'blue')

    # Left: all yellow
    draw_hollow_circle(-x, y, 'orange')

# Draw axes
ax.plot([-1.2, 1.2], [0, 0], color='black', linewidth=1.2)  # x-axis
ax.plot([0, 0], [-1.2, 1.2], color='black', linewidth=1.2)  # y-axis


def draw_reflection_arrow(x_pos):
    arrow = FancyArrowPatch(
        (x_pos, 0.3), (x_pos, -0.3),
        arrowstyle='<->',
        mutation_scale=15,
        color='black',
        linewidth=1.5,
        zorder=3
    )
    ax.add_patch(arrow)

# Arrows near y-axis on both left and right
draw_reflection_arrow(-0.25)
draw_reflection_arrow(0.25)

# Final formatting
ax.set_aspect('equal')
ax.set_xlim(-1.2, 1.2)
ax.set_ylim(-1.2, 1.2)
ax.set_title('Confidence Fiber of Reflection Invariant Model (b)', fontsize=20, pad=10)
ax.axis('off')
plt.savefig('../assets/ece_circle_of_life_b.pdf', dpi=300, bbox_inches='tight')
plt.close()

fig, ax = plt.subplots(figsize=(6, 6))

# Radii for annulus and circle placement
inner_radius = 0.5
outer_radius = 1.0
circle_radius = 0.75

# Create full circle
theta_full = np.linspace(0, 2 * np.pi, 500)
x_outer = outer_radius * np.cos(theta_full)
y_outer = outer_radius * np.sin(theta_full)
x_inner = inner_radius * np.cos(theta_full)
y_inner = inner_radius * np.sin(theta_full)

# Fill entire annulus light gray
ax.fill(x_outer, y_outer, color='lightgray')
ax.fill(x_inner, y_inner, color='white', zorder=2)

# Right half with cross-hatching (outline only here)
theta_right = np.linspace(-np.pi/2, np.pi/2, 250)
x_outer_r = outer_radius * np.cos(theta_right)
y_outer_r = outer_radius * np.sin(theta_right)
x_inner_r = inner_radius * np.cos(theta_right)
y_inner_r = inner_radius * np.sin(theta_right)

# Draw hatched annulus on right half (no hatch pattern requested; keep as outline)
x_right = np.concatenate([x_outer_r, x_inner_r[::-1]])
y_right = np.concatenate([y_outer_r, y_inner_r[::-1]])
ax.fill(x_right, y_right, facecolor='none', edgecolor='gray', linewidth=0.0, zorder=1)

# Hollow circle drawing helper
def draw_hollow_circle(x, y, color):
    circle = plt.Circle((x, y), 0.05, fill=True, edgecolor=color, facecolor=color, linewidth=2)
    ax.add_patch(circle)

# Use angles offset from axis lines to avoid overlap
n = 10
epsilon = np.pi / (2 * n + 1)
angles_upper = np.linspace(epsilon, np.pi - epsilon, n)
angles_lower = np.linspace(-np.pi + epsilon, -epsilon, n)

# Upper half (y > 0)
for i, angle in enumerate(angles_upper):
    x = circle_radius * np.cos(angle)
    y = circle_radius * np.sin(angle)
    if x > 0:
        # Right side: make the 3rd one yellow
        draw_hollow_circle(x, y, 'orange' if i == 2 else 'blue')
    else:
        # Left side: all blue
        draw_hollow_circle(x, y, 'blue')

# Lower half (y < 0): colors per spec
for i, angle in enumerate(angles_lower):
    x = circle_radius * np.cos(angle)
    y = circle_radius * np.sin(angle)
    # Right: 1 yellow, rest blue
    draw_hollow_circle(x, y, 'orange' if i == 0 else 'blue')
    # Left: all yellow
    draw_hollow_circle(-x, y, 'orange')

# Draw axes
ax.plot([-1.2, 1.2], [0, 0], color='black', linewidth=1.2)  # x-axis
ax.plot([0, 0], [-1.2, 1.2], color='black', linewidth=1.2)  # y-axis

# --- Curved rotation arrow helper ---
def draw_rotation_arrow(ax, radius=1.08, theta1_deg=40, theta2_deg=320,
                        linewidth=2, mutation_scale=18, color='black', zorder=4):
    """
    Draw a rotation arrow that *follows the circle* of given radius.
    Uses a path-based FancyArrowPatch so the arrowhead is tangent-aligned
    even for small radii.
    """
    arc_path = Path.arc(theta1_deg, theta2_deg)  # unit-circle arc
    trans = (Affine2D().scale(radius, radius).translate(0, 0) + ax.transData)

    arrow = FancyArrowPatch(
        path=arc_path,
        transform=trans,
        arrowstyle='-|>',
        mutation_scale=mutation_scale,  # head size in points
        lw=linewidth,
        color=color,
        facecolor=color,
        zorder=zorder
    )
    ax.add_patch(arrow)

# Draw one large CCW rotation arrow outside the annulus
draw_rotation_arrow(ax, radius=0.25, theta1_deg=40, theta2_deg=320)

# Final formatting
ax.set_aspect('equal')
ax.set_xlim(-1.2, 1.2)
ax.set_ylim(-1.2, 1.2)
ax.set_title('Confidence Fiber of Rotation Invariant Model (c)', fontsize=20, pad=10)
ax.axis('off')

plt.savefig('../assets/ece_circle_of_life_c.pdf', dpi=300, bbox_inches='tight')
plt.close()
