import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import numpy as np

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
ax.axis('off')
plt.savefig('../assets/ece_circle_of_life.pdf', dpi=300, bbox_inches='tight')
