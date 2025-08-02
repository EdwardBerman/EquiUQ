import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots(figsize=(6, 6))

# Radii for annulus and circle placement
inner_radius = 0.5
outer_radius = 1.0
circle_radius = 0.75

# Draw light gray annulus
theta = np.linspace(0, 2 * np.pi, 500)
x_outer = outer_radius * np.cos(theta)
y_outer = outer_radius * np.sin(theta)
x_inner = inner_radius * np.cos(theta)
y_inner = inner_radius * np.sin(theta)
ax.fill(x_outer, y_outer, color='lightgray')
ax.fill(x_inner, y_inner, color='white', zorder=2)

# Hollow circle drawing helper
def draw_hollow_circle(x, y, color):
    circle = plt.Circle((x, y), 0.05, fill=False, edgecolor=color, linewidth=2)
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

# Final formatting
ax.set_aspect('equal')
ax.set_xlim(-1.2, 1.2)
ax.set_ylim(-1.2, 1.2)
ax.axis('off')
plt.savefig('../assets/ece_circle_of_life.pdf', dpi=300, bbox_inches='tight')
