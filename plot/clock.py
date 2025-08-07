import matplotlib.pyplot as plt
import numpy as np

# Set up the figure and axis
fig, ax = plt.subplots(figsize=(6, 6))
ax.set_aspect('equal')

# Circle parameters
radius = 1.0
center = (0, 0)

# Draw the circle with thicker line
theta = np.linspace(0, 2 * np.pi, 500)
x = radius * np.cos(theta)
y = radius * np.sin(theta)
ax.plot(x, y, color='black', linewidth=2.5)

# 12 equally spaced points
angles = np.linspace(0, 2 * np.pi, 12, endpoint=False)
point_colors = plt.cm.tab20(np.linspace(0, 1, 12))  # 12 distinct colors

for angle, color in zip(angles, point_colors):
    px = radius * np.cos(angle)
    py = radius * np.sin(angle)
    ax.plot(px, py, 'o', color=color, markersize=24)  # Bigger dots

# Add two black arrows from the center
# Add two black arrows pointing directly up from the center
ax.arrow(0, 0, 0.0, 0.5, head_width=0.07, head_length=0.07, fc='black', ec='black')
ax.arrow(0, 0, 0.0, 0.8, head_width=0.07, head_length=0.07, fc='black', ec='black')

# Hide axes
ax.axis('off')
plt.savefig('../assets/clock.pdf', bbox_inches='tight', dpi=300)

