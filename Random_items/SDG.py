import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Define the loss function and its gradient
def loss(w):
    return (w - 3)**2

def grad(w):
    return 2 * (w - 3)

# Settings
w_vals = np.linspace(-1, 7, 400)
L_vals = loss(w_vals)
eta = 0.1  # learning rate
w_init = -1  # starting point
steps = 20

# Store the optimization path
w_path = [w_init]
for _ in range(steps):
    w_new = w_path[-1] - eta * grad(w_path[-1])
    w_path.append(w_new)

# Animation setup
fig, ax = plt.subplots()
line, = ax.plot(w_vals, L_vals, label='Loss Function $L(w) = (w - 3)^2$')
point, = ax.plot([], [], 'ro')
text = ax.text(0.05, 0.95, '', transform=ax.transAxes)

ax.set_xlim(-1, 7)
ax.set_ylim(0, 20)
ax.set_xlabel('$w$')
ax.set_ylabel('$L(w)$')
ax.legend()

def update(frame):
    w = w_path[frame]
    L = loss(w)
    point.set_data(w, L)
    text.set_text(f'Step {frame}\n$w$ = {w:.2f}\n$L(w)$ = {L:.2f}')
    return point, text

ani = FuncAnimation(fig, update, frames=len(w_path), interval=600, repeat=False)
plt.show()
