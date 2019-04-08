import matplotlib.pyplot as plt
import numpy as np

# non-maze
x_vals = [0.91, 0.93, 0.95, 0.97, 0.99]
y_vals_pi = [3651, 456, 552, 375, 497]
y_vals_vi = [119, 119, 134, 116, 79]

plt.title('Non-Maze Running Times')
plt.xlabel('Discount Rate')
plt.ylabel('Time (ms)')

plt.plot(x_vals, y_vals_pi, label='Policy Iteration')
plt.plot(x_vals, y_vals_vi, label='Value Iteration')

plt.xticks(x_vals)

plt.legend()
plt.show()

# non-maze iterations
x_vals = [0.91, 0.93, 0.95, 0.97, 0.99]
y_vals_pi = [4, 4, 6, 5, 6]
y_vals_vi = [48, 49, 53, 56, 57]

plt.title('Non-Maze Iterations')
plt.xlabel('Discount Rate')
plt.ylabel('Iterations')

plt.plot(x_vals, y_vals_pi, label='Policy Iteration')
plt.plot(x_vals, y_vals_vi, label='Value Iteration')

plt.xticks(x_vals)

plt.legend()
plt.show()

# non-maze q-learning
x_vals = [0.1, 0.3, 0.5, 0.7, 0.9]
y_vals_qc = [891, 694, 691, 723, 704]
y_vals_qd = [1643, 1320, 1449, 1240, 1539]

plt.title('Non-Maze Q-Learning Running Times')
plt.xlabel('Learning Rate')
plt.ylabel('Time (ms)')

plt.plot(x_vals, y_vals_qc, label='Constant Rate')
plt.plot(x_vals, y_vals_qd, label='Exponential Decay Rate')

plt.xticks(x_vals)

plt.legend()
plt.show()

# maze
x_vals = [0.91, 0.93, 0.95, 0.97, 0.99]
y_vals_pi = [46081, 37416, 51892, 54266, 114345]
y_vals_vi = [4157, 4236, 4514, 4720, 5213]

plt.title('Maze Running Times')
plt.xlabel('Discount Rate')
plt.ylabel('Time (ms)')

plt.plot(x_vals, y_vals_pi, label='Policy Iteration')
plt.plot(x_vals, y_vals_vi, label='Value Iteration')

plt.xticks(x_vals)

plt.legend()
plt.show()

# maze iterations
x_vals = [0.91, 0.93, 0.95, 0.97, 0.99]
y_vals_pi = [8, 8, 7, 8, 9]
y_vals_vi = [135, 140, 150, 158, 169]

plt.title('Maze Iterations')
plt.xlabel('Discount Rate')
plt.ylabel('Iterations')

plt.plot(x_vals, y_vals_pi, label='Policy Iteration')
plt.plot(x_vals, y_vals_vi, label='Value Iteration')

plt.xticks(x_vals)

plt.legend()
plt.show()

# maze q-learning
x_vals = [0.1, 0.3, 0.5, 0.7, 0.9]
y_vals_qc = [14852, 7797, 10532, 6738, 6658]
y_vals_qd = [250452, 273693, 252561, 326312, 278636]

plt.title('Maze Q-Learning Running Times')
plt.xlabel('Learning Rate')
plt.ylabel('Time (ms)')

plt.plot(x_vals, y_vals_qc, label='Constant Rate')
plt.plot(x_vals, y_vals_qd, label='Exponential Decay Rate')

plt.xticks(x_vals)

plt.legend()
plt.show()