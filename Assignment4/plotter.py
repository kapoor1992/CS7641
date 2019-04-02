import matplotlib.pyplot as plt
import numpy as np

# non-maze
x_vals = [0.9, 0.925, 0.95, 0.975, 1]
y_vals_pi = [4679, 376, 550, 1270, 2563]
y_vals_vi = [226, 262, 327, 276, 217]
y_vals_qc = [1099, 545, 274, 358, 304]
y_vals_qd = [1051, 2205, 822, 419, 750]

plt.title('Non-Maze Running Times')
plt.xlabel('Discount Rate')
plt.ylabel('Time (ms)')

plt.plot(x_vals, y_vals_pi, label='Policy Iteration')
plt.plot(x_vals, y_vals_vi, label='Value Iteration')
plt.plot(x_vals, y_vals_qc, label='Q-Learning Constant Rate')
plt.plot(x_vals, y_vals_qd, label='Q-Learning Exponential Decay Rate')

plt.xticks(x_vals)

plt.legend()
plt.show()

# maze
x_vals = [0.9, 0.925, 0.95, 0.975, 1]
y_vals_pi = [5421, 3099, 3867, 6050, 54009]
y_vals_vi = [410, 452, 486, 553, 517]
y_vals_qc = [4251, 3359, 2569, 3262, 2295]
y_vals_qd = [2437, 7179, 15360, 4255, 3342]

plt.title('Maze Running Times')
plt.xlabel('Discount Rate')
plt.ylabel('Time (ms)')

plt.plot(x_vals, y_vals_pi, label='Policy Iteration')
plt.plot(x_vals, y_vals_vi, label='Value Iteration')
plt.plot(x_vals, y_vals_qc, label='Q-Learning Constant Rate')
plt.plot(x_vals, y_vals_qd, label='Q-Learning Exponential Decay Rate')

plt.xticks(x_vals)

plt.legend()
plt.show()
