import numpy as np
from scipy.interpolate import make_interp_spline
import matplotlib.pyplot as plt

#Similarity Code

# Data from the table
cases = ['a', 'b', 'c', 'd']
similarity = [6.38e-04, 1.17e-03, 5.52e-03, 1.01e-02]

# Generate smooth line for similarity
x = np.arange(len(cases))
x_smooth = np.linspace(x.min(), x.max(), 300)
spl = make_interp_spline(x, similarity, k=2)
y_smooth = spl(x_smooth)

plt.figure(figsize=(8, 6))
plt.plot(x_smooth, y_smooth, color='b')  # Smooth line
plt.scatter(x, similarity, color='r', marker='s')  # Square markers for data points

# Customizing y-axis to use scientific notation
plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.2e}'))

plt.xlabel('Case', fontsize=30)
plt.xticks(x, cases, fontsize = 32)
plt.ylabel('Similarity Score (Intersect)', fontsize=27)
plt.yticks(fontsize= 27)
plt.title('Similarity', fontsize=30)

# Remove grid and add border
plt.grid(False)
plt.gca().spines['top'].set_visible(True)
plt.gca().spines['right'].set_visible(True)
plt.gca().spines['bottom'].set_visible(True)
plt.gca().spines['left'].set_visible(True)
plt.gca().spines['top'].set_color('black')
plt.gca().spines['right'].set_color('black')
plt.gca().spines['bottom'].set_color('black')
plt.gca().spines['left'].set_color('black')

plt.show()


#Distance Score
# Data for distance
distance = [0.5399, 0.3168, 0.2631, 0.1721]

# Generate smooth line for distance
spl_distance = make_interp_spline(x, distance, k=2)
y_smooth_distance = spl_distance(x_smooth)

plt.figure(figsize=(8, 6))
plt.plot(x_smooth, y_smooth_distance, color='b')  # Smooth line
plt.scatter(x, distance, color='r', marker='s')  # Square markers for data points

# Customizing y-axis
plt.xlabel('Case', fontsize=30)
plt.xticks(x, cases, fontsize=32)
plt.ylabel('Distance Score (Bhattacharyya)', fontsize=27)
plt.yticks(fontsize=30)
plt.title('Distance', fontsize=30)

# Remove grid and add border
plt.grid(False)
plt.gca().spines['top'].set_visible(True)
plt.gca().spines['right'].set_visible(True)
plt.gca().spines['bottom'].set_visible(True)
plt.gca().spines['left'].set_visible(True)
plt.gca().spines['top'].set_color('black')
plt.gca().spines['right'].set_color('black')
plt.gca().spines['bottom'].set_color('black')
plt.gca().spines['left'].set_color('black')

plt.show()