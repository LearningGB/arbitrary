import numpy as np
import matplotlib.pyplot as plt

# Load elevation data
csv_path = "mtfuji_data.csv"
np.set_printoptions(suppress=True)
fuji = np.loadtxt(csv_path, delimiter=",", skiprows=1)

# Define the gradient calculation function
def calculate_gradient(current_point):
    next_point = current_point - 1
    gradient = (fuji[current_point, 3] - fuji[next_point, 3]) / (fuji[current_point, 0] - fuji[next_point, 0])
    return gradient

# Define the destination calculation function
def calculate_destination(current_point, learning_rate=0.2):
    gradient = calculate_gradient(current_point)
    destination_point = int(current_point - learning_rate * gradient)
    if destination_point < 0 or destination_point >= len(fuji):
        destination_point = current_point
    return destination_point

# Define the descent function
def descend_mountain(starting_point):
    current_point = starting_point
    descent_path = [current_point]
    while True:
        next_point = calculate_destination(current_point)
        if next_point == current_point:
            break
        current_point = next_point
        descent_path.append(current_point)
    return descent_path

# Visualize the descent process
def visualize_descent(descent_path):
    plt.figure(figsize=(10, 6))

    # Elevation vs. point number
    plt.plot(fuji[:, 0], fuji[:, 3], label='Mt. Fuji')
    for point in descent_path:
        plt.scatter(fuji[point, 0], fuji[point, 3], c='red', marker='o')

    # Altitude and slope vs. repetition
    altitudes = []
    slopes = []
    for point in descent_path:
        altitudes.append(fuji[point, 3])
        slopes.append(calculate_gradient(point))
    plt.plot(range(len(descent_path)), altitudes, label='Altitude')
    plt.plot(range(len(descent_path)), slopes, label='Slope')

    plt.xlabel('Point Number')
    plt.ylabel('Elevation (m) / Slope')
    plt.title('Descent Process')
    plt.legend()
    plt.grid(True)
    plt.show()

# Visualize the descent process for different initial values
initial_values = [136, 142, 150, 158]
for initial_value in initial_values:
    descent_path = descend_mountain(initial_value)
    visualize_descent(descent_path)

# Visualize the descent process for different hyperparameters
learning_rates = [0.1, 0.2, 0.3]
for learning_rate in learning_rates:
    descent_path = descend_mountain(136, learning_rate=learning_rate)
    visualize_descent(descent_path)
