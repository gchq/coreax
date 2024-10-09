import json
from matplotlib import pyplot as plt

# Open the JSON file
with open('benchmark_results.json', 'r') as file:
    # Load the JSON data into a Python object
    data_by_solver = json.load(file)


# Visualization code
plt.figure(figsize=(12, 6))

for algo, data in data_by_solver.items():
    plt.plot(data['coreset_size'], data['accuracy'], 'o-', label=algo)

plt.xlabel('Iteration')
plt.ylabel('Performance Metric')
plt.title('Algorithm Performance Across Iterations')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)

plt.show()