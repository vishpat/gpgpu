import matplotlib.pyplot as plt

# Specify the file path and column number to read
file_path = "cost.txt"  # Replace with your actual file path
column_number = 1  # 1-based index of the column to plot

# Read the data from the file
data = []
with open(file_path, 'r') as file:
    for line in file:
        values = line.strip().split()
        if len(values) >= column_number:
            data.append(float(values[column_number - 1]))

# Create the graph
plt.figure()
plt.plot(data)
plt.xlabel("Iteration")  # Assuming no specific x-axis values
plt.ylabel("Cost")

# Customize the graph (optional)
plt.title("Learning Rate")
plt.grid(True)

# Save the graph as a PNG image
plt.savefig("learning_rate.png")

