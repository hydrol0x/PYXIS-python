import uproot
import numpy as np

# Parameters for your grid
N = 10  # Number of rows
M = 10  # Number of columns
num_events = 1000  # Number of scintillation events to simulate

# Type for each detection event
dtype = np.dtype([('energy', 'float32'), ('time_left', 'float32'), ('time_right', 'float32')])

# Create a dictionary to hold data arrays for each bar
bar_data = {}

# Initialize data arrays
for i in range(N):
    for j in range(M):
        bar_id = f"bar_{i}_{j}"
        bar_data[bar_id] = np.zeros(num_events, dtype=dtype)

# Fill the arrays with random data
for event in range(num_events):
    for i in range(N):
        for j in range(M):
            bar_id = f"bar_{i}_{j}"
            bar_data[bar_id][event]['energy'] = np.random.uniform(0, 10)
            bar_data[bar_id][event]['time_left'] = np.random.uniform(0, 100)
            bar_data[bar_id][event]['time_right'] = np.random.uniform(0, 100)

# Writing data to a ROOT file using uproot
with uproot.recreate("simulated_detector_output.root") as file:
    for bar_id in bar_data:
        file[f"DetectorData/{bar_id}"] = bar_data[bar_id]
