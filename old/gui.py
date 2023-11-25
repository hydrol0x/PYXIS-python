import tkinter as tk
from tkinter import ttk
from generator import Detector

# create window
root = tk.Tk()
root.title("Detector")

# create global variables
detectors = []
current_detector = None

# define functions


def generate_detector():
    global current_detector
    size = int(size_entry.get())
    num_points = int(num_points_entry.get())
    name = name_entry.get()
    detector = Detector(size, num_points)
    detector.name = name
    detector.generate_grid()
    if distribution_var.get() == "Normal":
        mu = list(map(float, mu_entry.get().split(",")))
        sigma = list(map(float, sigma_entry.get().split(",")))
        scale = int(scale_entry.get())
        detector.generate_normal(mu, sigma, scale)
    else:
        low = int(low_entry.get())
        high = int(high_entry.get())
        detector.generate_random(low, high)
    detectors.append(detector)
    current_detector = detector
    update_detector_dropdown()


def show_distribution(detector):
    detector.display_dist()


def select_detector(event):
    global current_detector
    detector_name = detector_dropdown.get()
    for detector in detectors:
        if detector.name == detector_name:
            current_detector = detector


def update_detector_dropdown():
    detector_names = [detector.name for detector in detectors]
    detector_dropdown.config(values=detector_names)
    detector_var.set(detector_names[0] if detector_names else "None")


# create widgets
name_label = ttk.Label(root, text="Name: ")
name_entry = ttk.Entry(root, width=10)
size_label = ttk.Label(root, text="Size (cm): ")
size_entry = ttk.Entry(root, width=10)
num_points_label = ttk.Label(root, text="Number of Points: ")
num_points_entry = ttk.Entry(root, width=10)
distribution_label = ttk.Label(root, text="Distribution Type: ")
distribution_var = tk.StringVar(value="Normal")
distribution_dropdown = ttk.Combobox(
    root, textvariable=distribution_var, values=["Normal", "Random"])
detector_label = ttk.Label(root, text="Select Detector:")
detector_var = tk.StringVar(value="None")
detector_names = [detector.name for detector in detectors]
detector_dropdown = ttk.Combobox(
    root, textvariable=detector_var, values=detector_names)
detector_dropdown.bind("<<ComboboxSelected>>", select_detector)
show_button = ttk.Button(root, text="Show Distribution",
                         command=lambda: show_distribution(current_detector))

# add widgets to window
name_label.grid(row=0, column=0, padx=5, pady=5)
name_entry.grid(row=0, column=1, padx=5, pady=5)
size_label.grid(row=1, column=0, padx=5, pady=5)
size_entry.grid(row=1, column=1, padx=5, pady=5)
num_points_label.grid(row=2, column=0, padx=5, pady=5)
num_points_entry.grid(row=2, column=1, padx=5, pady=5)
distribution_label.grid(row=3, column=0, padx=5, pady=5)
distribution_dropdown.grid(row=3, column=1, padx=5, pady=5)
detector_label.grid(row=5, column=0, padx=5, pady=5)
detector_dropdown.grid(row=5, column=1, padx=5, pady=5)
show_button.grid(row=6, column=0, columnspan=2, padx=5, pady=5)

# create widgets for distribution parameters
params_frame = ttk.Frame(root, padding=10)
params_frame.grid(row=4, column=0, columnspan=2, padx=5, pady=5)

# random distribution parameters
random_frame = ttk.Frame(params_frame, padding=5)
random_frame.grid(row=0, column=0, sticky="w")
random_label = ttk.Label(random_frame, text="Random Distribution Parameters:")
random_label.grid(row=0, column=0, columnspan=2, sticky="w")
low_label = ttk.Label(random_frame, text="Low:")
low_label.grid(row=1, column=0, sticky="w")
low_entry = ttk.Entry(random_frame, width=10)
low_entry.insert(0, "0")
low_entry.grid(row=1, column=1, sticky="w")
high_label = ttk.Label(random_frame, text="High:")
high_label.grid(row=2, column=0, sticky="w")
high_entry = ttk.Entry(random_frame, width=10)
high_entry.insert(0, "10")
high_entry.grid(row=2, column=1, sticky="w")

# normal distribution parameters
normal_frame = ttk.Frame(params_frame, padding=5)
normal_frame.grid(row=0, column=1, sticky="w")
normal_label = ttk.Label(normal_frame, text="Normal Distribution Parameters:")
normal_label.grid(row=0, column=0, columnspan=2, sticky="w")
mu_label = ttk.Label(normal_frame, text="µ:")
mu_label.grid(row=1, column=0, sticky="w")
mu_entry = ttk.Entry(normal_frame, width=10)
mu_entry.insert(0, "0.0,0.0")
mu_entry.grid(row=1, column=1, sticky="w")
sigma_label = ttk.Label(normal_frame, text="σ:")
sigma_label.grid(row=2, column=0, sticky="w")
sigma_entry = ttk.Entry(normal_frame, width=10)
sigma_entry.insert(0, "3,3")
sigma_entry.grid(row=2, column=1, sticky="w")
scale_label = ttk.Label(normal_frame, text="Scale:")
scale_label.grid(row=3, column=0, sticky="w")
scale_entry = ttk.Entry(normal_frame, width=10)
scale_entry.insert(0, "1")
scale_entry.grid(row=3, column=1, sticky="w")

generate_button = ttk.Button(
    params_frame, text="Generate", command=generate_detector)
generate_button.grid(row=4, column=0, columnspan=2, padx=5, pady=5)


# start event loop
root.mainloop()
