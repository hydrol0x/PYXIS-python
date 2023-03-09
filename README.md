# Detector Simulation

This is a Python program for simulating a detector, generating a grid of evenly spaced points, and generating a distribution of points on the grid. The program uses NumPy and SciPy for generating the grid and the point distribution, and Matplotlib for displaying the results.

## Installation

This program requires NumPy, SciPy, and Matplotlib to be installed. To install these dependencies using pip, run the following command:

`pip install numpy scipy matplotlib`

## Usage

The program is contained in a single Python file detector_simulation.py. To use the program, import the Detector class from the module and create a new instance of it with the desired parameters:

### Creating a detector
```
from detector import Detector

detector = Detector(size=100, num_points=10000)
```

The `size` parameter specifies the side length of the detector in cm, and `num_points` specifies the number of points to generate on the detector grid.

### Generating a grid
After creating a `Detector` instance, you can generate the grid of points and display it using the following methods:

```
detector.generate_grid()
detector.display_grid()
```

### Generate Distribution
You can also generate a distribution of points on the grid using the `generate_normal` or `generate_random` methods:

```
detector.generate_normal(mu=[0.0, 0.0], sigma=[3, 3], scale=1)
detector.display_dist()
```

The `generate_normal` method generates a normal distribution of points on the grid, with the specified mean `mu`, standard deviation `sigma`, and scaling factor `scale`. The `generate_random` method generates a random distribution of points with a minimum of ` low` and maximum of `high`.

### Example output
`Detector.distribution` is a list of coordinate points from the grid as well as the coresponding z-value from the generated distribution
```
[(-5.0, -5.0, 10.995223491546444), (-5.0, -4.666666666666667, 13.150677735695503), (-5.0, -4.333333333333333, 15.535690079304302)...]
```
`Detector.display_dist()` will provide a matplotlib 3D output.

Normal distribution with an 8x8 detector and 961 points. 
![3d normal distribution plot](https://user-images.githubusercontent.com/34951139/223906488-6f90a6f6-33b0-4f8d-a44d-67718a0c303e.png)

Random distribution with an 8x8 detector and 961 points.
![3d random distribution plot](https://user-images.githubusercontent.com/34951139/223906689-ad64c7d0-9fd0-4278-ad59-d2fe2770fe5a.png)
