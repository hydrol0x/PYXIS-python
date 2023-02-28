import numpy as np
import matplotlib.pyplot as plt
import math
import warnings


def is_square(i: int) -> bool:
    return i == math.isqrt(i) ** 2


def generate_grid(size, per_side):
    return (np.mgrid[-size/2:size /
                     2:complex(per_side, 1), -size/2:size/2:complex(per_side, 1)])


def runtime_warn(message):
    warnings.warn(message, RuntimeWarning, stacklevel=2)


class Detector:
    def __init__(self, size, num_points):
        if (not is_square(num_points)):
            runtime_warn(
                "numPoints not square; results rounded to square number")
        self.per_side = int(num_points**.5)
        self.size = size
        self.area = size**2
        self.num_points = num_points
        self.point_spacing = self.per_side / self.size
        self.distribution_function = None

    def generate_grid(self):
        # Generate a grid from -size/2 to size/2
        size = self.size
        per_side = self.per_side
        _X, _Y = np.mgrid[-size/2:size /
                          2:complex(self.per_side, 1), -self.size/2:self.size/2:complex(self.per_side, 1)]
        self.grid = np.vstack((_X.flatten(), _Y.flatten())).T
        self._X, self._Y = _X, _Y

    def display_grid(self):
        plt.plot(self._X, self._Y, marker='.', color='k', linestyle='none')
        plt.show()

    def print_properties(self):
        print(
            "Size: {size}cm\nArea: {area}cm²\nNum pts: {num_points}\nPt. spacing: ~{point_spacing:.3f}cm\nPt. Distrib: {point_distribution}".format(size=self.size, area=self.area, num_points=self.num_points, point_spacing=self.point_spacing, point_distribution=self.distribution_function))

    def generate_random(self):
        self.distribution_function = "Random"
        # TODO: implement random dist

    def generate_normal(self):
        self.distribution_function = "Normal"
        # TODO: implement normal dist gen


detector = Detector(20, 1000)
detector.generate_grid()
detector.display_grid()
