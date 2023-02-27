import numpy as np
import math
import warnings


def is_square(i: int) -> bool:
    return i == math.isqrt(i) ** 2


def runtime_warn(message):
    warnings.warn(message, RuntimeWarning, stacklevel=2)


class Detector:
    def __init__(self, size, num_points):
        if (not is_square(num_points)):
            runtime_warn(
                "numPoints not square; results rounded to square number")
        per_side = int(num_points**.5)
        self.size = size
        self.area = size**2
        self.num_points = num_points
        self.point_spacing = per_side / size
        self.points = np.zeros((per_side, per_side))
        self.distribution_function = None

    def print_properties(self):
        print(
            "Size: {size}cm\nArea: {area}cm²\nNum pts: {num_points}\nPt. spacing: ~{point_spacing:.3f}cm\nPt. Distrib: {point_distribution}".format(size=self.size, area=self.area, num_points=self.num_points, point_spacing=self.point_spacing, point_distribution=self.distribution_function))

    def generate_random(self):
        self.distribution_function = "Random"
        # TODO: implement random dist

    def generate_normal(self):
        self.distribution_function = "Normal"
        # TODO: implement normal dist gen


detector = Detector(11, 100)
detector.generate_normal()
detector.print_properties()
