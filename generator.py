import numpy as np
import matplotlib.pyplot as plt
import math
import warnings

from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import multivariate_normal


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
        self.point_spacing = self.size/(self.per_side - 1)
        self.distribution_function = None

    def generate_grid(self):
        # Generate a grid from -size/2 to size/2
        size = self.size
        per_side = self.per_side
        __X, __Y = np.mgrid[-size/2:size /
                            2:complex(self.per_side, 1), -self.size/2:self.size/2:complex(self.per_side, 1)]
        self.grid = np.vstack((__X.flatten(), __Y.flatten())).T
        self.__X, self.__Y = __X, __Y

    def display_grid(self):
        plt.plot(self.__X, self.__Y, marker='.', color='k', linestyle='none')
        plt.show()

    def display_dist(self, wireframe=False):
        distrib = self.distribution_function
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        x = self.__X
        y = self.__Y
        z = self.__Z
        if (distrib == "Normal"):
            if self.num_points <= 1000:
                ax.scatter(x, y, z)
            else:
                if wireframe:
                    ax.plot_wireframe(x, y, z)
                ax.plot_surface(x, y, z)
        else:
            ax.scatter(x, y, z)
        plt.show()

    def print_properties(self):
        print(
            "Size: {size}cm\nArea: {area}cm²\nNum pts: {num_points}\nPt. spacing: ~{point_spacing:.3f}cm\nPt. Distrib: {point_distribution}".format(size=self.size, area=self.area, num_points=self.num_points, point_spacing=self.point_spacing, point_distribution=self.distribution_function))

    def generate_random(self, low=0, high=10):
        self.distribution_function = "Random"
        grid = self.grid
        self.__Z = np.random.uniform(low=low, high=high, size=grid.shape[0])

    def generate_normal(self, mu=[0.0, 0.0], sigma=[3, 3], scale=1):
        self.distribution_function = "Normal"
        grid = self.grid
        x = self.__X
        mu = np.array(mu)

        sigma = np.array(sigma)
        covariance = np.diag(sigma**2)

        z = multivariate_normal.pdf(grid, mean=mu, cov=covariance)
        z = z.reshape(x.shape)
        z *= scale
        self.__Z = z


if __name__ == "__main__":
    detector = Detector(100, 100)
    detector.generate_grid()
    # detector.generate_normal(scale=1000, mu=[-10, -10], sigma=[10, 10])
    detector.generate_random()
    detector.display_dist()
