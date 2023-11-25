# TODO: add error checking (maybe use @decorator to try except)
import warnings
import numpy as np
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math
import csv

# TODO: add docstrings to future methods or functions


def is_square(i: int) -> bool:
    return i == math.isqrt(i) ** 2


def generate_grid(size: int, per_side: int):
    # Generates an interval of points that is evenly spaced (numpy uses complex num. to accomplish)
    """Generate an interval of evenly spaced points to fill the detector

    Args:
        size (int): side length (cm) of detector
        # TODO: consolidate per_side into this func, i.e calculate within this func
        per_side (int): number of points per side

    Returns:
        np.array: numpy array
    """
    return (np.mgrid[-size/2:size /
                     2:complex(per_side, 1), -size/2:size/2:complex(per_side, 1)])


def coordinates_from_axes(xx, yy, zz):
    """Generate coordinate pairs for each point on detector with the corresponding z-val from point distribution. Z represents counts.

    Args:
        xx np.array: numpy array from meshgrid
        yy np.array: numpy array from meshgrid
        zz np.array: numpy array from distribution generation

    Returns:
        list: List of coordinates
    """
    output = []
    for x, y, z in zip(xx, yy, zz):
        for i in range(len(x)):
            output.append((x[i], y[i], z[i]))
    return output


def runtime_warn(message):
    warnings.warn(message, RuntimeWarning, stacklevel=2)


class Detector:
    def __init__(self, size: int, num_points: int):
        if (not is_square(num_points)):
            runtime_warn(
                "numPoints not square; results rounded to square number")
        self.name = ""
        self.per_side = int(num_points**.5)
        self.size = size
        self.area = size**2
        self.num_points = int((num_points)**.5)**2
        self.point_spacing = self.size/(self.per_side - 1)
        self.distribution_function = None

    def generate_grid(self):
        """Generate the detector grid of coordinate points
        """
        # Generate a grid from -size/2 to size/2
        size = self.size
        per_side = self.per_side
        __X, __Y = np.mgrid[-size/2:size /
                            2: complex(self.per_side, 1), -self.size/2: self.size/2: complex(self.per_side, 1)]
        self.grid = np.vstack((__X.flatten(), __Y.flatten())).T
        self.__X, self.__Y = __X, __Y

    def display_grid(self):
        """Display detector coordinate grid (2d)
        """
        plt.plot(self.__X, self.__Y, marker='.', color='k', linestyle='none')
        plt.show()

    def display_dist(self, wireframe=False):
        """Display a plot of the detector distribution of counts (3d)

        Args:
            wireframe (bool, optional): Display a wireframe on plot. Defaults to False.
        """
        distrib = self.distribution_function
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        plt.suptitle(
            f"Detector '{self.name}' distribution ({self.distribution_function})")
        ax.set_xlabel('x cm', fontsize=10)
        ax.set_ylabel('y cm', fontsize=10)
        ax.set_zlabel('Counts', fontsize=10)
        x = self.__X
        y = self.__Y
        z = self.__Z
        title = f"{self.size}cm×{self.size}cm,  # pts. {self.num_points}, "
        if (distrib == "Normal"):
            title += f"µ={self.__mu}, σ={self.__sigma}"
            if self.num_points <= 1000:
                ax.scatter(x, y, z)
            else:
                if wireframe:
                    ax.plot_wireframe(x, y, z)
                ax.plot_surface(x, y, z)
        elif distrib == "Random":
            title += f"max={self.__random_high}, min={self.__random_low}"
            ax.scatter(x, y, z)
        else:
            ax.scatter(x, y, z)
        plt.title(title, fontsize=8)
        plt.show()

    def generate_random(self, low=0, high=10):
        """Generate a random point for each detector grid point. Points fall from low-high

        Args:
            low (int, optional): Lowest possible rnd value. Defaults to 0.
            high (int, optional): Highest possible rnd value. Defaults to 10.
        """
        self.distribution_function = "Random"
        self.__random_low = low
        self.__random_high = high
        grid = self.grid

        x, y = self.__X, self.__Y
        z = np.random.uniform(low=low, high=high,
                              size=grid.shape[0]).reshape(x.shape)
        self.__Z = z
        self.distribution = coordinates_from_axes(x, y, z)

    def generate_normal(self, mu=[0.0, 0.0], sigma=[3, 3], scale=1):
        """Generates a normal distribution of counts on the detector.

        Args:
            mu (list, optional): Mean of the distribution. Defaults to [0.0, 0.0].
            sigma (list, optional): Standard Deviation of the distribution. Defaults to [3, 3].
            scale (int, optional): Scale multiplier. Defaults to 1.
        """
        self.distribution_function = "Normal"
        self.__mu = mu
        self.__sigma = sigma
        grid = self.grid

        mu = np.array(mu)
        sigma = np.array(sigma)
        covariance = np.diag(sigma**2)

        x, y = self.__X, self.__Y
        z = multivariate_normal.pdf(
            grid, mean=mu, cov=covariance).reshape(x.shape)
        # Normal gaussian dist. has an area under curve of 1, so all the values will be low
        # Group focusing on math can determine how to scale graph
        z *= scale
        self.__Z = z
        self.distribution = coordinates_from_axes(x, y, z)
# TODO: maybe add generate random from nomral (ie a random sample from a normally distributed rand. var)

    def print_properties(self):
        """Generates a formatted output of detector properties.
        """
        print(
            "Size: {size}cm\nArea: {area}cm²\nNum pts: {num_points}\nPt. spacing: ~{point_spacing:.3f}cm\nPt. Distrib: {point_distribution}".format(size=self.size, area=self.area, num_points=self.num_points, point_spacing=self.point_spacing, point_distribution=self.distribution_function))

    def output_distribution(self, filename="distribution"):
        # TODO: maybe change format of output
        # Writing to file
        with open(f"{filename}.csv", "w", newline='') as outfile:
            # Writing data to a file
            writer = csv.writer(outfile, dialect='excel', delimiter=',')
            for coordinate in self.distribution:
                writer.writerow(coordinate)


if __name__ == "__main__":
    # Initiate each Detector
    detector_norm = Detector(10, 1000)
    detector_rand = Detector(10, 1000)

    # Generate the grid for detectors (in future this might be done on init, will see)
    detector_norm.generate_grid()
    detector_rand.generate_grid()

    # generate respective distribution (and distribution coordinates)
    detector_norm.generate_normal(mu=[0.0, 0.0], sigma=[3, 3], scale=10000)
    detector_rand.generate_random(low=0, high=1000)

    # Display a graph of each dist
    # (note: have to close out of one graph before next one displays)
    detector_norm.display_dist()
    detector_rand.display_dist()

    print("Normal dist. points: ")
    print(detector_norm.distribution)
    detector_norm.output_distribution("normal_distribution")

    print("\n Random dist. points")
    print(detector_rand.distribution)
    detector_rand.output_distribution("random_distribution")
