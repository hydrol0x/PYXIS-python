import numpy as np
import uproot
from abc import ABC, abstractmethod 
from enum import Enum
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import multivariate_normal

Dist = Enum('Dist', 'UNKNOWN RANDOM GAUSSIAN')
    # PYXIS? for example would use some heuristic to generate realistic ish data

class Detector(ABC):
    @abstractmethod
    def file_name(self)-> str: pass

    @property
    @abstractmethod
    def dtype(self) -> np.dtype:
        pass

    @property
    @abstractmethod
    # TODO: prob change name to channel_data or unit_data or something thats more agnostic
    # If there is only 1channel this would just have 1 element (like i think rutherford would count as this)
    def bar_data(self) -> dict[str, np.ndarray]:
        pass

    @abstractmethod
    def generate(self):
        pass

def generate_root_file(detector, file_path=""):
    # Set default file path if not provided
    if not file_path:
        file_path = f"{detector.file_name()}.root"

    def write_data(file, data, path=""):
        # Iterate through the dictionary
        for key, value in data.items():
            current_path = f"{path}/{key}" if path else key
            if isinstance(value, dict):
                # If the value is a dictionary, recurse
                write_data(file, value, current_path)
            else:
                # Otherwise, write the data to the file
                file[current_path] = value

    with uproot.recreate(file_path) as file:
        write_data(file, detector.bar_data)


# TODO: use somewhat believable heuristic or gaussian to generate the energy and time data
class PYXIS(Detector):
    def __init__(self, name: str, num_rows: int, num_cols: int, bar_length: float, index_of_refraction: float, num_events: int):
        if num_rows<0:
            raise ValueError("`num_rows` must be greater than 0")
        if num_cols<0:
            raise ValueError("`num_cols` must be greater than 0")
        if num_events<0:
            raise ValueError("`num_events` must be greater than 0")

        self.name = name
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.num_events = num_events # number of randomly generated detections, i.e number of counts on histogram
        self.bar_length = bar_length # in meters
        
        self.bar_index_of_refraction = index_of_refraction # to calculate c in bar

        self._dtype = np.dtype([('Energy', 'float32'), ('Time', 'float32')]) # pyxis data for each unit (bar), energy of detection & time it was detected on left and right PMT
        # TODO: bar_data should store tuple for xy position, not the string for the branch name.
        # Branch name should be created a root file generation time i.e in generate() not overall, so its separated from data analysis
        self._bar_data = {} # dictionary of above data, holds data for each unit (i.e bar) which is then represented in separate ROOT tree branches
        self._mean_energies = {} # used to generate position distribution, contains mean energy for each bar

        # Parameters for normal/gaussian distributions
        self.gaussian_param = {
            "mu_energy":0.0,
            "sigma_energy":1.0,
            "mu_left_time":0.0,
            "sigma_left_time":1.0,
            "mu_right_time":0.0,
            "sigma_right_time":1.0,
        }

        # parameters for uniform random distribution range
        self.random_param = {
            "low":0.0,
            "high":50.0
            # TODO: finish
        }     

    @property
    def dtype(self) -> np.dtype:
        # Satisfies abstract class specification
        return self._dtype

    @property
    def bar_data(self) -> dict[str, np.ndarray]:
        # Satisfies abstract class specification
        return self._bar_data
    
    def generate(self, type:Dist):
        """
        Depending on specified distribution type, will generate mock bar data that matches that distribution.
        This can then be used to generate a ROOT file w/ data
        """
        # TODO: in future will be multiple dist for generate, prob through multiple methods i.e generate_random()
        self.dist_type = type

        # TODO: make loop to fill every bar in here to not keep rewriting it
        match type:
            case Dist.RANDOM:
                print("generating random")
                self._generate_random()
            case Dist.GAUSSIAN:
                print("generating gaussian")
                self._generate_gaussian()
            case _:
                raise ValueError(f"`{type}` is not a valid distribution type.")
    
    def _generate_gaussian(self):
        self._init_bar_data()
        self._fill_gaussian()

    def _generate_random(self):
        self._init_bar_data()
        self._fill_random()

    def _init_bar_data(self):
        """
        Initializes bar data with 0s for all values.
        """
        if self._bar_data:
            return
        for i in range(self.num_rows):
            for j in range(self.num_cols):
                # bar_id = f"bar_{i}_{j}"
                bar_id = (i, j)
                self._bar_data[bar_id] = {} # dictionary containing data for each side of the bar
                for k in range(2): # left and right -- 0 and 1
                    self._bar_data[bar_id][k] = np.zeros(self.num_events, dtype=self._dtype)     
    
    def file_name(self):
        return f"PYXIS-{self.dist_type.name}-{self.name}-{self.num_rows}x{self.num_cols}"

    def _fill_random(self):
        for event in range(self.num_events):
            for i in range(self.num_rows):
                for j in range(self.num_cols):
                    bar_id = (i,j)
                    # bar_id = f"bar_{i}_{j}"
                    # TODO: add range specifier for uniform rnd
                    for k in range(2):
                        self._bar_data[bar_id][k][event]['Energy'] = np.random.uniform(0, 100)
                        self._bar_data[bar_id][k][event]['Time'] = np.random.uniform(0, 100)
    
    def _fill_gaussian(self):
        print("filling gaussian")
        mu_1, sigma_1, mu_2, sigma_2, mu_3, sigma_3 = self.gaussian_param.values()
        for event in range(self.num_events):
            for i in range(self.num_rows):
                for j in range(self.num_cols):
                    # TODO: make the iteration over the actual pre zeroed dictionary, not this way!!!!
                    bar_id = (i,j)
                    # bar_id = f"bar_{i}_{j}"
                    for k in range(2): 
                        # NOTE: this for loop is not really necessary... honestly if we know its range(2) for 0 and 1 i should prob. just make it all one step i was just lazy
                        self._bar_data[bar_id][k][event]['Energy'] = np.random.normal(mu_1, sigma_1)
                        self._bar_data[bar_id][k][event]['Time'] = np.random.normal(mu_3, sigma_3)
    
    def read_root(self):
        # Read in root file into dict structure
        pass

    def _get_z_from_time(self, bar_data: np.ndarray):
        # 
        
        c = 299_792_458 # speed of light
        bar_vel = c / self.bar_index_of_refraction # velocity of light in the bar



        # return distance_left
    
    def _get_energies(self, bar_data: np.ndarray):
        # for now just taking mean of left and right
        energy_left: np.ndarray = bar_data[0]['Energy']
        energy_right: np.ndarray = bar_data[0]['Energy']

        return (energy_left+energy_right) / 2.0

    def get_position_timeseries(self):
        # first get xyz positions, from x,y of bar + calcualted fromt time difference 

        z = 0
        energy = 10
        # must loop over all events
        # energy_position_timeseries= [xy_coord + (self._get_z_from_time(data), energy) for xy_coord, data in self._bar_data.items()]
        energy_position_timeseries = []
        for xy_coord, data in self._bar_data.items():
            z_coords = self._get_z_from_time(data) # z_coord for each event for the particular bar
            energies = self._get_energies(data)
            for i, z in enumerate(z_coords):
                print(xy_coord + (z,), energies[i])
            # energy_position_timeseries.append(xy_coord + (self._get_z_from_time(data), energy))
        
        # Then use time recorded (or avg or smth) for the xyzt timeseries

    # def other_get_positions(self):
    #     pass
    #     data = self._bar_data
    #     mean_energies = {}
    #     for bar_id, channels in data.items():
    #         # Concatenate energies from both channels
    #         energies = np.concatenate([channels[0]['Energy'], channels[1]['Energy']])
    #         # Calculate the mean energy
    #         mean_energies[bar_id] = np.mean(energies)

    #     fit_data = np.array([[bar_id[0], bar_id[1], mean_energy] for bar_id, mean_energy in mean_energies.items()])
    #     mean = np.mean(fit_data, axis=0)
    #     cov = np.cov(fit_data, rowvar=False)
    #     model = multivariate_normal(mean=mean, cov=cov) 
    #     self._mean_energies = mean_energies
    #     self._model = model
    
    # def display(self):
    #     pass
    #     # Prepare scatter plot data
    #     self.position() # generate model and mean energies
    #     x_coords = [bar_id[0] for bar_id in self.mean_energies.keys()]
    #     y_coords = [bar_id[1] for bar_id in self.mean_energies.keys()]
    #     z_coords = [mean_energy for mean_energy in mean_energies.values()]

    #     # Create a meshgrid for the surface plot
    #     x_range = np.linspace(min(x_coords), max(x_coords), 30)
    #     y_range = np.linspace(min(y_coords), max(y_coords), 30)
    #     X, Y = np.meshgrid(x_range, y_range)

    #     # Calculate Z values (probability density) for the meshgrid
    #     Z = np.array([model.pdf([x, y, np.mean(z_coords)]) for x, y in zip(np.ravel(X), np.ravel(Y))])
    #     Z = Z.reshape(X.shape)

    #     # Set up the 3D plot
    #     fig = plt.figure()
    #     ax = fig.add_subplot(111, projection='3d')

    #     # Plot the scatter plot
    #     ax.scatter(x_coords, y_coords, z_coords, c='r', marker='o')

    #     # Plot the surface
    #     ax.plot_surface(X, Y, Z, rstride=1, cstride=1, alpha=0.5, cmap='viridis')

    #     # Labeling the axes
    #     ax.set_xlabel('X Coordinate')
    #     ax.set_ylabel('Y Coordinate')
    #     ax.set_zlabel('Mean Energy / Probability Density')

    #     # Show the plot
    #     plt.show() 
                
class RUTHERFORD(Detector):
    def __init__(self, name: str, num_events: int):
        if num_events<0:
            raise ValueError("`num_events` must be greater than 0")

        self.name = name
        self.num_events = num_events # number of randomly generated detections, i.e number of counts on histogram

        self.angulardist_param = {
        }
    

        

if __name__ == "__main__":

    import numpy as np
    from scipy.stats import multivariate_normal

    # Example data (assuming this is similar to your data structure)
    
    pyxis =PYXIS("test", 1, 1, 1, .5, 3)
    pyxis.gaussian_param['sigma_energy'] = 5.0 # example of setting parameters
    pyxis.generate(Dist.GAUSSIAN)
    pyxis.get_position_timeseries()
    # generate_root_file(pyxis)
    # pyxis.display()

    # The model is now a fitted multivariate normal distribution

    # print("\n\n==== Debug/Dev ====\n\n")

   
    # pyxis.generate(Dist.RANDOM)
    # pyxis.generate(Dist.RANDOM)
    # generate_root_file(pyxis)