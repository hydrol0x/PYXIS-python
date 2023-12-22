import numpy as np
import uproot
from abc import ABC, abstractmethod 
from enum import Enum

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
    def __init__(self, name: str, num_rows: int, num_cols: int, num_events: int):
        self.name = name
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.num_events = num_events # number of randomly generated detections, i.e number of counts on histogram
        self._dtype = np.dtype([('Energy', 'float32'), ('Time', 'float32')]) # pyxis data for each unit (bar), energy of detection & time it was detected on left and right PMT
        self._bar_data = {} # dictionary of above data, holds data for each unit (i.e bar) which is then represented in separate ROOT tree branches

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
        match type:
            case Dist.RANDOM:
                print("generating random")
                self._generate_random()
            case Dist.GAUSSIAN:
                print("generating gaussian")
                self._generate_gaussian()
            case _:
                print("ERROR: not a valid distribution type")
    
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
                bar_id = f"bar_{i}_{j}"
                self._bar_data[bar_id] = {} # dictionary containing data for each side of the bar
                for i in range(2): # left and right -- 0 and 1
                    self._bar_data[bar_id][i] = np.zeros(self.num_events, dtype=self._dtype)     
    
    def file_name(self):
        return f"PYXIS-{self.dist_type.name}-{self.name}-{self.num_rows}x{self.num_cols}"

    def _fill_random(self):
        for event in range(self.num_events):
            for i in range(self.num_rows):
                for j in range(self.num_cols):
                    bar_id = f"bar_{i}_{j}"
                    # TODO: add range specifier for uniform rnd
                    for i in range(2):
                        self._bar_data[bar_id][i][event]['Energy'] = np.random.uniform(0, 10)
                        self._bar_data[bar_id][i][event]['Time'] = np.random.uniform(0, 100)
    
    def _fill_gaussian(self):
        print("filling gaussian")
        mu_1, sigma_1, mu_2, sigma_2, mu_3, sigma_3 = self.gaussian_param.values()
        for event in range(self.num_events):
            for i in range(self.num_rows):
                for j in range(self.num_cols):
                    bar_id = f"bar_{i}_{j}"
                    for i in range(2): 
                        # NOTE: this for loop is not really necessary... honestly if we know its range(2) for 0 and 1 i should prob. just make it all one step i was just lazy
                        self._bar_data[bar_id][i][event]['Energy'] = np.random.normal(mu_1, sigma_1)
                        self._bar_data[bar_id][i][event]['Time'] = np.random.normal(mu_3, sigma_3)

if __name__ == "__main__":
    print("\n\n==== Debug/Dev ====\n\n")

    pyxis =PYXIS("test", 10, 10, 1_000)
    pyxis.gaussian_param['sigma_energy'] = 5.0 # example of setting parameters
    pyxis.generate(Dist.GAUSSIAN)
    pyxis.generate(Dist.RANDOM)
    # pyxis.generate(Dist.RANDOM)
    generate_root_file(pyxis)