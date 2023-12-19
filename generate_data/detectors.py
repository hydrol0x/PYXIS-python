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

def generate_root_file(detector: Detector, file_path: str = ""):
    if not file_path:
        file_path = f"{detector.file_name()}.root"
    with uproot.recreate(file_path) as file:
        for bar_id in detector.bar_data:
            file[f"DetectorData/{bar_id}"] = detector.bar_data[bar_id]

# TODO: use somewhat believable heuristic or gaussian to generate the energy and time data
class PYXIS(Detector):
    def __init__(self, name: str, num_rows: int, num_cols: int, num_events: int):
        self.name = name
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.num_events = num_events # number of randomly generated detections, i.e number of counts on histogram
        self._dtype = np.dtype([('energy', 'float32'), ('time_left', 'float32'), ('time_right', 'float32')]) # pyxis data for each unit (bar), energy of detection & time it was detected on left and right PMT
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
                self._bar_data[bar_id] = np.zeros(self.num_events, dtype=self._dtype)     
    
    def file_name(self):
        # TODO: when more distributions add that in file name
        return f"PYXIS-{self.dist_type.name}-{self.name}-{self.num_rows}x{self.num_cols}"

    def _fill_random(self):
        for event in range(self.num_events):
            for i in range(self.num_rows):
                for j in range(self.num_cols):
                    bar_id = f"bar_{i}_{j}"
                    # TODO: add range specifier for uniform rnd
                    self._bar_data[bar_id][event]['energy'] = np.random.uniform(0, 10)
                    self._bar_data[bar_id][event]['time_left'] = np.random.uniform(0, 100)
                    self._bar_data[bar_id][event]['time_right'] = np.random.uniform(0, 100)
    
    def _fill_gaussian(self):
        print("filling gaussian")
        mu_1, sigma_1, mu_2, sigma_2, mu_3, sigma_3 = self.gaussian_param.values()
        for event in range(self.num_events):
            for i in range(self.num_rows):
                for j in range(self.num_cols):
                    bar_id = f"bar_{i}_{j}"
                    self._bar_data[bar_id][event]['energy'] = np.random.normal(mu_1, sigma_1)
                    self._bar_data[bar_id][event]['time_left'] = np.random.normal(mu_2, sigma_2)
                    self._bar_data[bar_id][event]['time_right'] = np.random.normal(mu_3, sigma_3)

if __name__ == "__main__":
    print("\n\n==== Debug/Dev ====\n\n")

    pyxis =PYXIS("test", 10, 10, 1000)
    pyxis.gaussian_param['sigma_energy'] = 5.0 # example of setting parameters
    pyxis.generate(Dist.GAUSSIAN)
    # pyxis.generate(Dist.RANDOM)
    generate_root_file(pyxis)