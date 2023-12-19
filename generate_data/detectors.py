import numpy as np
import uproot
from abc import ABC, abstractmethod 

class Detector(ABC):
    @abstractmethod
    def file_name(self)-> str: pass

    @property
    @abstractmethod
    def dtype(self) -> np.dtype:
        pass

    @property
    @abstractmethod
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
        self.num_events = num_events
        self._dtype = np.dtype([('energy', 'float32'), ('time_left', 'float32'), ('time_right', 'float32')]) 
        self._bar_data = {}

    @property
    def dtype(self) -> np.dtype:
        return self._dtype

    @property
    def bar_data(self) -> dict[str, np.ndarray]:
        return self._bar_data
    
    def generate(self):
        # TODO: in future will be multiple dist for generate, prob through multiple methods i.e generate_random()
        self._init_bar_data()
        self._fill_random()

    def _init_bar_data(self):
        for i in range(self.num_rows):
            for j in range(self.num_cols):
                bar_id = f"bar_{i}_{j}"
                self._bar_data[bar_id] = np.zeros(self.num_events, dtype=self._dtype)     
    
    def file_name(self):
        # TODO: when more distributions add that in file name
        return f"PYXIS-rnd-{self.name}-{self.num_rows}x{self.num_cols}"

    def _fill_random(self):
        for event in range(self.num_events):
            for i in range(self.num_rows):
                for j in range(self.num_cols):
                    bar_id = f"bar_{i}_{j}"
                    # TODO: add range specifier for uniform rnd
                    self._bar_data[bar_id][event]['energy'] = np.random.uniform(0, 10)
                    self._bar_data[bar_id][event]['time_left'] = np.random.uniform(0, 100)
                    self._bar_data[bar_id][event]['time_right'] = np.random.uniform(0, 100)

pyxis =PYXIS("test", 10, 10, 1000)
pyxis.generate()
generate_root_file(pyxis)