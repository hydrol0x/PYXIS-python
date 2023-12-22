import pytest
from detectors.detectors import PYXIS, Dist, generate_root_file
import numpy as np
import os

class TestPYXIS:
    @pytest.fixture
    def pyxis_instance(self):
        return PYXIS("test", 10, 10, 1000)

    def test_initialization(self, pyxis_instance):
        assert pyxis_instance.name == "test"
        assert pyxis_instance.num_rows == 10
        assert pyxis_instance.num_cols == 10
        assert pyxis_instance.num_events == 1000

    def test_dtype_property(self, pyxis_instance):
        assert pyxis_instance.dtype == np.dtype([('Energy', 'float32'), ('Time', 'float32')])

    def test_bar_data_property(self, pyxis_instance):
        assert isinstance(pyxis_instance.bar_data, dict)
        assert pyxis_instance.bar_data == {}

    @pytest.mark.parametrize("dist_type", [Dist.RANDOM, Dist.GAUSSIAN])
    def test_generate_method(self, pyxis_instance, dist_type):
        pyxis_instance.generate(dist_type)
        assert pyxis_instance.dist_type == dist_type
        assert pyxis_instance.bar_data

    def test_generate_method_invalid_type(self, pyxis_instance):
        with pytest.raises(ValueError):
            pyxis_instance.generate("invalid_type")

    def test_file_name_method(self, pyxis_instance):
        pyxis_instance.generate(Dist.GAUSSIAN)
        expected_name = f"PYXIS-GAUSSIAN-test-10x10"
        assert pyxis_instance.file_name() == expected_name

    def test_generate_root_file(self, pyxis_instance, tmp_path):
        pyxis_instance.generate(Dist.RANDOM)
        file_path = tmp_path / "test.root"
        generate_root_file(pyxis_instance, str(file_path))
        assert os.path.exists(file_path)

@pytest.mark.parametrize("num_rows, num_cols, num_events", [(10, 10, 1000), (1, 1, 1), (0, 0, 0)])
def test_data_generation(num_rows, num_cols, num_events):
    pyxis = PYXIS("test", num_rows, num_cols, num_events)
    pyxis.generate(Dist.RANDOM)
    for bar_data in pyxis.bar_data.values():
        for side_data in bar_data.values():
            assert len(side_data) == num_events
            assert side_data.dtype == pyxis.dtype

def test_negative_values_initialization():
    with pytest.raises(ValueError):
        PYXIS("test", -1, -1, -1)
