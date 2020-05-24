"""Test suite for zsvision/zs_utils.py
"""

import json
import pickle
import tempfile
from pathlib import Path

import numpy as np
import msgpack_numpy as msgpack_np
from zsvision.zs_utils import (
    memcache,
    load_json_config,
    seconds_to_timestr,
    dump_hickle_escaped
)


def test_load_json_config_inheritance():
    child_config_path = Path("zsvision_test/misc/dummy_child_config.json")
    config = load_json_config(child_config_path)
    expected = {
        "property-a": "a-parent",
        "property-b": "b-child",
        "property-c": "c-child"
    }
    for key, val in expected.items():
        assert config[key] == val, f"Mismatch for {key}: {config[key]} vs {val}"


def test_seconds_to_timestr():
    expected = {
        0: "00:00:00.000",
        0.1: "00:00:00.100",
        1: "00:00:01.000",
        25 * 3600: "25:00:00.000",
    }
    for key, val in expected.items():
        ts = seconds_to_timestr(key)
        assert ts == val, f"Mismatch for {key}: {ts} vs {val}"


def test_memcache():

    def pickle_dumper(obj, path):
        with open(path, "wb") as f:
            pickle.dump(obj, f)

    def numpy_dumper(obj, path):
        np.save(path, obj)

    def msgpack_dumper(obj, path):
        with open(path, "wb") as f:
            f.write(msgpack_np.packb(obj, use_bin_type=True))

    def json_dumper(obj, path):
        with open(path, "w") as f:
            json.dump(obj, f)

    storage_map = {
        "pickle": {
            "dumper": pickle_dumper,
            "suffix": ".pickle",
        },
        "hickle": {
            "dumper": dump_hickle_escaped,
            "suffix": ".hickle",
        },
        "numpy": {
            "dumper": numpy_dumper,
            "suffix": ".npy",
        },
        "msgpack": {
            "dumper": msgpack_dumper,
            "suffix": ".mp",
        },
        "json": {
            "dumper": json_dumper,
            "suffix": ".json",
        }
    }
    data = [
        [],
        ["abc"],
        {str(ii): np.random.rand(ii).tolist() for ii in range(10)},
        {f"parent/{ii}": np.random.rand(ii).tolist() for ii in range(10)},
    ]
    for sample_data in data:
        for storage_type, subdict in storage_map.items():
            tmp = tempfile.NamedTemporaryFile(suffix=subdict['suffix'], delete=0)
            path = Path(tmp.name)
            print(f"Testing memcache for {storage_type}")
            subdict["dumper"](sample_data, path)
            res = memcache(path)
            msg = f"{storage_map} serialization did not preserve: {sample_data}"
            np.testing.assert_equal(res, sample_data), msg
            path.unlink()


if __name__ == "__main__":
    test_memcache()
