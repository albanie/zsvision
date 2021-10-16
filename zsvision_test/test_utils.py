"""Test suite for zsvision/zs_utils.py
"""

import os
import json
import pickle
import logging
import tempfile
import torch
from pathlib import Path

import numpy as np
import hickle
import msgpack_numpy as msgpack_np
from zsvision.zs_utils import (
    BlockTimer,
    memcache,
    load_json_config,
    load_yaml_config,
    parse_tree_layout,
    seconds_to_timestr,
    list_visible_gpu_types,
    quote_and_escape_ffmpeg_path
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


def test_load_yaml_config_inheritance():
    child_config_path = Path("zsvision_test/misc/dummy_child_config.yaml")
    config = load_yaml_config(child_config_path)
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

    def hickle_dumper(obj, path):
        hickle.dump(obj, path)
    
    def torch_dumper(obj, path):
        torch.save(obj, path)

    storage_map = {
        "pickle": {
            "dumper": pickle_dumper,
            "suffix": ".pickle",
        },
        "hickle": {
            "dumper": hickle_dumper,
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
        },
        "torch": {
            "dumper": torch_dumper,
            "suffix": ".ptn",
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
            for verbose in (True, False):
                tmp = tempfile.NamedTemporaryFile(suffix=subdict['suffix'], delete=0)
                path = Path(tmp.name)
                print(f"Testing memcache (verbose: {verbose}) for {storage_type}")
                subdict["dumper"](sample_data, path)
                res = memcache(path, verbose=verbose)
                msg = f"{storage_map} serialization did not preserve: {sample_data}"
                np.testing.assert_equal(res, sample_data), msg
                path.unlink()


def test_list_visible_gpu_types():
    KNOWN_GPUS = {
        "GeForce GTX 1080 Ti",
    }
    visible_gpus = list_visible_gpu_types()
    for visible_gpu in visible_gpus:
        msg = f"GPU: {visible_gpu} was not a member of known gpus: {KNOWN_GPUS}"
        assert visible_gpu in KNOWN_GPUS, msg


def test_quote_and_escape_ffmpeg_path():
    test_cases = (
        {"input": "/a/b/c.ext", "expected": "'/a/b/c.ext'"},
        {"input": "/a/b/c d.ext", "expected": "'/a/b/c d.ext'"},
        {"input": "/a/b/c'd.ext", "expected": '"/a/b/c\'d.ext"'},
        {"input": "/a/b/c$d.ext", "expected": r"'/a/b/c\$d.ext'"},
        {"input": "/a/b/c%d.ext", "expected": r"'/a/b/c\%d.ext'"},
    )
    for test_case in test_cases:
        output = quote_and_escape_ffmpeg_path(test_case["input"])
        msg = f"Expected {test_case['expected']}, but found {output}"
        assert output == test_case["expected"], msg


def test_parse_tree_layout():
    test_cases = (
        (
            "a/b/c.ext",
        ),
        (
            "a/b/c.ext",
            "a/b/d.ext",
        ),
        (
            "a/b/c.ext",
            "a/b/c/d.ext",
            "a/b/c/d/e.ext",
        ),
        (
            "a /b /c .ext",
        ),
    )
    tmp_tree_output = tempfile.NamedTemporaryFile(delete=False)
    cwd = Path.cwd()
    for test_case in test_cases:
        with tempfile.TemporaryDirectory() as tmp_dir:
            for rel_path in test_case:
                path = Path(tmp_dir) / rel_path
                path.parent.mkdir(exist_ok=True, parents=True)
                path.touch()
            # Move to the temporary directory to create a relative path strcuture
            os.system(f"cd {tmp_dir}; tree > {tmp_tree_output.name} ; cd {cwd}")
            layout = parse_tree_layout(Path(tmp_tree_output.name))

            expected = set()
            for rel_path in test_case:
                for prefix_length in range(len(rel_path)):
                    expected.add(Path(*(Path(rel_path).parts[:prefix_length])))

            assert expected == layout, f"Expected {layout} to match {expected}"
    os.unlink(tmp_tree_output.name)


def test_BlockTimer():
    with BlockTimer(msg="testing block timer"):
        pass
    logger = logging.getLogger(name="dummy logger")
    logging.basicConfig()
    logger.setLevel(logging.INFO)
    with BlockTimer(msg="testing block timer with logger", logger=logger):
        pass


if __name__ == "__main__":
    test_quote_and_escape_ffmpeg_path()
    test_parse_tree_layout()
    test_BlockTimer()
    test_memcache()
    test_load_json_config_inheritance()
    test_load_yaml_config_inheritance()
