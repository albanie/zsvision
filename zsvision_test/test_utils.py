"""Test suite for zsvision/zs_utils.py
"""

from pathlib import Path
from zsvision.zs_utils import load_json_config, seconds_to_timestr


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
        1: "00:00:01.000",
        25 * 3600: "25:00:00.000",
    }
    for key, val in expected.items():
        ts = seconds_to_timestr(key)
        assert ts == val, f"Mismatch for {key}: {ts} vs {val}"
