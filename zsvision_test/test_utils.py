"""Test suite for zsvision/zs_utils.py
"""

from pathlib import Path
from zsvision.zs_utils import load_json_config


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
