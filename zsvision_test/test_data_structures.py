"""Test suite for zsvision/zs_data_structures.py
"""

import numpy as np
from zsvision.zs_data_structures import FeatureCache


def test_feature_cache():
    dim = 2048
    num_keys = 1000
    keylist = [str(x) for x in range(num_keys)]
    cache = FeatureCache(keylist=keylist, dim=dim)

    # check that we can add and retrieve (key, value) pairs
    sample_key = "0"
    sample_feat = np.ones(dim, dtype=np.float16)
    cache[sample_key] = sample_feat

    assert np.array_equal(sample_feat, cache[sample_key]), (
        f"Expected to retrieve {sample_feat} for {sample_key}"
    )

    # check that cache hits act as expected
    assert sample_key in cache, f"Expected cache hit for {sample_key}"

    unused_key = "1"
    assert unused_key not in cache, f"Did not expect cache hit for {unused_key}"

    unknown_key = "-1"
    assert unknown_key not in cache, f"Expected cache miss for {unused_key}"


if __name__ == "__main__":
    test_feature_cache()
