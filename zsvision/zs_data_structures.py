"""Datastructures for interacting with computer vision features.

The motivation for `ExpertStore` is to use memory backed by an array to avoid
fragmentation (rather than using a standard python dictionary), but still offer a
key-value accessor interface.

Relative comparison:
    Memory needs for dict: 417.1 MB
    Memory needs for np: 409.6 MB
    Memory needs for expert_store: 412.0 MB
"""

import argparse
import humanize
import pickle
import numpy as np
from typing import List


class ExpertStore:

    def __init__(self, keylist: List[str], dim: int, dtype: np.dtype = np.float16):
        self.keys = keylist
        self.dim = dim
        self.store_dtype = dtype
        self.store = np.zeros((len(keylist), dim), dtype=dtype)
        self.keymap = {}
        self.missing = set()

        for idx, key in enumerate(keylist):
            self.keymap[key] = idx

    def __setitem__(self, key, value):
        idx = self.keymap[key]
        if isinstance(value, np.ndarray):
            # non-nan values must be vectors of the appropriate size
            assert value.size == self.dim, f"cannot set value with size {value.size}"
        else:
            assert np.isnan(value)
        self.store[idx] = value

    def __getitem__(self, key):
        return self.store[self.keymap[key]]

    def todict(self):
        """Convert the current datastructure into a vanilla python dictionary

        Returns:
            a dictionary with the same keys and values as the current object.
        """
        return {key: self[key] for key in self.keymap}

    def __len__(self):
        return len(self.keys)

    def __repr__(self):
        keep_samples = 3
        samples = list(self.keymap.items())[:keep_samples]
        sample_str = "\n".join([f"{key}: {val}" for key, val in samples])
        summary = (
            f"ExpertStore object with {len(self.keys)} features (dim: {self.dim})"
            f" (storage is using {humanize.naturalsize(self.store.nbytes)})"
            f"\nFirst {keep_samples} elements of keymap: \n{sample_str}"
        )
        return summary


def gen_dict_store(keylist, dim):
    store = dict()
    for key in keylist:
        store[key] = np.random.rand(1, dim).astype(np.float16)
    return store


class HashableDict(dict):
    def __hash__(self):
        return hash(frozenset(self))


class HashableOrderedDict(dict):
    def __hash__(self):
        return hash(frozenset(self))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dim", type=int, default=2048)
    parser.add_argument("--num_keys", type=int, default=100000)
    args = parser.parse_args()
    keylist = [str(x) for x in range(args.num_keys)]
    for store_name in "dict", "np", "expert_store":
        if store_name == "dict":
            store = gen_dict_store(keylist=keylist, dim=args.dim)
        elif store_name == "np":
            store = np.random.rand(len(keylist), args.dim).astype(np.float16)
        elif store_name == "expert_store":
            store = ExpertStore(keylist=keylist, dim=args.dim)
            print(store)
        serialised = pickle.dumps(store)
        print(f"Memory used by {store_name}: {humanize.naturalsize(len(serialised))}")


if __name__ == "__main__":
    main()
