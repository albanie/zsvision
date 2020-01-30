import os
import glob
import time
import pickle
import socket
import fnmatch
import functools
from pathlib import Path

import numpy as np
import msgpack

import msgpack_numpy as msgpack_np

msgpack_np.patch()


@functools.lru_cache(maxsize=64, typed=False)
def memcache(path):
    suffix = Path(path).suffix
    print(f"loading features >>> ", end=" ")
    tic = time.time()
    if suffix in {".pkl", ".pickle"}:
        res = pickle_loader(path)
    elif suffix == ".npy":
        res = np_loader(path)
    elif suffix == ".mp":
        res = msgpack_loader(path)
    else:
        raise ValueError(f"unknown suffix: {suffix} for path {path}")
    print(f"[Total: {time.time() - tic:.1f}s] ({socket.gethostname() + ':' + str(path)})")
    return res


def pickle_loader(pkl_path):
    tic = time.time()
    with open(pkl_path, "rb") as f:
        buffer = f.read()
        print(f"[I/O: {time.time() - tic:.1f}s]", end=" ")
        tic = time.time()
        data = pickle.loads(buffer, encoding="latin1")
        print(f"[deserialisation: {time.time() - tic:.1f}s]", end=" ")
    return data


def msgpack_loader(mp_path):
    """Msgpack provides a faster serialisation routine than pickle, so is preferable
    for loading and deserialising large feature sets from disk."""
    tic = time.time()
    with open(mp_path, "rb") as f:
        buffer = f.read()
        print(f"[I/O: {time.time() - tic:.1f}s]", end=" ")
        tic = time.time()
        data = msgpack_np.unpackb(buffer, object_hook=msgpack_np.decode, encoding="utf-8")
        print(f"[deserialisation: {time.time() - tic:.1f}s]", end=" ")
    return data


def np_loader(np_path, l2norm=False):
    with open(np_path, "rb") as f:
        data = np.load(f, encoding="latin1", allow_pickle=True)
    if isinstance(data, np.ndarray) and data.size == 1:
        data = data[()]  # handle numpy dict storage convnetion
    if l2norm:
        print("L2 normalizing features")
        if isinstance(data, dict):
            for key in data:
                feats_ = data[key]
                feats_ = feats_ / max(np.linalg.norm(feats_), 1E-6)
                data[key] = feats_
        elif data.ndim == 2:
            data_norm = np.linalg.norm(data, axis=1)
            data = data / np.maximum(data_norm.reshape(-1, 1), 1E-6)
        else:
            raise ValueError("unexpected data format {}".format(type(data)))
    return data


def set_nested_key_val(key, val, target):
    """Use a prefix key (e.g. key1.key2.key3) to set a value in a nested dict"""

    # escape periods in keys
    key = key.replace("_.", "&&")
    subkeys = key.split(".")
    subkeys = [x.replace("&&", ".") for x in subkeys]

    nested = target
    print("subkeys", subkeys)
    for subkey in subkeys[:-1]:
        try:
            nested = nested.__getitem__(subkey)
        except Exception as exception:
            print(subkey)
            raise exception
    orig = nested[subkeys[-1]]
    if orig == "":
        if val == "":
            val = 0
        else:
            val = str(val)
    elif isinstance(orig, bool):
        if val.lower() in {"0", "False"}:
            val = False
        else:
            val = bool(val)
    elif isinstance(orig, list):
        if isinstance(val, str) and "," in val:
            val = val.split(",")
            # we use the convention that a trailing comma indicates a single item list
            if len(val) == 2 and val[1] == "":
                val.pop()
            if val and not orig:
                raise ValueError(f"Could not infer correct type from empty original list")
            else:
                val = [type(orig[0])(x) for x in val]
        assert isinstance(val, list), "Failed to pass a list where expected"
    elif isinstance(orig, int):
        val = int(val)
    elif isinstance(orig, float):
        val = float(val)
    elif isinstance(orig, str):
        val = str(val)
    else:
        raise ValueError(f"unrecognised type: {type(val)}")
    nested[subkeys[-1]] = val
