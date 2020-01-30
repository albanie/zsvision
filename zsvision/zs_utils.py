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
