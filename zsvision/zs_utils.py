import sys
import time
import pickle
import socket
import json
import functools
from typing import Union
from pathlib import Path
import scipy.io as spio
from typeguard import typechecked

import numpy as np
import msgpack_numpy as msgpack_np
import zsvision.zs_data_structures

msgpack_np.patch()


@functools.lru_cache(maxsize=64, typed=False)
@typechecked
def memcache(path: Union[Path, str]):
    path = Path(path)
    suffix = path.suffix
    print(f"loading data from {path} ({socket.gethostname()})", end=" ", flush=True)
    tic = time.time()
    if suffix in {".pkl", ".pickle"}:
        res = pickle_loader(path)
    elif suffix == ".npy":
        res = np_loader(path)
    elif suffix == ".mp":
        res = msgpack_loader(path)
    elif suffix == ".json":
        with open(path, "r") as f:
            res = json.load(f)
    elif suffix == ".mat":
        res = loadmat(path)
    else:
        raise ValueError(f"unknown suffix: {suffix} for path {path}")
    print(f"[Total: {time.time() - tic:.1f}s]")
    return res


@typechecked
def support_old_pickles(buffer: bytes) -> object:
    try:
        data = pickle.loads(buffer, encoding="latin1")
    except ModuleNotFoundError as exception:
        if "datastructures" in exception.msg:
            sys.modules['datastructures'] = zsvision.zs_data_structures
            data = pickle.loads(buffer, encoding="latin1")
    return data


@typechecked
def pickle_loader(pkl_path: Path, backwards_compatible: bool = True) -> object:
    """Deserialise object from pickle.

    Args:
        pkl_path: the location of the path where the pickle path is stored
        backwards_compatible: if true, support old pickle formats used with the.
            ExpertStore format

    Return:
        The deserialised object.
    """
    tic = time.time()
    with open(pkl_path, "rb") as f:
        buffer = f.read()
        print(f"[I/O: {time.time() - tic:.1f}s]", end=" ")
        tic = time.time()
        if backwards_compatible:
            data = support_old_pickles(buffer)
        else:
            data = pickle.loads(buffer, encoding="latin1")
        print(f"[deserialisation: {time.time() - tic:.1f}s]", end=" ")
    return data


@typechecked
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


def loadmat(filename):
    """This function should be called instead of direct spio.loadmat
    as it cures the problem of not properly recovering python dictionaries
    from mat files. It calls the function check keys to cure all entries
    which are still mat-objects.

    The function is heavily based on this reference:
    https://stackoverflow.com/a/29126361

    Args:
        filename (str): the location of the .mat file to load
    
    Returns:
        (dict): a parsed .mat file in the form of a python dictionary.
    """
    def _check_keys(d):
        """Checks if entries in dictionary are mat-objects. If yes
        todict is called to change them to nested dictionaries
        """
        for key in d:
            if isinstance(d[key], spio.matlab.mio5_params.mat_struct):
                d[key] = _todict(d[key])
            elif isinstance(d[key], np.ndarray):
                d[key] = _tolist(d[key])
            else:
                pass
        return d

    def _todict(matobj):
        """A recursive function which constructs from matobjects nested dictionaries
        """
        d = {}
        for strg in matobj._fieldnames:
            elem = matobj.__dict__[strg]
            if isinstance(elem, spio.matlab.mio5_params.mat_struct):
                d[strg] = _todict(elem)
            elif isinstance(elem, np.ndarray):
                d[strg] = _tolist(elem)
            else:
                d[strg] = elem
        return d

    def _tolist(ndarray):
        """A recursive function which constructs lists from cellarrays
        (which are loaded as numpy ndarrays), recursing into the elements
        if they contain matobjects or are non-numeric.
        """
        if np.issubdtype(ndarray.dtype, np.number):
            return ndarray
        elem_list = []
        for sub_elem in ndarray:
            if isinstance(sub_elem, spio.matlab.mio5_params.mat_struct):
                elem_list.append(_todict(sub_elem))
            elif isinstance(sub_elem, np.ndarray):
                elem_list.append(_tolist(sub_elem))
            else:
                elem_list.append(sub_elem)
        return elem_list

    data = spio.loadmat(filename, struct_as_record=False, squeeze_me=True)
    return _check_keys(data)


@functools.lru_cache(maxsize=64, typed=False)
def concat_features(feat_paths, axis):
    aggregates = [memcache(x) for x in feat_paths]
    tic = time.time()
    msg = "expected to concatenate datastructures of a single type"
    assert len(set(type(x) for x in aggregates)) == 1, msg
    if isinstance(aggregates[0], dict):
        keys = aggregates[0]  # for now, we assume that all aggregates share keys
        merged = {}
        for key in keys:
            merged[key] = np.concatenate([x[key] for x in aggregates], axis=axis)
    elif isinstance(aggregates[0], zsvision.zs_data_structures.ExpertStore):
        dims, stores = [], []
        keys = aggregates[0].keys
        for x in aggregates:
            dims.append(x.dim)
            stores.append(x.store)
            try:
                assert x.keys == keys, "all aggregates must share identical keys"
            except Exception as E:
                print(E)
                import ipdb; ipdb.set_trace()
        msg = "expected to concatenate ExpertStores with a common dimension"
        assert len(set(dims)) == 1, msg
        dim = dims[0]
        merged = zsvision.zs_data_structures.ExpertStore(keys, dim=dim)
        merged.store = np.concatenate(stores, axis=axis)
    else:
        raise ValueError(f"Unknown datastructure: {type(aggregates[0])}")
    # Force memory clearance
    for aggregate in aggregates:
        del aggregate
    print("done in {:.3f}s".format(time.time() - tic))
    return merged


class BlockTimer:
    """A minimal inline codeblock timer"""
    def __init__(self, msg, precise=False, mute=False):
        self.msg = msg
        self.mute = mute
        self.precise = precise

    def __enter__(self):
        self.start = time.time()
        print(f"{self.msg}...", end="", flush=True)
        return self

    def __exit__(self, *args):
        if self.precise:
            total = f"{time.time() - self.start:.3f}s"
        else:
            total = time.strftime('%Hh%Mm%Ss', time.gmtime(time.time() - self.start))
        if not self.mute:
            print(f" took {total}")
