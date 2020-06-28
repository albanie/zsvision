import sys
import json
import time
import pickle
import socket
import numbers
import functools
import unicodedata
import subprocess
from typing import Dict, List, Union
from pathlib import Path

import numpy as np
import hickle
import scipy.io as spio
import msgpack_numpy as msgpack_np
from beartype import beartype
from mergedeep import Strategy, merge
from typeguard import typechecked
from beartype.cave import AnyType
import zsvision.zs_data_structures


@functools.lru_cache(maxsize=64, typed=False)
@typechecked
def memcache(path: Union[Path, str]):
    path = Path(path)
    suffix = path.suffix
    print(f"loading data from {path} ({socket.gethostname()})", end=" ", flush=True)
    tic = time.time()
    if suffix in {".pkl", ".pickle"}:
        res = pickle_loader(path)
    elif suffix in {".hkl", ".hickle"}:
        res = hickle.load(path)
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


@beartype
def support_old_pickles(buffer: bytes) -> object:
    try:
        data = pickle.loads(buffer, encoding="latin1")
    except ModuleNotFoundError as exception:
        if "datastructures" in str(exception.msg):
            sys.modules['datastructures'] = zsvision.zs_data_structures
            data = pickle.loads(buffer, encoding="latin1")
    return data


@beartype
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


@beartype
def msgpack_loader(mp_path: Path):
    """Msgpack provides a faster serialisation routine than pickle, so is preferable
    for loading and deserialising large feature sets from disk."""
    tic = time.time()
    with open(mp_path, "rb") as f:
        buffer = f.read()
        print(f"[I/O: {time.time() - tic:.1f}s]", end=" ")
        tic = time.time()
        data = msgpack_np.unpackb(buffer, raw=False)
        print(f"[deserialisation: {time.time() - tic:.1f}s]", end=" ")
    return data


@beartype
def np_loader(np_path: Path, l2norm=False):
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


@beartype
def set_nested_key_val(key: str, val: AnyType, target: dict):
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


@beartype
def loadmat(src_path: Path) -> dict:
    """This function should be called instead of direct spio.loadmat as it addresses the
    problem of not properly recovering python dictionaries from mat files. It calls the
    function check keys to cure all entries which are still mat-objects.

    The function is heavily based on this reference:
    https://stackoverflow.com/a/29126361

    Args:
        src_path: the location of the .mat file to load

    Returns:
        a parsed .mat file in the form of a python dictionary.
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

    data = spio.loadmat(src_path, struct_as_record=False, squeeze_me=True)
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
        self.start = None

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


@typechecked
def find_ancestors(cfg_fname: Path) -> List[Dict]:
    """Search the hierarchy specified by the `inherit_from` attribute of a json config
    via post-order traversal.

    Args:
        cfg_fname: the location of the json config file

    Returns:
        a list of loaded configs in the order specified by the inheritance.
    """
    with open(cfg_fname, "r") as f:
        config = json.load(f)
    ancestors = []
    if "inherit_from" in config:
        immediate_ancestors = config["inherit_from"].split(",")
        for immediate_ancestor in immediate_ancestors:
            ancestors.extend(find_ancestors(Path(immediate_ancestor)))
    ancestors.append(config)
    return ancestors


@beartype
def load_json_config(cfg_fname: Path) -> dict:
    """Load a json configuration file into memory.

    Args:
        cfg_fname: the location of the json config file

    Returns:
        the loaded configuration

    NOTES: A json file may include an `inherit_from`: "<path>" key, value pair which
    points to a list of templates from which to inherit default values.  Inheritance
    specifiers are traversed in increasing order of importance, from left to right.
    E.g. given
        "inherit_from": "path-to-A,path-to-B",
    the values of B will override the values of A.
    """
    ancestors = find_ancestors(cfg_fname)
    config = ancestors.pop()
    ancestors = reversed(ancestors)
    for ancestor in ancestors:
        merge(ancestor, config, strategy=Strategy.REPLACE)
        config = ancestor
    return config


@beartype
def seconds_to_timestr(secs: numbers.Number) -> str:
    """Convert a total number of seconds into a formatted time string.

    Arguments:
        secs: the total number of seconds

    Returns:
        a formatted time (HH:MM:SS.mmm)

    NOTE: Probably this function is not needed. But I refuse to spend more of my life
    looking at datetime/time/strftime combinations.
    """
    assert secs >= 0, "Expected a non-negative number of seconds"
    mins, secs = divmod(secs, 60)
    hours, mins = divmod(mins, 60)
    ms = secs - int(secs)
    return f"{int(hours):02d}:{int(mins):02d}:{int(secs):02d}.{int(ms * 1000):03d}"


@typechecked
def list_visible_gpu_types() -> List[str]:
    """Provide a list of the NVIDIA GPUs that are visible on the current machine.

    Returns:
        a list of GPU device types.
    """
    cmd = ["nvidia-smi", "-L"]
    try:
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                             check=True)
        device_strs = res.stdout.decode("utf-8").splitlines()
        devices = [x.split(":")[1].split("(")[0].strip() for x in device_strs]
    except FileNotFoundError:
        devices = []
    return devices


@beartype
def quote_and_escape_ffmpeg_path(path: (str, Path)) -> str:
    """Quote and escape paths for use with ffmpeg/ffprobe.

    Args:
        path: the location of a file to be processed by ffmpeg

    Returns:
        a quoted, dollar-escaped path

    Example usage:
        `os.system("ffprobe {quote_and_escape_ffmpeg_path(path)}")`

    NOTE: This function is useful for processing file paths that may contain:
        1. spaces
        2. dollar characters ($)
        3. percent sign characters (%)
    when invoking ffmpeg or ffprobe from python.
    """
    # Dollar signs need to be escaped when used in paths
    escaped = str(path).replace("$", r"\$").replace("%", r"\%")
    if "'" in escaped:
        quoted = f'"{escaped}"'
    else:
        quoted = f"'{escaped}'"
    return quoted


@beartype
def parse_tree_layout(
        tree_layout_path: Path,
        prefix_token: str = "── ",
) -> set:
    """Given a text dump of the output of the linux `tree` command, this function will
    reconstruct the relative paths of the files in the tree.

    Args:
        tree_layout_path: the location of the text file containing the `tree` output
        prefix_token: the token used by the `tree` command to denote a new file.

    Returns:
        the collection of parsed paths.

    NOTES:
    1. This function assumes that it is parsing the output of the tree command that
    has been run in the directory of the structure it is displaying (i.e. `tree` is run)
    without arguments.
    2. The output of each row in the `tree` command is prefixed by a T-bar or an L-bar
    (see example formats (1) and (2) resp. below).
    3. If the file at `tree_layout_path` contains any rows that are not part of the tree
    output, they are ignored.

    Example:
        Given tree outputs of the forms (1) or (2) shown below:

        (1)
            ├── Conversation
            │   ├── Belfast
            │   │   ├── 11+12

        (2)
            └── Conversation
                └── Belfast
                    └── 11+12

        in both cases, this function will return a set of pathlib paths of the form:

        {
            "."
            "Conversation",
            "Conversation/Belfast",
            "Conversation/Belfast/11+12",
        }

    """
    with open(tree_layout_path, "r") as f:
        rows = f.read().splitlines()

    # filter the input to only contain the file tree structure by searching for the
    # presence of the tree prefix token
    rows = [x for x in rows if prefix_token in x]

    # convert nbsp escape codes into spaces
    rows = [unicodedata.normalize("NFKD", x) for x in rows]

    current_path = Path(".")
    paths = {current_path}
    known_prefix_heads = {"├", "└"}
    for row in rows:
        prefix, name = row.split(prefix_token)
        prefix, prefix_head = prefix[:-1], list(prefix).pop(-1)
        msg = f"Expected prefix head to be in {known_prefix_heads} found {prefix_head}"
        assert prefix_head in known_prefix_heads, msg
        assert len(prefix) % 4 == 0, "Expected prefix string length to be a multiple of 4"
        depth = int(len(prefix) / 4)
        current_path = Path(*current_path.parts[:depth]) / name
        paths.add(current_path)
    return paths


if __name__ == "__main__":
    print(list_visible_gpu_types())
