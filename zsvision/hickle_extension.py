"""A small wrapper around hickle to allow the usage of python dictionaries with
slashes in key names.

Feature proposal: https://github.com/telegraphic/hickle/issues/124
Hickle license: https://github.com/telegraphic/hickle/blob/master/LICENSE
Author: Samuel Albanie
"""
# ---------------------------------------------
#                                  Monkey patch
# ---------------------------------------------
import hickle
import numpy as np
from six import string_types
from hickle.lookup import (
    container_types_dict,
    container_key_types_dict
)

# Use a dict to provide a nonlocal variable that works with python2/3. This avoids the
# need to pass global load/dump kwargs around in the recursion
NON_LOCAL_ESCAPE_SLASH = {"status": False}

# Set the escape code - using a shorter uuid would avoid extra storage at increased
# collision risk
ESCAPE_SLASH = {"/": "24fecef0-9d7a"}


def patched_convert(self, **kwargs):
    """ Convert from PyContainer to python core data type.

    Returns: self, either as a list, tuple, set or dict
                (or other type specified in lookup.py)
    """
    if self.container_type in container_types_dict.keys():
        convert_fn = container_types_dict[self.container_type]
        return convert_fn(self)
    if self.container_type == str(dict).encode('ascii', 'ignore'):
        keys = []
        for item in self:
            key = item.name.split('/')[-1]
            key_type = item.key_type[0]
            if key_type in container_key_types_dict.keys():
                to_type_fn = container_key_types_dict[key_type]
                key = to_type_fn(key)
            if NON_LOCAL_ESCAPE_SLASH["status"]:
                key = key.replace(ESCAPE_SLASH["/"], "/")
            keys.append(key)

        items = [item[0] for item in self]
        return dict(zip(keys, items))
    else:
        return self


def create_dict_dataset(py_obj, h_group, call_id=0, **kwargs):
    """ Creates a data group for each key in dictionary

    Notes:
        This is a very important function which uses the recursive _dump
        method to build up hierarchical data models stored in the HDF5 file.
        As this is critical to functioning, it is kept in the main hickle.py
        file instead of in the loaders/ directory.

    Args:
        py_obj: python object to dump; should be dictionary
        h_group (h5.File.group): group to dump data into.
        call_id (int): index to identify object's relative location in the iterable.
    """
    h_dictgroup = h_group.create_group('data_%i' % call_id)

    h_dictgroup.attrs['type'] = [str(type(py_obj)).encode('ascii', 'ignore')]

    for key, py_subobj in py_obj.items():
        if isinstance(key, string_types):
            str_key = "%r" % (key)
        else:
            str_key = str(key)
        if NON_LOCAL_ESCAPE_SLASH["status"]:
            assert ESCAPE_SLASH["/"] not in str_key, ("found UUID escape code for / "
                                                      f"character ({ESCAPE_SLASH['/']})"
                                                      f" in {str_key}, invalid key")
            str_key = str_key.replace("/", ESCAPE_SLASH["/"])
        h_subgroup = h_dictgroup.create_group(str_key)
        h_subgroup.attrs["type"] = [b'dict_item']

        h_subgroup.attrs["key_type"] = [str(type(key)).encode('ascii', 'ignore')]

        hickle.hickle._dump(py_subobj, h_subgroup, call_id=0, **kwargs)


def dump_wrapper(py_obj, file_obj, mode='w', track_times=True, path='/',
                 escape_slash=False, **kwargs):
    NON_LOCAL_ESCAPE_SLASH["status"] = escape_slash
    hickle_dump(py_obj, file_obj, mode=mode, track_times=track_times, path=path, **kwargs)


def load_wrapper(fileobj, path='/', safe=True, escape_slash=False):
    NON_LOCAL_ESCAPE_SLASH["status"] = escape_slash
    return hickle_load(fileobj, path=path, safe=safe)

# ----------------
# apply patch
# ----------------
hickle.hickle.PyContainer.convert = patched_convert
hickle.hickle.create_dict_dataset = create_dict_dataset

hickle_dump = hickle.dump
hickle.dump = dump_wrapper
hickle_load = hickle.load
hickle.load = load_wrapper


if __name__ == "__main__":
    # -------------------------------------------------------------------------------
    # demo behaviour with standard data forms, as well as dictionary that will break
    # current hickle implementation
    # -------------------------------------------------------------------------------
    standard_data = [
        [],
        ["abc"],
        {str(ii): np.random.rand(ii).tolist() for ii in range(10)},
    ]
    test_samples = standard_data + [{"will/break": 0}]

    # use a temporary path for writing
    path = "/tmp/out.hickle"

    for sample in test_samples:
        print(f"Checking input: {sample}")
        print('=========================')

        # preserve current behaviour with `escape_slash=False` (which would be the
        # default value)
        hickle.dump(sample, path, escape_slash=False)
        try:
            res = hickle.load(path, escape_slash=False)
        except KeyError as KE:
            print(f"Without the escape, with input {sample} hickle raises: {KE}")

        # check that the new option preserves data and does not raise errors
        hickle.dump(sample, path, escape_slash=True)
        res = hickle.load(path, escape_slash=True)
        try:
            assert res == sample
        except:
            import ipdb; ipdb.set_trace()

    # validate the exception handler
    ESCAPE_SLASH_CODE = "24fecef0-9d7a"
    sample = {f"will_raise_exception{ESCAPE_SLASH_CODE}": 0}
    try:
        hickle.dump(sample, path, escape_slash=True)
    except AssertionError as AE:
        print(f"With `escape_slash=True`, with input {sample}, raised: {AE}")

# ------------------------------------------------------------
# Gives the following output (using hickle==3.4.6, python 3.7) >>>
# ------------------------------------------------------------

# Checking input: []
# =========================
# Checking input: ['abc']
# =========================
# Checking input: {'0': [], '1': [0.14481838022021565], '2': [0.6365788075805874, 0.24260517490063604], '3': [0.8579390773394745, 0.5166198155386421, 0.4254016202096744], '4': [0.8598153606750859, 0.3284407887378431, 0.9855339339799313, 0.5012603554945942], '5': [0.19304456798880265, 0.9169342763295387, 0.2607961714975783, 0.23814530638719988, 0.2627359472125166], '6': [0.667786053346103, 0.85234247192366, 0.32213689679303825, 0.9260819620498718, 0.33354064793451277, 0.29494135954584755], '7': [0.2769343700673663, 0.7896779451383129, 0.02710238460838199, 0.47624268249915325, 0.3334764166406614, 0.6053893427172244, 0.30312075310489506], '8': [0.9166907521325827, 0.1398051021743969, 0.41622172486076947, 0.8578345064153042, 0.4192547312313747, 0.729580942460437, 0.17950092076311364, 0.8407066968635177], '9': [0.7672039925928313, 0.512199543951006, 0.045217213577125936, 0.4861864015226359, 0.7236094765828289, 0.8495770452239637, 0.7371319870530006, 0.03462712530121592, 0.6094842017142903]}
# =========================
# Checking input: {'will/break': 0}
# =========================
# Without the escape, with input {'will/break': 0} hickle raises: "Can't open attribute (can't locate attribute: 'type')"
# With `escape_slash=True`, with input {'will_raise_exception24fecef0-9d7a': 0}, raised: found UUID escape code for / character (24fecef0-9d7a) in 'will_raise_exception24fecef0-9d7a', invalid key