"""Save and load the nested Python data of the loop figures (dicts of panels, contours, points) as compressed .npz
files, without pickle.

Arrays are stored as they are; everything else (dicts, tuples, lists, strings, numbers, None) goes into a JSON
description in the same file, with tags that keep the exact Python types, so that loading gives back objects that
compare equal to the saved ones (tuples stay tuples, numpy floats stay numpy floats). Object arrays (e.g. the
extremum structure at each grid point, a nested tuple of strings) are stored as a table of their distinct elements
plus an integer index array.
"""
import json

import numpy as np


def _encode(obj, arrays):
    """JSON-able description of obj; arrays found inside are added to arrays and referenced by name."""
    if isinstance(obj, np.ndarray):
        name = f"a{len(arrays)}"
        if obj.dtype == object:
            items = [json.dumps(_encode(x, arrays)) for x in obj.ravel()]
            table, index = np.unique(np.array(items, dtype=str), return_inverse=True)
            arrays[name] = index.reshape(obj.shape).astype(np.int32)
            arrays[name + "_table"] = table
            return {"objarray": name}
        arrays[name] = obj
        return {"array": name}
    if obj is None or isinstance(obj, (bool, str)):
        return obj
    if isinstance(obj, np.bool_):
        return {"bool_": bool(obj)}
    if isinstance(obj, np.floating):
        return {"float64": float(obj)}
    if isinstance(obj, np.integer):
        return {"int64": int(obj)}
    if isinstance(obj, int):
        return obj
    if isinstance(obj, float):
        return {"float": repr(obj)}          # repr round-trips exactly, and covers nan and inf
    if isinstance(obj, tuple):
        return {"tuple": [_encode(x, arrays) for x in obj]}
    if isinstance(obj, list):
        return {"list": [_encode(x, arrays) for x in obj]}
    if isinstance(obj, dict):
        return {"dict": [[_encode(k, arrays), _encode(v, arrays)] for k, v in obj.items()]}
    raise TypeError(f"cannot save {type(obj).__name__}")


def _decode(desc, arrays):
    """The object described by desc (inverse of _encode)."""
    if desc is None or isinstance(desc, (bool, str, int)):
        return desc
    (tag, value), = desc.items()
    if tag == "array":
        return arrays[value]
    if tag == "objarray":
        table = [_decode(json.loads(s), arrays) for s in arrays[value + "_table"]]
        index = arrays[value]
        out = np.empty(index.shape, dtype=object)
        for idx in np.ndindex(index.shape):
            out[idx] = table[index[idx]]
        return out
    if tag == "bool_":
        return np.bool_(value)
    if tag == "float64":
        return np.float64(value)
    if tag == "int64":
        return np.int64(value)
    if tag == "float":
        return float(value)
    if tag == "tuple":
        return tuple(_decode(x, arrays) for x in value)
    if tag == "list":
        return [_decode(x, arrays) for x in value]
    if tag == "dict":
        return {_decode(k, arrays): _decode(v, arrays) for k, v in value}
    raise ValueError(f"unknown tag {tag}")


def save(path, obj):
    """Save obj (nested dicts, tuples, lists, arrays, ...) to path as a compressed .npz file."""
    arrays = {}
    desc = _encode(obj, arrays)
    np.savez_compressed(path, __desc__=np.array(json.dumps(desc)), **arrays)


def load(path):
    """Load what save() saved."""
    with np.load(path, allow_pickle=False) as f:
        arrays = {k: f[k] for k in f.files}
    return _decode(json.loads(str(arrays.pop("__desc__"))), arrays)
