from __future__ import annotations
from collections.abc import Iterable
from typing import Any, Dict, List
import hashlib


def _recursive_repr(item: Any) -> str:
    """Hack around python `repr` to deterministically represent dictionaries.

    This is able to represent more things than json.dumps, since it does not require
    things to be JSON serializable (e.g. datetimes).
    """
    if isinstance(item, (str, bytes)):
        result = str(item)
    elif isinstance(item, list):
        result = f"[{', '.join(_recursive_repr(x) for x in item)}]"
    elif isinstance(item, dict):
        kv_pairs = [f"{_recursive_repr(k)}: {_recursive_repr(item[k])}" for k in sorted(item)]
        result = "{" + ", ".join(kv_pairs) + "}"
    else:
        result = repr(item)
    return result


def get_hash(item: Any) -> str:
    repr_ = _recursive_repr(item).encode("utf-8")
    return hashlib.md5(repr_).hexdigest()


def get_hash_int(item: Any) -> int:
    return int(get_hash(item), base=16)


def escape_chars(text: Any, chars: Iterable[str]) -> str:
    """Helper function to escape uncomfortable characters."""
    result = str(text)
    chars = list(set(chars))
    if "\\" in chars:
        chars.remove("\\")
        chars.insert(0, "\\")
    for ch in chars:
        result = result.replace(ch, "\\" + ch)
    return result


def convert_kwargs_to_cmd_line_args(kwargs: Dict) -> List[str]:
    """Helper function to build command line arguments out of dict."""
    args: List[str] = []
    for k in sorted(kwargs.keys()):
        v = kwargs[k]
        if isinstance(v, Iterable) and not isinstance(v, str):
            for value in v:
                args.append(f"-{k}")
                if value is not None:
                    args.append(f"{value}")
            continue
        args.append(f"-{k}")
        if v is not None:
            args.append(f"{v}")
    return args
