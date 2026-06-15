from .base import BaseDumper
from .numpy import NumpyDumper
from .devnull import DevNullDumper
from .dump_name_concat import dump_name_concat

__all__ = [
    "BaseDumper",
    "NumpyDumper",
    "DevNullDumper",
    "dump_name_concat",
]


def dumper() -> BaseDumper:
    """
    Returns
    -------
    ``BaseLogger``
        Current active dumper.
    """
    if BaseDumper.current_dumper is None:
        BaseDumper.current_dumper = NumpyDumper("./")
    return BaseDumper.current_dumper
