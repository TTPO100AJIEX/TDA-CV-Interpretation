import typing

import numpy
import sklearn.base

import cvtda.dumping
from .process_iter import process_iter


def process_iter_dump(
    transformer: sklearn.base.TransformerMixin,
    data: numpy.ndarray,
    do_fit: bool,
    dump_name: typing.Optional[str] = None,
    *args,
    **kwargs,
):
    """
    Processes the `data` with the `transformer`, optionally fitting the transformer, and dumps the result.

    Parameters
    ----------
    transformer : ``sklearn.base.TransformerMixin``
        The transformer.
    data : ``numpy.ndarray``
        The data to process.
    do_fit : ``bool``
        If True, will also fit the transformer.
    dump_name : ``str``, optional
        The name of the dump for the results.

    Returns
    -------
    ``Any``
        The output of `transformer.transform`
    """
    if do_fit and cvtda.dumping.dumper().has_dump(dump_name):
        # We must call fit() anyway, even if we have a dump
        transformer.fit(data, *args, **kwargs)

    def func():
        return process_iter(transformer, data, do_fit, *args, **kwargs)

    return func() if dump_name is None else cvtda.dumping.dumper().execute(func, dump_name)
