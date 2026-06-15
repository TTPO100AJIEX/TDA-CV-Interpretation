import numpy
import sklearn.base


def process_iter(transformer: sklearn.base.TransformerMixin, data: numpy.ndarray, do_fit: bool, *args, **kwargs):
    """
    Processes the `data` with the `transformer`, optionally fitting the transformer.

    Parameters
    ----------
    transformer : ``sklearn.base.TransformerMixin``
        The transformer.
    data : ``numpy.ndarray``
        The data to process.
    do_fit : ``bool``
        If True, will also fit the transformer.

    Returns
    -------
    ``Any``
        The output of `transformer.transform`
    """
    if do_fit:
        return transformer.fit_transform(data, *args, **kwargs)
    return transformer.transform(data, *args, **kwargs)
