import numpy
import numpy.ma.core


def base_(data: numpy.ndarray) -> numpy.ndarray:
    if isinstance(data, numpy.ma.core.MaskedArray):
        return numpy.ma.hstack(
            [
                numpy.ma.sum(numpy.ma.abs(data), axis=1, keepdims=True),  # manhattan norm
                numpy.ma.sqrt(numpy.ma.sum(data**2, axis=1, keepdims=True)),  # euclidean norm
                numpy.ma.max(numpy.ma.abs(data), axis=1, keepdims=True),  # infinity norm
            ]
        ).filled(0)
    else:
        return numpy.hstack(
            [
                numpy.trapz(data, axis=1).reshape(-1, 1),  # integral
                numpy.sum(numpy.abs(data), axis=1, keepdims=True),  # manhattan norm
                numpy.sqrt(numpy.sum(data**2, axis=1, keepdims=True)),  # euclidean norm
                numpy.max(numpy.abs(data), axis=1, keepdims=True),  # infinity norm
            ]
        )


def extension_(data: numpy.ndarray) -> numpy.ndarray:
    if isinstance(data, numpy.ma.core.MaskedArray):
        return numpy.ma.hstack(
            [
                numpy.ma.max(data, axis=1, keepdims=True),
                numpy.ma.sum(data, axis=1, keepdims=True),
                numpy.ma.mean(data, axis=1, keepdims=True),
                numpy.ma.std(data, axis=1, keepdims=True),
                numpy.ma.median(data, axis=1, keepdims=True),
            ]
        ).filled(0)
    else:
        return numpy.hstack(
            [
                numpy.max(data, axis=1, keepdims=True),
                numpy.sum(data, axis=1, keepdims=True),
                numpy.mean(data, axis=1, keepdims=True),
                numpy.std(data, axis=1, keepdims=True),
                numpy.median(data, axis=1, keepdims=True),
            ]
        )


def sequence2features(sequence_batch: numpy.ndarray, reduced: bool = True) -> numpy.ndarray:
    """
    Calculates statistical features of a numeric sequence for all objects in a batch:
    - Integral
    - Manhattan norm
    - Euclidean norm
    - Infinity norm
    - Maximum
    - Sum of elements
    - Mean of elements
    - Standard deviation
    - Median

    Parameters
    ----------
    sequence_batch : ``numpy.ndarray``
        (size `num_items x seq_length`) Input data: a batch of numeric sequences.
        Masked array are supported.
    reduced : ``bool``, default: ``True``
        Removes some features from computations to improve efficiency in reduced pipelines.

    Returns
    -------
    ``numpy.ndarray``
        (size `num_items x num_features`) The computed features.
    """
    base = base_(sequence_batch)
    if reduced:
        return base

    if isinstance(sequence_batch, numpy.ma.core.MaskedArray):
        return numpy.ma.hstack([base, extension_(sequence_batch)])
    else:
        return numpy.hstack([base, extension_(sequence_batch)])
