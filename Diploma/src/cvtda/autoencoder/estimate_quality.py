import numpy


def aggregate_(diff: numpy.ndarray) -> float:
    axes = tuple(set(range(len(diff.shape))) - set([0]))
    return numpy.sum(diff, axis=axes).mean()


def estimate_quality(decoded: numpy.ndarray, original: numpy.ndarray) -> dict:
    """
    Quality metrics for compression task.

    Parameters
    ----------
    decoded : ``numpy.ndarray``
        (size `num_items x width x height x num_channels`) Decoded images.
    original : ``numpy.ndarray``
        (size `num_items x width x height x num_channels`) Original images.

    Returns
    -------
    ``dict``
        1. MAE: mean absolute error between original and decoded images.
        2. MSE: mean squared error between original and decoded images.
    """
    decoded = numpy.squeeze(decoded)
    original = numpy.squeeze(original)
    return {"MAE": aggregate_(numpy.abs(original - decoded)), "MSE": aggregate_((original - decoded) ** 2)}
