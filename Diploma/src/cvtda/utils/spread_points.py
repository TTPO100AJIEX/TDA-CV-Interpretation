import numpy


def spread_points(size: int, num_points: int) -> numpy.ndarray:
    """
    Spread a given number of points evenly on a line of given size.

    Parameters
    ----------
    size : ``int``
        Size of the line. Must be an odd number.
    num_points : ``int``
        Number of points to spread.

    Returns
    -------
    ``numpy.ndarray``
        (size `num_points`) The coordinates of points.
    """
    assert size % 2 == 0, "Even size is not implemented"

    center, spacing = size // 2, size // (num_points + 1)
    centers = numpy.array(range((spacing + 1) // 2, center, spacing + 1))
    centers = numpy.array([*(center - centers - 1), *(center + centers)])
    if num_points % 2 == 1:
        centers = numpy.array([*centers, center])
    centers.sort()
    return centers
