import numpy
import typing

import cvtda.utils
import cvtda.logging


def image2pointcloud(images: numpy.ndarray, n_jobs: int = -1) -> typing.List[numpy.ndarray]:
    """
    Converts a set of images to point clouds.
    Each pixel of the image is converted into a point,
    with the first two dimensions being the coordinates of the pixel,
    and the color values of the pixel for each channel placed in the end.

    Parameters
    ----------
    images : ``numpy.ndarray``
        (size `num_items x width x height x num_channels`) Input images in any format.
    n_jobs : ``int``, default: ``-1``
        The number of jobs to use for the computation. See :mod:`joblib` for details.

    Returns
    -------
    ``numpy.ndarray``
        (size `num_items x (2 + num_channels)`) The resulting point clouds.
    """

    def _impl(image: numpy.ndarray) -> numpy.ndarray:
        width, height = image.shape[0:2]
        x = numpy.indices((width, height))[0]
        y = numpy.indices((width, height))[1]
        return numpy.dstack([x, y, image]).reshape((width * height, -1))

    return cvtda.utils.parallel(
        _impl,
        cvtda.logging.logger().pbar(images, desc="image2pointcloud"),
        n_jobs=n_jobs,
        return_as="list",
    )
