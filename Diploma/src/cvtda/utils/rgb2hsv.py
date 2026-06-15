import numpy

import skimage.color

import cvtda.utils
import cvtda.logging


def rgb2hsv(images: numpy.ndarray, n_jobs: int = -1) -> numpy.ndarray:
    """
    Converts a set of images in RGB color space to HSV.
    See :func:`skimage.color.rgb2hsv` for details.

    Parameters
    ----------
    images : ``numpy.ndarray``
        (size `num_items x width x height x 3`) Input images in RGB format.
    n_jobs : ``int``, default: ``-1``
        The number of jobs to use for the computation. See :mod:`joblib` for details.

    Returns
    -------
    ``numpy.ndarray``
        (size `num_items x width x height x 3`) The images in RGB format.
    """
    return numpy.stack(
        cvtda.utils.parallel(
            skimage.color.rgb2hsv,
            cvtda.logging.logger().pbar(images, desc="rgb2hsv"),
            n_jobs=n_jobs,
            return_as="list",
            num_parameters=1,
        )
    )
