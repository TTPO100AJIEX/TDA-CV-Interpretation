import numpy

import skimage.color

import cvtda.utils
import cvtda.logging


def rgb2gray(images: numpy.ndarray, n_jobs: int = -1) -> numpy.ndarray:
    """
    Converts a set of images in RGB color space to grayscale.
    See :func:`skimage.color.rgb2gray` for details.

    Parameters
    ----------
    images : ``numpy.ndarray``
        (size `num_items x width x height x 3`) Input images in RGB format.
    n_jobs : ``int``, default: ``-1``
        The number of jobs to use for the computation. See :mod:`joblib` for details.

    Returns
    -------
    ``numpy.ndarray``
        (size `num_items x width x height`) The images in grayscale.
    """
    return numpy.stack(
        cvtda.utils.parallel(
            skimage.color.rgb2gray,
            cvtda.logging.logger().pbar(images, desc="rgb2gray"),
            n_jobs=n_jobs,
            return_as="list",
            num_parameters=1,
        )
    )
