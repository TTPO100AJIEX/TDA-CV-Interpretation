import numpy

import cvtda.utils


def test_rgb2gray():
    input = numpy.random.rand(100, 32, 32, 3)
    assert cvtda.utils.rgb2gray(input, n_jobs=1).shape == (100, 32, 32)
