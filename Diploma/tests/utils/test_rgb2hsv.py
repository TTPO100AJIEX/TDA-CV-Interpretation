import numpy

import cvtda.utils


def test_rgb2hsv():
    input = numpy.random.rand(100, 32, 32, 3)
    assert cvtda.utils.rgb2hsv(input, n_jobs=1).shape == (100, 32, 32, 3)
