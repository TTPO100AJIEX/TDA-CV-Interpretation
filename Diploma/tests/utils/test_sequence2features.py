import numpy

import cvtda.utils


def test_reduced():
    input = numpy.random.rand(50, 100)
    assert cvtda.utils.sequence2features(input).shape == (50, 4)


def test_full():
    input = numpy.random.rand(50, 100)
    assert cvtda.utils.sequence2features(input, reduced=False).shape == (50, 9)
