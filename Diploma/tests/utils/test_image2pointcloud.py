import numpy

import cvtda.utils


def test_onechannel():
    image = numpy.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    output = cvtda.utils.image2pointcloud([image], n_jobs=1)

    expected_output = numpy.array(
        [[0, 0, 1], [0, 1, 2], [0, 2, 3], [1, 0, 4], [1, 1, 5], [1, 2, 6], [2, 0, 7], [2, 1, 8], [2, 2, 9]]
    )

    assert len(output) == 1
    assert output[0].shape == (9, 3)
    assert (output[0] == expected_output).all()


def test_multichannel():
    image3d = numpy.array(
        [
            [[1, 2], [3, 4], [5, 6]],
            [[7, 8], [9, 1], [2, 3]],
            [[4, 5], [6, 7], [8, 9]],
        ]
    )
    output = cvtda.utils.image2pointcloud([image3d], n_jobs=1)

    expected_output = numpy.array(
        [
            [0, 0, 1, 2],
            [0, 1, 3, 4],
            [0, 2, 5, 6],
            [1, 0, 7, 8],
            [1, 1, 9, 1],
            [1, 2, 2, 3],
            [2, 0, 4, 5],
            [2, 1, 6, 7],
            [2, 2, 8, 9],
        ]
    )

    assert len(output) == 1
    assert output[0].shape == (9, 4)
    assert (output[0] == expected_output).all()
