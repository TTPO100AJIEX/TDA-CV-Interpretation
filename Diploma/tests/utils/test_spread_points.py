import pytest

import cvtda.utils


@pytest.mark.parametrize(
    ["size", "num_points", "expected_output"],
    [
        pytest.param(28, 4, [4, 10, 17, 23]),
        pytest.param(32, 4, [5, 12, 19, 26]),
        pytest.param(28, 7, [3, 7, 11, 14, 16, 20, 24]),
    ],
)
def test_ok(size, num_points, expected_output):
    output = cvtda.utils.spread_points(size, num_points)
    assert (output == expected_output).all()


def test_unimplemented():
    with pytest.raises(AssertionError):
        cvtda.utils.spread_points(25, 5)
