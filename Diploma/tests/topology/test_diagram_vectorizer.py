import numpy
import pytest

import cvtda.topology

DIAGRAM = [[0.0, 1, 1], [0, 2, 1], [1, 3, 1]]


@pytest.mark.parametrize(
    ["preset", "expected_features"],
    [
        pytest.param(cvtda.topology.DiagramVectorizer.PRESETS.full, 126, id="full"),
        pytest.param(cvtda.topology.DiagramVectorizer.PRESETS.reduced, 32, id="reduced"),
        pytest.param(cvtda.topology.DiagramVectorizer.PRESETS.quick, 32, id="quick"),
    ],
)
@pytest.mark.parametrize(
    ["num_objects"],
    [pytest.param(1, id="one_object"), pytest.param(2, id="two_objects"), pytest.param(25, id="many_objects")],
)
@pytest.mark.parametrize(
    ["batch_size"],
    [pytest.param(None, id="default_batch_size"), pytest.param(1, id="batch_one"), pytest.param(10, id="batch_ten")],
)
def test_vectorizer(preset, expected_features, num_objects, batch_size):
    vectorizer = cvtda.topology.DiagramVectorizer(settings=preset, batch_size=batch_size, n_jobs=1)
    output = vectorizer.fit_transform(numpy.array([DIAGRAM for _ in range(num_objects)]))
    assert output.shape == (num_objects, expected_features)


def test_transform_before_fit():
    with pytest.raises(AssertionError):
        cvtda.topology.DiagramVectorizer(n_jobs=1).transform(numpy.array([DIAGRAM]))
