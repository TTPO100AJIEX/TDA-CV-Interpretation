import numpy
import pytest
import skimage.data

from cvtda.topology import PointCloudsExtractor


def make_rgb():
    return skimage.data.astronaut()[:5, :5, :]


def make_gray():
    return skimage.data.astronaut()[:5, :5, 0]


@pytest.mark.parametrize(
    ["preset", "expected_features"],
    [
        pytest.param(PointCloudsExtractor.PRESETS.full, 126 * 3, id="full"),
        pytest.param(PointCloudsExtractor.PRESETS.reduced, 0, id="reduced"),
        pytest.param(PointCloudsExtractor.PRESETS.quick, 0, id="quick"),
    ],
)
@pytest.mark.parametrize(["num_objects"], [pytest.param(1, id="one_object"), pytest.param(2, id="two_objects")])
def test_gray(preset, expected_features, num_objects):
    extractor = PointCloudsExtractor(n_jobs=1, settings=preset)
    output = extractor.fit_transform(numpy.array([make_gray()] * num_objects))
    assert output.shape == (num_objects, expected_features)
    assert numpy.isnan(output).sum() == 0


@pytest.mark.parametrize(
    ["preset", "expected_diagrams"],
    [
        pytest.param(PointCloudsExtractor.PRESETS.full, True, id="full"),
        pytest.param(PointCloudsExtractor.PRESETS.reduced, False, id="reduced"),
        pytest.param(PointCloudsExtractor.PRESETS.quick, False, id="quick"),
    ],
)
@pytest.mark.parametrize(["num_objects"], [pytest.param(1, id="one_object"), pytest.param(2, id="two_objects")])
def test_gray_return_diagrams(preset, expected_diagrams, num_objects):
    extractor = PointCloudsExtractor(n_jobs=1, settings=preset, return_diagrams=True)
    output = extractor.fit_transform(numpy.array([make_gray()] * num_objects))
    if not expected_diagrams:
        assert len(output) == 0
    else:
        assert len(output) == num_objects
        for item in output:
            assert len(item) == 1
            assert item[0].shape == (26, 3)


@pytest.mark.parametrize(
    ["preset", "expected_features"],
    [
        pytest.param(PointCloudsExtractor.PRESETS.full, 126 * 3 * 5, id="full"),
        pytest.param(PointCloudsExtractor.PRESETS.reduced, 0, id="reduced"),
        pytest.param(PointCloudsExtractor.PRESETS.quick, 0, id="quick"),
    ],
)
@pytest.mark.parametrize(["num_objects"], [pytest.param(1, id="one_object"), pytest.param(2, id="two_objects")])
def test_rgb(preset, expected_features, num_objects):
    extractor = PointCloudsExtractor(n_jobs=1, settings=preset)
    output = extractor.fit_transform(numpy.array([make_rgb()] * num_objects))
    assert output.shape == (num_objects, expected_features)
    assert numpy.isnan(output).sum() == 0


@pytest.mark.parametrize(
    ["preset", "expected_diagrams"],
    [
        pytest.param(PointCloudsExtractor.PRESETS.full, True, id="full"),
        pytest.param(PointCloudsExtractor.PRESETS.reduced, False, id="reduced"),
        pytest.param(PointCloudsExtractor.PRESETS.quick, False, id="quick"),
    ],
)
@pytest.mark.parametrize(["num_objects"], [pytest.param(1, id="one_object"), pytest.param(2, id="two_objects")])
def test_rgb_return_diagrams(preset, expected_diagrams, num_objects):
    extractor = PointCloudsExtractor(n_jobs=1, settings=preset, return_diagrams=True)
    output = extractor.fit_transform(numpy.array([make_rgb()] * num_objects))
    if not expected_diagrams:
        assert len(output) == 0
    else:
        assert len(output) == num_objects
        for item in output:
            assert len(item) == 5
            assert item[0].shape == (26, 3)
            assert item[1].shape == (41, 3)
            assert item[2].shape == (26, 3)
            assert item[3].shape == (26, 3)
            assert item[4].shape == (26, 3)


def test_transform_before_fit():
    input = numpy.array([make_gray()])
    with pytest.raises(AssertionError):
        PointCloudsExtractor(n_jobs=1).transform(input)


def test_dimensions_mismatch():
    input1 = numpy.array([make_gray()])
    extractor = PointCloudsExtractor(n_jobs=1).fit(input1)

    input2 = numpy.array([make_rgb()])
    with pytest.raises(AssertionError):
        extractor.transform(input2)
