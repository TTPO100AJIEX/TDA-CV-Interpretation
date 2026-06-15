import numpy
import pytest
import skimage.data

from cvtda.topology import GeometryExtractor


def make_rgb():
    return skimage.transform.resize(skimage.data.astronaut(), (32, 32))


def make_gray():
    return make_rgb()[:, :, 0]


@pytest.mark.parametrize(
    ["preset", "expected_features"],
    [
        pytest.param(GeometryExtractor.PRESETS.full, 4687, id="full"),
        pytest.param(GeometryExtractor.PRESETS.reduced, 2617, id="reduced"),
        pytest.param(GeometryExtractor.PRESETS.quick, 2593, id="quick"),
    ],
)
@pytest.mark.parametrize(["num_objects"], [pytest.param(1, id="one_object"), pytest.param(2, id="two_objects")])
def test_gray(preset, expected_features, num_objects):
    extractor = GeometryExtractor(n_jobs=1, settings=preset)
    output = extractor.fit_transform(numpy.array([make_gray()] * num_objects))
    assert output.shape == (num_objects, expected_features)
    assert numpy.isnan(output).sum() == 0


@pytest.mark.parametrize(
    ["preset", "expected_features"],
    [
        pytest.param(GeometryExtractor.PRESETS.full, 4687 * 4 + 1019, id="full"),
        pytest.param(GeometryExtractor.PRESETS.reduced, 2617 * 4 + 1019, id="reduced"),
        pytest.param(GeometryExtractor.PRESETS.quick, 2593 * 4 + 1019, id="quick"),
    ],
)
@pytest.mark.parametrize(["num_objects"], [pytest.param(1, id="one_object"), pytest.param(2, id="two_objects")])
def test_rgb(preset, expected_features, num_objects):
    extractor = GeometryExtractor(n_jobs=1, settings=preset)
    output = extractor.fit_transform(numpy.array([make_rgb()] * num_objects))
    assert output.shape == (num_objects, expected_features)
    assert numpy.isnan(output).sum() == 0


def test_constant_pixels():
    extractor = GeometryExtractor(n_jobs=1, settings=GeometryExtractor.PRESETS.full)
    output = extractor.fit_transform(numpy.zeros((1, 32, 32, 3)))
    assert output.shape == (1, 4687 * 4 + 1019)
    assert numpy.isnan(output).sum() == 0


def test_transform_before_fit():
    with pytest.raises(AssertionError):
        GeometryExtractor(n_jobs=1).transform(numpy.array([make_gray()]))


def test_dimensions_mismatch():
    extractor = GeometryExtractor(n_jobs=1).fit(numpy.array([make_gray()]))
    with pytest.raises(AssertionError):
        extractor.transform(numpy.array([make_rgb()]))
