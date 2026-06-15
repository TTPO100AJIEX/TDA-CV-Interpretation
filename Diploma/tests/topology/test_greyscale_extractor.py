import numpy
import pytest
import skimage.data

from cvtda.topology import GreyscaleExtractor


def make_rgb():
    return skimage.transform.resize(skimage.data.astronaut(), (32, 32))


def make_gray():
    return make_rgb()[:, :, 0]


@pytest.mark.parametrize(
    ["preset", "expected_features"],
    [
        pytest.param(GreyscaleExtractor.PRESETS.full, 126 * 2, id="full"),
        pytest.param(GreyscaleExtractor.PRESETS.reduced, 32 * 2, id="reduced"),
        pytest.param(GreyscaleExtractor.PRESETS.quick, 32 * 2, id="quick"),
        pytest.param(GreyscaleExtractor.Settings(enabled=False), 0, id="disabled"),
    ],
)
@pytest.mark.parametrize(["num_objects"], [pytest.param(1, id="one_object"), pytest.param(2, id="two_objects")])
def test_gray(preset, expected_features, num_objects):
    extractor = GreyscaleExtractor(n_jobs=1, settings=preset)
    output = extractor.fit_transform(numpy.array([make_gray()] * num_objects))
    assert output.shape == (num_objects, expected_features)
    assert numpy.isnan(output).sum() == 0


@pytest.mark.parametrize(
    ["preset", "expected_diagrams"],
    [
        pytest.param(GreyscaleExtractor.PRESETS.full, True, id="full"),
        pytest.param(GreyscaleExtractor.PRESETS.reduced, True, id="reduced"),
        pytest.param(GreyscaleExtractor.PRESETS.quick, True, id="quick"),
        pytest.param(GreyscaleExtractor.Settings(enabled=False), False, id="disabled"),
    ],
)
@pytest.mark.parametrize(["num_objects"], [pytest.param(1, id="one_object"), pytest.param(2, id="two_objects")])
def test_gray_return_diagrams(preset, expected_diagrams, num_objects):
    extractor = GreyscaleExtractor(n_jobs=1, settings=preset, return_diagrams=True)
    output = extractor.fit_transform(numpy.array([make_gray()] * num_objects))
    if not expected_diagrams:
        assert len(output) == 0
    else:
        assert len(output) == num_objects
        for item in output:
            assert len(item) == 1
            assert item[0].shape == (89, 3)


@pytest.mark.parametrize(
    ["preset", "expected_features"],
    [
        pytest.param(GreyscaleExtractor.PRESETS.full, 126 * 2 * 4, id="full"),
        pytest.param(GreyscaleExtractor.PRESETS.reduced, 32 * 2 * 4, id="reduced"),
        pytest.param(GreyscaleExtractor.PRESETS.quick, 32 * 2 * 4, id="quick"),
        pytest.param(GreyscaleExtractor.Settings(enabled=False), 0, id="disabled"),
    ],
)
@pytest.mark.parametrize(["num_objects"], [pytest.param(1, id="one_object"), pytest.param(2, id="two_objects")])
def test_rgb(preset, expected_features, num_objects):
    extractor = GreyscaleExtractor(n_jobs=1, settings=preset)
    output = extractor.fit_transform(numpy.array([make_rgb()] * num_objects))
    assert output.shape == (num_objects, expected_features)
    assert numpy.isnan(output).sum() == 0


@pytest.mark.parametrize(
    ["preset", "expected_diagrams"],
    [
        pytest.param(GreyscaleExtractor.PRESETS.full, True, id="full"),
        pytest.param(GreyscaleExtractor.PRESETS.reduced, True, id="reduced"),
        pytest.param(GreyscaleExtractor.PRESETS.quick, True, id="quick"),
        pytest.param(GreyscaleExtractor.Settings(enabled=False), False, id="disabled"),
    ],
)
@pytest.mark.parametrize(["num_objects"], [pytest.param(1, id="one_object"), pytest.param(2, id="two_objects")])
def test_rgb_return_diagrams(preset, expected_diagrams, num_objects):
    extractor = GreyscaleExtractor(n_jobs=1, settings=preset, return_diagrams=True)
    output = extractor.fit_transform(numpy.array([make_rgb()] * num_objects))
    if not expected_diagrams:
        assert len(output) == 0
    else:
        assert len(output) == num_objects
        for item in output:
            assert len(item) == 4
            assert item[0].shape == (88, 3)
            assert item[1].shape == (89, 3)
            assert item[2].shape == (92, 3)
            assert item[3].shape == (85, 3)


def test_transform_before_fit():
    input = numpy.array([make_gray()])
    with pytest.raises(AssertionError):
        GreyscaleExtractor(n_jobs=1).transform(input)


def test_dimensions_mismatch():
    input1 = numpy.array([make_gray()])
    extractor = GreyscaleExtractor(n_jobs=1).fit(input1)

    input2 = numpy.array([make_rgb()])
    with pytest.raises(AssertionError):
        extractor.transform(input2)
