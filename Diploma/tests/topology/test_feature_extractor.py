import numpy
import pytest
import skimage.data

from cvtda.topology import FeatureExtractor


def make_rgb():
    return skimage.transform.resize(skimage.data.astronaut(), (32, 32))


def make_gray():
    return make_rgb()[:, :, 0]


NUM_FILTRATIONS_QUICK = (4 + 4) * 1


@pytest.mark.parametrize(["num_objects"], [pytest.param(1, id="one_object"), pytest.param(2, id="two_objects")])
def test_gray(num_objects):
    extractor = FeatureExtractor(settings=FeatureExtractor.PRESETS.quick, n_jobs=1)
    output = extractor.fit_transform(numpy.array([make_gray()] * num_objects))
    assert output.shape == (num_objects, NUM_FILTRATIONS_QUICK * 32 * 2 + 32 * 2 + 2593)
    assert numpy.isnan(output).sum() == 0


@pytest.mark.parametrize(["num_objects"], [pytest.param(1, id="one_object"), pytest.param(2, id="two_objects")])
def test_rgb(num_objects):
    extractor = FeatureExtractor(settings=FeatureExtractor.PRESETS.quick, n_jobs=1)
    output = extractor.fit_transform(numpy.array([make_rgb()] * num_objects))
    assert output.shape == (num_objects, NUM_FILTRATIONS_QUICK * 32 * 2 * 4 + 32 * 2 * 4 + 2593 * 4 + 1019)
    assert numpy.isnan(output).sum() == 0


def test_transform_before_fit():
    input = numpy.array([make_gray()])
    with pytest.raises(AssertionError):
        FeatureExtractor(settings=FeatureExtractor.PRESETS.quick, n_jobs=1).transform(input)


def test_dimensions_mismatch():
    input1 = numpy.array([make_gray()])
    extractor = FeatureExtractor(settings=FeatureExtractor.PRESETS.quick, n_jobs=1).fit(input1)

    input2 = numpy.array([make_rgb()])
    with pytest.raises(AssertionError):
        extractor.transform(input2)
