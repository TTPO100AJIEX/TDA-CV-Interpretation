import numpy
import pytest

import cvtda.topology.interface


class Extractor(cvtda.topology.interface.Extractor):
    def __init__(self, cb, n_jobs: int = 1, only_get_from_dump: bool = False, test_param: str = "abc"):
        assert n_jobs == 1
        assert test_param == "abc"
        super().__init__(n_jobs=n_jobs, only_get_from_dump=only_get_from_dump, test_param=test_param, cb=cb)
        if cb:
            cb(self)

        self.rgb_calls_ = []
        self.gray_calls_ = []

    def process_rgb_(self, rgb_images, do_fit, dump_name):
        self.rgb_calls_.append({"rgb_images": rgb_images.shape, "do_fit": do_fit, "dump_name": dump_name})
        return numpy.random.rand(len(rgb_images), 3)

    def feature_names_rgb_(self):
        return ["a", "b", "c"]

    def process_gray_(self, gray_images, do_fit, dump_name):
        self.gray_calls_.append({"gray_images": gray_images.shape, "do_fit": do_fit, "dump_name": dump_name})
        return numpy.random.rand(len(gray_images), 2)

    def feature_names_gray_(self):
        return ["a", "b"]


def test_grayscale_fit():
    def callback_(_):
        callback_.times_called += 1

    callback_.times_called = 0

    extractor = Extractor(callback_)
    assert callback_.times_called == 1
    assert extractor.rgb_calls_ == []
    assert extractor.gray_calls_ == []

    assert extractor.fit(numpy.random.rand(10, 32, 32)) == extractor
    assert extractor.fitted_ is True
    assert extractor.fit_dimensions_ == (32, 32)
    assert callback_.times_called == 1
    assert extractor.rgb_calls_ == []
    assert extractor.gray_calls_ == [{"gray_images": (10, 32, 32), "do_fit": True, "dump_name": None}]


def test_grayscale_fit_transform():
    def callback_(_):
        callback_.times_called += 1

    callback_.times_called = 0

    extractor = Extractor(callback_)
    assert callback_.times_called == 1
    assert extractor.rgb_calls_ == []
    assert extractor.gray_calls_ == []

    extractor.fit_transform(numpy.random.rand(10, 32, 32))
    assert extractor.fitted_ is True
    assert extractor.fit_dimensions_ == (32, 32)
    assert callback_.times_called == 1
    assert extractor.rgb_calls_ == []
    assert extractor.gray_calls_ == [{"gray_images": (10, 32, 32), "do_fit": True, "dump_name": None}]


def test_grayscale_transform():
    def callback_(_):
        callback_.times_called += 1

    callback_.times_called = 0

    extractor = Extractor(callback_)
    assert callback_.times_called == 1
    assert extractor.rgb_calls_ == []
    assert extractor.gray_calls_ == []

    with pytest.raises(AssertionError):
        extractor.transform(numpy.random.rand(10, 32, 32))
        assert callback_.times_called == 1
        assert extractor.rgb_calls_ == []
        assert extractor.gray_calls_ == []


def test_grayscale_from_dump():
    def callback_(_):
        callback_.times_called += 1

    callback_.times_called = 0

    extractor = Extractor(callback_, only_get_from_dump=True)
    assert callback_.times_called == 1
    assert extractor.rgb_calls_ == []
    assert extractor.gray_calls_ == []

    with pytest.raises(AssertionError):
        extractor.fit(numpy.random.rand(10, 32, 32))


def test_rgb_fit():
    def callback_(_):
        callback_.times_called += 1

    callback_.times_called = 0

    extractor = Extractor(callback_)
    assert callback_.times_called == 1
    assert extractor.rgb_calls_ == []
    assert extractor.gray_calls_ == []

    assert extractor.fit(numpy.random.rand(10, 32, 32, 3)) == extractor
    assert extractor.fitted_ is True
    assert extractor.fit_dimensions_ == (32, 32, 3)
    assert callback_.times_called == 5

    assert extractor.rgb_calls_ == [{"rgb_images": (10, 32, 32, 3), "do_fit": True, "dump_name": None}]
    assert extractor.gray_calls_ == []

    for e in [
        extractor.gray_extractor_,
        extractor.red_extractor_,
        extractor.green_extractor_,
        extractor.blue_extractor_,
    ]:
        assert e.rgb_calls_ == []
        assert e.gray_calls_ == [{"gray_images": (10, 32, 32), "do_fit": True, "dump_name": None}]


def test_rgb_fit_transform():
    def callback_(_):
        callback_.times_called += 1

    callback_.times_called = 0

    extractor = Extractor(callback_)
    assert callback_.times_called == 1
    assert extractor.rgb_calls_ == []
    assert extractor.gray_calls_ == []

    extractor.fit_transform(numpy.random.rand(10, 32, 32, 3))
    assert extractor.fitted_ is True
    assert extractor.fit_dimensions_ == (32, 32, 3)
    assert callback_.times_called == 5

    assert extractor.rgb_calls_ == [{"rgb_images": (10, 32, 32, 3), "do_fit": True, "dump_name": None}]
    assert extractor.gray_calls_ == []

    for e in [
        extractor.gray_extractor_,
        extractor.red_extractor_,
        extractor.green_extractor_,
        extractor.blue_extractor_,
    ]:
        assert e.rgb_calls_ == []
        assert e.gray_calls_ == [{"gray_images": (10, 32, 32), "do_fit": True, "dump_name": None}]
