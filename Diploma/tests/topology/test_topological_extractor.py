import numpy
import pytest
import sklearn.utils.validation

import cvtda.topology
import cvtda.topology.interface

DIAGRAMS = numpy.array([[[0.0, 1, 1], [0, 2, 1], [1, 3, 1]] for _ in range(10)])


class TopologicalExtractor(cvtda.topology.interface.TopologicalExtractor):
    def __init__(
        self,
        cb,
        enabled: bool = True,
        supports_rgb: bool = True,
        n_jobs: int = 1,
        only_get_from_dump: bool = False,
        topo_only_get_from_dump: bool = False,
        return_diagrams: bool = False,
        test_param: str = "abc",
    ):
        assert n_jobs == 1
        assert test_param == "abc"
        super().__init__(
            enabled=enabled,
            vectorizer_settings=cvtda.topology.DiagramVectorizer.PRESETS.reduced,
            supports_rgb=supports_rgb,
            n_jobs=n_jobs,
            only_get_from_dump=only_get_from_dump,
            topo_only_get_from_dump=topo_only_get_from_dump,
            return_diagrams=return_diagrams,
            test_param=test_param,
            cb=cb,
        )
        if cb:
            cb(self)

        self.diagrams_calls_ = []

    def get_diagrams_(self, images: numpy.ndarray, do_fit, dump_name):
        self.diagrams_calls_.append({"images": images.shape, "do_fit": do_fit, "dump_name": dump_name})
        return DIAGRAMS


def test_grayscale_fit():
    def callback_(_):
        callback_.times_called += 1

    callback_.times_called = 0

    extractor = TopologicalExtractor(callback_, supports_rgb=True)
    assert callback_.times_called == 1
    assert extractor.diagrams_calls_ == []

    assert extractor.fit(numpy.random.rand(10, 32, 32)) == extractor
    assert callback_.times_called == 1
    assert extractor.diagrams_calls_ == [{"images": (10, 32, 32), "do_fit": True, "dump_name": None}]
    sklearn.utils.validation.check_is_fitted(extractor.scaler_)
    assert extractor.vectorizer_.fitted_ is True


def test_grayscale_fit_transform():
    def callback_(_):
        callback_.times_called += 1

    callback_.times_called = 0

    extractor = TopologicalExtractor(callback_, supports_rgb=True)
    assert callback_.times_called == 1
    assert extractor.diagrams_calls_ == []

    assert extractor.fit_transform(numpy.random.rand(10, 32, 32)).shape != DIAGRAMS.shape
    assert callback_.times_called == 1
    assert extractor.diagrams_calls_ == [{"images": (10, 32, 32), "do_fit": True, "dump_name": None}]
    sklearn.utils.validation.check_is_fitted(extractor.scaler_)
    assert extractor.vectorizer_.fitted_ is True


def test_grayscale_fit_return_diagrams():
    def callback_(_):
        callback_.times_called += 1

    callback_.times_called = 0

    extractor = TopologicalExtractor(callback_, supports_rgb=True, return_diagrams=True)
    assert callback_.times_called == 1
    assert extractor.diagrams_calls_ == []

    assert extractor.fit(numpy.random.rand(10, 32, 32)) == extractor
    assert callback_.times_called == 1
    assert extractor.diagrams_calls_ == [{"images": (10, 32, 32), "do_fit": True, "dump_name": None}]
    sklearn.utils.validation.check_is_fitted(extractor.scaler_)
    assert extractor.vectorizer_.fitted_ is False


def test_grayscale_fit_transform_return_diagrams():
    def callback_(_):
        callback_.times_called += 1

    callback_.times_called = 0

    extractor = TopologicalExtractor(callback_, supports_rgb=True, return_diagrams=True)
    assert callback_.times_called == 1
    assert extractor.diagrams_calls_ == []

    assert (extractor.fit_transform(numpy.random.rand(10, 32, 32)) == DIAGRAMS).all()
    assert callback_.times_called == 1
    assert extractor.diagrams_calls_ == [{"images": (10, 32, 32), "do_fit": True, "dump_name": None}]
    sklearn.utils.validation.check_is_fitted(extractor.scaler_)
    assert extractor.vectorizer_.fitted_ is False


def test_rgb_fit():
    def callback_(_):
        callback_.times_called += 1

    callback_.times_called = 0

    extractor = TopologicalExtractor(callback_, supports_rgb=True)
    assert callback_.times_called == 1
    assert extractor.diagrams_calls_ == []

    assert extractor.fit(numpy.random.rand(10, 32, 32, 3)) == extractor
    assert callback_.times_called == 5

    assert extractor.diagrams_calls_ == [{"images": (10, 32, 32, 3), "do_fit": True, "dump_name": None}]
    sklearn.utils.validation.check_is_fitted(extractor.scaler_)
    assert extractor.vectorizer_.fitted_ is True

    for e in [
        extractor.gray_extractor_,
        extractor.red_extractor_,
        extractor.green_extractor_,
        extractor.blue_extractor_,
    ]:
        assert e.diagrams_calls_ == [{"images": (10, 32, 32), "do_fit": True, "dump_name": None}]
        sklearn.utils.validation.check_is_fitted(e.scaler_)
        assert e.vectorizer_.fitted_ is True


def test_rgb_fit_transform():
    def callback_(_):
        callback_.times_called += 1

    callback_.times_called = 0

    extractor = TopologicalExtractor(callback_, supports_rgb=True)
    assert callback_.times_called == 1
    assert extractor.diagrams_calls_ == []

    assert extractor.fit_transform(numpy.random.rand(10, 32, 32, 3)).shape == (10, 32 * 5)
    assert callback_.times_called == 5

    assert extractor.diagrams_calls_ == [{"images": (10, 32, 32, 3), "do_fit": True, "dump_name": None}]
    sklearn.utils.validation.check_is_fitted(extractor.scaler_)
    assert extractor.vectorizer_.fitted_ is True

    for e in [
        extractor.gray_extractor_,
        extractor.red_extractor_,
        extractor.green_extractor_,
        extractor.blue_extractor_,
    ]:
        assert e.diagrams_calls_ == [{"images": (10, 32, 32), "do_fit": True, "dump_name": None}]
        sklearn.utils.validation.check_is_fitted(e.scaler_)
        assert e.vectorizer_.fitted_ is True


def test_rgb_fit_return_diagrams():
    def callback_(_):
        callback_.times_called += 1

    callback_.times_called = 0

    extractor = TopologicalExtractor(callback_, supports_rgb=True, return_diagrams=True)
    assert callback_.times_called == 1
    assert extractor.diagrams_calls_ == []

    assert extractor.fit(numpy.random.rand(10, 32, 32, 3)) == extractor
    assert callback_.times_called == 5

    assert extractor.diagrams_calls_ == [{"images": (10, 32, 32, 3), "do_fit": True, "dump_name": None}]
    sklearn.utils.validation.check_is_fitted(extractor.scaler_)
    assert extractor.vectorizer_.fitted_ is False

    for e in [
        extractor.gray_extractor_,
        extractor.red_extractor_,
        extractor.green_extractor_,
        extractor.blue_extractor_,
    ]:
        assert e.diagrams_calls_ == [{"images": (10, 32, 32), "do_fit": True, "dump_name": None}]
        sklearn.utils.validation.check_is_fitted(e.scaler_)
        assert e.vectorizer_.fitted_ is False


def test_rgb_fit_transform_return_diagrams():
    def callback_(_):
        callback_.times_called += 1

    callback_.times_called = 0

    extractor = TopologicalExtractor(callback_, supports_rgb=True, return_diagrams=True)
    assert callback_.times_called == 1
    assert extractor.diagrams_calls_ == []

    result = extractor.fit_transform(numpy.random.rand(10, 32, 32, 3))
    assert callback_.times_called == 5
    assert len(result) == 10
    for res in result:
        assert len(res) == 5
        for item in res:
            assert (item == DIAGRAMS[0]).all()

    assert extractor.diagrams_calls_ == [{"images": (10, 32, 32, 3), "do_fit": True, "dump_name": None}]
    sklearn.utils.validation.check_is_fitted(extractor.scaler_)
    assert extractor.vectorizer_.fitted_ is False

    for e in [
        extractor.gray_extractor_,
        extractor.red_extractor_,
        extractor.green_extractor_,
        extractor.blue_extractor_,
    ]:
        assert e.diagrams_calls_ == [{"images": (10, 32, 32), "do_fit": True, "dump_name": None}]
        sklearn.utils.validation.check_is_fitted(e.scaler_)
        assert e.vectorizer_.fitted_ is False


def test_rgb_fit_transform_not_supports_rgb():
    def callback_(_):
        callback_.times_called += 1

    callback_.times_called = 0

    extractor = TopologicalExtractor(callback_, supports_rgb=False)
    assert callback_.times_called == 1
    assert extractor.diagrams_calls_ == []

    assert extractor.fit_transform(numpy.random.rand(10, 32, 32, 3)).shape == (10, 32 * 4)
    assert callback_.times_called == 5

    assert extractor.diagrams_calls_ == []
    with pytest.raises(sklearn.utils.validation.NotFittedError):
        sklearn.utils.validation.check_is_fitted(extractor.scaler_)
    assert extractor.vectorizer_.fitted_ is False

    for e in [
        extractor.gray_extractor_,
        extractor.red_extractor_,
        extractor.green_extractor_,
        extractor.blue_extractor_,
    ]:
        assert e.diagrams_calls_ == [{"images": (10, 32, 32), "do_fit": True, "dump_name": None}]
        sklearn.utils.validation.check_is_fitted(e.scaler_)
        assert e.vectorizer_.fitted_ is True


def test_rgb_fit_transform_not_supports_rgb_return_diagrams():
    def callback_(_):
        callback_.times_called += 1

    callback_.times_called = 0

    extractor = TopologicalExtractor(callback_, supports_rgb=False, return_diagrams=True)
    assert callback_.times_called == 1
    assert extractor.diagrams_calls_ == []

    result = extractor.fit_transform(numpy.random.rand(10, 32, 32, 3))
    assert callback_.times_called == 5
    assert len(result) == 10
    for res in result:
        assert len(res) == 4
        for item in res:
            assert (item == DIAGRAMS[0]).all()

    assert extractor.diagrams_calls_ == []
    with pytest.raises(sklearn.utils.validation.NotFittedError):
        sklearn.utils.validation.check_is_fitted(extractor.scaler_)
    assert extractor.vectorizer_.fitted_ is False

    for e in [
        extractor.gray_extractor_,
        extractor.red_extractor_,
        extractor.green_extractor_,
        extractor.blue_extractor_,
    ]:
        assert e.diagrams_calls_ == [{"images": (10, 32, 32), "do_fit": True, "dump_name": None}]
        sklearn.utils.validation.check_is_fitted(e.scaler_)
        assert e.vectorizer_.fitted_ is False
