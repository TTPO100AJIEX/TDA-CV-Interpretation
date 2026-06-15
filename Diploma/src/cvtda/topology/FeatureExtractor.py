import typing
import dataclasses

import numpy
import gtda.images
import sklearn.preprocessing

import cvtda.utils
import cvtda.logging
import cvtda.dumping

from . import utils
from .FiltrationsExtractor import FiltrationsExtractor
from .GreyscaleExtractor import GreyscaleExtractor
from .PointCloudsExtractor import PointCloudsExtractor
from .GeometryExtractor import GeometryExtractor


class FeatureExtractor(cvtda.utils.FeatureExtractorBase):
    """
    The complete feature extraction pipeline.
    Combines all other extractors in one comprehensive process.
    Additionally, applies standardization to the resulting features via :class:`sklearn.preprocessing.StandardScaler`.
    """

    @dataclasses.dataclass(frozen=True)
    class Settings:
        """
        Attributes
        ----------
        greyscale : ``GreyscaleExtractor.Settings``
            Settings for the :class:`GreyscaleExtractor` applied directly to input images.
        inverted : ``GreyscaleExtractor.Settings``
            Settings for the :class:`GreyscaleExtractor` applied to inverted images.
        filtrations : ``FiltrationsExtractor.Settings``
            Settings for the :class:`FiltrationsExtractor`.
        point_clouds : ``PointCloudsExtractor.Settings``
            Settings for the :class:`PointCloudsExtractor`.
        geometry : ``GeometryExtractor.Settings``
            Settings for the :class:`GeometryExtractor`.
        """

        greyscale: GreyscaleExtractor.Settings = GreyscaleExtractor.Settings()
        inverted: GreyscaleExtractor.Settings = GreyscaleExtractor.Settings()
        filtrations: FiltrationsExtractor.Settings = FiltrationsExtractor.Settings()
        point_clouds: PointCloudsExtractor.Settings = PointCloudsExtractor.Settings()
        geometry: GeometryExtractor.Settings = GeometryExtractor.Settings()

    PRESETS = cvtda.utils.FeatureExtractorBase.Presets(
        full=Settings(
            greyscale=GreyscaleExtractor.PRESETS.full,
            inverted=GreyscaleExtractor.PRESETS.full,
            filtrations=FiltrationsExtractor.PRESETS.full,
            point_clouds=PointCloudsExtractor.PRESETS.full,
            geometry=GeometryExtractor.PRESETS.full,
        ),
        reduced=Settings(
            greyscale=GreyscaleExtractor.PRESETS.reduced,
            inverted=GreyscaleExtractor.PRESETS.reduced,
            filtrations=FiltrationsExtractor.PRESETS.reduced,
            point_clouds=PointCloudsExtractor.PRESETS.reduced,
            geometry=GeometryExtractor.PRESETS.reduced,
        ),
        quick=Settings(
            greyscale=GreyscaleExtractor.PRESETS.quick,
            inverted=GreyscaleExtractor.Settings(enabled=False),
            filtrations=FiltrationsExtractor.PRESETS.quick,
            point_clouds=PointCloudsExtractor.PRESETS.quick,
            geometry=GeometryExtractor.PRESETS.quick,
        ),
    )

    def __init__(
        self,
        n_jobs: int = -1,
        return_diagrams: bool = False,
        only_get_from_dump: bool = False,
        settings: Settings = PRESETS.reduced,
    ):
        self.fitted_ = False
        self.return_diagrams_ = return_diagrams
        self.only_get_from_dump_ = only_get_from_dump

        extractor_kwargs = {"n_jobs": n_jobs, "only_get_from_dump": only_get_from_dump}
        topological_extractor_kwargs = {**extractor_kwargs, "return_diagrams": return_diagrams}

        self.inverter_ = gtda.images.Inverter()
        self.greyscale_ = GreyscaleExtractor(settings=settings.greyscale, **topological_extractor_kwargs)
        self.inverted_ = GreyscaleExtractor(settings=settings.inverted, **topological_extractor_kwargs)
        self.filtrations_ = FiltrationsExtractor(settings=settings.filtrations, **topological_extractor_kwargs)
        self.point_clouds_ = PointCloudsExtractor(settings=settings.point_clouds, **topological_extractor_kwargs)
        self.geometry_ = GeometryExtractor(settings=settings.geometry, **extractor_kwargs)
        self.scaler_ = sklearn.preprocessing.StandardScaler(copy=False)

    def feature_names(self) -> typing.List[str]:
        return [
            *self.greyscale_.feature_names(),
            *self.inverted_.feature_names(),
            *self.filtrations_.feature_names(),
            *self.point_clouds_.feature_names(),
            *self.geometry_.feature_names(),
        ]

    def fit(self, images: numpy.ndarray, dump_name: typing.Optional[str] = None):
        self.fit_transform(images, dump_name)
        return self

    def transform(self, images: numpy.ndarray, dump_name: typing.Optional[str] = None):
        assert self.fitted_ is True, "fit() must be called before transform()"
        return self.process_(images, do_fit=False, dump_name=dump_name)

    def fit_transform(self, images: numpy.ndarray, dump_name: typing.Optional[str] = None):
        result = self.process_(images, do_fit=True, dump_name=dump_name)
        self.fitted_ = True
        return result

    def process_(self, images: numpy.ndarray, do_fit: bool, dump_name: typing.Optional[str] = None):
        """
        Performs the actual computations for a batch of images.

        Parameters
        ----------
        images : ``numpy.ndarray``
            (size `num_items x width x height x num_channels`) Input images in grayscale or rgb format.
        do_fit : ``bool``
            If True, will also fit the extractor.
        dump_name : ``str``, optional
            The root to dump the results in, if dumping is enabled.

        Returns
        -------
        ``numpy.ndarray`` or ``list[numpy.ndarray]``
            The result.
        """
        results = []

        greyscale_dump = cvtda.dumping.dump_name_concat(dump_name, "greyscale")
        results.append(utils.process_iter(self.greyscale_, images, do_fit, greyscale_dump))

        inverted_images = utils.process_iter(self.inverter_, images, do_fit=do_fit)
        inverted_greyscale_dump = cvtda.dumping.dump_name_concat(dump_name, "inverted_greyscale")
        results.append(utils.process_iter(self.inverted_, inverted_images, do_fit, inverted_greyscale_dump))

        filtrations_dump = cvtda.dumping.dump_name_concat(dump_name, "filtrations")
        results.append(utils.process_iter(self.filtrations_, images, do_fit, filtrations_dump))

        point_clouds_dump = cvtda.dumping.dump_name_concat(dump_name, "point_clouds")
        results.append(utils.process_iter(self.point_clouds_, images, do_fit, point_clouds_dump))

        if not self.return_diagrams_:
            # Geometry features do not use persistence diagrams, so we skip those.
            geometry_dump = cvtda.dumping.dump_name_concat(dump_name, "geometry")
            results.append(utils.process_iter(self.geometry_, images, do_fit, geometry_dump))

        results = utils.hstack(results, not self.return_diagrams_)
        if self.return_diagrams_:
            cvtda.logging.logger().print("Diagrams requested. Returning diagrams.")
            return results

        cvtda.logging.logger().print("Applying StandardScaler.")
        features = utils.process_iter(self.scaler_, results, do_fit)
        if not self.only_get_from_dump_:
            assert features.shape == (len(images), len(self.feature_names()))
        return features
