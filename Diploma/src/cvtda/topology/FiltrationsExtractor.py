import math
import typing
import dataclasses

import numpy
import joblib
import itertools
import gtda.images
import gtda.homology

import cvtda.utils
import cvtda.logging

from . import utils
import cvtda.dumping
from .interface import TopologicalExtractor
from .DiagramVectorizer import DiagramVectorizer


class FiltrationExtractor(TopologicalExtractor):
    """
    Binarizes the images, computes the filtration, and uses cubical persistence to calculate the features.
    """

    def __init__(
        self,
        filtration_class,
        filtation_kwargs: dict,
        binarizer_threshold: float,
        diagram_settings: DiagramVectorizer.Settings,
        n_jobs: int = -1,
        return_diagrams: bool = False,
        only_get_from_dump: bool = False,
        **kwargs,
    ):
        """
        Parameters
        ----------
        filtration_class : ``Any``
            The filtration class from :mod:`gtda.images` used for filtration.
        filtation_kwargs : ``dict``
            Kwargs to pass to the filtration class constructor.
        binarizer_threshold : ``float``
            Threshold for image binarization.
        diagram_settings : ``DiagramVectorizer.Settings``
            Settings for diagram vectorization.
        n_jobs : ``int``, default: ``-1``
            The number of jobs to use for the computation. See :mod:`joblib` for details.
        return_diagrams : ``bool``, default `False`
            If true, :meth:`transform` returns raw persistence diagrams rather then final features.
        only_get_from_dump : ``bool``, default `False`
            If true, all results will be obtained from dump, and no computations will be performed.
        """
        super().__init__(
            enabled=True,
            filtration_class=filtration_class,
            filtation_kwargs=filtation_kwargs,
            binarizer_threshold=binarizer_threshold,
            vectorizer_settings=diagram_settings,
            supports_rgb=False,
            n_jobs=n_jobs,
            return_diagrams=return_diagrams,
            diagram_settings=diagram_settings,
            only_get_from_dump=only_get_from_dump,
            **kwargs,
        )

        self.binarizer_ = gtda.images.Binarizer(threshold=binarizer_threshold, n_jobs=self.n_jobs_)
        self.filtration_ = filtration_class(**filtation_kwargs, n_jobs=self.n_jobs_)
        self.persistence_ = None

    def get_diagrams_(self, images: numpy.ndarray, do_fit: bool, dump_name: typing.Optional[str] = None):
        cvtda.logging.logger().print(
            f"FiltrationExtractor: processing {dump_name}, do_fit = {do_fit}, filtration = {self.filtration_}"
        )

        if do_fit and (self.persistence_ is None):
            dims = list(range(len(images.shape) - 1))
            self.persistence_ = gtda.homology.CubicalPersistence(homology_dimensions=dims, n_jobs=self.n_jobs_)

        # Binarization
        bin_images = utils.process_iter(self.binarizer_, images, do_fit)
        assert bin_images.shape == images.shape

        # Filtration
        filtrations = utils.process_iter(self.filtration_, bin_images, do_fit)
        assert filtrations.shape == images.shape

        # Persistence
        return utils.process_iter_dump(self.persistence_, filtrations, do_fit, self.diagrams_dump_(dump_name))


class FiltrationsExtractor(cvtda.utils.FeatureExtractorBase):
    """
    Extracts features from grayscale images with a set of
    :class:`FiltrationExtractor` instances with different hyperparameters.
    """

    @dataclasses.dataclass(frozen=True)
    class Settings:
        """
        Attributes
        ----------
        binarizer_thresholds : ``list[float]``
            Threshold for image binarization.
        height_directions : ``list[float]``, optional, default None
            Directions for height filtrations. See :class:`gtda.images.HeightFiltration` for details.
            If not specified, the directions will be initialized dynamically on the training set
            with a 45 degree angle between any two adjacent directions.
        num_radial : ``int``, default 4
            The number of points to sample over each dimension for radial filtration centers.
            The points are samples uniformly using :func:`cvtda.utils.spread_points`.
            See :class:`gtda.images.RadialFiltration` for details on the filtration.
        dilation : ``bool``, default False
            Whether to use a dilation filtration. See :class:`gtda.images.DilationFiltration` for details.
        dilation : ``bool``, default False
            Whether to use a dilation filtration. See :class:`gtda.images.DilationFiltration` for details.
        erosion : ``bool``, default False
            Whether to use an erosion filtration. See :class:`gtda.images.ErosionFiltration` for details.
        signed_distance : ``bool``, default False
            Whether to use a signed distance filtration. See :class:`gtda.images.SignedDistanceFiltration` for details.
        density_radiuses : ``Iterable[int]``, default []
            Radiuses for density filtrations. See :class:`gtda.images.DensityFiltration` for details.
        vectorizer : ``DiagramVectorizer.Settings``
            Settings for diagram vectorization.
        """

        binarizer_thresholds: typing.List[float] = dataclasses.field(default_factory=lambda: [0.25, 0.5, 0.75])

        height_directions: typing.Optional[typing.Iterable[typing.Tuple[float, float]]] = None
        num_radial: int = 4
        dilation: bool = False
        erosion: bool = False
        signed_distance: bool = False
        density_radiuses: typing.Iterable[int] = dataclasses.field(default_factory=lambda: [])

        vectorizer: DiagramVectorizer.Settings = DiagramVectorizer.Settings()

    PRESETS = cvtda.utils.FeatureExtractorBase.Presets(
        full=Settings(
            binarizer_thresholds=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
            height_directions=None,
            num_radial=4,
            dilation=True,
            erosion=True,
            signed_distance=True,
            density_radiuses=[1, 3],
            vectorizer=DiagramVectorizer.PRESETS.full,
        ),
        reduced=Settings(
            binarizer_thresholds=[0.25, 0.5, 0.75],
            height_directions=None,
            num_radial=4,
            dilation=False,
            erosion=False,
            signed_distance=False,
            density_radiuses=[],
            vectorizer=DiagramVectorizer.PRESETS.reduced,
        ),
        quick=Settings(
            binarizer_thresholds=[0.5],
            height_directions=[(-1, -1), (1, -1), (-1, 1), (1, 1)],
            num_radial=2,
            dilation=False,
            erosion=False,
            signed_distance=False,
            density_radiuses=[],
            vectorizer=DiagramVectorizer.PRESETS.quick,
        ),
    )

    def __init__(
        self,
        settings: Settings,
        n_jobs: int = -1,
        return_diagrams: bool = False,
        only_get_from_dump: bool = False,
    ):
        self.fitted_ = False
        self.n_jobs_ = n_jobs
        self.settings_ = settings
        self.return_diagrams_ = return_diagrams
        self.only_get_from_dump_ = only_get_from_dump
        self.filtration_extractors_: typing.List[typing.Tuple[FiltrationExtractor, str, str]] = []

    def feature_names(self) -> typing.List[str]:
        feature_names = []
        for extractor, _, readable_name in self.filtration_extractors_:
            feature_names.extend(self.nest_feature_names(readable_name, extractor.feature_names()))
        return feature_names

    def fit(self, images: numpy.ndarray, dump_name: typing.Optional[str] = None):
        self.fit_transform(images, dump_name)
        return self

    def transform(self, images: numpy.ndarray, dump_name: typing.Optional[str] = None) -> numpy.ndarray:
        assert self.fitted_ is True, "fit() must be called before transform()"
        cvtda.logging.logger().print("Applying filtrations")
        return self.do_work_(images, do_fit=False, dump_name=dump_name)

    def fit_transform(self, images: numpy.ndarray, dump_name: typing.Optional[str] = None) -> numpy.ndarray:
        assert len(images.shape) >= 3, f"{len(images.shape) - 1}d images are not supported"
        cvtda.logging.logger().print("Fitting filtrations")

        shape = images.shape
        if (len(shape) == 4) and (shape[-1] == 3):
            shape = shape[:-1]
        self.fill_filtrations_(*shape[1:])

        result = self.do_work_(images, do_fit=True, dump_name=dump_name)
        self.fitted_ = True
        return result

    def do_work_(self, images: numpy.ndarray, do_fit: bool, dump_name: typing.Optional[str] = None) -> numpy.ndarray:
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

        def do_work_one(extractor: FiltrationExtractor, name, readable_name):
            # Do work with one filtration
            extractor_dump_name = cvtda.dumping.dump_name_concat(dump_name, name)
            with cvtda.logging.DevNullLogger():
                result = utils.process_iter(extractor, images, do_fit, dump_name=extractor_dump_name)
                return ((extractor, name, readable_name), result) if do_fit else result

        parallel = cvtda.utils.parallel(
            do_work_one, self.filtration_extractors_, return_as="generator", n_jobs=self.outer_n_jobs_
        )
        outputs = list(cvtda.logging.logger().pbar(parallel, total=len(self.filtration_extractors_)))
        if do_fit:
            self.filtration_extractors_ = [output[0] for output in outputs]
            outputs = [output[1] for output in outputs]
        result = utils.hstack(outputs, not self.return_diagrams_)
        if not self.return_diagrams_ and not self.only_get_from_dump_:
            assert result.shape == (len(images), len(self.feature_names()))
        return result

    def fill_filtrations_(self, *shape: typing.List[int]):
        self.inner_n_jobs_ = 1
        self.do_fill_filtrations_(*shape)

        # We try to utilize the CPU cores as efficiently as possible
        # and dynamically balance the number of filtrations processed in parallel
        # and the parallelization level used within each filtration.
        n_jobs = joblib.effective_n_jobs(self.n_jobs_)
        if len(self.filtration_extractors_) > n_jobs * 3:
            self.outer_n_jobs_ = -1
            self.inner_n_jobs_ = 1
        else:
            self.outer_n_jobs_ = math.gcd(n_jobs, len(self.filtration_extractors_))
            self.inner_n_jobs_ = n_jobs // self.outer_n_jobs_
        self.filtration_extractors_ = []
        self.do_fill_filtrations_(*shape)

    def do_fill_filtrations_(self, *shape: typing.List[int]):
        self.filtration_extractors_ = []
        for binarizer_threshold in self.settings_.binarizer_thresholds:
            self.add_height_filtrations_(binarizer_threshold, *shape)
            self.add_radial_filtrations_(binarizer_threshold, *shape)
            self.add_dilation_filtrations_(binarizer_threshold)
            self.add_erosion_filtrations_(binarizer_threshold)
            self.add_signed_distance_filtrations_(binarizer_threshold)
            self.add_density_filtrations_(binarizer_threshold)

    def make_filtration_(self, filtration_class, filtation_kwargs: dict, binarizer_threshold: float):
        return FiltrationExtractor(
            filtration_class,
            filtation_kwargs,
            binarizer_threshold,
            self.settings_.vectorizer,
            n_jobs=self.inner_n_jobs_,
            return_diagrams=self.return_diagrams_,
            only_get_from_dump=self.only_get_from_dump_,
        )

    def add_height_filtrations_(self, binarizer_threshold: float, *shape: typing.List[int]):
        directions = []
        if self.settings_.height_directions is None:
            # Initialize directions
            directions = list(itertools.product(*([[-1, 0, 1]] * len(shape))))
            directions = filter(lambda item: not all(i == 0 for i in item), directions)
            directions = list(directions)
        else:
            directions = self.settings_.height_directions

        for direction in directions:
            self.filtration_extractors_.append(
                (
                    self.make_filtration_(
                        gtda.images.HeightFiltration, {"direction": numpy.array(direction)}, binarizer_threshold
                    ),
                    f"{int(binarizer_threshold * 10)}/HeightFiltration_{direction[0]}_{direction[1]}",
                    f"HeightFiltration with d = ({direction[0]}, {direction[1]}), bin. thr. = 0.{int(binarizer_threshold * 10)}",
                )
            )

    def add_radial_filtrations_(self, binarizer_threshold: float, *shape: typing.List[int]):
        points = [cvtda.utils.spread_points(coord, self.settings_.num_radial) for coord in shape]
        for center in list(itertools.product(*points)):
            self.filtration_extractors_.append(
                (
                    self.make_filtration_(
                        gtda.images.RadialFiltration, {"center": numpy.array(center)}, binarizer_threshold
                    ),
                    f"{int(binarizer_threshold * 10)}/RadialFiltration_{center[0]}_{center[1]}",
                    f"RadialFiltration with c = ({center[0]}, {center[1]}), bin. thr. = 0.{int(binarizer_threshold * 10)}",
                )
            )

    def add_dilation_filtrations_(self, binarizer_threshold: float):
        if not self.settings_.dilation:
            return
        self.filtration_extractors_.append(
            (
                self.make_filtration_(gtda.images.DilationFiltration, {}, binarizer_threshold),
                f"{int(binarizer_threshold * 10)}/DilationFiltration",
                f"DilationFiltration, bin. thr. = 0.{int(binarizer_threshold * 10)}",
            )
        )

    def add_erosion_filtrations_(self, binarizer_threshold: float):
        if not self.settings_.erosion:
            return
        self.filtration_extractors_.append(
            (
                self.make_filtration_(gtda.images.ErosionFiltration, {}, binarizer_threshold),
                f"{int(binarizer_threshold * 10)}/ErosionFiltration",
                f"ErosionFiltration, bin. thr. = 0.{int(binarizer_threshold * 10)}",
            )
        )

    def add_signed_distance_filtrations_(self, binarizer_threshold: float):
        if not self.settings_.signed_distance:
            return
        self.filtration_extractors_.append(
            (
                self.make_filtration_(gtda.images.SignedDistanceFiltration, {}, binarizer_threshold),
                f"{int(binarizer_threshold * 10)}/SignedDistanceFiltration",
                f"SignedDistanceFiltration, bin. thr. = 0.{int(binarizer_threshold * 10)}",
            )
        )

    def add_density_filtrations_(self, binarizer_threshold: float):
        for radius in self.settings_.density_radiuses:
            self.filtration_extractors_.append(
                (
                    self.make_filtration_(gtda.images.DensityFiltration, {"radius": radius}, binarizer_threshold),
                    f"{int(binarizer_threshold * 10)}/DensityFiltration_{radius}",
                    f"DensityFiltration with r = {radius}, bin. thr. = 0.{int(binarizer_threshold * 10)}",
                )
            )
