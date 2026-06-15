import abc
import typing

import numpy
import gtda.diagrams

import cvtda.logging
import cvtda.dumping

from .. import utils
from .Extractor import Extractor
from ..DiagramVectorizer import DiagramVectorizer


class TopologicalExtractor(Extractor, abc.ABC):
    """
    Base class for transforming images into features with persistence diagrams calculated in some way.
    Basically abstracts the vectorization of the diagrams.
    """

    def __init__(
        self,
        enabled: bool,
        vectorizer_settings: DiagramVectorizer.Settings,
        supports_rgb: bool,
        n_jobs: int = -1,
        return_diagrams: bool = False,
        only_get_from_dump: bool = False,
        topo_only_get_from_dump: bool = False,
        **kwargs,
    ):
        """
        Parameters
        ----------
        enabled : ``bool``
            Whether the extractor is enabled in the pipeline.
        vectorizer_settings : ``DiagramVectorizer.Settings``
            Settings for diagram vectorization.
        supports_rgb : ``bool``
            Whether the extractor supports processing RGB images directly.
        n_jobs : ``int``, default: ``-1``
            The number of jobs to use for the computation. See :mod:`joblib` for details.
        return_diagrams : ``bool``, default `False`
            If true, :meth:`transform` returns raw persistence diagrams rather then final features.
        only_get_from_dump : ``bool``, default `False`
            If true, all results will be obtained from dump, and no computations will be performed.
        """
        super().__init__(
            n_jobs=n_jobs,
            return_diagrams=return_diagrams,
            only_get_from_dump=False,
            topo_only_get_from_dump=(topo_only_get_from_dump or only_get_from_dump),
            **kwargs,
        )

        self.topo_only_get_from_dump_ = topo_only_get_from_dump or only_get_from_dump
        self.return_diagrams_ = return_diagrams
        self.supports_rgb_ = supports_rgb

        self.enabled_ = enabled
        self.vectorizer_ = DiagramVectorizer(n_jobs=self.n_jobs_, settings=vectorizer_settings)
        self.scaler_ = gtda.diagrams.Scaler(n_jobs=self.n_jobs_, function=lambda x: numpy.max(x))

    def final_dump_name_(self, dump_name: typing.Optional[str] = None):
        # Disable taking the final dump in Extractor. We need to process reading from dump here (to apply Scaler, e.g.)!
        return None

    def diagrams_dump_(self, dump_name: typing.Optional[str]):
        """
        The name of the dump with the computed persistence diagrams.
        """
        return cvtda.dumping.dump_name_concat(dump_name, "diagrams")

    def force_numpy_(self):
        return not self.return_diagrams_

    def nothing_(self, num_objects: int):
        """
        Produces an empty output for a batch of given size.
        """
        return [] if self.return_diagrams_ else numpy.empty((num_objects, 0))

    def process_rgb_(self, rgb_images: numpy.ndarray, do_fit: bool, dump_name: typing.Optional[str] = None):
        if not self.supports_rgb_:
            return self.nothing_(len(rgb_images))
        return self.do_work_(rgb_images, do_fit, dump_name)

    def feature_names_rgb_(self) -> typing.List[str]:
        if not self.supports_rgb_ or not self.enabled_:
            return []
        return self.vectorizer_.feature_names()

    def process_gray_(self, gray_images: numpy.ndarray, do_fit: bool, dump_name: typing.Optional[str] = None):
        return self.do_work_(gray_images, do_fit, dump_name)

    def feature_names_gray_(self) -> typing.List[str]:
        if not self.enabled_:
            return []
        return self.vectorizer_.feature_names()

    def do_work_(self, images: numpy.ndarray, do_fit: bool, dump_name: typing.Optional[str] = None):
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
        if not self.enabled_:
            # If not enabled, just return empty output for compatibility.
            return self.nothing_(len(images))

        # If the user requests to read all results from dumps, do that.
        # Only diagram scaler needs to be applied in this case.
        if self.topo_only_get_from_dump_:
            if self.return_diagrams_:
                diagrams = cvtda.dumping.dumper().get_dump(self.diagrams_dump_(dump_name))
                cvtda.logging.logger().print("Applying Scaler to persistence diagrams.")
                return utils.process_iter(self.scaler_, diagrams, do_fit)
            else:
                return cvtda.dumping.dumper().get_dump(self.features_dump_(dump_name))

        # Get persistence diagrams
        diagrams = self.get_diagrams_(images, do_fit, dump_name)

        # Reduce persistence diagrams to a common scale
        cvtda.logging.logger().print("Applying Scaler to persistence diagrams.")
        diagrams = numpy.nan_to_num(utils.process_iter(self.scaler_, diagrams, do_fit), 0)
        if self.return_diagrams_:
            return diagrams

        # Vectorize the diagrams into features
        features = utils.process_iter_dump(self.vectorizer_, diagrams, do_fit, self.features_dump_(dump_name))
        assert features.shape == (len(images), len(self.vectorizer_.feature_names()))
        return features

    @abc.abstractmethod
    def get_diagrams_(
        self, images: numpy.ndarray, do_fit: bool, dump_name: typing.Optional[str] = None
    ) -> typing.List[numpy.ndarray]:
        """
        Computes persistence diagrams for a set of images.

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
        ``list[numpy.ndarray]``
            Persistence diagrams.
        """
        pass
