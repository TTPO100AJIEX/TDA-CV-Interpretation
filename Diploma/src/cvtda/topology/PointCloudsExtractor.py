import typing
import dataclasses

import numpy
import gtda.homology

import cvtda.utils
import cvtda.logging

from . import utils
from .interface import TopologicalExtractor
from .DiagramVectorizer import DiagramVectorizer


class PointCloudsExtractor(TopologicalExtractor):
    """
    Extracts features from images by transforming them
    into point clouds and using the Vietoris-Rips complex.
    """

    @dataclasses.dataclass(frozen=True)
    class Settings:
        """
        Attributes
        ----------
        enabled : ``bool``
            Whether the extractor is enabled in the pipeline.
        vectorizer_settings : ``DiagramVectorizer.Settings``
            Settings for diagram vectorization.
        """

        enabled: bool = True
        vectorizer: DiagramVectorizer.Settings = DiagramVectorizer.Settings()

    PRESETS = cvtda.utils.FeatureExtractorBase.Presets(
        full=Settings(vectorizer=DiagramVectorizer.PRESETS.full),
        reduced=Settings(enabled=False, vectorizer=DiagramVectorizer.PRESETS.reduced),
        quick=Settings(enabled=False, vectorizer=DiagramVectorizer.PRESETS.quick),
    )

    def __init__(
        self,
        n_jobs: int = -1,
        return_diagrams: bool = False,
        settings: Settings = Settings(),
        only_get_from_dump: bool = False,
        **kwargs,
    ):
        super().__init__(
            enabled=settings.enabled,
            vectorizer_settings=settings.vectorizer,
            supports_rgb=True,
            n_jobs=n_jobs,
            return_diagrams=return_diagrams,
            settings=settings,
            only_get_from_dump=only_get_from_dump,
            **kwargs,
        )

        self.persistence_ = gtda.homology.VietorisRipsPersistence(homology_dimensions=[0, 1, 2], n_jobs=self.n_jobs_)

    def get_diagrams_(self, images: numpy.ndarray, do_fit: bool, dump_name: typing.Optional[str] = None):
        cvtda.logging.logger().print(f"PointCloudsExtractor: processing {dump_name}, do_fit = {do_fit}")

        point_clouds = cvtda.utils.image2pointcloud(images, self.n_jobs_)
        return utils.process_iter_dump(self.persistence_, point_clouds, do_fit, self.diagrams_dump_(dump_name))
