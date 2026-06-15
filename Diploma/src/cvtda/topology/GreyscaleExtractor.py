import typing
import dataclasses

import numpy
import gtda.homology

import cvtda.utils
import cvtda.logging

from . import utils
from .interface import TopologicalExtractor
from .DiagramVectorizer import DiagramVectorizer


class GreyscaleExtractor(TopologicalExtractor):
    """
    Extracts features from grayscale images by directly applying a cubical filtration.
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
        reduced=Settings(vectorizer=DiagramVectorizer.PRESETS.reduced),
        quick=Settings(vectorizer=DiagramVectorizer.PRESETS.quick),
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
            supports_rgb=False,
            n_jobs=n_jobs,
            return_diagrams=return_diagrams,
            settings=settings,
            only_get_from_dump=only_get_from_dump,
            **kwargs,
        )
        self.persistence_ = None

    def get_diagrams_(self, images: numpy.ndarray, do_fit: bool, dump_name: typing.Optional[str] = None):
        cvtda.logging.logger().print(f"GreyscaleExtractor: processing {dump_name}, do_fit = {do_fit}")
        if do_fit and (self.persistence_ is None):
            dims = list(range(len(images.shape) - 1))
            self.persistence_ = gtda.homology.CubicalPersistence(homology_dimensions=dims, n_jobs=self.n_jobs_)
        return utils.process_iter_dump(self.persistence_, images, do_fit, self.diagrams_dump_(dump_name))
