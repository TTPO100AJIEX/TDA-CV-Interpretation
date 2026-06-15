import typing
import dataclasses

import numpy
import gudhi
import gtda.homology

import cvtda.utils
import cvtda.logging

from . import utils
from .interface import TopologicalExtractor
from .DiagramVectorizer import DiagramVectorizer


def get_representative_mask(img, coords_birth, coords_death, dim):
    def fill(img, start_point, is_in):
        mask = numpy.zeros_like(img, dtype=bool)

        def _fill(i, j):
            if i < 0 or j < 0 or i >= img.shape[0] or j >= img.shape[1] or mask[i, j] or not is_in(i, j):
                return
            mask[i, j] = 1.0
            _fill(i - 1, j)
            _fill(i + 1, j)
            _fill(i, j - 1)
            _fill(i, j + 1)

        _fill(start_point[0], start_point[1])
        return mask

    match dim:
        case 0:

            def _is_in(i, j):
                return img[i, j] >= img[coords_birth] and img[i, j] < img[coords_death]

            return fill(img, coords_birth, _is_in)
        case 1:

            def _is_in(i, j):
                return img[i, j] > img[coords_birth] and img[i, j] <= img[coords_death]

            return fill(img, coords_death, _is_in)
        case __:
            raise NotImplementedError


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

    def explain_gray_diagram_(
        self,
        feature_name: str,
        diagram: numpy.ndarray,
        diagram_explanation: cvtda.utils.FeatureExplanation.PersistenceDiagram,
        image: numpy.ndarray,
    ) -> cvtda.utils.FeatureExplanation:
        cc = gudhi.CubicalComplex(top_dimensional_cells=image)
        cc.persistence(homology_coeff_field=2, min_persistence=0)
        reps, _ = cc.cofaces_of_persistence_pairs()

        result = cvtda.utils.FeatureExplanation(feature_name=feature_name)
        for point in diagram_explanation.get_best_points():
            birth, death, dim = diagram[point]
            for i, (birth_idx, death_idx) in enumerate(reps[int(dim)]):
                coords_birth = numpy.unravel_index(birth_idx, image.shape, order="F")
                coords_death = numpy.unravel_index(death_idx, image.shape, order="F")
                if birth != image[coords_birth] or death != image[coords_death]:
                    continue
                result.visualizations.append(
                    cvtda.utils.FeatureExplanation.Visualization(
                        image=image,
                        title=f"H{int(dim)}: {image[coords_birth]:.2f} - {image[coords_death]:.2f}",
                        mask=get_representative_mask(image, coords_birth, coords_death, dim),
                        points=[
                            cvtda.utils.FeatureExplanation.Visualization.Point(
                                x=coords_birth[1], y=coords_birth[0], label="Birth"
                            ),
                            cvtda.utils.FeatureExplanation.Visualization.Point(
                                x=coords_death[1], y=coords_death[0], label="Death"
                            ),
                        ],
                    )
                )
                reps[int(dim)] = numpy.delete(reps[int(dim)], i, axis=0)
                break
            else:
                assert False, f"No representative found for point {point}"
        return result
