import typing
import dataclasses

import gph
import numpy
import ripser
import gtda.homology

import cvtda.utils
import cvtda.logging

from . import utils
from .interface import TopologicalExtractor
from .DiagramVectorizer import DiagramVectorizer


def get_explanation_mask(point_cloud: numpy.ndarray, explanation, dimension: int) -> numpy.ndarray:
    point_cloud = point_cloud.astype(int)
    mask = numpy.zeros((point_cloud[:, 0].max() + 1, point_cloud[:, 1].max() + 1))
    if dimension == 0:
        points = [point_cloud[point] for point in explanation]
        death = numpy.linalg.norm(points[1] - points[2])

        queue = set([tuple(points[0])])
        while len(queue) != 0:
            point = queue.pop()
            mask[point[0], point[1]] = 1.0
            for other in point_cloud[numpy.linalg.norm(point_cloud - point, axis=1) < death]:
                if mask[other[0], other[1]] == 1.0:
                    continue
                queue.add(tuple(other))
    else:
        points = point_cloud[explanation[:, :2].flatten()][:, :2]
        mask[points[:, 0], points[:, 1]] = 1.0
    return mask


def get_explanation_points(
    point_cloud: numpy.ndarray, generator: typing.List[int]
) -> typing.List[cvtda.utils.FeatureExplanation.Visualization.Point]:
    generator = [point_cloud[point_idx] for point_idx in generator]
    return [
        cvtda.utils.FeatureExplanation.Visualization.Point(
            point[0], point[1], label=("Birth" if i < len(generator) - 2 else "Death")
        )
        for i, point in enumerate(generator)
    ]


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

    def explain_gray_diagram_(
        self,
        feature_name: str,
        diagram: numpy.ndarray,
        diagram_explanation: cvtda.utils.FeatureExplanation.PersistenceDiagram,
        image: numpy.ndarray,
    ) -> cvtda.utils.FeatureExplanation:
        diagram = diagram[diagram[:, 1] > diagram[:, 0]]
        point_cloud = cvtda.utils.image2pointcloud([image], 1)[0]
        raw_vr = gph.ripser_parallel(point_cloud, maxdim=2, return_generators=True)
        ripser_vr = ripser.ripser(point_cloud, maxdim=2, do_cocycles=True)

        generators = list(raw_vr["gens"][0]) + list(raw_vr["gens"][1][0]) + list(raw_vr["gens"][1][1])
        explanations = list(raw_vr["gens"][0]) + ripser_vr["cocycles"][1]
        assert len(diagram) == len(generators), f"Diagram size: {len(diagram)}, generators size: {len(generators)}"
        assert len(diagram) == len(explanations), f"Diagram size: {len(diagram)}, explanation size: {len(explanations)}"

        result = cvtda.utils.FeatureExplanation(feature_name=feature_name)
        for diagram_point in diagram_explanation.get_best_points():
            birth, death, dim = diagram[diagram_point]
            result.visualizations.append(
                cvtda.utils.FeatureExplanation.Visualization(
                    image=image,
                    title=f"H{int(dim)}: {birth:.2f} - {death:.2f}",
                    mask=get_explanation_mask(point_cloud, explanations[diagram_point], int(dim)),
                    points=get_explanation_points(point_cloud, generators[diagram_point]),
                )
            )
        return result
