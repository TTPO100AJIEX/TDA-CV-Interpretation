from __future__ import annotations

import abc
import typing
import inspect
import dataclasses

import numpy
import sklearn.base
import matplotlib.axes
import matplotlib.pyplot as plt


@dataclasses.dataclass
class FeatureExplanation:
    @dataclasses.dataclass
    class PersistenceDiagram:
        diagram: numpy.ndarray
        per_point_stats: numpy.ndarray

        def get_best_points(self):
            non_zero_stats = self.per_point_stats[self.diagram[:, 1] - self.diagram[:, 0] > 0]
            threshold = max(numpy.percentile(non_zero_stats, 75), 1e-8)
            best_idx = numpy.argsort(self.per_point_stats)[::-1]
            return best_idx[self.per_point_stats[best_idx] >= threshold]

        def display(self, ax: matplotlib.axes.Axes):
            def draw(diagram, description: str):
                for dim in range(int(numpy.max(diagram[:, 2], initial=0)) + 1):
                    points = diagram[diagram[:, 2] == dim]
                    if len(points) == 0:
                        continue
                    ax.scatter(points[:, 0], points[:, 1], label=f"{description}: H{dim}")

            limits = [-0.1, self.diagram[:, 1].max() * 1.1]
            draw(self.diagram[self.get_best_points(), :], "Good")
            draw(numpy.delete(self.diagram, self.get_best_points(), axis=0), "Bad")
            ax.plot(limits, limits, linestyle="dashed", color="black")
            ax.set_xlim(*limits)
            ax.set_ylim(*limits)
            ax.legend(loc="lower right")
            ax.set_aspect('equal')

    @dataclasses.dataclass
    class Visualization:
        @dataclasses.dataclass
        class Point:
            x: float
            y: float
            s: typing.Optional[float] = None
            label: typing.Optional[str] = None
            facecolor: typing.Optional[str] = None
            edgecolor: typing.Optional[str] = None

        @dataclasses.dataclass
        class Line:
            x: typing.List[float]
            y: typing.List[float]

        image: typing.Optional[numpy.ndarray] = None
        title: typing.Optional[str] = None
        mask: typing.Optional[numpy.ndarray] = None
        points: typing.List[Point] = dataclasses.field(default_factory=lambda: [])
        lines: typing.List[Line] = dataclasses.field(default_factory=lambda: [])

        def display(self, ax: matplotlib.axes.Axes):
            if self.image is not None:
                ax.imshow(self.image, cmap="gray")
            if self.title is not None:
                ax.set_title(self.title)
            for point in self.points:
                ax.scatter(
                    point.x, point.y, point.s, label=point.label, facecolor=point.facecolor, edgecolor=point.edgecolor
                )
            for line in self.lines:
                ax.plot(line.x, line.y)
            if self.mask is not None:
                ax.imshow(self.mask, cmap="gray", alpha=0.75)
            _, labels = ax.get_legend_handles_labels()
            if labels:
                ax.legend()
            ax.axis("off")

    def readeable_feature_name(self):
        parts = self.feature_name.split(" -> ")
        return f"{parts[0]}\n{' · '.join(parts[1:-1])} [{parts[-1]}]"

    feature_name: str
    persistence_diagrams: typing.List[PersistenceDiagram] = dataclasses.field(default_factory=lambda: [])
    messages: typing.List[str] = dataclasses.field(default_factory=lambda: [])
    visualizations: typing.List[Visualization] = dataclasses.field(default_factory=lambda: [])

    @staticmethod
    def display_many(items: typing.List[FeatureExplanation], title: str):
        nrows = max(len(exp.persistence_diagrams) + len(exp.visualizations) or 1 for exp in items)
        fig, axes = plt.subplots(nrows, len(items), figsize=(len(items) * 3, nrows * 3), squeeze=False)
        for col, exp in enumerate(items):
            panels = (exp.persistence_diagrams + exp.visualizations) or [None]
            for row, panel in enumerate(panels):
                ax = axes[row][col]
                if panel is not None:
                    panel.display(ax)
                else:
                    text = "\n".join(exp.messages) or "(no visualization)"
                    ax.text(0.5, 0.5, text, ha="center", va="center", color="#555555", wrap=True)
                    ax.set_facecolor("#F5F5F5")
                    ax.axis("off")

            for row in range(len(panels) or 1, nrows):
                axes[row][col].axis("off")
            axes[0][col].set_title(exp.readeable_feature_name())

        fig.suptitle(title, fontweight="bold")
        fig.tight_layout()
        return fig

    def display(self, with_diagrams: bool = True):
        FeatureExplanation.display_many([self if with_diagrams else dataclasses.replace(self, persistence_diagrams=[])])

    def extend(self, other):
        self.persistence_diagrams.extend(other.persistence_diagrams)
        self.messages.extend(other.messages)
        self.visualizations.extend(other.visualizations)


class FeatureExtractorBase(sklearn.base.TransformerMixin, abc.ABC):
    """
    Base feature extractor class.

    Attributes
    ----------
    PRESETS : ``Presets``
        Settings presets of the feature extractor.
    """

    @dataclasses.dataclass(frozen=True)
    class Presets:
        """
        Settings presets container of the feature extractor.

        Attributes
        ----------
        full : ``object``
            The full, slow pipeline.
        reduced : ``object``
            The reduced pipeline with good balance between speed and quality.
        quick : ``object``
            The quick pipeline.
        """

        full: object
        reduced: object
        quick: object

    PRESETS: Presets = None

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if "settings" not in inspect.signature(cls.__init__).parameters.keys():
            return
        if cls.PRESETS is None:
            raise TypeError(f"{cls.__name__} must define PRESETS")

    def nest_feature_name(self, prefix: str, name: str) -> str:
        return f"{prefix} -> {name}"

    def nest_feature_names(self, prefix: str, names: typing.List[str]) -> typing.List[str]:
        return [self.nest_feature_name(prefix, name) for name in names]

    def unnest_feature_name(self, name: str) -> typing.Tuple[str, str]:
        idx = name.index(" -> ")
        return name[:idx], name[idx + 4 :]

    @abc.abstractmethod
    def feature_names(self) -> typing.List[str]:
        """
        Gives a list of features extracted by this class.

        Returns
        -------
        ``list[str]``
            Feature names.
        """
        pass

    @abc.abstractmethod
    def explain(self, feature_name: str, input: numpy.ndarray) -> FeatureExplanation:
        pass
