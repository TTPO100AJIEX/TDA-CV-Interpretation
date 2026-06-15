import abc
import typing
import inspect
import dataclasses

import sklearn.base


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
        if not isinstance(cls.PRESETS, FeatureExtractorBase.Presets):
            raise TypeError(f"{cls.__name__} must be an instance of Presets")

    def nest_feature_names(self, prefix: str, names: typing.List[str]) -> typing.List[str]:
        return [f"{prefix} -> {name}" for name in names]

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
