import typing
import dataclasses

import numpy
import skimage.measure
import skimage.feature
import skimage.feature.util

import cvtda.utils
import cvtda.logging
from cvtda.utils import FeatureExplanation

from . import utils
from .interface import Extractor


class BaseGeometricFeature(cvtda.utils.FeatureExtractorBase):
    def fit(self, image: numpy.ndarray):
        self.feature_names_ = list(map(str, range(len(self.transform(image)))))
        return self

    def feature_names(self) -> typing.List[str]:
        return self.feature_names_


class DAISY(BaseGeometricFeature):
    def transform(self, image: numpy.ndarray) -> numpy.ndarray:
        return self.run_(image).flatten()

    def explain(self, feature_name: str, image: numpy.ndarray) -> FeatureExplanation:
        _, visualization = self.run_(image, visualize=True)
        return FeatureExplanation(
            feature_name=feature_name,
            visualizations=[FeatureExplanation.Visualization(image=image, mask=visualization, title="DAISY")],
        )

    def run_(self, image: numpy.ndarray, **extra_params):
        image_shape = max(*image.shape)
        return skimage.feature.daisy(
            image,
            step=(6 * image_shape // 32),
            radius=(12 * image_shape // 32),
            rings=5,
            histograms=5,
            orientations=8,
            **extra_params,
        )


class SkimageDetectorBase(BaseGeometricFeature):
    def __init__(self, extractor: skimage.feature.util.DescriptorExtractor, num_features: int, reduced_stats: bool):
        self.extractor_ = extractor
        self.num_features_ = num_features
        self.reduced_stats_ = reduced_stats

    def transform(self, image: numpy.ndarray) -> numpy.ndarray:
        try:
            self.extractor_.detect_and_extract(image)
            descriptors = self.extractor_.descriptors.transpose()
            if descriptors.shape[1] == 0:
                raise "How is this possible?"
        except:  # noqa: E722
            # DescriptorExtractor may throw errors for degenerate images. We substitute those with zeros.
            descriptors = numpy.zeros((self.num_features_, 1))
        assert descriptors.shape[0] == self.num_features_
        return cvtda.utils.sequence2features(numpy.ma.array(descriptors), reduced=self.reduced_stats_).flatten()

    def explain(self, feature_name: str, image: numpy.ndarray) -> FeatureExplanation:
        try:
            self.extractor_.detect(image)
            points = [
                FeatureExplanation.Visualization.Point(x=x, y=y, s=2**s, facecolor="none", edgecolor="r")
                for ((x, y), s) in zip(self.extractor_.keypoints, self.extractor_.scales)
            ]
            return FeatureExplanation(
                feature_name=feature_name,
                visualizations=[FeatureExplanation.Visualization(image=image, points=points, title="SIFT")],
            )
        except RuntimeError as exc:
            # sift may throw errors for degenerate images. We substitute those with zeros.
            return FeatureExplanation(feature_name=feature_name, messages=[str(exc)])


class SIFT(SkimageDetectorBase):
    def __init__(self, reduced_stats: bool):
        super().__init__(skimage.feature.SIFT(), 128, reduced_stats)


class ORB(SkimageDetectorBase):
    def __init__(self, reduced_stats: bool):
        super().__init__(skimage.feature.ORB(), 256, reduced_stats)


class HOG(BaseGeometricFeature):
    def transform(self, image: numpy.ndarray) -> numpy.ndarray:
        return self.run_(image)

    def explain(self, feature_name: str, image: numpy.ndarray) -> FeatureExplanation:
        _, hog = self.run_(image, visualize=True)
        return FeatureExplanation(
            feature_name=feature_name,
            visualizations=[FeatureExplanation.Visualization(image=image, mask=hog / hog.max(), title="HOG")],
        )

    def run_(self, image: numpy.ndarray, **extra_params):
        extra_params["pixels_per_cell"] = (image.shape[0] // 4, image.shape[1] // 4)
        if image.ndim == 3:
            extra_params["channel_axis"] = 2
        return skimage.feature.hog(image, **extra_params)


class BasicFeatures(BaseGeometricFeature):
    def __init__(self, reduced_stats: bool):
        self.reduced_stats_ = reduced_stats

    def transform(self, image: numpy.ndarray) -> numpy.ndarray:
        basic_features = self.run_(image)
        basic_features = basic_features.reshape((-1, basic_features.shape[-1]))
        return cvtda.utils.sequence2features(
            numpy.ma.array(basic_features.transpose()), reduced=self.reduced_stats_
        ).flatten()

    def explain(self, feature_name: str, image: numpy.ndarray) -> FeatureExplanation:
        features = self.run_(image)
        feature = features[:, :, int(feature_name) // features.shape[2]]
        return FeatureExplanation(
            feature_name=feature_name,
            visualizations=[FeatureExplanation.Visualization(image=feature, title=f"Basic Feature {feature_name}")],
        )

    def run_(self, image: numpy.ndarray):
        return skimage.feature.multiscale_basic_features(image)


class BlurEffect(BaseGeometricFeature):
    def transform(self, image: numpy.ndarray) -> numpy.ndarray:
        return numpy.array([skimage.measure.blur_effect(image)])

    def explain(self, feature_name: str, image: numpy.ndarray) -> FeatureExplanation:
        return FeatureExplanation(
            feature_name=feature_name, messages=[f"Blur effect is {skimage.measure.blur_effect(image):.2f}"]
        )


class IntertiaTensorEigvals(BaseGeometricFeature):
    def transform(self, image: numpy.ndarray) -> numpy.ndarray:
        try:
            return numpy.array(skimage.measure.inertia_tensor_eigvals(image))
        except:  # noqa: E722
            # inertia tensor is undefined for degenerate images.
            return numpy.zeros((image.ndim))

    def explain(self, feature_name: str, image: numpy.ndarray) -> FeatureExplanation:
        return FeatureExplanation(
            feature_name=feature_name,
            messages=[f"Inertia tensor eigenvalue {feature_name} is {self.transform(image)[int(feature_name)]:.2f}"],
        )


class Centroid(BaseGeometricFeature):
    def transform(self, image: numpy.ndarray) -> numpy.ndarray:
        return skimage.measure.centroid(image)

    def explain(self, feature_name: str, image: numpy.ndarray) -> FeatureExplanation:
        return FeatureExplanation(feature_name=feature_name, messages=[f"Centroid is {self.transform(image).round(2)}"])


class Moments(BaseGeometricFeature):
    def __init__(self, order: int):
        self.order_ = order

    def transform(self, image: numpy.ndarray) -> numpy.ndarray:
        return skimage.measure.moments(image, order=self.order_).flatten()

    def explain(self, feature_name: str, image: numpy.ndarray) -> FeatureExplanation:
        return FeatureExplanation(
            feature_name=feature_name, messages=[f"Moment {feature_name} is {self.transform(image)[int(feature_name)]:.2f}"]
        )


class MomentsCentral(BaseGeometricFeature):
    def __init__(self, order: int):
        self.order_ = order

    def transform(self, image: numpy.ndarray) -> numpy.ndarray:
        return skimage.measure.moments_central(image, order=self.order_).flatten()

    def explain(self, feature_name: str, image: numpy.ndarray) -> FeatureExplanation:
        return FeatureExplanation(
            feature_name=feature_name,
            messages=[f"Central moment {feature_name} is {self.transform(image)[int(feature_name)]:.2f}"],
        )


class MomentsHu(BaseGeometricFeature):
    def __init__(self, order: int):
        self.order_ = order

    def transform(self, image: numpy.ndarray) -> numpy.ndarray:
        moments_central = skimage.measure.moments_central(image, order=self.order_)
        moments_normalized = skimage.measure.moments_normalized(moments_central)
        return skimage.measure.moments_hu(moments_normalized).flatten()

    def explain(self, feature_name: str, image: numpy.ndarray) -> FeatureExplanation:
        return FeatureExplanation(
            feature_name=feature_name,
            messages=[f"Hu's moment {feature_name} is {self.transform(image)[int(feature_name)]:.2f}"],
        )


class ShannonEntropy(BaseGeometricFeature):
    def transform(self, image: numpy.ndarray) -> numpy.ndarray:
        return numpy.array([skimage.measure.shannon_entropy(image)])

    def explain(self, feature_name: str, image: numpy.ndarray) -> FeatureExplanation:
        return FeatureExplanation(
            feature_name=feature_name, messages=[f"Shannon entropy is {skimage.measure.shannon_entropy(image):.2f}"]
        )


class PearsonCorrCoeff(BaseGeometricFeature):
    def transform(self, image: numpy.ndarray) -> numpy.ndarray:
        return numpy.array(
            [
                skimage.measure.pearson_corr_coeff(image[:, :, 0], image[:, :, 1])[0],
                skimage.measure.pearson_corr_coeff(image[:, :, 0], image[:, :, 2])[0],
                skimage.measure.pearson_corr_coeff(image[:, :, 1], image[:, :, 2])[0],
            ]
        )

    def explain(self, feature_name: str, image: numpy.ndarray) -> FeatureExplanation:
        return FeatureExplanation(
            feature_name=feature_name,
            messages=[f"Pearson correlation coefficient {feature_name} is {self.transform(image)[int(feature_name)]:.2f}"],
        )


class Curvature(BaseGeometricFeature):
    THRESHOLDS = numpy.arange(256) / 255.0

    def __init__(self, reduced_stats: bool):
        self.reduced_stats_ = reduced_stats

    def fit(self, image: numpy.ndarray):
        series_features = cvtda.utils.sequence2features(self.run_(image), reduced=self.reduced_stats_)
        names = list(map(str, range(series_features.shape[1])))
        self.feature_names_ = (
            self.nest_feature_names("euler_number", names)
            + self.nest_feature_names("area", names)
            + self.nest_feature_names("perimeter", names)
            + self.nest_feature_names("euler_number_diff", names)
            + self.nest_feature_names("area_diff", names)
            + self.nest_feature_names("perimeter_diff", names)
        )
        return self

    def transform(self, image: numpy.ndarray) -> numpy.ndarray:
        series = self.run_(image)
        series_diff = numpy.diff(series, axis=1)
        return numpy.concatenate(
            [
                cvtda.utils.sequence2features(series, reduced=self.reduced_stats_).flatten(),
                cvtda.utils.sequence2features(series_diff, reduced=self.reduced_stats_).flatten(),
            ]
        )

    def explain(self, feature_name: str, image: numpy.ndarray) -> FeatureExplanation:
        series = self.run_(image)
        match self.unnest_feature_name(feature_name)[0]:
            case "euler_number":
                return self.make_explanation_(feature_name, "Euler number", series[0])
            case "area":
                return self.make_explanation_(feature_name, "Area", series[1])
            case "perimeter":
                return self.make_explanation_(feature_name, "Perimeter", series[2])
            case "euler_number_diff":
                return self.make_explanation_(feature_name, "Euler number (derivative)", numpy.diff(series[0]))
            case "area_diff":
                return self.make_explanation_(feature_name, "Area (derivative)", numpy.diff(series[1]))
            case "perimeter_diff":
                return self.make_explanation_(feature_name, "Perimeter (derivative)", numpy.diff(series[2]))
            case __:
                raise NotImplementedError(f"Unknown feature {feature_name}")

    def make_explanation_(self, feature_name: str, title: str, series: numpy.ndarray) -> FeatureExplanation:
        line = FeatureExplanation.Visualization.Line(x=Curvature.THRESHOLDS[: len(series)], y=series)
        return FeatureExplanation(
            feature_name=feature_name, visualizations=[FeatureExplanation.Visualization(lines=[line], title=title)]
        )

    def run_(self, image: numpy.ndarray):
        min_, max_ = image.min(), image.max()
        assert (min_ >= 0) and (max_ <= 1), f"Bad image format: should be [0, 1]; received [{min_}, {max_}]"

        euler_numbers, area, perimeter = [], [], []
        for threshold in Curvature.THRESHOLDS:
            bin = image > threshold
            euler_numbers.append(skimage.measure.euler_number(bin))
            area.append(bin.sum())
            perimeter.append(skimage.measure.perimeter(bin))
        return numpy.array([euler_numbers, area, perimeter])


class BaseGeometryExtractor(cvtda.utils.FeatureExtractorBase):
    def __init__(self, name: str, n_jobs: int, features: typing.List[typing.Tuple[str, BaseGeometricFeature]]):
        self.name_ = name
        self.fitted_ = False
        self.n_jobs_ = n_jobs
        self.features_ = features

    def feature_names(self) -> typing.List[str]:
        assert self.fitted_ is True, "fit() must be called before feature_names()"
        return sum(map(lambda pair: self.nest_feature_names(pair[0], pair[1].feature_names()), self.features_), [])

    def fit(self, images: numpy.ndarray):
        for name, feature in self.features_:
            feature.fit(images[0])
        self.fitted_ = True
        return self

    def transform(self, images: numpy.ndarray) -> numpy.ndarray:
        assert self.fitted_ is True, "fit() must be called before transform()"

        def process_one_(image: numpy.ndarray) -> numpy.ndarray:
            return self.calc_raw_(image)

        pbar = cvtda.logging.logger().pbar(images, desc=self.name_)
        features = numpy.stack(cvtda.utils.parallel(process_one_, pbar, n_jobs=self.n_jobs_))
        assert features.shape == (len(images), len(self.feature_names()))
        return features

    def explain(self, feature_name: str, input: numpy.ndarray) -> FeatureExplanation:
        prefix, suffix = self.unnest_feature_name(feature_name)
        for name, feature in self.features_:
            if name == prefix:
                result = feature.explain(suffix, input)
                result.feature_name = self.nest_feature_name(prefix, result.feature_name)
                return result
        raise NotImplementedError(f"Unknown feature {feature_name}")

    def calc_raw_(self, image: numpy.ndarray) -> typing.List[numpy.ndarray]:
        features = list(map(lambda pair: pair[1].transform(image), self.features_))
        if len(features) == 0:
            return numpy.empty((0,))
        return numpy.nan_to_num(numpy.concatenate(features), 0)


class GrayGeometryExtractor(BaseGeometryExtractor):
    """
    Extracts geometric features from grayscale images:
    - DAISY [1], see :func:`skimage.feature.daisy`
    - SIFT [2], see :class:`skimage.feature.SIFT`
    - ORB [3], see :class:`skimage.feature.ORB`
    - HOG [4], see :func:`skimage.feature.hog`
    - Basic local features, see :func:`skimage.feature.multiscale_basic_features`
    - Blur effect [5], see :func:`skimage.measure.blur_effect`
    - Moments ($M$): see :func:`skimage.measure.moments`
    - Centroid: $M_{10} / M{00}$, see :func:`skimage.measure.centroid`
    - Central moments: see :func:`skimage.measure.moments_central`
    - Hu's image moments: see :func:`skimage.measure.moments_hu`
    - Eigenvalues of the intertia vector, see :func:`skimage.measure.inertia_tensor_eigvals`
    - Shannon entropy: see :func:`skimage.measure.shannon_entropy`
    - Statistics of the local curvature for 256 uniform binarization thresholds:
        - Euler number: see :func:`skimage.measure.euler_number`
        - Perimeter: see :func:`skimage.measure.perimeter`
        - Area

    References
    ----------
    .. [1] Engin Tola, Vincent Lepetit, Pascal Fua
            "A fast local descriptor for dense matching."
            https://ieeexplore.ieee.org/document/4587673/
    .. [2] McConnell RK (1986)
            "Method of and apparatus for pattern recognition."
            https://patents.google.com/patent/US4567610A/en
    .. [3] Ethan Rublee, Vincent Rabaud, Kurt Konolige, Gary Bradski
            ORB: An efficient alternative to SIFT or SURF
            https://ieeexplore.ieee.org/document/6126544
    .. [4] D.G. Lowe "Object recognition from local scale-invariant features"
            https://ieeexplore.ieee.org/document/790410
    .. [5] Frederique Crete, Thierry Dolmiere, Patricia Ladret, Marina Nicolas
            "The blur effect: perception and estimation with a new no-reference perceptual blur metric"
            https://doi.org/10.1117/12.702790
    """

    @dataclasses.dataclass(frozen=True)
    class Settings:
        """
        Attributes
        ----------
        reduced_stats : ``bool``, default True
            Whether to reduce the set of statistical features. See :func:`utils.sequence2features` for details.
        daisy : ``bool``, default True
            Whether to compute DAISY features.
        sift : ``bool``, default True
            Whether to compute SIFT features.
        orb : ``bool``, default True
            Whether to compute ORB features.
        hog : ``bool``, default True
            Whether to compute HOG features.
        basic : ``bool``, default True
            Whether to compute basic local features.
        blur_effect : ``bool``, default True
            Whether to compute blur effect.
        moments : ``bool``, default True
            Whether to compute the moments.
        centroid : ``bool``, default True
            Whether to compute the centroid.
        moments_central : ``bool``, default True
            Whether to compute the central moments.
        moments_hu : ``bool``, default True
            Whether to compute Hu's image moments.
        inertia_tensor_eigvals : ``bool``, default True
            Whether to compute the eigenvalues of the intertia vector.
        shannon_entropy : ``bool``, default True
            Whether to compute the Shannon entropy.
        curvature : ``bool``, default True
            Whether to compute the local curvature features.
        """

        enabled: bool = True
        reduced_stats: bool = True

        daisy: bool = True
        sift: bool = True
        orb: bool = True
        hog: bool = True
        basic: bool = True
        blur_effect: bool = True
        moments: bool = True
        centroid: bool = True
        moments_central: bool = True
        moments_hu: bool = True
        inertia_tensor_eigvals: bool = True
        shannon_entropy: bool = True
        curvature: bool = True

    PRESETS = cvtda.utils.FeatureExtractorBase.Presets(
        full=Settings(reduced_stats=False), reduced=Settings(), quick=Settings(curvature=False)
    )

    def __init__(self, n_jobs: int = -1, settings=Settings()):
        features: typing.List[typing.Tuple[str, BaseGeometricFeature]] = []
        if settings.enabled and settings.daisy:
            features.append(("daisy", DAISY()))
        if settings.enabled and settings.sift:
            features.append(("sift", SIFT(settings.reduced_stats)))
        if settings.enabled and settings.orb:
            features.append(("orb", ORB(settings.reduced_stats)))
        if settings.enabled and settings.hog:
            features.append(("hog", HOG()))
        if settings.enabled and settings.basic:
            features.append(("basic", BasicFeatures(settings.reduced_stats)))
        if settings.enabled and settings.blur_effect:
            features.append(("blur_effect", BlurEffect()))
        if settings.enabled and settings.centroid:
            features.append(("centroid", Centroid()))
        if settings.enabled and settings.inertia_tensor_eigvals:
            features.append(("inertia_tensor_eigvals", IntertiaTensorEigvals()))
        if settings.enabled and settings.moments:
            features.append(("moments", Moments(order=9)))
        if settings.enabled and settings.moments_central:
            features.append(("moments_central", MomentsCentral(order=9)))
        if settings.enabled and settings.moments_hu:
            features.append(("moments_hu", MomentsHu(order=9)))
        if settings.enabled and settings.shannon_entropy:
            features.append(("shannon_entropy", ShannonEntropy()))
        if settings.enabled and settings.curvature:
            features.append(("curvature", Curvature(settings.reduced_stats)))
        super().__init__("GrayGeometryExtractor", n_jobs, features)


class RGBGeometryExtractor(BaseGeometryExtractor):
    """
    Extracts geometric features from RGB images:
    - HOG [1], see :func:`skimage.feature.hog`
    - Moments ($M$): see :func:`skimage.measure.moments`
    - Centroid: $M_{10} / M{00}$, see :func:`skimage.measure.centroid`
    - Eigenvalues of the intertia vector, see :func:`skimage.measure.inertia_tensor_eigvals`
    - Central moments: see :func:`skimage.measure.moments_central`
    - Pearson correlations between color channels, see :func:`skimage.measure.pearson_corr_coeff`

    References
    ----------
    .. [1] D.G. Lowe "Object recognition from local scale-invariant features"
            https://ieeexplore.ieee.org/document/790410
    """

    @dataclasses.dataclass(frozen=True)
    class Settings:
        """
        Attributes
        ----------
        hog : ``bool``, default True
            Whether to compute HOG features.
        moments : ``bool``, default True
            Whether to compute the moments.
        centroid : ``bool``, default True
            Whether to compute the centroid.
        moments_central : ``bool``, default True
            Whether to compute the central moments.
        inertia_tensor_eigvals : ``bool``, default True
            Whether to compute the eigenvalues of the intertia vector.
        corr_coef : ``bool``, default True
            Whether to compute the correlations between color channels.
        """

        enabled: bool = True

        hog: bool = True
        moments: bool = True
        centroid: bool = True
        moments_central: bool = True
        inertia_tensor_eigvals: bool = True
        corr_coef: bool = True

    PRESETS = cvtda.utils.FeatureExtractorBase.Presets(full=Settings(), reduced=Settings(), quick=Settings())

    def __init__(self, n_jobs: int = -1, settings=Settings()):
        features: typing.List[typing.Tuple[str, BaseGeometricFeature]] = []
        if settings.enabled and settings.hog:
            features.append(("hog", HOG()))
        if settings.enabled and settings.centroid:
            features.append(("centroid", Centroid()))
        if settings.enabled and settings.inertia_tensor_eigvals:
            features.append(("inertia_tensor_eigvals", IntertiaTensorEigvals()))
        if settings.enabled and settings.moments:
            features.append(("moments", Moments(order=6)))
        if settings.enabled and settings.moments_central:
            features.append(("moments_central", MomentsCentral(order=6)))
        if settings.enabled and settings.corr_coef:
            features.append(("corr_coef", PearsonCorrCoeff()))
        super().__init__("RGBGeometryExtractor", n_jobs, features)


class MultidimensionalGeometryExtractor(BaseGeometryExtractor):
    """
    Extracts geometric features from grayscale images of higher dimensions (4D, 5D, etc.):
    - Basic local features, see :func:`skimage.feature.multiscale_basic_features`
    - Blur effect [1], see :func:`skimage.measure.blur_effect`
    - Moments ($M$): see :func:`skimage.measure.moments`
    - Centroid: $M_{10} / M{00}$, see :func:`skimage.measure.centroid`
    - Central moments: see :func:`skimage.measure.moments_central`
    - Eigenvalues of the intertia vector, see :func:`skimage.measure.inertia_tensor_eigvals`

    References
    ----------
    .. [1] Frederique Crete, Thierry Dolmiere, Patricia Ladret, Marina Nicolas
            "The blur effect: perception and estimation with a new no-reference perceptual blur metric"
            https://doi.org/10.1117/12.702790
    """

    @dataclasses.dataclass(frozen=True)
    class Settings:
        """
        Attributes
        ----------
        reduced_stats : ``bool``, default True
            Whether to reduce the set of statistical features. See :func:`utils.sequence2features` for details.
        basic : ``bool``, default True
            Whether to compute basic local features.
        blur_effect : ``bool``, default True
            Whether to compute blur effect.
        moments : ``bool``, default True
            Whether to compute the moments.
        centroid : ``bool``, default True
            Whether to compute the centroid.
        moments_central : ``bool``, default True
            Whether to compute the central moments.
        inertia_tensor_eigvals : ``bool``, default True
            Whether to compute the eigenvalues of the intertia vector.
        """

        enabled: bool = True
        reduced_stats: bool = True

        basic: bool = True
        blur_effect: bool = True
        moments: bool = True
        centroid: bool = True
        moments_central: bool = True
        inertia_tensor_eigvals: bool = True

    PRESETS = cvtda.utils.FeatureExtractorBase.Presets(
        full=Settings(reduced_stats=False), reduced=Settings(), quick=Settings()
    )

    def __init__(self, n_jobs: int = -1, settings=Settings()):
        features: typing.List[typing.Tuple[str, BaseGeometricFeature]] = []
        if settings.enabled and settings.basic:
            features.append(("basic", BasicFeatures(settings.reduced_stats)))
        if settings.enabled and settings.blur_effect:
            features.append(("blur_effect", BlurEffect()))
        if settings.enabled and settings.centroid:
            features.append(("centroid", Centroid()))
        if settings.enabled and settings.inertia_tensor_eigvals:
            features.append(("inertia_tensor_eigvals", IntertiaTensorEigvals()))
        if settings.enabled and settings.moments:
            features.append(("moments", Moments(order=9)))
        if settings.enabled and settings.moments_central:
            features.append(("moments_central", MomentsCentral(order=9)))
        super().__init__("MultidimensionalGeometryExtractor", n_jobs, features)


class GeometryExtractor(Extractor):
    """
    Extracts geometric features from images in any format, determines the format internally.
    """

    @dataclasses.dataclass(frozen=True)
    class Settings:
        """
        Attributes
        ----------
        rgb : ``RGBGeometryExtractor.Settings``
            Settings for the RGB extractor.
        gray : ``GrayGeometryExtractor.Settings``
            Settings for the grayscale extractor.
        multidimensional : ``MultidimensionalGeometryExtractor.Settings``
            Settings for the multidimensional extractor.
        """

        rgb: RGBGeometryExtractor.Settings = RGBGeometryExtractor.Settings()
        gray: GrayGeometryExtractor.Settings = GrayGeometryExtractor.Settings()
        multidimensional: MultidimensionalGeometryExtractor.Settings = MultidimensionalGeometryExtractor.Settings()

    PRESETS = cvtda.utils.FeatureExtractorBase.Presets(
        full=Settings(
            rgb=RGBGeometryExtractor.PRESETS.full,
            gray=GrayGeometryExtractor.PRESETS.full,
            multidimensional=MultidimensionalGeometryExtractor.PRESETS.full,
        ),
        reduced=Settings(
            rgb=RGBGeometryExtractor.PRESETS.reduced,
            gray=GrayGeometryExtractor.PRESETS.reduced,
            multidimensional=MultidimensionalGeometryExtractor.PRESETS.reduced,
        ),
        quick=Settings(
            rgb=RGBGeometryExtractor.PRESETS.quick,
            gray=GrayGeometryExtractor.PRESETS.quick,
            multidimensional=MultidimensionalGeometryExtractor.PRESETS.quick,
        ),
    )

    def __init__(self, n_jobs: int = -1, settings=Settings(), only_get_from_dump: bool = False):
        super().__init__(n_jobs=n_jobs, settings=settings, only_get_from_dump=only_get_from_dump)

        self.rgb_extractor_ = RGBGeometryExtractor(n_jobs=self.n_jobs_, settings=settings.rgb)
        self.gray_extractor_ = GrayGeometryExtractor(n_jobs=self.n_jobs_, settings=settings.gray)
        self.multidimensional_extractor_ = MultidimensionalGeometryExtractor(
            n_jobs=self.n_jobs_, settings=settings.multidimensional
        )

    def process_rgb_(
        self, rgb_images: numpy.ndarray, do_fit: bool, dump_name: typing.Optional[str] = None
    ) -> numpy.ndarray:
        return utils.process_iter_dump(self.rgb_extractor_, rgb_images, do_fit, self.features_dump_(dump_name))

    def feature_names_rgb_(self) -> typing.List[str]:
        return self.rgb_extractor_.feature_names()

    def process_gray_(
        self, gray_images: numpy.ndarray, do_fit: bool, dump_name: typing.Optional[str] = None
    ) -> numpy.ndarray:
        if len(gray_images.shape) == 3:
            return utils.process_iter_dump(self.gray_extractor_, gray_images, do_fit, self.features_dump_(dump_name))
        else:
            return utils.process_iter_dump(
                self.multidimensional_extractor_, gray_images, do_fit, self.features_dump_(dump_name)
            )

    def feature_names_gray_(self) -> typing.List[str]:
        if len(self.fit_dimensions_) == 2:
            return self.gray_extractor_.feature_names()
        else:
            return self.multidimensional_extractor_.feature_names()

    def explain_rgb_(self, feature_name: str, image: numpy.ndarray) -> FeatureExplanation:
        return self.rgb_extractor_.explain(feature_name, image)

    def explain_gray_(self, feature_name: str, image: numpy.ndarray) -> FeatureExplanation:
        if len(image.shape) == 2:
            return self.gray_extractor_.explain(feature_name, image)
        else:
            return self.multidimensional_extractor_.explain(feature_name, image)
