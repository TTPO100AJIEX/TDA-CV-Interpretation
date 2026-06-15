import numpy
import sklearn.base
import matplotlib.pyplot as plt

import cvtda.utils
import cvtda.logging


def calculate_binary_information_value(feature: numpy.ndarray, target: numpy.ndarray, bins: int) -> float:
    """
    Calculates information value for one feature in a binary classification problem.

    Parameters
    ----------
    feature : ``numpy.ndarray``
        (size `num_items`) One feature value for each object.
    target : ``numpy.ndarray``
        (size `num_items`) Target class labels for each object. Must be 0 or 1.
    bins : ``int``, default `10`
        The number of bins to categorize the quantitative features.

    Returns
    -------
    ``float``
        Information value score of the feature.
    """
    # First, we need to transform the quantitive feature into categorical.
    quantiles = numpy.linspace(0, 1, bins + 1)
    bins = numpy.unique(numpy.quantile(feature, quantiles))  # Make bins with quantiles
    bins[0] -= 0.0001
    bins[-1] += 0.0001
    feature = numpy.digitize(feature, bins, right=True)  # Digitize the feature

    # Compute events and non-events
    n, events = [], []
    for bin in range(1, len(bins)):
        mask = feature == bin
        n.append(mask.sum())
        events.append((target * mask).sum())
    n = numpy.array(n)
    events = numpy.array(events)

    non_events = n - events
    events_prc = numpy.maximum(events, 0.5) / events.sum()
    non_events_prc = numpy.maximum(non_events, 0.5) / non_events.sum()

    woe = numpy.log(events_prc / non_events_prc)  # weight of evidence
    iv = woe * (events_prc - non_events_prc)  # information value
    return iv.sum()


def calculate_information_value_one_feature(feature: numpy.ndarray, y_true: numpy.ndarray, bins: int) -> dict:
    """
    Calculates information value for one feature in a multi-class classification problem.
    We define information value for a multi-class classification problem as the average
    information value in the corresponding binary classification problems.

    Parameters
    ----------
    feature : ``numpy.ndarray``
        (size `num_items`) One feature value for each object.
    y_true : ``numpy.ndarray``
        (size `num_items`) Target class labels for each object.
    bins : ``int``, default `10`
        The number of bins to categorize the quantitative features.

    Returns
    -------
    ``float``
        Information value score of the feature.
    """
    IVs = []
    for class_idx in range(numpy.max(y_true)):
        target = (y_true == class_idx).astype(int)
        IVs.append(calculate_binary_information_value(feature, target, bins))
    return numpy.mean(IVs)


def calculate_information_value(
    features: numpy.ndarray, y_true: numpy.ndarray, bins: int = 10, n_jobs: int = -1
) -> numpy.ndarray:
    """
    Calculates information value for multiple features in a multi-class classification problem.

    Parameters
    ----------
    features : ``numpy.ndarray``
        (size `num_items x num_features`) Features for each object.
    y_true : ``numpy.ndarray``
        (size `num_items`) Target class labels for each object.
    bins : ``int``, default `10`
        The number of bins to categorize the quantitative features.
    n_jobs : ``int``, default: ``-1``
        The number of jobs to use for the computation. See :mod:`joblib` for details.

    Returns
    -------
    ``numpy.ndarray``
        (size `num_features`) Information value score for each feature.
    """
    params = [(features[:, idx], y_true, bins) for idx in range(features.shape[1])]
    iv_generator = cvtda.utils.parallel(
        calculate_information_value_one_feature, params, return_as="generator", n_jobs=n_jobs
    )
    pbar = cvtda.logging.logger().pbar(iv_generator, total=features.shape[1], desc="information values")
    return numpy.array(list(pbar))


class InformationValueFeatureSelector(sklearn.base.TransformerMixin):
    """
    Selects best features based on their information value on the training set.

    References
    ----------
    .. [1] David Bridston Osteyee , Irving John Good
            "Information, Weight of Evidence. The Singularity Between Probability Measures and Signal Detection"
            https://doi.org/10.1007/BFb0064126
    """

    def __init__(self, n_jobs: int = -1, bins: int = 10, threshold: float = 0.5):
        self.fitted_ = False
        self.n_jobs_ = n_jobs

        self.bins_ = bins
        self.threshold_ = threshold

    def fit(self, features: numpy.ndarray, target: numpy.ndarray):
        """
        Leaves only good features as selected by :meth:`fit` in the feature matrix.

        Parameters
        ----------
        features : ``numpy.ndarray``
            (size `num_items x num_features`) Features for each object.
        target : ``numpy.ndarray``
            (size `num_items`) Target class labels for each object.
        """
        cvtda.logging.logger().print("Fitting the information value feature selector")
        self.IV_ = calculate_information_value(features, target, bins=self.bins_, n_jobs=self.n_jobs_)
        self.good_features_idx_ = numpy.where(self.IV_ > self.threshold_)[0]

        cvtda.logging.logger().print("Fitting complete")
        self.fitted_ = True
        return self

    def transform(self, features: numpy.ndarray) -> numpy.ndarray:
        """
        Leaves only good features as selected by :meth:`fit` in the feature matrix.

        Parameters
        ----------
        features : ``numpy.ndarray``
            (size `num_items x num_features`) Features for each object.

        Returns
        -------
        ``numpy.ndarray``
            (size `num_items x num_good_features`) Selected features for each object.
        """
        assert self.fitted_ is True, "fit() must be called before transform()"
        return features[:, self.good_features_idx_]

    def hist(self, bins: int = 50) -> plt.Figure:
        """
        Histogram of information value scores for all features seen in :meth:`fit`.

        Parameters
        ----------
        bins : ``int``, default `50`
            Number of bins on the histogram.

        Returns
        -------
        ``matplotlib.pyplot.Figure``
            A :mod:`matplotlib` figure with the histogram.
        """
        assert self.fitted_ is True, "fit() must be called before hist()"
        return plt.hist(self.IV_, bins=bins)
