import numpy
import sklearn.metrics


def estimate_quality(pred_proba: numpy.ndarray, masks: numpy.ndarray) -> dict:
    """
    Quality metrics for segmentation task.

    Parameters
    ----------
    pred_proba : ``numpy.ndarray``
        (size `num_items x width x height`) Predicted probability estimates for each pixel.
    masks : ``numpy.ndarray``
        (size `num_items x width x height`) Target segmentation masks.

    Returns
    -------
    ``dict``
        1. Accuracy. See :func:`sklearn.metrics.accuracy_score`.
        2. AUC-ROC. See :func:`sklearn.metrics.roc_auc_score`.
        3. Precision. See :func:`sklearn.metrics.precision_score`.
        4. Recall. See :func:`sklearn.metrics.recall_score`.
        5. F1-score. See :func:`sklearn.metrics.f1_score`.
        6. Jaccard score. See :func:`sklearn.metrics.jaccard_score`.
    """

    # Segmentation is basically just per-pixel classification
    scores = pred_proba.flatten()
    preds = (scores >= 0.5).astype(numpy.int8)
    targets = masks.flatten().astype(numpy.int8)
    return {
        "Accuracy": sklearn.metrics.accuracy_score(targets, preds),
        "AUC-ROC": sklearn.metrics.roc_auc_score(targets, scores),
        "Precision": sklearn.metrics.precision_score(targets, preds, zero_division=0),
        "Recall": sklearn.metrics.recall_score(targets, preds, zero_division=0),
        "F1-score": sklearn.metrics.f1_score(targets, preds, zero_division=0),
        "Jaccard score": sklearn.metrics.jaccard_score(targets, preds, zero_division=0),
    }
