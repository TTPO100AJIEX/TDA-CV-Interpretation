import os
import sys

sys.path.append(os.path.abspath(os.path.join("../src")))

import numpy
import torchvision

train = torchvision.datasets.MNIST("mnist", train=True, download=True)
train_images = numpy.array([item[0] for item in train])[:100] / 255
train_labels = numpy.array([item[1] for item in train])[:100]

test = torchvision.datasets.MNIST("mnist", train=False, download=True)
test_images = numpy.array([item[0] for item in test])[:100] / 255
test_labels = numpy.array([item[1] for item in test])[:100]


import cvtda.topology

extractor = cvtda.topology.FeatureExtractor(n_jobs=1)
train_features = extractor.fit_transform(train_images)
test_features = extractor.transform(test_images)

import cvtda.classification

cvtda.classification.classify(
    train_images, train_features, train_labels, None, test_images, test_features, test_labels, None
)
