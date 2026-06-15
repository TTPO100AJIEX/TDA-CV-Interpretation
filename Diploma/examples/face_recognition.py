import os
import sys

sys.path.append(os.path.abspath(os.path.join("../src")))

import numpy
import torchvision
import torchvision.transforms.v2

transform = torchvision.transforms.v2.Compose(
    [torchvision.transforms.v2.Grayscale(), torchvision.transforms.v2.Resize((32, 32))]
)

train = torchvision.datasets.ImageFolder("../experiments/labeled_faces_in_the_wild/lfw/training", transform=transform)
train_images = numpy.array([train[i][0] for i in range(40)]) / 255
train_labels = numpy.array([train[i][1] for i in range(40)])

test = torchvision.datasets.ImageFolder("../experiments/labeled_faces_in_the_wild/lfw/testing", transform=transform)
test_images = numpy.array([test[i][0] for i in range(4, 14)]) / 255
test_labels = numpy.array([test[i][1] for i in range(4, 14)])


import cvtda.topology

extractor = cvtda.topology.FeatureExtractor()
train_features = extractor.fit_transform(train_images, "train")
test_features = extractor.transform(test_images, "test")

extractor = cvtda.topology.FeatureExtractor(return_diagrams=True, only_get_from_dump=True)
train_diagrams = extractor.fit_transform(train_images, "train")
test_diagrams = extractor.transform(test_images, "test")

import cvtda.face_recognition

cvtda.face_recognition.learn(
    train_images, train_features, train_labels, train_diagrams, test_images, test_features, test_labels, test_diagrams
)
