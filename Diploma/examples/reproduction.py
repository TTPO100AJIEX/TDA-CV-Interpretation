import sys

sys.path.append("../src")

import warnings

warnings.filterwarnings("ignore")

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--quick", action=argparse.BooleanOptionalAction, default=False)
parser.add_argument("--clean", action=argparse.BooleanOptionalAction, default=False)
args = parser.parse_args()

import gc
import os
import glob
import time
import numpy
import torch
import shutil
import PIL.Image
import torchvision
import cvtda.utils
import cvtda.logging
import cvtda.topology
import cvtda.autoencoder
import cvtda.segmentation
import cvtda.classification
import torchvision.tv_tensors
import cvtda.face_recognition
import sklearn.model_selection
import torchvision.transforms.v2

if args.quick:
    FRACTION = {
        "mnist": 0.2,
        "fmnist": 0.2,
        "cifar-10": 0.25,
        "imagenette": 0.4,
        "gtsrb": 0.25,
        "faces": 1,
        "lfw": 0.4,
        "midv500p": 1,
    }

if args.quick:
    preset = cvtda.topology.FeatureExtractor.PRESETS.quick
    classification_params = dict(
        nn_epochs=6,
        grad_boost_max_iter=15,
        grad_boost_max_depth=3,
        xgboost_n_classifiers=20,
        xgboost_max_depth=3,
        catboost_iterations=50,
        catboost_depth=3,
    )
    face_recognition_params = dict(nn_epochs=6)
    autoencoders_params = dict(nn_epochs=6)
    segmentation_params = dict(n_epochs=20)
else:
    preset = cvtda.topology.FeatureExtractor.PRESETS.reduced
    classification_params = dict()
    face_recognition_params = dict()
    autoencoders_params = dict()
    segmentation_params = dict()


def subset(images: numpy.ndarray, labels: numpy.ndarray, fraction: float):
    if fraction == 1:
        return images, labels
    idxs, _ = sklearn.model_selection.train_test_split(
        numpy.arange(len(labels)), stratify=labels, train_size=fraction, random_state=42
    )
    return images[idxs], labels[idxs]


def process_impl(
    name, train_images, train_labels, test_images, test_labels, with_labels: bool = True, should_clean: bool = True
):
    folder = f"{name}/results"
    if args.clean and should_clean:
        shutil.rmtree(folder, ignore_errors=True)

    if args.quick:
        train_images, train_labels = subset(train_images, train_labels, FRACTION[name])
        test_images, test_labels = subset(test_images, test_labels, FRACTION[name])

        if len(train_images.shape) == 4:
            train_images = cvtda.utils.rgb2gray(train_images)
            test_images = cvtda.utils.rgb2gray(test_images)

    print(f"Processing {name}: train = {train_images.shape}, test = {test_images.shape}")
    features_extractor = cvtda.topology.FeatureExtractor(
        settings=preset, n_jobs=-1, return_diagrams=False, only_get_from_dump=False
    )
    train_features = features_extractor.fit_transform(train_images, f"{folder}/train")
    test_features = features_extractor.transform(test_images, f"{folder}/test")

    diagrams_extractor = cvtda.topology.FeatureExtractor(
        settings=preset, n_jobs=1, return_diagrams=True, only_get_from_dump=True
    )
    train_diagrams = diagrams_extractor.fit_transform(train_images, f"{folder}/train")
    test_diagrams = diagrams_extractor.transform(test_images, f"{folder}/test")

    gc.collect()
    if with_labels:
        return (
            train_images,
            train_features,
            train_labels,
            train_diagrams,
            test_images,
            test_features,
            test_labels,
            test_diagrams,
        )
    else:
        return (train_images, train_features, train_diagrams, test_images, test_features, test_diagrams)


def process(name, train, test, with_labels: bool = True, should_clean: bool = True):
    folder = f"{name}/results"
    if args.clean and should_clean:
        shutil.rmtree(folder, ignore_errors=True)

    train_images = numpy.array([numpy.array(item[0]) / 255 for item in train])
    train_labels = numpy.array([item[1] for item in train])

    test_images = numpy.array([numpy.array(item[0]) / 255 for item in test])
    test_labels = numpy.array([item[1] for item in test])

    return process_impl(name, train_images, train_labels, test_images, test_labels, with_labels, should_clean)


def classification(name, train, test):
    start = time.time()
    results = cvtda.classification.classify(
        *process(name, train, test),
        dump_name=f"{name}/results/classification",
        only_get_from_dump=False,
        **classification_params,
    )
    print(f"{name} classification done in {int(time.time() - start)}s")
    print(results)
    gc.collect()


def face_recognition(name, train, test):
    start = time.time()
    folder = f"{name}/results/face_recognition"
    fig = cvtda.face_recognition.learn(*process(name, train, test), dump_name=folder, **face_recognition_params)
    fig.tight_layout()
    fig.savefig(f"{folder}/distributions.svg")
    fig.savefig(f"{folder}/distributions.png")
    print(f"{name} face_recognition done in {int(time.time() - start)}s")
    gc.collect()


def compression(name, train, test):
    start = time.time()
    results = cvtda.autoencoder.try_autoencoders(
        *process(name, train, test, with_labels=False, should_clean=False),
        dump_name=f"{name}/results/compression",
        only_get_from_dump=False,
        **autoencoders_params,
    )
    print(f"{name} compression done in {int(time.time() - start)}s")
    print(results)
    gc.collect()


def segmentation(name):
    start = time.time()

    transforms = torchvision.transforms.v2.Compose(
        [
            torchvision.transforms.v2.CenterCrop((224, 224)),
            torchvision.transforms.v2.Resize((64, 64)),
            torchvision.transforms.v2.ToDtype(torch.float32, scale=True),
        ]
    )

    def transforms_wrapper(image, mask):
        image = torchvision.tv_tensors.Image(image)
        mask = torchvision.tv_tensors.Mask(mask)
        return transforms(image, mask)

    def load_dataset(path):
        images, masks = [], []
        for filename in glob.glob(f"{path}/image/*"):
            image, mask = transforms_wrapper(
                PIL.Image.open(filename), PIL.Image.open(filename.replace("image", "mask").replace("jpg", "png"))
            )
            images.append(image.permute((1, 2, 0)).numpy())
            masks.append(mask.squeeze().numpy())
        return numpy.array(images), numpy.array(masks)

    train_images, train_features, train_masks, _, test_images, test_features, test_masks, _ = process_impl(
        name, *load_dataset(f"{name}/train"), *load_dataset(f"{name}/val")
    )
    result = cvtda.segmentation.segment(
        train_images,
        train_features,
        train_masks,
        test_images,
        test_features,
        test_masks,
        dump_name=f"{name}/results/segmentation",
        only_get_from_dump=False,
        remove_cross_maps=True,
        **segmentation_params,
    )
    print(f"{name} segmentation done in {int(time.time() - start)}s")
    print(result)


start = time.time()
with cvtda.logging.DevNullLogger():
    classification(
        "mnist",
        torchvision.datasets.MNIST("mnist", train=True, download=True),
        torchvision.datasets.MNIST("mnist", train=False, download=True),
    )

    classification(
        "fmnist",
        torchvision.datasets.FashionMNIST("fmnist", train=True, download=True),
        torchvision.datasets.FashionMNIST("fmnist", train=False, download=True),
    )

    classification(
        "cifar-10",
        torchvision.datasets.CIFAR10("cifar-10", train=True, download=True),
        torchvision.datasets.CIFAR10("cifar-10", train=False, download=True),
    )

    do_download = not os.path.exists("./imagenette/imagenette2-160")
    transform = torchvision.transforms.v2.Resize((32, 32), antialias=True)
    classification(
        "imagenette",
        torchvision.datasets.Imagenette(
            "imagenette", split="train", size="160px", transform=transform, download=do_download
        ),
        torchvision.datasets.Imagenette("imagenette", split="val", size="160px", transform=transform, download=False),
    )

    transform = torchvision.transforms.v2.Resize((32, 32), antialias=True)
    classification(
        "gtsrb",
        torchvision.datasets.GTSRB("gtsrb", split="train", transform=transform, download=True),
        torchvision.datasets.GTSRB("gtsrb", split="test", transform=transform, download=True),
    )

    # Download manually into 'faces' folder from https://cam-orl.co.uk/facedatabase.html
    transform = torchvision.transforms.v2.Compose(
        [
            torchvision.transforms.v2.Resize((32, 32), antialias=True),
            torchvision.transforms.v2.Grayscale(),
        ]
    )
    face_recognition(
        "faces",
        torchvision.datasets.ImageFolder("faces/training", transform=transform),
        torchvision.datasets.ImageFolder("faces/testing", transform=transform),
    )

    # Download manually into 'lfw' folder from https://vis-www.cs.umass.edu/lfw
    transform = torchvision.transforms.v2.Compose(
        [
            torchvision.transforms.v2.CenterCrop((128, 128)),
            torchvision.transforms.v2.Resize((32, 32), antialias=True),
        ]
    )
    face_recognition(
        "lfw",
        torchvision.datasets.ImageFolder("lfw/training", transform=transform),
        torchvision.datasets.ImageFolder("lfw/testing", transform=transform),
    )

    compression(
        "mnist",
        torchvision.datasets.MNIST("mnist", train=True),
        torchvision.datasets.MNIST("mnist", train=False),
    )

    compression(
        "cifar-10",
        torchvision.datasets.CIFAR10("cifar-10", train=True),
        torchvision.datasets.CIFAR10("cifar-10", train=False),
    )

    transform = torchvision.transforms.v2.Resize((32, 32), antialias=True)
    compression(
        "imagenette",
        torchvision.datasets.Imagenette("imagenette", split="train", size="160px", transform=transform),
        torchvision.datasets.Imagenette("imagenette", split="val", size="160px", transform=transform),
    )

    transform = torchvision.transforms.v2.Resize((32, 32), antialias=True)
    compression(
        "gtsrb",
        torchvision.datasets.GTSRB("gtsrb", split="train", transform=transform),
        torchvision.datasets.GTSRB("gtsrb", split="test", transform=transform),
    )

    # Download manually into 'midv500' folder: https://doi.org/10.48550/arXiv.1807.05786
    segmentation("midv500p")

print(f"Total runtime: {int(time.time() - start)}s")
