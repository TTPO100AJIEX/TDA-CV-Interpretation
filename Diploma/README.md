# A feature engineering framework for computer vision based on topological data analysis

The library provides tools to simplify the application of topological data analysis to arbitrary computer vision problems and is equipped with related tools to solve classification, face recognition, compression, and segmentation problems.

The package is maintained by Aleksandr Abramov at HSE University, Faculty of Computer Science. The work was supervised by Dr. Vsevolod Chernyshev at Ulm University, suggesting the design and methodologies for the methods implemented.

## Installation

The library can be installed directly from the Python package index using `pip install cv-tda`. The core package contains the algebraic topology toolkit necessary to extract topological features from images.

The library comes with four optional packages providing methods to solve classification, face recognition, compression, and segmentation problems. To use them, you need to install additional dependencies with the respective commands: `pip install cv-tda[classification]`, `pip install cv-tda[facerecognition]`, `pip install cv-tda[autoencoder]`, and `pip install cv-tda[segmentation]`. Moreover, to use the trainable persistence diagram vectorization technique described in the study, you should install version 0.0.0 of the torchph utility with `pip install git+https://github.com/c-hofer/torchph.git@master`.

Please note that the library requires Python 3.10. Correct functioning with other versions of the interpreter is not guaranteed.

We identified no limitations in using the tools on various operating systems.

There are no CPU requirements, but limiting the resource availability will increase the running time of most methods. The amount of RAM required depends on the quantity and size of analyzed images: at least 32 GB is required to process 60,000 color photographs of size 48 x 48. A GPU is not required to use the library, although it can speed up the operation of individual functions.

## API

### Topological feature extraction algorithm

The main element of the library is the `FeatureExtractor` class of `cvtda.topology`, which implements a comprehensive algorithm for topological analysis of a set of images.

The class implements the scikit-learn's `TransformerMixin` interface, thus providing three standard methods: `fit` for tuning the algorithm to the training set, `transform` for transforming the test set, and `fit_transform`, which combines the previous two.

The input for these methods is a single numpy array containing the images with pixel values between 0 and 1. For monochrome images, the array must have three dimensions: the image index, its height, and width, respectively. In the case of color images, the RGB channels should be placed as the last (fourth) dimension. Working with other image formats is not supported.

The main parameters of the class are the following values
1) `n_jobs` is the maximum number of simultaneously executed tasks, the degree of parallelization. The default value (`-1`) corresponds to the use of all available CPU resources.
2) `reduced` is a flag indicating which version of the algorithm to use: reduced (`True`) or full (`False`). A detailed description of both methods is presented in the main study.
3) `return_diagrams` indicates the algorithm to return the persistence diagrams instead of the final features.

The output without the `return_diagrams` parameter is a two-dimensional numpy array containing a feature description for each image of the original dataset. When `return_diagrams` is specified, the algorithm returns a list with each element containing a set of persistence diagrams for the corresponding image.

### Classification quality evaluation

To evaluate the topological approach applied to image classification, the `classification` package provides the `classify` method, which generates predictions and quality estimates with nine machine-learning models described in the paper.

The inputs are the training and test sets of images with target classes, extracted features, and, optionally, persistence diagrams. If the latter is not specified, the corresponding method (a neural network based on a trainable persistence diagram vectorization technique) is excluded from the analysis.

The output is a pandas table with quality metrics for all models and a matplotlib image of their confusion matrices.

The parameters include:
1) `label_names` – class names to include in confusion matrices.
2) `confusion_matrix_include_values` – when disabled, will remove the quantities of each type of errors from the confusion matrices.
3) `nn_device`, `xgboost_device`, and `catboost_device` let you use the GPU to train the corresponding models.
4) A set of specific values to fine-tune the classifiers.

### Face recognition quality evaluation

Similar to classification, to evaluate the quality of the extracted features in a face recognition task, the `face_recognition` package provides the `learn` method, which trains multiple models and evaluates their quality.

The inputs are also two sets of photographs with the identifiers of the respective people, feature matrices, and persistence diagrams. Unlike classification, persistence diagrams are required for face recognition.

The outputs are the scatter plots showing the distributions of distances between latent representations of photographs of the same person and different people, formed by corresponding models.

### Compression quality evaluation

To analyze the behavior of topological features in image compression, the `autoencoder` package provides `try_autoencoders` - a method similar to the previously described, which fits four autoencoders to a set of training data and evaluates the models with a test dataset. The output is a pandas table with quality metrics of all models.

### Segmentation quality evaluation

Similarly, the `segmentation` package provides the `segment` function to evaluate the topological features in finding individual objects in images. The inputs are also training and test samples, but, unlike the methods described earlier, the segmentation maps of all images must be provided as the target variable. The output is a pandas table with quality metrics of estimated models and the control neural network with the first part of U-Net removed, allowing one to assess the quality of virtually random predictions made without any information about the original image. 

It is important to note that the segmentation methods, including the segment function, do not use persistence diagrams and do not accept them as input.

### Logging utilities

To monitor the library's runtime, the `logging` package implements a respective mechanism, allowing for flexible customization of the output and the progress indicators for some methods. For this, we provide the `BaseLogger` interface and two implementations: the `CLILogger`, which outputs messages to the command line, and `DevNullLogger`, which hides all messages.

To select one of the implementations, the user should create a context, initializing the corresponding class as follows:

```python
with CLILogger():
    # Code that will print to the command line
with DevNullLogger():
    # Code that will not print anything
```

The architecture allows, if necessary, to provide custom logging implementations without changing the library's source code. To do this, you need to create a new class implementing the `BaseLogger` interface and use it when initializing the context as follows:

```python
class CustomLogger(BaseLogger):
    def print(self, data):
        # User code
    def pbar(self, data], total,  desc):
        # User code
    def zip(self,  *iterables, desc):
        # User code
    def set_pbar_postfix(self, pbar, data):
        # User code

with CustomLogger():
    # Code that uses CustomLogger to print runtime information
```

### Backup utilities

The `dumping` package implements a similar context-based mechanism to backup intermediate results and restore them from persistent storage. The `BaseDumper` class describes the interface of the corresponding methods, and two implementations are supplied with the library, `NumpyDumper` and `DevNullDumper`, which allow you to save numpy arrays to disk and disable backup processes, respectively. To choose the implementation, you should create a context, initializing the corresponding class:

```python
with NumpyDumper():
    # Code that will save the results to disk
with DevNullDumper():
    # Code that will not backup intermediate results
```

To specify the names of backups (for `NumpyDumper`, it coincides with the directory to save the files), you should pass the `dump_name` parameter to the method that supports this feature. Thus, most implementations in the library, including the `FeatureExtractor`, accept this value as an optional parameter in `fit`, `transform`, and `fit_transform` methods.

Moreover, all methods supporting this feature provide an `only_get_from_dump` parameter telling the library to unconditionally read previously computed results from the backup, which speeds up the recovery.

Finally, the context-based mechanism lets you provide custom implementations without changing the library's source code, if necessary. To do this, you need to create a new class implementing the `BaseDumper` interface and use it when creating a context as follows:

```python
class CustomDumper(BaseDumper):
    def execute(self, function, name, *function_args):
        # User code
    def save_dump(self, data, name):
        # User code
    def has_dump(self, name):
        # User code
    def get_dump_impl_(self, name):
        # User code

with CustomDumper():
    # Code that uses CustomDumper to backup intermediate results
```

### Other elements provided by the library

In addition to the main tools described earlier, the library provides several auxiliary implementations, a brief documentation of which is presented further. Please refer to the source code for more detailed descriptions of those methods, including the parameters and data requirements.

1) `utils` package:
    1) `DuplicateFeaturesRemover` implements an efficient algorithm to remove the features with the same values for all images.
    2) `image2pointcloud` transforms a set of images of any dimension into a set of metric spaces as described in the paper.
    3) `rgb2gray` and `rgb2hsv` convert a set of color images into the corresponding representation: monochrome or HSV.
    4) `sequence2features` calculates the statistical characteristics of a numerical sequence as described in the study.
    5) `spread_points` provides a given number of uniformly distributed points on a segment of a certain length.
2) `topology` package:
    1) `DiagramVectorizer` implements the statistical method of persistence diagram vectorization.
    2) `FiltrationExtractor` and `FiltrationsExtractor` implement techniques to extract topological features from monochrome images by binarizing them with subsequent application of various filtrations.
    3) `GrayGeometryExtractor` and `GeometryExtractor` compute geometric features for a set of images.
    4) `GreyscaleExtractor` implements the simplest method to extract topological features from monochrome images by directly constructing cubical complexes for them.
    5) `PointCloudsExtractor` analyzes images as metric spaces using Vietoris–Rips complexes.
3) The `neural_network` package provides utilities to develop neural networks to evaluate the quality when solving classification and face recognition problems.
4) `classification` package:
    1) `estimate_quality` calculates quality metrics when solving a classification problem.
    2) `NNClassifier` is a general implementation of all neural network classification models described in the paper.
5) `BaseLearner`, `DiagramsLearner`, `NNLearner`, and `SimpleTopologicalLearner` of the `face_recognition` package implement corresponding face recognition models.
6) The `autoencoder` package implements corresponding machine-learning models (class `Autoencoder`) and tools (`estimate_quality` method) to evaluate image compression with various techniques.
7) The `segmentation` package provides a basic implementation for neural networks with U-Net architecture (`MiniUnet` class) and a function to evaluate the segmentation quality (`estimate_quality`).

## Typical use cases

### Topological feature extraction

The key use case of the library is to extract topological features from a set of images, performed by the following program:

```python
import cvtda.topology
extractor = cvtda.topology.FeatureExtractor()
train_features = extractor.fit_transform(train_images)
test_features = extractor.transform(test_images)
```
Moreover, specialized methods of the `topology` package, which have a similar API to the `FeatureExtractor` class, can be employed to develop more complex models.

### Classification quality evaluation

For a preliminary assessment of the applicability of topological data analysis to classify a set of images, the following program can be used:

```python
# Read the training set into train_images and train_labels
# Read the test set into test_images and test_labels

import cvtda.topology
extractor = cvtda.topology.FeatureExtractor()
train_features = extractor.fit_transform(train_images)
test_features = extractor.transform(test_images)

import cvtda.classification
cvtda.classification.classify(
    train_images, train_features, train_labels, None,
    test_images, test_features, test_labels, None
)
```

### Face recognition quality evaluation

To evaluate the quality of face recognition, one should additionally restore the persistence diagrams from a backup before calling `learn` as follows:

```python
# Read the training set into train_images and train_labels
# Read the test set into test_images and test_labels

import cvtda.topology
extractor = cvtda.topology.FeatureExtractor()
train_features = extractor.fit_transform(train_images, "train")
test_features = extractor.transform(test_images, "test")

extractor = cvtda.topology.FeatureExtractor(
    return_diagrams = True,
    only_get_from_dump = True
)
train_diagrams = extractor.fit_transform(train_images, "train")
test_diagrams = extractor.transform(test_images, "test")

import cvtda.face_recognition
cvtda.face_recognition.learn(
    train_images, train_features, train_labels, train_diagrams,
    test_images, test_features, test_labels, test_diagrams
)
```

## Typical error messages

Typical messages arising from the library and the actions required to fix them are presented in the Table below.

<table>
    <thead>
        <tr>
            <th>Error message</th>
            <th>Description</th>
            <th>Actions to fix</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>There is no dump at …</td>
            <td>An attempt to load a non-existent backup</td>
            <td>

1) Ensure the correct path to an existing backup is specified
2) Call the method without the only_get_from_dump flag to create a backup
            </td>
        </tr>
        <tr>
            <td>…d images are not supported</td>
            <td rowspan=2>The format of input images in not supported</td>
            <td rowspan=2>Ensure the input data represents a set of monochrome or RGB images</td>
        </tr>
        <tr>
            <td>Images with … channels are not supported</td>
        </tr>
        <tr>
            <td>Bad image format: should be [0, 1]; received …</td>
            <td>The format of input images in not supported</td>
            <td>Ensure the pixel values ​​of all images are between 0 and 1</td>
        </tr>
        <tr>
            <td>fit() must be called before transform()</td>
            <td>`transform` called for a class, that was not configured beforehand using `fit`</td>
            <td>Call `fit`</td>
        </tr>
        <tr>
            <td>The pipeline is fit for … Cannot use it with …</td>
            <td>`transform` called with images in a different format than the one seen in `fit`</td>
            <td>Ensure the image formats in the training and test sets match</td>
        </tr>
    </tbody>
</table>
