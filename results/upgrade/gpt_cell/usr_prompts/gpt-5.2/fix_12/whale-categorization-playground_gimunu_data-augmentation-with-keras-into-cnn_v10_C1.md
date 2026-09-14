# Goal

I want you to improve my Kaggle competition solution to increase the score toward a target. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (evaluation score improvement); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Ensure it runs end-to-end and produces a valid submission file.


# 1. Kaggle task description

## Task
Predict the individual whale species in images.

## Metric
Mean Average Precision @ 5 (MAP@5).

## Submission Format
For each `Image` in the test set, you may predict up to 5 labels for the whale `Id`. Whales that are not predicted to be one of the labels in the training data should be labeled as `new_whale`. The file should contain a header and have the following format:

```
Image,Id
00029b3a.jpg,new_whale w_1287fbc w_98baff9 w_7554f44 w_1eafe46
0003c693.jpg,new_whale w_1287fbc w_98baff9 w_7554f44 w_1eafe46
...
```

## Dataset
This training data contains thousands of images of humpback whale flukes. Individual whales have been identified by researchers and given an `Id`. The challenge is to predict the whale `Id` of images in the test set. What makes this such a challenge is that there are only a few examples for each of 3,000+ whale Ids.

- **train.zip** - a folder containing the training images
- **train.csv** - maps the training `Image` to the appropriate whale `Id`. Whales that are not predicted to have a label identified in the training data should be labeled as `new_whale`.
- **test.zip** - a folder containing the test images to predict the whale `Id`
- **sample_submission.csv** - a sample submission file in the correct format

# 2. Python version

3.6

# 3. Installed packages

No external packages required in the script and installed.

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (51 lines)
            sample_submission.csv (2611 lines)
            sample_submission.csv.zip (15.5 kB)
            test.zip (72.1 MB)
            train.csv (7241 lines)
            train.csv.zip (68.3 kB)
            train.zip (200.4 MB)
            test/
                11482c0d.jpg (27.8 kB)
                68edd063.jpg (11.5 kB)
                ... and 2608 other files
                test/
            train/
                6fc790ac.jpg (17.1 kB)
                f4bb7bb0.jpg (11.6 kB)
                ... and 7238 other files
                train/
            whale-categorization-playground/
                description.md (51 lines)
                sample_submission.csv (2611 lines)
                ... and 5 other files
                test/
                    11482c0d.jpg (27.8 kB)
                    68edd063.jpg (11.5 kB)
                    ... and 2608 other files
                    test/
                train/
                    6fc790ac.jpg (17.1 kB)
                    f4bb7bb0.jpg (11.6 kB)
                    ... and 7238 other files
                    train/
                whale-categorization-playground/
        input/
            description.md (51 lines)
            sample_submission.csv (2611 lines)
            sample_submission.csv.zip (15.5 kB)
            test.zip (72.1 MB)
            train.csv (7241 lines)
            train.csv.zip (68.3 kB)
            train.zip (200.4 MB)
            test/
                11482c0d.jpg (27.8 kB)
                68edd063.jpg (11.5 kB)
                ... and 2608 other files
                test/
                    11482c0d.jpg (27.8 kB)
                    68edd063.jpg (11.5 kB)
                    ... and 2608 other files
                    test/
            train/
                6fc790ac.jpg (17.1 kB)
                f4bb7bb0.jpg (11.6 kB)
                ... and 7238 other files
                train/
                    6fc790ac.jpg (17.1 kB)
                    f4bb7bb0.jpg (11.6 kB)
                    ... and 7238 other files
                    train/
            whale-categorization-playground/
                description.md (51 lines)
                sample_submission.csv (2611 lines)
                ... and 5 other files
                test/
                    11482c0d.jpg (27.8 kB)
                    68edd063.jpg (11.5 kB)
                    ... and 2608 other files
                    test/
                train/
                    6fc790ac.jpg (17.1 kB)
                    f4bb7bb0.jpg (11.6 kB)
                    ... and 7238 other files
                    train/
                whale-categorization-playground/
        working/
            whale-categorization-playground/
                description.md (51 lines)
                sample_submission.csv (2611 lines)
                ... and 5 other files
                test/
                    11482c0d.jpg (27.8 kB)
                    68edd063.jpg (11.5 kB)
                    ... and 2608 other files
                    test/
                train/
                    6fc790ac.jpg (17.1 kB)
                    f4bb7bb0.jpg (11.6 kB)
                    ... and 7238 other files
                    train/
                whale-categorization-playground/
```

-> data/sample_submission.csv has 2610 rows and 2 columns.
The columns are: Image, Id

-> data/train.csv has 7240 rows and 2 columns.
The columns are: Image, Id

-> data/whale-categorization-playground/sample_submission.csv has 2610 rows and 2 columns.
The columns are: Image, Id

-> data/whale-categorization-playground/train.csv has 7240 rows and 2 columns.
The columns are: Image, Id

-> input/sample_submission.csv has 2610 rows and 2 columns.
The columns are: Image, Id

-> input/train.csv has 7240 rows and 2 columns.
The columns are: Image, Id

-> (stopped after 10 files for performance)

# 5. Target score

0.32756

# 6. Current score

0.00068

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plan

- What this solution (achieved 0.00068) has done: 'Diagnosis: The crash happens in cell 9 because `x` is being overwritten inside the loop with a standardized array whose shape no longer matches a single image. In particular, the fallback `ImageDataGenerator._standardize` expects 4D batches, but the code passes a 3D array, leading to broadcasting that expands the data and makes `x` much larger (262144 elements). When the code then reshapes `x` to `(1, 64, 64, 1)`, the element count mismatch triggers the `ValueError`.  
Patch summary: Keep the core logic the same but ensure we always standardize a 4D single-image batch and avoid reusing `x` as a mutable intermediate that can accidentally grow in size. We write predictions using a per-image variable (`x_batch`) shaped `(1, SIZE, SIZE, 1)` and call the appropriate standardization method with that same 4D shape.  
Updated cells: Only cell 9 is modified.  
Compatibility notes for cell k+1: No variables or interfaces used by later cells are changed; we only adjust local variables inside the submission-writing loop. Output file name and CSV format remain identical.  
Assumptions: We assume the environment is using the fallback `ImageDataGenerator` without a public `.standardize` method (as in the provided code), but the patch also works if a real Keras generator is present by standardizing a 4D batch.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd
import seaborn as sns
from glob import glob
from PIL import Image
import matplotlib.pylab as plt
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split

from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION"] = "2"

import types

tf = types.SimpleNamespace()
keras = types.SimpleNamespace()
tf.keras = keras


class _TFImportStubError(RuntimeError):
    pass


def _unavailable(*args, **kwargs):
    raise _TFImportStubError(
        "TensorFlow could not be imported due to a protobuf incompatibility in this environment."
    )


Sequential = _unavailable
Dense = Dropout = Flatten = Conv2D = MaxPooling2D = _unavailable
K = types.SimpleNamespace()
ImageDataGenerator = _unavailable


## === cell 1
train_images = glob("../input/train/*jpg")
test_images = glob("../input/test/*jpg")
df = pd.read_csv("../input/train.csv")

df["Image"] = df["Image"].map( lambda x : "../input/train/"+x)
ImageToLabelDict = dict( zip( df["Image"], df["Id"]))


## === cell 2
SIZE = 64
def ImportImage( filename):
    img = Image.open(filename).convert("LA").resize( (SIZE,SIZE))
    return np.array(img)[:,:,0]
train_img = np.array([ImportImage( img) for img in train_images])
x = train_img


## === cell 3
print("%d training images" % x.shape[0])

print("Nbr of samples/class\tNbr of classes")
for index, val in df["Id"].value_counts().value_counts().sort_index().items():
    print("%d\t\t\t%d" % (index, val))


## === cell 4
class LabelOneHotEncoder():
    def __init__(self):
        self.ohe = OneHotEncoder()
        self.le = LabelEncoder()
    def fit_transform(self, x):
        features = self.le.fit_transform( x)
        return self.ohe.fit_transform( features.reshape(-1,1))
    def transform( self, x):
        return self.ohe.transform( self.la.transform( x.reshape(-1,1)))
    def inverse_tranform( self, x):
        return self.le.inverse_transform( self.ohe.inverse_tranform( x))
    def inverse_labels( self, x):
        return self.le.inverse_transform( x)

y = list(map(ImageToLabelDict.get, train_images))
lohe = LabelOneHotEncoder()
y_cat = lohe.fit_transform(y)


## === cell 5
def plotImages( images_arr, n_images=4):
    fig, axes = plt.subplots(n_images, n_images, figsize=(12,12))
    axes = axes.flatten()
    for img, ax in zip( images_arr, axes):
        if img.ndim != 2:
            img = img.reshape( (SIZE,SIZE))
        ax.imshow( img, cmap="Greys_r")
        ax.set_xticks(())
        ax.set_yticks(())
    plt.tight_layout()


## === cell 6
plotImages( x)


## === cell 7
x = x.reshape((-1, SIZE, SIZE, 1))
input_shape = x[0].shape
x_train = x.astype("float32")
y_train = y_cat

try:
    _ = ImageDataGenerator  # may be stubbed
    _test = ImageDataGenerator  # reference for NameError
    image_gen = ImageDataGenerator(
        featurewise_center=True,
        featurewise_std_normalization=True,
        rotation_range=15,
        width_shift_range=0.15,
        height_shift_range=0.15,
        horizontal_flip=True,
    )
except Exception:

    class ImageDataGenerator(object):
        def __init__(
            self,
            featurewise_center=False,
            featurewise_std_normalization=False,
            rotation_range=0,
            width_shift_range=0.0,
            height_shift_range=0.0,
            horizontal_flip=False,
        ):
            self.featurewise_center = featurewise_center
            self.featurewise_std_normalization = featurewise_std_normalization
            self.rotation_range = rotation_range
            self.width_shift_range = width_shift_range
            self.height_shift_range = height_shift_range
            self.horizontal_flip = horizontal_flip
            self.mean = None
            self.std = None

        def fit(self, x, augment=False):
            x = np.asarray(x, dtype=np.float32)
            if self.featurewise_center or self.featurewise_std_normalization:
                self.mean = x.mean(axis=0, keepdims=True)
            if self.featurewise_std_normalization:
                self.std = x.std(axis=0, keepdims=True)
                self.std[self.std == 0] = 1.0

        def _standardize(self, batch_x):
            if self.featurewise_center and self.mean is not None:
                batch_x = batch_x - self.mean
            if self.featurewise_std_normalization and self.std is not None:
                batch_x = batch_x / self.std
            return batch_x

        def flow(self, x, y=None, batch_size=32, shuffle=True):
            x = np.asarray(x, dtype=np.float32)
            if y is not None:
                y = np.asarray(y)
            n = x.shape[0]
            rng = np.random.RandomState(123)  # deterministic
            indices = np.arange(n)

            while True:
                if shuffle:
                    rng.shuffle(indices)
                for start in range(0, n, batch_size):
                    batch_idx = indices[start : start + batch_size]
                    batch_x = x[batch_idx].copy()
                    batch_x = self._standardize(batch_x)
                    if y is None:
                        yield batch_x
                    else:
                        yield batch_x, y[batch_idx]

    image_gen = ImageDataGenerator(
        featurewise_center=True,
        featurewise_std_normalization=True,
        rotation_range=15,
        width_shift_range=0.15,
        height_shift_range=0.15,
        horizontal_flip=True,
    )

image_gen.fit(x_train, augment=True)

augmented_images, _ = next(image_gen.flow(x_train, y_train.toarray(), batch_size=4 * 4))
plotImages(augmented_images)


## === cell 8
batch_size = 128
num_classes = len(y_cat.toarray()[0])
epochs = 5  # 0

print("x_train shape:", x_train.shape)
print(x_train.shape[0], "train samples")

try:
    model = Sequential()
    model.add(
        Conv2D(48, kernel_size=(3, 3), activation="relu", input_shape=input_shape)
    )
    model.add(Conv2D(48, (3, 3), activation="relu"))
    model.add(MaxPooling2D(pool_size=(3, 3)))
    model.add(Dropout(0.33))
    model.add(Flatten())
    model.add(Dense(24, activation="relu"))
    model.add(Dropout(0.33))
    model.add(Dense(num_classes, activation="softmax"))

    model.compile(
        loss=keras.losses.categorical_crossentropy,
        optimizer=keras.optimizers.Adadelta(),
        metrics=["accuracy"],
    )
    model.summary()
    model.fit_generator(
        image_gen.flow(x_train, y_train.toarray(), batch_size=batch_size),
        steps_per_epoch=25,
        epochs=epochs,
        verbose=1,
    )
except _TFImportStubError:

    class NumpySoftmaxModel(object):
        def __init__(self, num_classes, input_shape):
            self.num_classes = int(num_classes)
            self.input_shape = tuple(input_shape)
            self._compiled = False

        def add(self, *args, **kwargs):
            return

        def compile(self, *args, **kwargs):
            self._compiled = True

        def summary(self):
            print("Using NumpySoftmaxModel fallback (TensorFlow unavailable).")
            print("Input shape:", self.input_shape)
            print("Num classes:", self.num_classes)

        def fit_generator(
            self, generator, steps_per_epoch=1, epochs=1, verbose=1, **kwargs
        ):
            steps_per_epoch = int(steps_per_epoch) if steps_per_epoch is not None else 1
            epochs = int(epochs) if epochs is not None else 1
            for ep in range(epochs):
                for _ in range(steps_per_epoch):
                    _ = next(generator)
                if verbose:
                    print("Epoch %d/%d" % (ep + 1, epochs))
            return

        def predict_proba(self, x, **kwargs):
            x = np.asarray(x, dtype=np.float32)
            n = x.shape[0]
            sample_mean = x.reshape(n, -1).mean(axis=1, keepdims=True)  # (n,1)
            class_offsets = np.arange(self.num_classes, dtype=np.float32)[
                None, :
            ] / max(1.0, float(self.num_classes))
            logits = sample_mean + class_offsets
            logits = logits - logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            return probs

    model = NumpySoftmaxModel(num_classes=num_classes, input_shape=input_shape)
    model.compile(loss=None, optimizer=None, metrics=None)
    model.summary()
    model.fit_generator(
        image_gen.flow(x_train, y_train.toarray(), batch_size=batch_size),
        steps_per_epoch=25,
        epochs=epochs,
        verbose=1,
    )


## === cell 9
import warnings
from os.path import split

with open("sample_submission.csv", "w") as f:
    with warnings.catch_warnings():
        f.write("Image,Id\n")
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        for image in test_images:
            img = ImportImage(image).astype("float32")  # (SIZE, SIZE)
            x_batch = img.reshape(1, SIZE, SIZE, 1)  # (1, SIZE, SIZE, 1)

            if hasattr(image_gen, "standardize"):
                x_batch = image_gen.standardize(x_batch)
            else:
                x_batch = image_gen._standardize(x_batch)

            y = model.predict_proba(x_batch)
            predicted_args = np.argsort(y)[0][::-1][:5]
            predicted_tags = lohe.inverse_labels(predicted_args)
            image = split(image)[-1]
            predicted_tags = " ".join(predicted_tags)
            f.write("%s,%s\n" % (image, predicted_tags))
