# Goal

You will receive environment details and a partial notebook export.

# Requirements

- Fix the bug that causes the error in cell k.
- Do NOT adjust any other non-buggy cells.
- You may reference cell k+1 only to preserve variable/interface compatibility.
- Do not complete or extend code logic in cell k, k+1, or later cells.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (bug fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Output must follow your strict format: Diagnosis / Patch summary / Updated cells / Compatibility notes for cell k+1 / Assumptions.


# 1. Python version

3.6

# 2. Installed packages

No external packages required in the script and installed.

# 3. Data file paths

```
/
    kaggle/
        data/
            description.md (70 lines)
            images.zip (22.0 MB)
            sample_submission.csv (100 lines)
            sample_submission.csv.zip (2.3 kB)
            test.csv (100 lines)
            test.csv.zip (39.3 kB)
            train.csv (892 lines)
            train.csv.zip (357.1 kB)
            images/
                42.jpg (32.6 kB)
                168.jpg (16.5 kB)
                ... and 988 other files
            leaf-classification/
                description.md (70 lines)
                images.zip (22.0 MB)
                ... and 6 other files
                images/
                    42.jpg (32.6 kB)
                    168.jpg (16.5 kB)
                    ... and 988 other files
                leaf-classification/
        input/
            description.md (70 lines)
            images.zip (22.0 MB)
            sample_submission.csv (100 lines)
            sample_submission.csv.zip (2.3 kB)
            test.csv (100 lines)
            test.csv.zip (39.3 kB)
            train.csv (892 lines)
            train.csv.zip (357.1 kB)
            images/
                42.jpg (32.6 kB)
                168.jpg (16.5 kB)
                ... and 988 other files
            leaf-classification/
                description.md (70 lines)
                images.zip (22.0 MB)
                ... and 6 other files
                images/
                    42.jpg (32.6 kB)
                    168.jpg (16.5 kB)
                    ... and 988 other files
                leaf-classification/
        working/
            leaf-classification/
                description.md (70 lines)
                images.zip (22.0 MB)
                ... and 6 other files
                images/
                    42.jpg (32.6 kB)
                    168.jpg (16.5 kB)
                    ... and 988 other files
                leaf-classification/
```

-> data/leaf-classification/sample_submission.csv has 99 rows and 100 columns.
The columns are: id, Acer_Capillipes, Acer_Circinatum, Acer_Mono, Acer_Opalus, Acer_Palmatum, Acer_Pictum, Acer_Platanoids, Acer_Rubrum, Acer_Rufinerve, Acer_Saccharinum, Alnus_Cordata, Alnus_Maximowiczii, Alnus_Rubra, Alnus_Sieboldiana... and 85 more columns

-> data/leaf-classification/test.csv has 99 rows and 193 columns.
The columns are: id, margin1, margin2, margin3, margin4, margin5, margin6, margin7, margin8, margin9, margin10, margin11, margin12, margin13, margin14... and 178 more columns

-> data/leaf-classification/train.csv has 891 rows and 194 columns.
The columns are: id, species, margin1, margin2, margin3, margin4, margin5, margin6, margin7, margin8, margin9, margin10, margin11, margin12, margin13... and 179 more columns

-> data/sample_submission.csv has 99 rows and 100 columns.
The columns are: id, Acer_Capillipes, Acer_Circinatum, Acer_Mono, Acer_Opalus, Acer_Palmatum, Acer_Pictum, Acer_Platanoids, Acer_Rubrum, Acer_Rufinerve, Acer_Saccharinum, Alnus_Cordata, Alnus_Maximowiczii, Alnus_Rubra, Alnus_Sieboldiana... and 85 more columns

-> data/test.csv has 99 rows and 193 columns.
The columns are: id, margin1, margin2, margin3, margin4, margin5, margin6, margin7, margin8, margin9, margin10, margin11, margin12, margin13, margin14... and 178 more columns

-> data/train.csv has 891 rows and 194 columns.
The columns are: id, species, margin1, margin2, margin3, margin4, margin5, margin6, margin7, margin8, margin9, margin10, margin11, margin12, margin13... and 179 more columns

-> (stopped after 10 files for performance)

# 4. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "cpp"
os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", None)
os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_PYTHON", None)

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit

try:
    from keras.utils.np_utils import to_categorical  # type: ignore
    from keras.preprocessing.image import img_to_array, load_img  # type: ignore
except Exception:
    from PIL import Image

    def to_categorical(y, num_classes=None, dtype="float32"):
        y = np.array(y, dtype="int64").ravel()
        if num_classes is None:
            num_classes = int(np.max(y)) + 1 if y.size else 0
        out = np.zeros((y.shape[0], num_classes), dtype=dtype)
        if y.shape[0]:
            out[np.arange(y.shape[0]), y] = 1
        return out

    def load_img(path, grayscale=False, target_size=None):
        img = Image.open(path)
        img = img.convert("L") if grayscale else img.convert("RGB")
        if target_size is not None:
            img = img.resize((target_size[1], target_size[0]))
        return img

    def img_to_array(img):
        x = np.asarray(img, dtype="float32")
        if x.ndim == 2:
            x = x[:, :, np.newaxis]
        return x


root = "../input"
np.random.seed(2016)
split_random_state = 7
split = 0.9


def load_numeric_training(standardize=True):
    """
    Loads the pre-extracted features for the training data
    and returns a tuple of the image ids, the data, and the labels
    """
    data = pd.read_csv(os.path.join(root, "train.csv"))
    ID = data.pop("id")

    y = data.pop("species")
    y = LabelEncoder().fit(y).transform(y)
    X = StandardScaler().fit(data).transform(data) if standardize else data.values

    return ID, X, y


def load_numeric_test(standardize=True):
    """
    Loads the pre-extracted features for the test data
    and returns a tuple of the image ids, the data
    """
    test = pd.read_csv(os.path.join(root, "test.csv"))
    ID = test.pop("id")
    test = StandardScaler().fit(test).transform(test) if standardize else test.values
    return ID, test


def resize_img(img, max_dim=96):
    """
    Resize the image to so the maximum side is of size max_dim
    Returns a new image of the right size
    """
    max_ax = max((0, 1), key=lambda i: img.size[i])
    scale = max_dim / float(img.size[max_ax])
    return img.resize((int(img.size[0] * scale), int(img.size[1] * scale)))


def load_image_data(ids, max_dim=96, center=True):
    """
    Takes as input an array of image ids and loads the images as numpy
    arrays with the images resized so the longest side is max-dim length.
    If center is True, then will place the image in the center of
    the output array, otherwise it will be placed at the top-left corner.
    """
    X = np.empty((len(ids), max_dim, max_dim, 1))
    for i, idee in enumerate(ids):
        x = resize_img(
            load_img(os.path.join(root, "images", str(idee) + ".jpg"), grayscale=True),
            max_dim=max_dim,
        )
        x = img_to_array(x)
        length = x.shape[0]
        width = x.shape[1]
        if center:
            h1 = int((max_dim - length) / 2)
            h2 = h1 + length
            w1 = int((max_dim - width) / 2)
            w2 = w1 + width
        else:
            h1, w1 = 0, 0
            h2, w2 = (length, width)
        X[i, h1:h2, w1:w2, 0:1] = x
    return np.around(X / 255.0)


def load_train_data(split=split, random_state=None):
    """
    Loads the pre-extracted feature and image training data and
    splits them into training and cross-validation.
    Returns one tuple for the training data and one for the validation
    data. Each tuple is in the order pre-extracted features, images,
    and labels.
    """
    ID, X_num_tr, y = load_numeric_training()
    X_img_tr = load_image_data(ID)

    n_samples = len(y)
    n_classes = int(np.unique(y).shape[0])
    min_test = n_classes
    max_train = (n_samples - min_test) / float(n_samples)
    if split > max_train:
        split = max_train

    sss = StratifiedShuffleSplit(
        n_splits=1, train_size=split, random_state=random_state
    )
    train_ind, test_ind = next(sss.split(X_num_tr, y))
    X_num_val, X_img_val, y_val = X_num_tr[test_ind], X_img_tr[test_ind], y[test_ind]
    X_num_tr, X_img_tr, y_tr = X_num_tr[train_ind], X_img_tr[train_ind], y[train_ind]
    return (X_num_tr, X_img_tr, y_tr), (X_num_val, X_img_val, y_val)


def load_test_data():
    """
    Loads the pre-extracted feature and image test data.
    Returns a tuple in the order ids, pre-extracted features,
    and images.
    """
    ID, X_num_te = load_numeric_test()
    X_img_te = load_image_data(ID)
    return ID, X_num_te, X_img_te


print("Loading the training data...")
(X_num_tr, X_img_tr, y_tr), (X_num_val, X_img_val, y_val) = load_train_data(
    random_state=split_random_state
)
y_tr_cat = to_categorical(y_tr)
y_val_cat = to_categorical(y_val)
print("Training data loaded!")


## === cell 1
try:
    from tensorflow.keras.preprocessing.image import (  # type: ignore
        ImageDataGenerator,
        NumpyArrayIterator,
        array_to_img,
    )
except Exception:
    try:
        from keras.preprocessing.image import ImageDataGenerator, NumpyArrayIterator, array_to_img  # type: ignore
    except Exception:
        from PIL import Image

        def array_to_img(x, data_format=None, scale=True):
            x = np.asarray(x)
            if x.ndim == 3 and x.shape[-1] == 1:
                x = x[:, :, 0]
            if scale:
                x = np.clip(x, 0.0, 1.0)
                x = (x * 255.0).astype("uint8")
            else:
                x = np.clip(x, 0, 255).astype("uint8")
            return Image.fromarray(x)

        class ImageDataGenerator(object):
            def __init__(
                self,
                rotation_range=0.0,
                zoom_range=0.0,
                horizontal_flip=False,
                vertical_flip=False,
                fill_mode="nearest",
            ):
                self.rotation_range = rotation_range
                self.zoom_range = zoom_range
                self.horizontal_flip = horizontal_flip
                self.vertical_flip = vertical_flip
                self.fill_mode = fill_mode
                self.dim_ordering = "tf"

            def random_transform(self, x):
                x = np.asarray(x)
                if x.ndim != 3:
                    return x
                img = array_to_img(x, scale=False)

                if self.rotation_range:
                    theta = np.random.uniform(-self.rotation_range, self.rotation_range)
                    img = img.rotate(theta, resample=Image.BILINEAR)

                if self.zoom_range:
                    if (
                        isinstance(self.zoom_range, (tuple, list))
                        and len(self.zoom_range) == 2
                    ):
                        z = np.random.uniform(self.zoom_range[0], self.zoom_range[1])
                    else:
                        z = np.random.uniform(
                            1.0 - float(self.zoom_range), 1.0 + float(self.zoom_range)
                        )
                    if z != 1.0:
                        w, h = img.size
                        new_w, new_h = max(1, int(w * z)), max(1, int(h * z))
                        img_z = img.resize((new_w, new_h), resample=Image.BILINEAR)
                        if z > 1.0:
                            left = (new_w - w) // 2
                            top = (new_h - h) // 2
                            img = img_z.crop((left, top, left + w, top + h))
                        else:
                            canvas = Image.new(img.mode, (w, h))
                            left = (w - new_w) // 2
                            top = (h - new_h) // 2
                            canvas.paste(img_z, (left, top))
                            img = canvas

                if self.horizontal_flip and np.random.rand() < 0.5:
                    img = img.transpose(Image.FLIP_LEFT_RIGHT)
                if self.vertical_flip and np.random.rand() < 0.5:
                    img = img.transpose(Image.FLIP_TOP_BOTTOM)

                arr = np.asarray(img, dtype="float32")
                if arr.ndim == 2:
                    arr = arr[:, :, np.newaxis]
                return arr

            def standardize(self, x):
                return x

            def flow(
                self,
                X,
                y=None,
                batch_size=32,
                shuffle=True,
                seed=None,
                save_to_dir=None,
                save_prefix="",
                save_format="jpeg",
            ):
                return NumpyArrayIterator(
                    X,
                    y,
                    self,
                    batch_size,
                    shuffle,
                    seed,
                    save_to_dir,
                    save_prefix,
                    save_format,
                )

        class NumpyArrayIterator(object):
            def __init__(
                self,
                X,
                y,
                image_data_generator,
                batch_size=32,
                shuffle=True,
                seed=None,
                save_to_dir=None,
                save_prefix="",
                save_format="jpeg",
            ):
                self.X = np.asarray(X)
                self.y = None if y is None else np.asarray(y)
                self.image_data_generator = image_data_generator
                self.batch_size = int(batch_size)
                self.shuffle = bool(shuffle)
                self.seed = seed
                self.save_to_dir = save_to_dir
                self.save_prefix = save_prefix
                self.save_format = save_format
                self.dim_ordering = getattr(image_data_generator, "dim_ordering", "tf")

                self.n = self.X.shape[0]
                self.batch_index = 0
                self.total_batches_seen = 0
                self.lock = None  # used by our overridden .next() via context manager in keras; keep compatible

                self.index_array = np.arange(self.n)
                if self.seed is not None:
                    np.random.seed(self.seed)
                if self.shuffle:
                    np.random.shuffle(self.index_array)

                self.index_generator = self._flow_index()

            def _flow_index(self):
                while True:
                    if self.batch_index == 0 and self.shuffle:
                        if self.seed is not None:
                            np.random.seed(self.seed + self.total_batches_seen)
                        np.random.shuffle(self.index_array)

                    current_index = (self.batch_index * self.batch_size) % self.n
                    if self.n > current_index + self.batch_size:
                        current_batch_size = self.batch_size
                        self.batch_index += 1
                    else:
                        current_batch_size = self.n - current_index
                        self.batch_index = 0
                    self.total_batches_seen += 1
                    yield self.index_array[
                        current_index : current_index + current_batch_size
                    ], current_index, current_batch_size

            def next(self):
                index_array, current_index, current_batch_size = next(
                    self.index_generator
                )
                batch_x = np.zeros(
                    tuple([current_batch_size] + list(self.X.shape)[1:]),
                    dtype="float32",
                )
                for i, j in enumerate(index_array):
                    x = self.X[j]
                    x = self.image_data_generator.random_transform(x.astype("float32"))
                    x = self.image_data_generator.standardize(x)
                    batch_x[i] = x
                if self.y is None:
                    return batch_x
                batch_y = self.y[index_array]
                return batch_x, batch_y


class ImageDataGenerator2(ImageDataGenerator):
    def flow(
        self,
        X,
        y=None,
        batch_size=32,
        shuffle=True,
        seed=None,
        save_to_dir=None,
        save_prefix="",
        save_format="jpeg",
    ):
        return NumpyArrayIterator2(
            X,
            y,
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            seed=seed,
            save_to_dir=save_to_dir,
            save_prefix=save_prefix,
            save_format=save_format,
        )


class NumpyArrayIterator2(NumpyArrayIterator):
    def next(self):
        if getattr(self, "lock", None) is not None:
            with self.lock:
                self.index_array, current_index, current_batch_size = next(
                    self.index_generator
                )
        else:
            self.index_array, current_index, current_batch_size = next(
                self.index_generator
            )

        batch_x = np.zeros(tuple([current_batch_size] + list(self.X.shape)[1:]))
        for i, j in enumerate(self.index_array):
            x = self.X[j]
            x = self.image_data_generator.random_transform(x.astype("float32"))
            x = self.image_data_generator.standardize(x)
            batch_x[i] = x
        if getattr(self, "save_to_dir", None):
            for i in range(current_batch_size):
                img = array_to_img(
                    batch_x[i], getattr(self, "dim_ordering", None), scale=True
                )
                fname = "{prefix}_{index}_{hash}.{format}".format(
                    prefix=self.save_prefix,
                    index=current_index + i,
                    hash=np.random.randint(1e4),
                    format=self.save_format,
                )
                img.save(os.path.join(self.save_to_dir, fname))
        if self.y is None:
            return batch_x
        batch_y = self.y[self.index_array]
        return batch_x, batch_y


print("Creating Data Augmenter...")
imgen = ImageDataGenerator2(
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode="nearest",
)
imgen_train = imgen.flow(X_img_tr, y_tr_cat, seed=np.random.randint(1, 10000))
print("Finished making data augmenter...")


## === cell 2

import numpy as np


class _Tensor(object):
    def __init__(self, shape=None, name=None):
        self.shape = shape
        self.name = name


class _Layer(object):
    def __call__(self, x):
        return _Tensor(
            shape=getattr(x, "shape", None), name=getattr(self, "name", None)
        )


class Input(object):
    def __new__(cls, shape=None, name=None, **kwargs):
        return _Tensor(shape=shape, name=name)


class Dense(_Layer):
    def __init__(self, units, activation=None, **kwargs):
        self.units = units
        self.activation = activation


class Dropout(_Layer):
    def __init__(self, rate, **kwargs):
        self.rate = rate


class Activation(_Layer):
    def __init__(self, activation, **kwargs):
        self.activation = activation


class Conv2D(_Layer):
    def __init__(
        self, filters, kernel_size, input_shape=None, padding="valid", **kwargs
    ):
        self.filters = filters
        self.kernel_size = kernel_size
        self.input_shape = input_shape
        self.padding = padding


class MaxPooling2D(_Layer):
    def __init__(self, pool_size=(2, 2), strides=None, **kwargs):
        self.pool_size = pool_size
        self.strides = strides


class Flatten(_Layer):
    def __init__(self, **kwargs):
        pass


class Concatenate(_Layer):
    def __init__(self, axis=-1, **kwargs):
        self.axis = axis

    def __call__(self, tensors):
        return _Tensor(shape=None, name=None)


class Model(object):
    def __init__(self, inputs=None, outputs=None, **kwargs):
        self.inputs = inputs
        self.outputs = outputs
        self._compiled = False
        self.history = {"loss": [], "val_loss": [], "accuracy": [], "val_accuracy": []}

    def compile(self, loss=None, optimizer=None, metrics=None, **kwargs):
        self.loss = loss
        self.optimizer = optimizer
        self.metrics = metrics or []
        self._compiled = True

    def fit_generator(
        self,
        generator,
        samples_per_epoch=None,
        nb_epoch=1,
        validation_data=None,
        nb_val_samples=None,
        verbose=0,
        callbacks=None,
        **kwargs
    ):
        callbacks = callbacks or []
        for cb in callbacks:
            if hasattr(cb, "set_model"):
                cb.set_model(self)
            if hasattr(cb, "on_train_begin"):
                cb.on_train_begin({})

        steps = 0
        if samples_per_epoch is not None:
            first = next(generator)
            batch_size = len(first[1]) if hasattr(first[1], "__len__") else 1
            steps = int(np.ceil(float(samples_per_epoch) / float(max(1, batch_size))))

            def _regen():
                yield first
                while True:
                    yield next(generator)

            generator = _regen()
        else:
            steps = 1

        for epoch in range(int(nb_epoch)):
            for cb in callbacks:
                if hasattr(cb, "on_epoch_begin"):
                    cb.on_epoch_begin(epoch, {})

            for step in range(int(steps)):
                _ = next(generator)
                for cb in callbacks:
                    if hasattr(cb, "on_batch_end"):
                        cb.on_batch_end(step, {})

            logs = {"loss": 0.0, "val_loss": 0.0, "accuracy": 0.0, "val_accuracy": 0.0}
            self.history["loss"].append(logs["loss"])
            self.history["val_loss"].append(logs["val_loss"])
            self.history["accuracy"].append(logs["accuracy"])
            self.history["val_accuracy"].append(logs["val_accuracy"])

            for cb in callbacks:
                if hasattr(cb, "on_epoch_end"):
                    cb.on_epoch_end(epoch, logs)

        for cb in callbacks:
            if hasattr(cb, "on_train_end"):
                cb.on_train_end({})

        class _History(object):
            def __init__(self, history):
                self.history = history

        return _History(self.history)


def combined_model():

    image = Input(shape=(96, 96, 1), name="image")
    x = Conv2D(8, (5, 5), input_shape=(96, 96, 1), padding="same")(image)
    x = (Activation("relu"))(x)
    x = (MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))(x)

    x = (Conv2D(32, (5, 5), padding="same"))(x)
    x = (Activation("relu"))(x)
    x = (MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))(x)

    x = Flatten()(x)
    numerical = Input(shape=(192,), name="numerical")
    concatenated = Concatenate(axis=-1)([x, numerical])

    x = Dense(100, activation="relu")(concatenated)
    x = Dropout(0.5)(x)

    out = Dense(99, activation="softmax")(x)
    model = Model(inputs=[image, numerical], outputs=out)
    model.compile(
        loss="categorical_crossentropy", optimizer="rmsprop", metrics=["accuracy"]
    )

    return model


print("Creating the model...")
model = combined_model()
print("Model created!")


## === cell 3
import os
import pickle

try:
    from keras.callbacks import ModelCheckpoint  # type: ignore
    from keras.models import load_model  # type: ignore
except Exception:

    class ModelCheckpoint(object):
        def __init__(
            self,
            filepath,
            monitor="val_loss",
            verbose=0,
            save_best_only=False,
            **kwargs
        ):
            self.filepath = filepath
            self.monitor = monitor
            self.verbose = int(verbose)
            self.save_best_only = bool(save_best_only)
            self.best = None
            self.model = None

        def set_model(self, model):
            self.model = model

        def on_train_begin(self, logs=None):
            self.best = None

        def on_epoch_end(self, epoch, logs=None):
            logs = logs or {}
            current = logs.get(self.monitor, None)
            do_save = True
            if self.save_best_only:
                if current is None:
                    do_save = False
                else:
                    if (self.best is None) or (current < self.best):
                        self.best = current
                        do_save = True
                    else:
                        do_save = False

            if do_save and self.model is not None:
                try:
                    with open(self.filepath, "wb") as f:
                        pickle.dump(self.model, f, protocol=pickle.HIGHEST_PROTOCOL)
                    if self.verbose:
                        print(
                            "Epoch {0}: saving model to {1}".format(
                                int(epoch) + 1, self.filepath
                            )
                        )
                except Exception:
                    pass

    def load_model(filepath, **kwargs):
        try:
            with open(filepath, "rb") as f:
                return pickle.load(f)
        except Exception:
            return globals().get("model", None)


def combined_generator(imgen, X):
    """
    A generator to train our keras neural network. It
    takes the image augmenter generator and the array
    of the pre-extracted features.
    It yields a minibatch and will run indefinitely
    """
    while True:
        for i in range(X.shape[0]):
            if hasattr(imgen, "next") and callable(getattr(imgen, "next")):
                batch_img, batch_y = imgen.next()
            else:
                batch_img, batch_y = next(imgen)
            x = X[imgen.index_array]
            yield [batch_img, x], batch_y


best_model_file = "leafnet.h5"
best_model = ModelCheckpoint(
    best_model_file, monitor="val_loss", verbose=1, save_best_only=True
)

try:
    _orig_fit_generator = model.fit_generator

    def _fit_generator_no_reentrancy(
        generator,
        samples_per_epoch=None,
        nb_epoch=1,
        validation_data=None,
        nb_val_samples=None,
        verbose=0,
        callbacks=None,
        **kwargs
    ):
        if samples_per_epoch is not None:
            first = next(generator)
            batch_size = len(first[1]) if hasattr(first[1], "__len__") else 1
            steps = int(np.ceil(float(samples_per_epoch) / float(max(1, batch_size))))

            _gen = generator

            def _regen():
                yield first
                while True:
                    yield next(_gen)

            generator = _regen()
        return _orig_fit_generator(
            generator,
            samples_per_epoch=samples_per_epoch,
            nb_epoch=nb_epoch,
            validation_data=validation_data,
            nb_val_samples=nb_val_samples,
            verbose=verbose,
            callbacks=callbacks,
            **kwargs
        )

    model.fit_generator = _fit_generator_no_reentrancy
except Exception:
    pass

print("Training model...")
history = model.fit_generator(
    combined_generator(imgen_train, X_num_tr),
    samples_per_epoch=X_num_tr.shape[0],
    nb_epoch=89,
    validation_data=([X_img_val, X_num_val], y_val_cat),
    nb_val_samples=X_num_val.shape[0],
    verbose=0,
    callbacks=[best_model],
)

print("Loading the best model...")
model = load_model(best_model_file)
print("Best Model loaded!")


## --- ERROR in cell 3, traceback:
[0;31m---------------------------------------------------------------------------[0m
[0;31mValueError[0m                                Traceback (most recent call last)
[0;32m/tmp/ipykernel_11/2226273130.py[0m in [0;36m<cell line: 0>[0;34m()[0m
[1;32m    131[0m [0;34m[0m[0m
[1;32m    132[0m [0mprint[0m[0;34m([0m[0;34m"Training model..."[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 133[0;31m history = model.fit_generator(
[0m[1;32m    134[0m     [0mcombined_generator[0m[0;34m([0m[0mimgen_train[0m[0;34m,[0m [0mX_num_tr[0m[0;34m)[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m
[1;32m    135[0m     [0msamples_per_epoch[0m[0;34m=[0m[0mX_num_tr[0m[0;34m.[0m[0mshape[0m[0;34m[[0m[0;36m0[0m[0;34m][0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m

[0;32m/tmp/ipykernel_11/2226273130.py[0m in [0;36m_fit_generator_no_reentrancy[0;34m(generator, samples_per_epoch, nb_epoch, validation_data, nb_val_samples, verbose, callbacks, **kwargs)[0m
[1;32m    115[0m [0;34m[0m[0m
[1;32m    116[0m             [0mgenerator[0m [0;34m=[0m [0m_regen[0m[0;34m([0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 117[0;31m         return _orig_fit_generator(
[0m[1;32m    118[0m             [0mgenerator[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m
[1;32m    119[0m             [0msamples_per_epoch[0m[0;34m=[0m[0msamples_per_epoch[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m

[0;32m/tmp/ipykernel_11/4137817686.py[0m in [0;36mfit_generator[0;34m(self, generator, samples_per_epoch, nb_epoch, validation_data, nb_val_samples, verbose, callbacks, **kwargs)[0m
[1;32m    135[0m [0;34m[0m[0m
[1;32m    136[0m             [0;32mfor[0m [0mstep[0m [0;32min[0m [0mrange[0m[0;34m([0m[0mint[0m[0;34m([0m[0msteps[0m[0;34m)[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 137[0;31m                 [0m_[0m [0;34m=[0m [0mnext[0m[0;34m([0m[0mgenerator[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    138[0m                 [0;32mfor[0m [0mcb[0m [0;32min[0m [0mcallbacks[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m    139[0m                     [0;32mif[0m [0mhasattr[0m[0;34m([0m[0mcb[0m[0;34m,[0m [0;34m"on_batch_end"[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m

[0;32m/tmp/ipykernel_11/4137817686.py[0m in [0;36m_regen[0;34m()[0m
[1;32m    123[0m                 [0;32myield[0m [0mfirst[0m[0;34m[0m[0;34m[0m[0m
[1;32m    124[0m                 [0;32mwhile[0m [0;32mTrue[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 125[0;31m                     [0;32myield[0m [0mnext[0m[0;34m([0m[0mgenerator[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    126[0m [0;34m[0m[0m
[1;32m    127[0m             [0mgenerator[0m [0;34m=[0m [0m_regen[0m[0;34m([0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m

[0;31mValueError[0m: generator already executing

## === cell 4
LABELS = sorted(pd.read_csv(os.path.join(root, 'train.csv')).species.unique())

index, test, X_img_te = load_test_data()

yPred_proba = model.predict([X_img_te, test])

yPred = pd.DataFrame(yPred_proba,index=index,columns=LABELS)

print('Creating and writing submission...')
fp = open('submit.csv', 'w')
fp.write(yPred.to_csv())
print('Finished writing submission')
yPred.tail()
