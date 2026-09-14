# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


# 1. Kaggle task description

## Task
Create a classifier to predict the severity of diabetic retinopathy.

## Metric
Quadratic weighted kappa, which measures the agreement between two ratings. This metric typically varies from 0 (random agreement between raters) to 1 (complete agreement between raters). In the event that there is less agreement between the raters than expected by chance, this metric may go below 0. The quadratic weighted kappa is calculated between the scores assigned by the human rater and the predicted scores.

Images have five possible ratings, 0,1,2,3,4.  Each image is characterized by a tuple *(e*,*e)*, which corresponds to its scores by *Rater A* (human) and *Rater B* (predicted).  The quadratic weighted kappa is calculated as follows. First, an N x N histogram matrix *O* is constructed, such that *O* corresponds to the number of images that received a rating *i* by *A* and a rating *j* by *B*. An *N-by-N* matrix of weights, *w*, is calculated based on the difference between raters' scores:

An *N-by-N* histogram matrix of expected ratings, *E*, is calculated, assuming that there is no correlation between rating scores.  This is calculated as the outer product between each rater's histogram vector of ratings, normalized such that *E* and *O* have the same sum.

## Submission Format
```
id_code,diagnosis
0005cfc8afb6,0
003f0afdcd15,0
etc.
```

## Dataset
You are provided with a large set of retina images taken using [fundus photography](https://en.wikipedia.org/wiki/Fundus_photography) under a variety of imaging conditions.

Labels are on a scale of 0 to 4:

> 0 - No DR
> 1 - Mild
> 2 - Moderate
> 3 - Severe
> 4 - Proliferative DR

Images may contain artifacts, be out of focus, underexposed, or overexposed. The images were gathered from multiple clinics using a variety of cameras over an extended period of time, which will introduce further variation.

- **train.csv** - the training labels
- **test.csv** - the test set (you must predict the `diagnosis` value for these variables)
- **sample_submission.csv** - a sample submission file in the correct format
- **train.zip** - the training set images
- **test.zip** - the public test set images

# 2. Python version

3.7

# 3. Installed packages

No external packages required in the script and installed.

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (118 lines)
            sample_submission.csv (368 lines)
            sample_submission.csv.zip (3.2 kB)
            test.csv (368 lines)
            test.csv.zip (2.9 kB)
            test.zip (160 Bytes)
            test_images.zip (902.9 MB)
            train.csv (3296 lines)
            train.csv.zip (27.5 kB)
            train.zip (162 Bytes)
            train_images.zip (7.7 GB)
            aptos2019-blindness-detection/
                description.md (118 lines)
                sample_submission.csv (368 lines)
                ... and 9 other files
                aptos2019-blindness-detection/
                test_images/
                    218c822a3dd9.png (5.7 MB)
                    0e82bcacc475.png (5.2 MB)
                    ... and 365 other files
                    test_images/
                train_images/
                    184a185e7447.png (337.5 kB)
                    c4aef0d88d1b.png (876.6 kB)
                    ... and 3293 other files
                    train_images/
            test_images/
                218c822a3dd9.png (5.7 MB)
                0e82bcacc475.png (5.2 MB)
                ... and 365 other files
                test_images/
            train_images/
                184a185e7447.png (337.5 kB)
                c4aef0d88d1b.png (876.6 kB)
                ... and 3293 other files
                train_images/
        input/
            description.md (118 lines)
            sample_submission.csv (368 lines)
            sample_submission.csv.zip (3.2 kB)
            test.csv (368 lines)
            test.csv.zip (2.9 kB)
            test.zip (160 Bytes)
            test_images.zip (902.9 MB)
            train.csv (3296 lines)
            train.csv.zip (27.5 kB)
            train.zip (162 Bytes)
            train_images.zip (7.7 GB)
            aptos2019-blindness-detection/
                description.md (118 lines)
                sample_submission.csv (368 lines)
                ... and 9 other files
                aptos2019-blindness-detection/
                test_images/
                    218c822a3dd9.png (5.7 MB)
                    0e82bcacc475.png (5.2 MB)
                    ... and 365 other files
                    test_images/
                train_images/
                    184a185e7447.png (337.5 kB)
                    c4aef0d88d1b.png (876.6 kB)
                    ... and 3293 other files
                    train_images/
            test_images/
                218c822a3dd9.png (5.7 MB)
                0e82bcacc475.png (5.2 MB)
                ... and 365 other files
                test_images/
                    218c822a3dd9.png (5.7 MB)
                    0e82bcacc475.png (5.2 MB)
                    ... and 365 other files
                    test_images/
            train_images/
                184a185e7447.png (337.5 kB)
                c4aef0d88d1b.png (876.6 kB)
                ... and 3293 other files
                train_images/
                    184a185e7447.png (337.5 kB)
                    c4aef0d88d1b.png (876.6 kB)
                    ... and 3293 other files
                    train_images/
        working/
            aptos2019-blindness-detection/
                description.md (118 lines)
                sample_submission.csv (368 lines)
                ... and 9 other files
                aptos2019-blindness-detection/
                test_images/
                    218c822a3dd9.png (5.7 MB)
                    0e82bcacc475.png (5.2 MB)
                    ... and 365 other files
                    test_images/
                train_images/
                    184a185e7447.png (337.5 kB)
                    c4aef0d88d1b.png (876.6 kB)
                    ... and 3293 other files
                    train_images/
```

-> data/aptos2019-blindness-detection/sample_submission.csv has 367 rows and 2 columns.
The columns are: id_code, diagnosis

-> data/aptos2019-blindness-detection/test.csv has 367 rows and 1 columns.
The columns are: id_code

-> data/aptos2019-blindness-detection/train.csv has 3295 rows and 2 columns.
The columns are: id_code, diagnosis

-> data/sample_submission.csv has 367 rows and 2 columns.
The columns are: id_code, diagnosis

-> data/test.csv has 367 rows and 1 columns.
The columns are: id_code

-> data/train.csv has 3295 rows and 2 columns.
The columns are: id_code, diagnosis

-> input/aptos2019-blindness-detection/sample_submission.csv has 367 rows and 2 columns.
The columns are: id_code, diagnosis

-> (stopped after 10 files for performance)

# 5. Code solution

## === cell 0
import os
import math
import random

import numpy as np
import pandas as pd

TRAINING = False

SEED = 123
random.seed(SEED)
np.random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)

os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
os.environ.setdefault("TF_CUDNN_DETERMINISTIC", "1")

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "2")




## === cell 1
BASE_CANDIDATES = [
    "../input/aptos2019-blindness-detection",
    "../input/aptos2019-blindness-detection/aptos2019-blindness-detection",
    "/kaggle/input/aptos2019-blindness-detection",
    "/kaggle/input/aptos2019-blindness-detection/aptos2019-blindness-detection",
]
DATA_DIR = None
for p in BASE_CANDIDATES:
    if os.path.exists(os.path.join(p, "train.csv")):
        DATA_DIR = p
        break

if DATA_DIR is None:
    raise FileNotFoundError(
        "Could not locate aptos2019-blindness-detection dataset directory."
    )

TRAIN_CSV = os.path.join(DATA_DIR, "train.csv")
TEST_CSV = os.path.join(DATA_DIR, "test.csv")
TRAIN_IMG_DIR = os.path.join(DATA_DIR, "train_images")
TEST_IMG_DIR = os.path.join(DATA_DIR, "test_images")

print("Using DATA_DIR:", DATA_DIR)
print("Train images dir exists:", os.path.isdir(TRAIN_IMG_DIR))
print("Test images dir exists:", os.path.isdir(TEST_IMG_DIR))




## === cell 2
WEIGHTS_DIR = "../input/aptos-2019-densenet121-weights"
if os.path.isdir(WEIGHTS_DIR):
    print("Found weights dir:", WEIGHTS_DIR)
    try:
        print("Files:", os.listdir(WEIGHTS_DIR)[:10])
    except Exception as e:
        print("Could not list weights dir:", repr(e))
else:
    print(
        "Weights dir not found (will rely on locally trained weights if available):",
        WEIGHTS_DIR,
    )




## === cell 3
train = pd.read_csv(TRAIN_CSV)
test = pd.read_csv(TEST_CSV)

print("Number of train samples: ", train.shape[0])
print("Number of test samples: ", test.shape[0])
print(train.head())
print(test.head())




## === cell 4
train = train.copy()
test = test.copy()

train["id_code"] = train["id_code"].astype(str) + ".png"
test["id_code"] = test["id_code"].astype(str) + ".png"
train["diagnosis"] = train["diagnosis"].astype(int)

label_cols = ["lbl_0", "lbl_1", "lbl_2", "lbl_3", "lbl_4"]
d = train["diagnosis"].to_numpy(np.int32)
label_mat = (np.arange(len(label_cols), dtype=np.int32)[None, :] <= d[:, None]).astype(
    np.int32
)

train = pd.concat([train, pd.DataFrame(label_mat, columns=label_cols)], axis=1)
train["diagnosis"] = train["diagnosis"].astype(str)
print(train.head(3))




## === cell 5
import cv2

try:
    cv2.setNumThreads(0)
except Exception:
    pass

TF_AVAILABLE = True
TF_IMPORT_ERROR = None
tf = None

try:
    import google.protobuf.message_factory as _message_factory  # noqa: E402

    if not hasattr(_message_factory.MessageFactory, "GetPrototype"):

        def _GetPrototype(self, descriptor):
            if hasattr(self, "GetMessageClass"):
                return self.GetMessageClass(descriptor)
            if hasattr(_message_factory, "GetMessageClass"):
                return _message_factory.GetMessageClass(descriptor)
            raise AttributeError(
                "Cannot provide GetPrototype: no GetMessageClass available in this protobuf version."
            )

        try:
            _message_factory.MessageFactory.GetPrototype = _GetPrototype
        except Exception:
            pass
except Exception:
    pass

try:
    import tensorflow as tf  # noqa: E402
except Exception as e:
    TF_AVAILABLE = False
    TF_IMPORT_ERROR = repr(e)
    tf = None

IMG_SIZE = 224
NB_CHANNELS = 3
NB_CLASSES = 5  # 0, 1, 2, 3, 4

BATCH_SIZE = 16
TEST_BATCH_SIZE = 32

if TF_AVAILABLE:
    try:
        tf.random.set_seed(SEED)
    except Exception:
        pass

    try:
        tf.config.threading.set_intra_op_parallelism_threads(1)
        tf.config.threading.set_inter_op_parallelism_threads(1)
    except Exception:
        pass
else:
    print(
        "WARNING: TensorFlow could not be imported; will write a fallback submission."
    )
    print("TensorFlow import error:", TF_IMPORT_ERROR)




## === cell 6
"""
crops black parts around the image (intensity is <= tol)
"""


def crop_image(img, tol=10):
    def crop_image_1(img2d):
        mask = img2d > tol
        return img2d[np.ix_(mask.any(1), mask.any(0))]

    if img.ndim == 2:
        return crop_image_1(img)
    elif img.ndim == 3:
        try:
            img_cpy = img.copy()
            h, w, _ = img.shape
            img1 = cv2.resize(crop_image_1(img[:, :, 0]), (w, h))
            img2 = cv2.resize(crop_image_1(img[:, :, 1]), (w, h))
            img3 = cv2.resize(crop_image_1(img[:, :, 2]), (w, h))
            img[:, :, 0] = img1
            img[:, :, 1] = img2
            img[:, :, 2] = img3
        except Exception:
            return img_cpy
        return img
    else:
        return img


"""
crops black parts and enhances image (Ben Graham's method)
"""


def preprocess_image(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = crop_image(img)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = cv2.addWeighted(img, 4, cv2.GaussianBlur(img, (0, 0), IMG_SIZE / 10), -4, 128)
    return img




## === cell 7
if TF_AVAILABLE:

    class _ImageCache:
        __slots__ = ("max_items", "_d", "_keys")

        def __init__(self, max_items=4096):
            self.max_items = int(max_items)
            self._d = {}
            self._keys = []

        def get(self, k):
            return self._d.get(k, None)

        def put(self, k, v):
            if k in self._d:
                return
            self._d[k] = v
            self._keys.append(k)
            if len(self._keys) > self.max_items:
                old = self._keys.pop(0)
                self._d.pop(old, None)

    _GLOBAL_IMG_CACHE = _ImageCache(max_items=4096)

    class DataFrameSequence(tf.keras.utils.Sequence):
        def __init__(
            self,
            df,
            directory,
            x_col,
            y_cols=None,
            batch_size=32,
            shuffle=True,
            rescale=1.0 / 255.0,
            preprocess_fn=None,
            horizontal_flip=False,
        ):
            self.df = df.reset_index(drop=True)
            self.directory = directory
            self.x_col = x_col
            self.y_cols = y_cols
            self.batch_size = int(batch_size)
            self.shuffle = shuffle
            self.rescale = float(rescale)
            self.preprocess_fn = preprocess_fn
            self.horizontal_flip = horizontal_flip

            self._x = self.df[self.x_col].astype(str).to_numpy()
            self._paths = np.array(
                [os.path.join(self.directory, f) for f in self._x], dtype=object
            )

            if self.y_cols is None:
                self._y = None
            else:
                self._y = self.df[self.y_cols].to_numpy(dtype=np.float32, copy=True)

            self.indices = np.arange(len(self.df), dtype=np.int32)
            self.on_epoch_end()

        @property
        def n(self):
            return len(self.df)

        @property
        def filenames(self):
            return self._x.tolist()

        def __len__(self):
            return int(math.ceil(len(self.df) / self.batch_size))

        def on_epoch_end(self):
            if self.shuffle:
                np.random.shuffle(self.indices)

        def reset(self):
            self.on_epoch_end()

        def __getitem__(self, idx):
            start = idx * self.batch_size
            end = min((idx + 1) * self.batch_size, len(self.indices))
            batch_idx = self.indices[start:end]

            bs = len(batch_idx)
            X = np.empty((bs, IMG_SIZE, IMG_SIZE, NB_CHANNELS), dtype=np.float32)

            rescale = self.rescale
            preprocess_fn = self.preprocess_fn
            horizontal_flip = self.horizontal_flip
            paths = self._paths
            cache = _GLOBAL_IMG_CACHE

            for i, ridx in enumerate(batch_idx):
                fpath = paths[ridx]

                img = None
                if preprocess_fn is not None and not horizontal_flip:
                    img = cache.get(fpath)
                    if img is None:
                        raw = cv2.imread(fpath)
                        if raw is None:
                            raw = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
                        img = preprocess_fn(raw)
                        cache.put(fpath, img)

                if img is None:
                    raw = cv2.imread(fpath)
                    if raw is None:
                        raw = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
                    if preprocess_fn is not None:
                        img = preprocess_fn(raw)
                    else:
                        img = cv2.resize(raw, (IMG_SIZE, IMG_SIZE))
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                if horizontal_flip and (np.random.rand() < 0.5):
                    img = np.ascontiguousarray(img[:, ::-1, :])

                X[i] = img.astype(np.float32) * rescale

            if self._y is None:
                return X
            y = self._y[batch_idx]
            return X, y

    def _build_inference_dataset(df, directory, batch_size, rescale=1.0 / 255.0):
        paths = (
            df["id_code"].astype(str).map(lambda f: os.path.join(directory, f))
        ).values

        ds = tf.data.Dataset.from_tensor_slices(paths)

        def _py_load(path_t):
            path = path_t.decode("utf-8")
            raw = cv2.imread(path)
            if raw is None:
                raw = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
            img = preprocess_image(raw)
            img = img.astype(np.float32) * float(rescale)
            return img

        def _tf_load(path_t):
            img = tf.numpy_function(_py_load, [path_t], Tout=tf.float32)
            img.set_shape((IMG_SIZE, IMG_SIZE, NB_CHANNELS))
            return img

        opts = tf.data.Options()
        try:
            opts.experimental_deterministic = True
        except Exception:
            pass

        ds = ds.with_options(opts)
        ds = ds.map(_tf_load, num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.batch(int(batch_size), drop_remainder=False)
        ds = ds.prefetch(tf.data.AUTOTUNE)
        return ds




## === cell 8
if TF_AVAILABLE:
    val_fraction = 0.2
    n_total = len(train)
    n_val = int(round(n_total * val_fraction))

    perm = np.random.RandomState(SEED).permutation(n_total)
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]

    train_df = train.iloc[train_idx].reset_index(drop=True)
    val_df = train.iloc[val_idx].reset_index(drop=True)

    train_gen = DataFrameSequence(
        df=train_df,
        directory=TRAIN_IMG_DIR,
        x_col="id_code",
        y_cols=label_cols,
        batch_size=BATCH_SIZE,
        shuffle=True,
        rescale=1.0 / 255.0,
        preprocess_fn=preprocess_image,
        horizontal_flip=True,
    )

    val_gen = DataFrameSequence(
        df=val_df,
        directory=TRAIN_IMG_DIR,
        x_col="id_code",
        y_cols=label_cols,
        batch_size=BATCH_SIZE,
        shuffle=False,
        rescale=1.0 / 255.0,
        preprocess_fn=preprocess_image,
        horizontal_flip=False,
    )

    test_gen = DataFrameSequence(
        df=test,
        directory=TEST_IMG_DIR,
        x_col="id_code",
        y_cols=None,
        batch_size=TEST_BATCH_SIZE,
        shuffle=False,
        rescale=1.0 / 255.0,
        preprocess_fn=preprocess_image,
        horizontal_flip=False,
    )

    print(
        "train batches:",
        len(train_gen),
        "val batches:",
        len(val_gen),
        "test batches:",
        len(test_gen),
    )




## === cell 9
if TF_AVAILABLE:
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import (
        Input,
        GlobalAveragePooling2D,
        Dense,
        Dropout,
        BatchNormalization,
        Conv2D,
        MaxPooling2D,
    )
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint
    from tensorflow.keras.applications.resnet50 import ResNet50
    from tensorflow.keras.applications.densenet import DenseNet121

    MODEL_NAME = "densenet121"

    NB_WARMUP_EPOCHS = 2

    NB_EPOCHS = 6

    INITIAL_LR = 1e-3

    weights_path_template = os.path.join(WEIGHTS_DIR, "{}_weights.h5")
    log_path_template = os.path.join("logs/", "{}_training_log.csv")




## === cell 10
if TF_AVAILABLE:
    os.makedirs("weights", exist_ok=True)
    os.makedirs("logs", exist_ok=True)




## === cell 11
if TF_AVAILABLE:
    """
    ResNet50 based model
    """

    def get_resnet50(input_shape, nb_out):
        inputs = Input(shape=input_shape)
        base_model = ResNet50(
            weights="imagenet", include_top=False, input_tensor=inputs
        )

        x = GlobalAveragePooling2D()(base_model.output)
        x = Dropout(0.5)(x)

        x = Dense(2048, activation="relu")(x)
        x = Dropout(0.5)(x)

        x = Dense(1024, activation="relu")(x)
        x = Dropout(0.5)(x)

        output = Dense(nb_out, activation="softmax", name="final_output")(x)
        model = Model(inputs, output)
        return model




## === cell 12
if TF_AVAILABLE:

    def get_densenet121(input_shape, nb_out):
        inputs = Input(shape=input_shape)
        base_model = DenseNet121(weights=None, include_top=False, input_tensor=inputs)

        x = GlobalAveragePooling2D()(base_model.output)
        x = Dropout(0.5)(x)

        x = Dense(1024, activation="relu")(x)
        x = Dropout(0.5)(x)

        output = Dense(nb_out, activation="sigmoid")(x)
        model = Model(inputs, output)
        return model




## === cell 13
if TF_AVAILABLE:
    """
    simple CNN (conv1)
    """

    def get_conv1(input_shape, nb_out):
        inputs = Input(shape=input_shape)

        x = Conv2D(64, (7, 7), activation="relu")(inputs)
        x = MaxPooling2D((2, 2))(x)
        x = BatchNormalization()(x)

        x = Conv2D(64, (7, 7), activation="relu")(x)
        x = MaxPooling2D((2, 2))(x)
        x = BatchNormalization()(x)

        x = Conv2D(128, (5, 5), activation="relu")(x)
        x = MaxPooling2D((2, 2))(x)
        x = BatchNormalization()(x)

        x = Conv2D(256, (3, 3), activation="relu")(x)
        x = MaxPooling2D((2, 2))(x)
        x = BatchNormalization()(x)

        x = Conv2D(512, (3, 3), activation="relu")(x)
        x = MaxPooling2D((2, 2))(x)
        x = BatchNormalization()(x)

        x = GlobalAveragePooling2D()(x)
        x = Dropout(0.5)(x)

        x = Dense(2048, activation="relu")(x)
        x = Dropout(0.5)(x)

        x = Dense(1024, activation="relu")(x)
        x = Dropout(0.5)(x)

        output = Dense(nb_out, activation="softmax", name="final_output")(x)
        model = Model(inputs, output)
        return model




## === cell 14
if TF_AVAILABLE:
    """
    simple CNN v2 (conv2)
    """

    def get_conv2(input_shape, nb_out):
        inputs = Input(shape=input_shape)

        x = Conv2D(32, (7, 7), activation="relu")(inputs)
        x = MaxPooling2D((2, 2))(x)
        x = BatchNormalization()(x)

        x = Conv2D(64, (5, 5), activation="relu")(x)
        x = MaxPooling2D((2, 2))(x)
        x = BatchNormalization()(x)

        x = Conv2D(128, (3, 3), activation="relu")(x)
        x = MaxPooling2D((2, 2))(x)
        x = BatchNormalization()(x)

        x = GlobalAveragePooling2D()(x)
        x = Dropout(0.5)(x)

        x = Dense(1024, activation="relu")(x)
        x = Dropout(0.5)(x)

        output = Dense(nb_out, activation="softmax")(x)
        model = Model(inputs, output)
        return model




## === cell 15
if TF_AVAILABLE:
    """
    returns model
    """

    def _candidate_weight_paths(name: str):
        candidates = [
            weights_path_template.format(name),  # ../input/... if exists
            os.path.join("weights", f"{name}_weights.h5"),  # local training output
            os.path.join("weights", f"{name}_weights.hdf5"),  # legacy local output
        ]
        out = []
        for p in candidates:
            if p not in out:
                out.append(p)
        return out

    def _find_first_existing_weight(name: str):
        for wp in _candidate_weight_paths(name):
            if os.path.isfile(wp):
                return wp
        return None

    def get_model(name, input_shape, nb_out):
        models = {
            "resnet50": get_resnet50,
            "conv1": get_conv1,
            "conv2": get_conv2,
            "densenet121": get_densenet121,
        }

        if name not in models:
            print(f"No model named '{name}'")
            return None

        model = models[name](input_shape, nb_out)

        loaded = False
        for wp in _candidate_weight_paths(name):
            if os.path.isfile(wp):
                model.load_weights(wp)
                print(f"loaded model weights from {wp}")
                loaded = True
                break
        if not loaded:
            print("weights not found in any of:", _candidate_weight_paths(name))
            print("continuing without pretrained weights")

        return model




## === cell 16
if TF_AVAILABLE:
    """
    trains a ResNet50-based model
    """

    def _safe_steps(gen):
        return max(1, gen.n // gen.batch_size)

    def train_resnet50(model, train_generator, val_generator, weights_path, log_path):
        for i in range(len(model.layers)):
            model.layers[i].trainable = False
        for i in range(-5, 0):
            model.layers[i].trainable = True

        metrics_list = ["accuracy"]
        optimizer = Adam(learning_rate=INITIAL_LR)
        model.compile(
            optimizer=optimizer, loss="categorical_crossentropy", metrics=metrics_list
        )

        mc = ModelCheckpoint(
            weights_path, monitor="val_loss", save_best_only=True, verbose=1
        )
        cl = CSVLogger(log_path)

        STEP_SIZE_TRAIN = _safe_steps(train_generator)
        STEP_SIZE_VAL = _safe_steps(val_generator)

        model.fit(
            train_generator,
            steps_per_epoch=STEP_SIZE_TRAIN,
            validation_data=val_generator,
            validation_steps=STEP_SIZE_VAL,
            epochs=NB_WARMUP_EPOCHS,
            callbacks=[mc, cl],
            verbose=1,
        )

        train_generator.reset()
        val_generator.reset()

        for i in range(len(model.layers)):
            model.layers[i].trainable = True

        optimizer = Adam(learning_rate=INITIAL_LR)
        model.compile(
            optimizer=optimizer, loss="categorical_crossentropy", metrics=metrics_list
        )

        mc = ModelCheckpoint(
            weights_path, monitor="val_loss", save_best_only=True, verbose=1
        )
        cl = CSVLogger(log_path)

        STEP_SIZE_TRAIN = _safe_steps(train_generator)
        STEP_SIZE_VAL = _safe_steps(val_generator)

        model.fit(
            train_generator,
            steps_per_epoch=STEP_SIZE_TRAIN,
            validation_data=val_generator,
            validation_steps=STEP_SIZE_VAL,
            epochs=NB_WARMUP_EPOCHS,
            callbacks=[mc, cl],
            verbose=1,
        )




## === cell 17
if TF_AVAILABLE:

    def train_densenet121(
        model, train_generator, val_generator, weights_path, log_path
    ):
        metrics_list = ["accuracy"]
        optimizer = Adam(learning_rate=INITIAL_LR)
        model.compile(
            optimizer=optimizer, loss="binary_crossentropy", metrics=metrics_list
        )

        mc = ModelCheckpoint(
            weights_path, monitor="val_loss", save_best_only=True, verbose=1
        )
        cl = CSVLogger(log_path)

        STEP_SIZE_TRAIN = max(1, train_generator.n // train_generator.batch_size)
        STEP_SIZE_VAL = max(1, val_generator.n // val_generator.batch_size)

        model.fit(
            train_generator,
            steps_per_epoch=STEP_SIZE_TRAIN,
            validation_data=val_generator,
            validation_steps=STEP_SIZE_VAL,
            epochs=NB_EPOCHS,
            callbacks=[mc, cl],
            verbose=1,
        )




## === cell 18
if TF_AVAILABLE:
    """
    trains the simple CNN
    """

    def train_conv1(model, train_generator, val_generator, weights_path, log_path):
        metrics_list = ["accuracy"]
        optimizer = Adam(learning_rate=INITIAL_LR)
        model.compile(
            optimizer=optimizer, loss="categorical_crossentropy", metrics=metrics_list
        )

        mc = ModelCheckpoint(
            weights_path, monitor="val_loss", save_best_only=True, verbose=1
        )
        cl = CSVLogger(log_path)

        STEP_SIZE_TRAIN = max(1, train_generator.n // train_generator.batch_size)
        STEP_SIZE_VAL = max(1, val_generator.n // val_generator.batch_size)

        model.fit(
            train_generator,
            steps_per_epoch=STEP_SIZE_TRAIN,
            validation_data=val_generator,
            validation_steps=STEP_SIZE_VAL,
            epochs=NB_EPOCHS,
            callbacks=[mc, cl],
            verbose=1,
        )




## === cell 19
if TF_AVAILABLE:

    def train_conv2(model, train_generator, val_generator, weights_path, log_path):
        metrics_list = ["accuracy"]
        optimizer = Adam(learning_rate=INITIAL_LR)
        model.compile(
            optimizer=optimizer, loss="categorical_crossentropy", metrics=metrics_list
        )

        mc = ModelCheckpoint(
            weights_path, monitor="val_loss", save_best_only=True, verbose=1
        )
        cl = CSVLogger(log_path)

        STEP_SIZE_TRAIN = max(1, train_generator.n // train_generator.batch_size)
        STEP_SIZE_VAL = max(1, val_generator.n // val_generator.batch_size)

        model.fit(
            train_generator,
            steps_per_epoch=STEP_SIZE_TRAIN,
            validation_data=val_generator,
            validation_steps=STEP_SIZE_VAL,
            epochs=NB_EPOCHS,
            callbacks=[mc, cl],
            verbose=1,
        )




## === cell 20
if TF_AVAILABLE:

    def train_model(name, input_shape, nb_out, train_generator, val_generator):
        model = get_model(name, input_shape, nb_out)

        trainers = {
            "resnet50": train_resnet50,
            "conv1": train_conv1,
            "conv2": train_conv2,
            "densenet121": train_densenet121,
        }

        if name not in trainers:
            print(f"No model named '{name}'")
            return None

        weights_path = weights_path_template.format(name)
        if not os.path.isdir(os.path.dirname(weights_path)):
            weights_path = os.path.join("weights", f"{name}_weights.h5")

        trainers[name](
            model,
            train_generator,
            val_generator,
            weights_path,
            log_path_template.format(name),
        )

        return weights_path




## === cell 21
if TF_AVAILABLE:
    existing_wp = None
    try:
        existing_wp = _find_first_existing_weight(MODEL_NAME)
    except Exception:
        existing_wp = None

    if existing_wp is None:
        TRAINING = True
        print(
            "No existing weights found; enabling TRAINING to avoid untrained inference (0.0-ish score)."
        )
    else:
        TRAINING = False
        print("Existing weights found; keeping TRAINING disabled. Using:", existing_wp)

if TF_AVAILABLE and TRAINING:
    trained_wp = train_model(
        MODEL_NAME, (IMG_SIZE, IMG_SIZE, NB_CHANNELS), NB_CLASSES, train_gen, val_gen
    )
    if trained_wp is not None and os.path.isfile(trained_wp):
        print("Training complete; weights saved at:", trained_wp)




## === cell 22
id_codes = pd.read_csv(TEST_CSV)["id_code"].astype(str).values

if not TF_AVAILABLE:
    submission = pd.DataFrame(
        {"id_code": id_codes, "diagnosis": np.zeros_like(id_codes, dtype=int)}
    )
    submission.to_csv("submission.csv", index=False)
    print(submission.head())
    print("Wrote submission.csv with shape:", submission.shape)
    print("submission.csv exists:", os.path.isfile("submission.csv"))
else:
    best_wp = _find_first_existing_weight(MODEL_NAME)
    if best_wp is not None:
        print("Inference will use weights:", best_wp)
    else:
        print("WARNING: Still no weights found for inference; predictions likely poor.")

    model = get_model(MODEL_NAME, (IMG_SIZE, IMG_SIZE, NB_CHANNELS), NB_CLASSES)

    test_ds = _build_inference_dataset(
        df=test,
        directory=TEST_IMG_DIR,
        batch_size=TEST_BATCH_SIZE,
        rescale=1.0 / 255.0,
    )

    n_test = len(test)
    pred_steps = int(math.ceil(n_test / float(TEST_BATCH_SIZE)))

    preds = model.predict(
        test_ds,
        steps=pred_steps,
        verbose=1,
    )

    preds_bin = preds > 0.5
    predictions = preds_bin.astype(int).sum(axis=1) - 1
    predictions = np.clip(predictions, 0, 4).astype(int)

    submission = pd.DataFrame({"id_code": id_codes, "diagnosis": predictions})
    submission.to_csv("submission.csv", index=False)

    print(submission.head())
    print("Wrote submission.csv with shape:", submission.shape)
    print("submission.csv exists:", os.path.isfile("submission.csv"))
