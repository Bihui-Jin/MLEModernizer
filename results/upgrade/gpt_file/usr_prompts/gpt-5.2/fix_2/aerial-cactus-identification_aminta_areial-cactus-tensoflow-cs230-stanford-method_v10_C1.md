# Goal

I want you to fix bugs and increase the score toward a target for a Kaggle competition solution. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (big fix and/or evaluation score improvement); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Ensure it runs end-to-end and produces a valid submission file.


# 1. Kaggle task description

## Task
Create a classifier to predict whether an image contains a cactus.

## Metric
Area under the ROC curve.

## Submission Format
For each ID in the test set, you must predict a probability for the `has_cactus` variable. The file should contain a header and have the following format:

```
id,has_cactus
000940378805c44108d287872b2f04ce.jpg,0.5
0017242f54ececa4512b4d7937d1e21e.jpg,0.5
001ee6d8564003107853118ab87df407.jpg,0.5
etc.
```

## Dataset
This dataset contains a large number of 32 x 32 thumbnail images containing aerial photos of a cactus. The file name of an image corresponds to its `id`.

- **train/** - the training set images
- **test/** - the test set images (you must predict the labels of these)
- **train.csv** - the training set labels, indicates whether the image has a cactus (`has_cactus = 1`)
- **sample_submission.csv** - a sample submission file in the correct format

# 2. Python version

3.7

# 3. Installed packages

geopandas==0.14.4
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
numpy==1.26.4
opencv-python==4.12.0.88
opencv-python-headless==4.12.0.88
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
protobuf==6.33.0
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
sklearn-pandas==2.2.0
tensorflow==2.18.0
tensorflow-cloud==0.1.5
tensorflow-datasets==4.9.9
tensorflow_decision_forests==1.11.0
tensorflow-hub==0.16.1
tensorflow-io==0.37.1
tensorflow-io-gcs-filesystem==0.37.1
tensorflow-metadata==1.17.2
tensorflow-probability==0.25.0
tensorflow-text==2.18.1
tqdm==4.67.1

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (56 lines)
            sample_submission.csv (3326 lines)
            sample_submission.csv.zip (67.3 kB)
            test.zip (3.5 MB)
            train.csv (14176 lines)
            train.csv.zip (285.6 kB)
            train.zip (15.0 MB)
            aerial-cactus-identification/
                description.md (56 lines)
                sample_submission.csv (3326 lines)
                ... and 5 other files
                aerial-cactus-identification/
                test/
                    76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                    f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                    ... and 3323 other files
                    test/
                train/
                    775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                    65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                    ... and 14173 other files
                    train/
            test/
                76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                ... and 3323 other files
                test/
            train/
                775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                ... and 14173 other files
                train/
        input/
            description.md (56 lines)
            sample_submission.csv (3326 lines)
            sample_submission.csv.zip (67.3 kB)
            test.zip (3.5 MB)
            train.csv (14176 lines)
            train.csv.zip (285.6 kB)
            train.zip (15.0 MB)
            aerial-cactus-identification/
                description.md (56 lines)
                sample_submission.csv (3326 lines)
                ... and 5 other files
                aerial-cactus-identification/
                test/
                    76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                    f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                    ... and 3323 other files
                    test/
                train/
                    775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                    65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                    ... and 14173 other files
                    train/
            test/
                76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                ... and 3323 other files
                test/
                    76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                    f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                    ... and 3323 other files
                    test/
            train/
                775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                ... and 14173 other files
                train/
                    775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                    65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                    ... and 14173 other files
                    train/
        working/
            aerial-cactus-identification/
                description.md (56 lines)
                sample_submission.csv (3326 lines)
                ... and 5 other files
                aerial-cactus-identification/
                test/
                    76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                    f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                    ... and 3323 other files
                    test/
                train/
                    775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                    65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                    ... and 14173 other files
                    train/
```

-> data/aerial-cactus-identification/sample_submission.csv has 3325 rows and 2 columns.
The columns are: id, has_cactus

-> data/aerial-cactus-identification/train.csv has 14175 rows and 2 columns.
The columns are: id, has_cactus

-> data/sample_submission.csv has 3325 rows and 2 columns.
The columns are: id, has_cactus

-> data/train.csv has 14175 rows and 2 columns.
The columns are: id, has_cactus

-> input/aerial-cactus-identification/sample_submission.csv has 3325 rows and 2 columns.
The columns are: id, has_cactus

-> input/aerial-cactus-identification/train.csv has 14175 rows and 2 columns.
The columns are: id, has_cactus

-> (stopped after 10 files for performance)

# 5. Target score

0.9983

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os
import random
import json
import logging

import numpy as np
import pandas as pd
import cv2 as cv
import matplotlib.pyplot as plt

import sklearn.utils
from tqdm import tqdm

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import (
    Conv2D,
    Dense,
    Flatten,
    BatchNormalization,
    Dropout,
    DepthwiseConv2D,
)
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

print("TensorFlow:", tf.__version__)



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
DATA_ROOT = "/kaggle/input/aerial-cactus-identification"

train_img_dir = os.path.join(DATA_ROOT, "train", "train")
test_img_dir = os.path.join(DATA_ROOT, "test", "test")
csv_dir = os.path.join(DATA_ROOT, "train.csv")
sample_sub_path = os.path.join(DATA_ROOT, "sample_submission.csv")

print("cwd:", os.getcwd())
print("train_img_dir:", train_img_dir)
print("test_img_dir:", test_img_dir)
print("csv_dir:", csv_dir)
print("sample_sub_path:", sample_sub_path)

assert os.path.isfile(csv_dir), f"Missing {csv_dir}"
assert os.path.isfile(sample_sub_path), f"Missing {sample_sub_path}"
assert os.path.isdir(train_img_dir), f"Missing {train_img_dir}"
assert os.path.isdir(test_img_dir), f"Missing {test_img_dir}"



## === cell 2
seed_value = 1372
random.seed(seed_value)
np.random.seed(seed_value)
tf.random.set_seed(seed_value)



## === cell 3
from PIL import Image


def resize_and_save(filename, input_dir, output_dir, size=32):
    """Resize the image contained in `filename` and save it to the `output_dir`"""
    image = Image.open(os.path.join(input_dir, filename))
    image = image.resize((size, size))
    image.save(os.path.join(output_dir, filename))




## === cell 4
class Params:
    """Class that loads hyperparameters from a json file."""

    def __init__(self, json_path):
        self.update(json_path)

    def save(self, json_path):
        with open(json_path, "w") as f:
            json.dump(self.__dict__, f, indent=4)

    def update(self, json_path):
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    @property
    def dict(self):
        return self.__dict__


def set_logger(log_path):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s:%(levelname)s: %(message)s")
        )
        logger.addHandler(file_handler)

        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(stream_handler)


def save_dict_to_json(d, json_path):
    with open(json_path, "w") as f:
        d = {k: float(v) for k, v in d.items()}
        json.dump(d, f, indent=4)




## === cell 5
df = pd.read_csv(csv_dir)
df = sklearn.utils.shuffle(df, random_state=seed_value).reset_index(drop=True)

filenames = (train_img_dir + "/" + df["id"].astype(str)).values
labels = df["has_cactus"].astype(np.float32).values

print("num train rows:", len(df))
print("sample filename:", filenames[0])
print("sample label:", labels[0], type(labels[0]))




## === cell 6
def _parse_function(filename, label, size):
    """Decode JPEG, convert to float32 [0,1], resize."""
    image_string = tf.io.read_file(filename)
    image_decoded = tf.image.decode_jpeg(image_string, channels=3)
    image = tf.image.convert_image_dtype(image_decoded, tf.float32)
    resized_image = tf.image.resize(image, [size, size], method="bilinear")
    return resized_image, label




## === cell 7
def train_preprocess(image, label, use_random_flip):
    """Training-time augmentation (kept as in original)."""
    if use_random_flip:
        image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, max_delta=25.0 / 255.0)
    image = tf.image.random_saturation(image, lower=0.6, upper=1.4)
    image = tf.clip_by_value(image, 0.0, 1.0)
    return image, label




## === cell 8
def input_fn(is_training, filenames, labels, params):
    num_samples = len(filenames)
    assert len(filenames) == len(labels), "Filenames and labels should have same length"

    parse_fn = lambda f, l: _parse_function(f, l, params.image_size)
    train_fn = lambda img, lab: train_preprocess(img, lab, params.use_random_flip)

    ds = tf.data.Dataset.from_tensor_slices(
        (tf.constant(filenames), tf.constant(labels))
    )
    if is_training:
        ds = ds.shuffle(num_samples)
        ds = ds.map(parse_fn, num_parallel_calls=params.num_parallel_calls)
        ds = ds.map(train_fn, num_parallel_calls=params.num_parallel_calls)
    else:
        ds = ds.map(parse_fn, num_parallel_calls=params.num_parallel_calls)

    ds = ds.batch(params.batch_size).repeat().prefetch(tf.data.AUTOTUNE)
    return ds




## === cell 9
params_dict = {
    "learning_rate": 1.5e-3,
    "batch_size": 64,
    "num_epochs": 50,
    "image_size": 32,
    "use_random_flip": False,
    "num_labels": 2,
    "num_parallel_calls": 8,
    "save_summary_steps": 1,
}
json_path = os.path.join("./", "params.json")
with open(json_path, "w") as f:
    json.dump(params_dict, f, indent=2)

assert os.path.isfile(json_path), f"No json configuration file found at {json_path}"
params = Params(json_path)
print("Loaded params:", params.dict)



## === cell 10
split = int(len(filenames) * 0.15)
valid_filenames, valid_labels = filenames[:split], labels[:split]
train_filenames, train_labels = filenames[split:], labels[split:]

train_dataset = input_fn(True, train_filenames, train_labels, params)
valid_dataset = input_fn(False, valid_filenames, valid_labels, params)

steps_per_epoch = int(np.ceil(len(train_filenames) / params.batch_size))
validation_steps = int(np.ceil(len(valid_filenames) / params.batch_size))

print("train size:", len(train_filenames), "valid size:", len(valid_filenames))
print("steps_per_epoch:", steps_per_epoch, "validation_steps:", validation_steps)



## === cell 11
one_batch = next(iter(train_dataset.take(1)))
print(one_batch[0].shape, "= batch_size x 32 x 32 x 3")
for i in range(3):
    plt.figure()
    plt.imshow(one_batch[0][i].numpy())
    plt.title(f"label={float(one_batch[1][i].numpy())}")
    plt.grid(False)
plt.show()



## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
NotFoundError                             Traceback (most recent call last)
/tmp/ipykernel_11/1085924859.py in <cell line: 0>()
      1 # ---- Fix: TF2 runs eagerly; remove TF1 iterator/session usage.
      2 # Provide a quick sanity-check batch preview without affecting training logic.
----> 3 one_batch = next(iter(train_dataset.take(1)))
      4 print(one_batch[0].shape, "= batch_size x 32 x 32 x 3")
      5 for i in range(3):

/usr/local/lib/python3.11/dist-packages/tensorflow/python/data/ops/iterator_ops.py in __next__(self)
    824   def __next__(self):
    825     try:
--> 826       return self._next_internal()
    827     except errors.OutOfRangeError:
    828       raise StopIteration

/usr/local/lib/python3.11/dist-packages/tensorflow/python/data/ops/iterator_ops.py in _next_internal(self)
    774     # to communicate that there is no more data to iterate over.
    775     with context.execution_mode(context.SYNC):
--> 776       ret = gen_dataset_ops.iterator_get_next(
    777           self._iterator_resource,
    778           output_types=self._flat_output_types,

/usr/local/lib/python3.11/dist-packages/tensorflow/python/ops/gen_dataset_ops.py in iterator_get_next(iterator, output_types, output_shapes, name)
   3084       return _result
   3085     except _core._NotOkStatusException as e:
-> 3086       _ops.raise_from_not_ok_status(e, name)
   3087     except _core._FallbackException:
   3088       pass

/usr/local/lib/python3.11/dist-packages/tensorflow/python/framework/ops.py in raise_from_not_ok_status(e, name)
   6000 def raise_from_not_ok_status(e, name) -> NoReturn:
   6001   e.message += (" name: " + str(name if name is not None else ""))
-> 6002   raise core._status_to_exception(e) from None  # pylint: disable=protected-access
   6003 
   6004 

NotFoundError: {{function_node __wrapped__IteratorGetNext_output_types_2_device_/job:localhost/replica:0/task:0/device:CPU:0}} Error in user-defined function passed to ParallelMapDatasetV2:2 transformation with iterator: Iterator::Root::Prefetch::FiniteTake::Prefetch::ForeverRepeat[0]::MapAndBatch::ParallelMapV2: /kaggle/input/aerial-cactus-identification/train/train/75a3823cd2db7ec0dfa9e8dad8ef3407.jpg; No such file or directory
	 [[{{node ReadFile}}]] [Op:IteratorGetNext] name: 

## === cell 12
model = Sequential()

model.add(Conv2D(3, kernel_size=3, activation="relu", input_shape=(32, 32, 3)))

model.add(Conv2D(filters=16, kernel_size=3, activation="relu"))
model.add(Conv2D(filters=16, kernel_size=3, activation="relu"))
model.add(BatchNormalization())
model.add(Dropout(0.5))

model.add(DepthwiseConv2D(kernel_size=3, strides=1, padding="same", use_bias=True))
model.add(Conv2D(filters=32, kernel_size=1, activation="relu"))
model.add(Conv2D(filters=64, kernel_size=1, activation="relu"))
model.add(BatchNormalization())
model.add(Dropout(0.5))

model.add(DepthwiseConv2D(kernel_size=3, strides=2, padding="same", use_bias=True))
model.add(Conv2D(filters=128, kernel_size=1, activation="relu"))
model.add(Conv2D(filters=256, kernel_size=1, activation="relu"))
model.add(BatchNormalization())
model.add(Dropout(0.5))

model.add(DepthwiseConv2D(kernel_size=3, strides=1, padding="same", use_bias=True))
model.add(Conv2D(filters=256, kernel_size=1, activation="relu"))
model.add(BatchNormalization())
model.add(Conv2D(filters=512, kernel_size=1, activation="relu"))
model.add(BatchNormalization())
model.add(Dropout(0.5))

model.add(DepthwiseConv2D(kernel_size=3, strides=2, padding="same", use_bias=True))
model.add(Conv2D(filters=512, kernel_size=1, activation="relu"))
model.add(BatchNormalization())
model.add(Conv2D(filters=512, kernel_size=1, activation="relu"))
model.add(BatchNormalization())
model.add(Dropout(0.5))

model.add(DepthwiseConv2D(kernel_size=3, strides=1, padding="same", use_bias=True))
model.add(Conv2D(filters=1024, kernel_size=1, activation="relu"))
model.add(BatchNormalization())
model.add(Conv2D(filters=1024, kernel_size=1, activation="relu"))
model.add(BatchNormalization())
model.add(Dropout(0.5))

model.add(Flatten())

model.add(Dense(512, activation="relu"))
model.add(BatchNormalization())
model.add(Dropout(0.5))

model.add(Dense(256, activation="relu"))
model.add(BatchNormalization())
model.add(Dropout(0.5))

model.add(Dense(128, activation="elu"))
model.add(Dense(1, activation="sigmoid"))



## === cell 13
Adam = tf.keras.optimizers.Adam(learning_rate=params.learning_rate, amsgrad=True)
model.compile(optimizer=Adam, loss="binary_crossentropy", metrics=["accuracy"])
model.summary()



## === cell 14
file_path = "weights-aerial-cactus.keras"

callbacks = [
    ModelCheckpoint(
        file_path, monitor="val_accuracy", verbose=1, save_best_only=True, mode="max"
    ),
    ReduceLROnPlateau(
        monitor="val_loss", factor=0.2, patience=3, verbose=1, mode="min", min_lr=1e-5
    ),
    EarlyStopping(
        monitor="val_loss",
        min_delta=1e-10,
        patience=15,
        verbose=1,
        restore_best_weights=True,
    ),
]



## === cell 15
history = model.fit(
    train_dataset,
    validation_data=valid_dataset,
    epochs=params.num_epochs,
    verbose=True,
    steps_per_epoch=steps_per_epoch,
    validation_steps=validation_steps,
    callbacks=callbacks,
)



## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
NotFoundError                             Traceback (most recent call last)
/tmp/ipykernel_11/2939001477.py in <cell line: 0>()
----> 1 history = model.fit(
      2     train_dataset,
      3     validation_data=valid_dataset,
      4     epochs=params.num_epochs,
      5     verbose=True,

/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py in error_handler(*args, **kwargs)
    120             # To get the full stack trace, call:
    121             # `keras.config.disable_traceback_filtering()`
--> 122             raise e.with_traceback(filtered_tb) from None
    123         finally:
    124             del filtered_tb

/usr/local/lib/python3.11/dist-packages/tensorflow/python/eager/execute.py in quick_execute(op_name, num_outputs, inputs, attrs, ctx, name)
     57       e.message += " name: " + name
     58     raise core._status_to_exception(e) from None
---> 59   except TypeError as e:
     60     keras_symbolic_tensors = [x for x in inputs if _is_keras_symbolic_tensor(x)]
     61     if keras_symbolic_tensors:

NotFoundError: Graph execution error:

Detected at node ReadFile defined at (most recent call last):
<stack traces unavailable>
Detected at node ReadFile defined at (most recent call last):
<stack traces unavailable>
2 root error(s) found.
  (0) NOT_FOUND:  Error in user-defined function passed to ParallelMapDatasetV2:2 transformation with iterator: Iterator::Root::Prefetch::ForeverRepeat[0]::MapAndBatch::ParallelMapV2: /kaggle/input/aerial-cactus-identification/train/train/1deacae1785209f0429b9053ba85c727.jpg; No such file or directory
	 [[{{node ReadFile}}]]
	 [[IteratorGetNext]]
	 [[IteratorGetNext/_2]]
  (1) NOT_FOUND:  Error in user-defined function passed to ParallelMapDatasetV2:2 transformation with iterator: Iterator::Root::Prefetch::ForeverRepeat[0]::MapAndBatch::ParallelMapV2: /kaggle/input/aerial-cactus-identification/train/train/1deacae1785209f0429b9053ba85c727.jpg; No such file or directory
	 [[{{node ReadFile}}]]
	 [[IteratorGetNext]]
0 successful operations.
0 derived errors ignored. [Op:__inference_multi_step_on_iterator_15631]

## === cell 16
if os.path.isfile(file_path):
    model.load_weights(file_path)
    print("Loaded best weights from", file_path)
else:
    print("No weight file found at", file_path, "- using in-memory trained weights.")




## === cell 17
def plot_training_curves(history):
    acc = history.history.get("accuracy", [])
    val_acc = history.history.get("val_accuracy", [])
    loss = history.history.get("loss", [])
    val_loss = history.history.get("val_loss", [])

    epochs = range(1, len(loss) + 1)

    plt.figure()
    plt.plot(epochs, loss, "r", label="Training loss")
    plt.plot(epochs, val_loss, "g", label="Validation loss")
    plt.title("Losses")
    plt.legend()

    if acc and val_acc:
        plt.figure()
        plt.plot(epochs, acc, "r", label="Training accuracy")
        plt.plot(epochs, val_acc, "g", label="Validation accuracy")
        plt.title("Accuracies")
        plt.legend()
    plt.show()




## === cell 18
plot_training_curves(history)



## --- ERROR in cell 18, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2210949289.py in <cell line: 0>()
----> 1 plot_training_curves(history)
      2 

NameError: name 'history' is not defined

## === cell 19
test_df = pd.read_csv(sample_sub_path)
images_test = test_df["id"].values

X_test = []
for img_id in tqdm(images_test, desc="Loading test images"):
    img = cv.imread(os.path.join(test_img_dir, img_id))
    if img is None:
        raise FileNotFoundError(f"Failed to read test image: {img_id}")
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    X_test.append(img)

X_test = np.asarray(X_test, dtype=np.float32) / 255.0

y_test_pred = model.predict(X_test, batch_size=params.batch_size, verbose=0).reshape(-1)
y_test_pred = np.clip(y_test_pred, 0.0, 1.0)

submission = pd.DataFrame(
    {"id": images_test, "has_cactus": y_test_pred.astype(np.float32)}
)
submission.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", submission.shape)
print(submission.head())

## --- ERROR in cell 19, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/2563308391.py in <cell line: 0>()
      7     img = cv.imread(os.path.join(test_img_dir, img_id))
      8     if img is None:
----> 9         raise FileNotFoundError(f"Failed to read test image: {img_id}")
     10     # cv2 reads BGR; training pipeline uses decoded JPEG (RGB). Convert to RGB for consistency.
     11     img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

FileNotFoundError: Failed to read test image: 09034a34de0e2015a8a28dfe18f423f6.jpg
