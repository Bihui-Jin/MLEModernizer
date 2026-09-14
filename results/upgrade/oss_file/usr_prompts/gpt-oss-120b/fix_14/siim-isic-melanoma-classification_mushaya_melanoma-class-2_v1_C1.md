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
Predict whether a lesion is malignant (0 denotes **benign**, and 1 indicates **malignant**).

## Metric
Area under the ROC curve.

## Submission Format
For each `image_name` in the test set, you must predict the probability (`target`) that the sample is **malignant**. The file should contain a header and have the following format:

```
image_name,target
ISIC_0052060,0.7
ISIC_0052349,0.9
ISIC_0058510,0.8
ISIC_0073313,0.5
ISIC_0073502,0.5
etc.
```

## Dataset 
The images are provided in DICOM format.

Images are also provided in JPEG and TFRecord format (in the `jpeg` and `tfrecords` directories, respectively). Images in TFRecord format have been resized to a uniform 1024x1024.

Metadata is also provided outside of the DICOM format, in CSV files. See the `Columns` section for a description.

### Files
- **train.csv** - the training set
- **test.csv** - the test set
- **sample_submission.csv** - a sample submission file in the correct format

### Columns
- `image_name` - unique identifier, points to filename of related DICOM image
- `patient_id` - unique patient identifier
- `sex` - the sex of the patient (when unknown, will be blank)
- `age_approx` - approximate patient age at time of imaging
- `anatom_site_general_challenge` - location of imaged site
- `diagnosis` - detailed diagnosis information (train only)
- `benign_malignant` - indicator of malignancy of imaged lesion
- `target` - binarized version of the target variable

# 2. Python version

3.8

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
pillow==11.3.0
protobuf==6.33.0
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
seaborn==0.12.2
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
            description.md (176 lines)
            jpeg.zip (24.7 GB)
            sample_submission.csv (4143 lines)
            sample_submission.csv.zip (16.4 kB)
            test.csv (4143 lines)
            test.csv.zip (42.5 kB)
            test.zip (6.4 GB)
            tfrecords.zip (9.3 GB)
            train.csv (28985 lines)
            train.csv.zip (299.7 kB)
            train.zip (46.0 GB)
            jpeg/
                test/
                    ISIC_1440063.jpg (1.1 MB)
                    ISIC_0815802.jpg (853.1 kB)
                    ... and 4140 other files
                train/
                    ISIC_1845271.jpg (1.0 MB)
                    ISIC_1970027.jpg (138.4 kB)
                    ... and 28982 other files
            siim-isic-melanoma-classification/
                description.md (176 lines)
                jpeg.zip (24.7 GB)
                ... and 9 other files
                jpeg/
                    test/
                        ISIC_1440063.jpg (1.1 MB)
                        ISIC_0815802.jpg (853.1 kB)
                        ... and 4140 other files
                    train/
                        ISIC_1845271.jpg (1.0 MB)
                        ISIC_1970027.jpg (138.4 kB)
                        ... and 28982 other files
                siim-isic-melanoma-classification/
                test/
                    ISIC_0052212.dcm (1.5 MB)
                    ISIC_0076545.dcm (4.0 MB)
                    ... and 4140 other files
                    test/
                tfrecords/
                    test00-2071.tfrec (579.6 MB)
                    test01-2071.tfrec (583.5 MB)
                    ... and 14 other files
                train/
                    ISIC_0015719.dcm (2.4 MB)
                    ISIC_0068279.dcm (1.3 MB)
                    ... and 28982 other files
                    train/
            test/
                ISIC_0052212.dcm (1.5 MB)
                ISIC_0076545.dcm (4.0 MB)
                ... and 4140 other files
                test/
            tfrecords/
                test00-2071.tfrec (579.6 MB)
                test01-2071.tfrec (583.5 MB)
                ... and 14 other files
            train/
                ISIC_0015719.dcm (2.4 MB)
                ISIC_0068279.dcm (1.3 MB)
                ... and 28982 other files
                train/
        input/
            description.md (176 lines)
            jpeg.zip (24.7 GB)
            sample_submission.csv (4143 lines)
            sample_submission.csv.zip (16.4 kB)
            test.csv (4143 lines)
            test.csv.zip (42.5 kB)
            test.zip (6.4 GB)
            tfrecords.zip (9.3 GB)
            train.csv (28985 lines)
            train.csv.zip (299.7 kB)
            train.zip (46.0 GB)
            jpeg/
                test/
                    ISIC_1440063.jpg (1.1 MB)
                    ISIC_0815802.jpg (853.1 kB)
                    ... and 4140 other files
                train/
                    ISIC_1845271.jpg (1.0 MB)
                    ISIC_1970027.jpg (138.4 kB)
                    ... and 28982 other files
            siim-isic-melanoma-classification/
                description.md (176 lines)
                jpeg.zip (24.7 GB)
                ... and 9 other files
                jpeg/
                    test/
                        ISIC_1440063.jpg (1.1 MB)
                        ISIC_0815802.jpg (853.1 kB)
                        ... and 4140 other files
                    train/
                        ISIC_1845271.jpg (1.0 MB)
                        ISIC_1970027.jpg (138.4 kB)
                        ... and 28982 other files
                siim-isic-melanoma-classification/
                test/
                    ISIC_0052212.dcm (1.5 MB)
                    ISIC_0076545.dcm (4.0 MB)
                    ... and 4140 other files
                    test/
                tfrecords/
                    test00-2071.tfrec (579.6 MB)
                    test01-2071.tfrec (583.5 MB)
                    ... and 14 other files
                train/
                    ISIC_0015719.dcm (2.4 MB)
                    ISIC_0068279.dcm (1.3 MB)
                    ... and 28982 other files
                    train/
            test/
                ISIC_0052212.dcm (1.5 MB)
                ISIC_0076545.dcm (4.0 MB)
                ... and 4140 other files
                test/
                    ISIC_0052212.dcm (1.5 MB)
                    ISIC_0076545.dcm (4.0 MB)
                    ... and 4140 other files
                    test/
            tfrecords/
                test00-2071.tfrec (579.6 MB)
                test01-2071.tfrec (583.5 MB)
                ... and 14 other files
            train/
                ISIC_0015719.dcm (2.4 MB)
                ISIC_0068279.dcm (1.3 MB)
                ... and 28982 other files
                train/
                    ISIC_0015719.dcm (2.4 MB)
                    ISIC_0068279.dcm (1.3 MB)
                    ... and 28982 other files
                    train/
        working/
            siim-isic-melanoma-classification/
                description.md (176 lines)
                jpeg.zip (24.7 GB)
                ... and 9 other files
                jpeg/
                    test/
                        ISIC_1440063.jpg (1.1 MB)
                        ISIC_0815802.jpg (853.1 kB)
                        ... and 4140 other files
                    train/
                        ISIC_1845271.jpg (1.0 MB)
                        ISIC_1970027.jpg (138.4 kB)
                        ... and 28982 other files
                siim-isic-melanoma-classification/
                test/
                    ISIC_0052212.dcm (1.5 MB)
                    ISIC_0076545.dcm (4.0 MB)
                    ... and 4140 other files
                    test/
                tfrecords/
                    test00-2071.tfrec (579.6 MB)
                    test01-2071.tfrec (583.5 MB)
                    ... and 14 other files
                train/
                    ISIC_0015719.dcm (2.4 MB)
                    ISIC_0068279.dcm (1.3 MB)
                    ... and 28982 other files
                    train/
```

-> data/sample_submission.csv has 4142 rows and 2 columns.
The columns are: image_name, target

-> data/siim-isic-melanoma-classification/sample_submission.csv has 4142 rows and 2 columns.
The columns are: image_name, target

-> data/siim-isic-melanoma-classification/test.csv has 4142 rows and 5 columns.
The columns are: image_name, patient_id, sex, age_approx, anatom_site_general_challenge

-> data/siim-isic-melanoma-classification/train.csv has 28984 rows and 8 columns.
The columns are: image_name, patient_id, sex, age_approx, anatom_site_general_challenge, diagnosis, benign_malignant, target

-> data/test.csv has 4142 rows and 5 columns.
The columns are: image_name, patient_id, sex, age_approx, anatom_site_general_challenge

-> data/train.csv has 28984 rows and 8 columns.
The columns are: image_name, patient_id, sex, age_approx, anatom_site_general_challenge, diagnosis, benign_malignant, target

-> input/sample_submission.csv has 4142 rows and 2 columns.
The columns are: image_name, target

-> (stopped after 10 files for performance)

# 5. Target score

0.7838

# 6. Current score

0.63188

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.66839) has done: 'I add a ModelCheckpoint callback that saves the best model based on validation AUC, load that checkpoint before making predictions, and increase the training epochs slightly so the model has a better chance of reaching the target AUC while keeping all other logic unchanged.'
- What this solution (achieved 0.63188) has done: 'I keep the overall model and training pipeline unchanged but add a small amount of extra augmentation (random rotation) and make the learning‑rate scheduler and early‑stopping a bit less aggressive so the model can keep improving a little longer. These tweaks are minimal, keep the same architecture, and are expected to raise the validation AUC toward the target 0.7838 without over‑hauling the solution.'

# 9. Code solution

## === cell 0
import os, math, random, re, time

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras import layers, Model
import matplotlib.pylab as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import PIL
import gc
import cv2
import seaborn as sns

from tqdm import tqdm
from sklearn.utils import class_weight  # added for class weighting

policy = tf.keras.mixed_precision.Policy("mixed_float16")
tf.keras.mixed_precision.set_global_policy(policy)




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()  # TPU detection.
    print("Running on TPU ", tpu.master())
except ValueError:
    tpu = None

if tpu:
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.experimental.TPUStrategy(tpu)
else:
    strategy = tf.distribute.get_strategy()  # default strategy (CPU/GPU)

AUTO = tf.data.experimental.AUTOTUNE
REPLICAS = strategy.num_replicas_in_sync
print(f"REPLICAS: {REPLICAS}")




## === cell 2
dirname = "../input/siim-isic-melanoma-classification/"
train = pd.read_csv(os.path.join(dirname, "train.csv"))
test = pd.read_csv(os.path.join(dirname, "test.csv"))
print(train.head())
print(len(train))
print(len(test))
print(train["target"].value_counts())

class_weights_array = class_weight.compute_class_weight(
    class_weight="balanced", classes=np.unique(train["target"]), y=train["target"]
)
class_weights = {i: w for i, w in enumerate(class_weights_array)}
print("Class weights:", class_weights)




## === cell 3
sns.countplot(train["target"])




## === cell 4
GCS_PATH = os.path.join("..", "input", "siim-isic-melanoma-classification")
train_filenames = tf.io.gfile.glob(os.path.join(GCS_PATH, "tfrecords", "train*.tfrec"))
test_filenames = tf.io.gfile.glob(os.path.join(GCS_PATH, "tfrecords", "test*.tfrec"))
print(
    f"Found {len(train_filenames)} training TFRecord files and {len(test_filenames)} test TFRecord files."
)




## === cell 5
train_filenames, valid_filenames = train_test_split(
    train_filenames, test_size=0.2, shuffle=True
)
print(
    f"Training TFRecords: {len(train_filenames)}, Validation TFRecords: {len(valid_filenames)}"
)




## === cell 6
BATCH_SIZE = 8 * strategy.num_replicas_in_sync
IMAGE_SIZE = [1024, 1024]
AUTO = tf.data.experimental.AUTOTUNE
imSize = 1024




## === cell 7
def decode_image(image_data):
    image = tf.image.decode_jpeg(image_data, channels=3)
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.reshape(image, [*IMAGE_SIZE, 3])
    image = tf.image.resize(image, [imSize, imSize])
    return image


def read_labeled_tfrecord(example):
    LABELED_TFREC_FORMAT = {
        "image": tf.io.FixedLenFeature([], tf.string),
        "target": tf.io.FixedLenFeature([], tf.int64),
    }
    example = tf.io.parse_single_example(example, LABELED_TFREC_FORMAT)
    image = decode_image(example["image"])
    label = tf.cast(example["target"], tf.int32)
    return image, label


def read_unlabeled_tfrecord(example):
    UNLABELED_TFREC_FORMAT = {
        "image": tf.io.FixedLenFeature([], tf.string),
        "image_name": tf.io.FixedLenFeature([], tf.string),
    }
    example = tf.io.parse_single_example(example, UNLABELED_TFREC_FORMAT)
    image = decode_image(example["image"])
    idnum = example["image_name"]
    return image, idnum


def load_dataset(filenames, labeled=True, ordered=False):
    ignore_order = tf.data.Options()
    if not ordered:
        ignore_order.experimental_deterministic = False
    dataset = tf.data.TFRecordDataset(filenames, num_parallel_reads=AUTO)
    dataset = dataset.with_options(ignore_order)
    dataset = dataset.map(
        read_labeled_tfrecord if labeled else read_unlabeled_tfrecord,
        num_parallel_calls=AUTO,
    )
    return dataset


def data_augment(image, label):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    angle = tf.random.uniform([], -0.1, 0.1)  # radians ~ +/-5 degrees
    image = tfa.image.rotate(image, angle)
    image = tf.image.random_saturation(image, 0.5, 1.5)
    image = tf.image.random_brightness(image, max_delta=0.1)
    image = tf.image.random_contrast(image, lower=0.9, upper=1.1)
    return image, label




## === cell 8
def get_training_dataset():
    dataset = load_dataset(train_filenames, labeled=True)
    dataset = dataset.cache()  # <<< added cache
    dataset = dataset.map(data_augment, num_parallel_calls=AUTO)
    dataset = dataset.repeat()
    dataset = dataset.shuffle(2048)
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(AUTO)
    return dataset


def get_val_dataset():
    dataset = load_dataset(valid_filenames, labeled=True)
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.cache()
    dataset = dataset.prefetch(AUTO)
    return dataset




## === cell 9
def count_data_items(filenames):
    n = [
        int(re.compile(r"-([0-9]*)\.").search(filename).group(1))
        for filename in filenames
    ]
    return np.sum(n)


NUM_TRAINING_IMAGES = count_data_items(train_filenames)
NUM_VALID_IMAGES = count_data_items(valid_filenames)

STEPS_PER_EPOCH = math.ceil(NUM_TRAINING_IMAGES / BATCH_SIZE)
VALIDATION_STEPS = math.ceil(NUM_VALID_IMAGES / BATCH_SIZE)

print(
    f"Dataset: {NUM_TRAINING_IMAGES} training images, {NUM_VALID_IMAGES} validation images"
)
print(f"Steps per epoch: {STEPS_PER_EPOCH}, Validation steps: {VALIDATION_STEPS}")




## === cell 10
def get_test_dataset(ordered=False):
    dataset = load_dataset(test_filenames, labeled=False, ordered=ordered)
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(AUTO)
    return dataset




## === cell 11
def res_block(X_in, channels):
    X = layers.Conv2D(channels, (3, 3), strides=(1, 1), padding="same")(X_in)
    X = layers.BatchNormalization()(X)
    X = layers.LeakyReLU()(X)

    X = layers.Conv2D(channels, (3, 3), strides=(1, 1), padding="same")(X)
    X = layers.BatchNormalization()(X)
    X = layers.Add()([X, X_in])
    X = layers.LeakyReLU()(X)

    return X




## === cell 12
def my_model():
    X_in = layers.Input((1024, 1024, 3))

    X = layers.AveragePooling2D(pool_size=(2, 2), strides=2, name="avg_pool1")(X_in)

    X = layers.Conv2D(64, (3, 3), strides=(1, 1), padding="same", name="conv1")(X)
    X = layers.BatchNormalization()(X)
    X = layers.Activation("relu")(X)

    X = res_block(X, 64)

    X = layers.MaxPool2D(pool_size=(2, 2), strides=2, name="max_pool1")(X)
    X = layers.MaxPool2D(pool_size=(2, 2), strides=2, name="max_pool1.2")(X)

    X = layers.Conv2D(128, (3, 3), strides=(1, 1), padding="same", name="conv2")(X)
    X = layers.BatchNormalization()(X)
    X = layers.Activation("relu")(X)

    X = res_block(X, 128)

    X = layers.MaxPool2D(pool_size=(2, 2), strides=2, name="max_pool2")(X)

    X = layers.Conv2D(256, (3, 3), strides=(1, 1), padding="same", name="conv3")(X)
    X = layers.BatchNormalization()(X)
    X = layers.Activation("relu")(X)

    X = res_block(X, 256)

    X = layers.MaxPool2D(pool_size=(2, 2), strides=2, name="max_pool3")(X)

    X = res_block(X, 256)

    X = layers.MaxPool2D(pool_size=(2, 2), strides=2, name="max_pool4")(X)

    X = res_block(X, 256)

    X = layers.MaxPool2D(pool_size=(2, 2), strides=2, name="max_pool5")(X)

    X = layers.Flatten()(X)
    X = layers.Dense(4096, activation="relu", name="fc1")(X)
    X = layers.Dense(1024, activation="relu", name="fc2")(X)
    X = layers.Dropout(0.3)(X)
    X_out = layers.Dense(1, activation="sigmoid", name="answer")(X)

    model = Model(inputs=X_in, outputs=X_out, name="pinnet")
    return model




## === cell 13
with strategy.scope():
    model = my_model()
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
    )




## === cell 14
model.summary()




## === cell 15
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping


def callback():
    cb = []
    reduceLROnPlat = ReduceLROnPlateau(
        monitor="val_auc",
        factor=0.5,  # less aggressive LR reduction
        patience=6,  # give more epochs before reduction
        verbose=1,
        mode="max",
        min_delta=0.0001,
        cooldown=1,
        min_lr=1e-5,
    )
    cb.append(reduceLROnPlat)

    checkpoint = ModelCheckpoint(
        filepath="best_model.weights.h5",
        monitor="val_auc",
        mode="max",
        save_best_only=True,
        verbose=1,
        save_weights_only=True,
    )
    cb.append(checkpoint)

    early_stop = EarlyStopping(
        monitor="val_auc",
        mode="max",
        patience=10,  # allow more epochs before stopping
        restore_best_weights=True,
        verbose=1,
    )
    cb.append(early_stop)

    return cb




## === cell 16
cb = callback()
epochs = 120  # modestly longer training window
history = model.fit(
    get_training_dataset(),
    epochs=epochs,
    verbose=True,
    steps_per_epoch=STEPS_PER_EPOCH,
    validation_data=get_val_dataset(),
    validation_steps=VALIDATION_STEPS,
    callbacks=cb,
    class_weight=class_weights,
)




## --- ERROR in cell 16, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/116252442.py in <cell line: 0>()
      2 epochs = 120  # modestly longer training window
      3 history = model.fit(
----> 4     get_training_dataset(),
      5     epochs=epochs,
      6     verbose=True,

/tmp/ipykernel_55/3747007273.py in get_training_dataset()
      2     dataset = load_dataset(train_filenames, labeled=True)
      3     dataset = dataset.cache()  # <<< added cache
----> 4     dataset = dataset.map(data_augment, num_parallel_calls=AUTO)
      5     dataset = dataset.repeat()
      6     dataset = dataset.shuffle(2048)

/usr/local/lib/python3.11/dist-packages/tensorflow/python/data/ops/dataset_ops.py in map(self, map_func, num_parallel_calls, deterministic, synchronous, use_unbounded_threadpool, name)
   2339     from tensorflow.python.data.ops import map_op
   2340 
-> 2341     return map_op._map_v2(
   2342         self,
   2343         map_func,

/usr/local/lib/python3.11/dist-packages/tensorflow/python/data/ops/map_op.py in _map_v2(input_dataset, map_func, num_parallel_calls, deterministic, synchronous, use_unbounded_threadpool, name)
     55           num_parallel_calls,
     56       )
---> 57     return _ParallelMapDataset(
     58         input_dataset,
     59         map_func,

/usr/local/lib/python3.11/dist-packages/tensorflow/python/data/ops/map_op.py in __init__(self, input_dataset, map_func, num_parallel_calls, deterministic, use_inter_op_parallelism, preserve_cardinality, use_legacy_function, use_unbounded_threadpool, name)
    200     self._input_dataset = input_dataset
    201     self._use_inter_op_parallelism = use_inter_op_parallelism
--> 202     self._map_func = structured_function.StructuredFunctionWrapper(
    203         map_func,
    204         self._transformation_name(),

/usr/local/lib/python3.11/dist-packages/tensorflow/python/data/ops/structured_function.py in __init__(self, func, transformation_name, dataset, input_classes, input_shapes, input_types, input_structure, add_to_graph, use_legacy_function, defun_kwargs)
    263         fn_factory = trace_tf_function(defun_kwargs)
    264 
--> 265     self._function = fn_factory()
    266     # There is no graph to add in eager mode.
    267     add_to_graph &= not context.executing_eagerly()

/usr/local/lib/python3.11/dist-packages/tensorflow/python/eager/polymorphic_function/polymorphic_function.py in get_concrete_function(self, *args, **kwargs)
   1249   def get_concrete_function(self, *args, **kwargs):
   1250     # Implements PolymorphicFunction.get_concrete_function.
-> 1251     concrete = self._get_concrete_function_garbage_collected(*args, **kwargs)
   1252     concrete._garbage_collector.release()  # pylint: disable=protected-access
   1253     return concrete

/usr/local/lib/python3.11/dist-packages/tensorflow/python/eager/polymorphic_function/polymorphic_function.py in _get_concrete_function_garbage_collected(self, *args, **kwargs)
   1219       if self._variable_creation_config is None:
   1220         initializers = []
-> 1221         self._initialize(args, kwargs, add_initializers_to=initializers)
   1222         self._initialize_uninitialized_variables(initializers)
   1223 

/usr/local/lib/python3.11/dist-packages/tensorflow/python/eager/polymorphic_function/polymorphic_function.py in _initialize(self, args, kwds, add_initializers_to)
    694     )
    695     # Force the definition of the function for these arguments
--> 696     self._concrete_variable_creation_fn = tracing_compilation.trace_function(
    697         args, kwds, self._variable_creation_config
    698     )

/usr/local/lib/python3.11/dist-packages/tensorflow/python/eager/polymorphic_function/tracing_compilation.py in trace_function(args, kwargs, tracing_options)
    176       kwargs = {}
    177 
--> 178     concrete_function = _maybe_define_function(
    179         args, kwargs, tracing_options
    180     )

/usr/local/lib/python3.11/dist-packages/tensorflow/python/eager/polymorphic_function/tracing_compilation.py in _maybe_define_function(args, kwargs, tracing_options)
    281         else:
    282           target_func_type = lookup_func_type
--> 283         concrete_function = _create_concrete_function(
    284             target_func_type, lookup_func_context, func_graph, tracing_options
    285         )

/usr/local/lib/python3.11/dist-packages/tensorflow/python/eager/polymorphic_function/tracing_compilation.py in _create_concrete_function(function_type, type_context, func_graph, tracing_options)
    308       attributes_lib.DISABLE_ACD, False
    309   )
--> 310   traced_func_graph = func_graph_module.func_graph_from_py_func(
    311       tracing_options.name,
    312       tracing_options.python_function,

/usr/local/lib/python3.11/dist-packages/tensorflow/python/framework/func_graph.py in func_graph_from_py_func(name, python_func, args, kwargs, signature, func_graph, add_control_dependencies, arg_names, op_return_value, collections, capture_by_value, create_placeholders)
   1057 
   1058     _, original_func = tf_decorator.unwrap(python_func)
-> 1059     func_outputs = python_func(*func_args, **func_kwargs)
   1060 
   1061     # invariant: `func_outputs` contains only Tensors, CompositeTensors,

/usr/local/lib/python3.11/dist-packages/tensorflow/python/eager/polymorphic_function/polymorphic_function.py in wrapped_fn(*args, **kwds)
    597         # the function a weak reference to itself to avoid a reference cycle.
    598         with OptionalXlaContext(compile_with_xla):
--> 599           out = weak_wrapped_fn().__wrapped__(*args, **kwds)
    600         return out
    601 

/usr/local/lib/python3.11/dist-packages/tensorflow/python/data/ops/structured_function.py in wrapped_fn(*args)
    229       # Note: wrapper_helper will apply autograph based on context.
    230       def wrapped_fn(*args):  # pylint: disable=missing-docstring
--> 231         ret = wrapper_helper(*args)
    232         ret = structure.to_tensor_list(self._output_structure, ret)
    233         return [ops.convert_to_tensor(t) for t in ret]

/usr/local/lib/python3.11/dist-packages/tensorflow/python/data/ops/structured_function.py in wrapper_helper(*args)
    159       if not _should_unpack(nested_args):
    160         nested_args = (nested_args,)
--> 161       ret = autograph.tf_convert(self._func, ag_ctx)(*nested_args)
    162       ret = variable_utils.convert_variables_to_tensors(ret)
    163       if _should_pack(ret):

/usr/local/lib/python3.11/dist-packages/tensorflow/python/autograph/impl/api.py in wrapper(*args, **kwargs)
    691       except Exception as e:  # pylint:disable=broad-except
    692         if hasattr(e, 'ag_error_metadata'):
--> 693           raise e.ag_error_metadata.to_exception(e)
    694         else:
    695           raise

/usr/local/lib/python3.11/dist-packages/tensorflow/python/autograph/impl/api.py in wrapper(*args, **kwargs)
    688       try:
    689         with conversion_ctx:
--> 690           return converted_call(f, args, kwargs, options=options)
    691       except Exception as e:  # pylint:disable=broad-except
    692         if hasattr(e, 'ag_error_metadata'):

/usr/local/lib/python3.11/dist-packages/tensorflow/python/autograph/impl/api.py in converted_call(f, args, kwargs, caller_fn_scope, options)
    437     try:
    438       if kwargs is not None:
--> 439         result = converted_f(*effective_args, **kwargs)
    440       else:
    441         result = converted_f(*effective_args)

/tmp/__autograph_generated_filezrdxuo4t.py in tf__data_augment(image, label)
     11                 image = ag__.converted_call(ag__.ld(tf).image.random_flip_up_down, (ag__.ld(image),), None, fscope)
     12                 angle = ag__.converted_call(ag__.ld(tf).random.uniform, ([], -0.1, 0.1), None, fscope)
---> 13                 image = ag__.converted_call(ag__.ld(tfa).image.rotate, (ag__.ld(image), ag__.ld(angle)), None, fscope)
     14                 image = ag__.converted_call(ag__.ld(tf).image.random_saturation, (ag__.ld(image), 0.5, 1.5), None, fscope)
     15                 image = ag__.converted_call(ag__.ld(tf).image.random_brightness, (ag__.ld(image),), dict(max_delta=0.1), fscope)

NameError: in user code:

    File "/tmp/ipykernel_55/316458610.py", line 49, in data_augment  *
        image = tfa.image.rotate(image, angle)

    NameError: name 'tfa' is not defined


## === cell 17
model.load_weights("best_model.weights.h5")




## --- ERROR in cell 17, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_55/1273657460.py in <cell line: 0>()
----> 1 model.load_weights("best_model.weights.h5")
      2 
      3 

/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py in error_handler(*args, **kwargs)
    120             # To get the full stack trace, call:
    121             # `keras.config.disable_traceback_filtering()`
--> 122             raise e.with_traceback(filtered_tb) from None
    123         finally:
    124             del filtered_tb

/usr/local/lib/python3.11/dist-packages/h5py/_hl/files.py in __init__(self, name, mode, driver, libver, userblock_size, swmr, rdcc_nslots, rdcc_nbytes, rdcc_w0, track_order, fs_strategy, fs_persist, fs_threshold, fs_page_size, page_buf_size, min_meta_keep, min_raw_keep, locking, alignment_threshold, alignment_interval, meta_block_size, **kwds)
    562                                  fs_persist=fs_persist, fs_threshold=fs_threshold,
    563                                  fs_page_size=fs_page_size)
--> 564                 fid = make_fid(name, mode, userblock_size, fapl, fcpl, swmr=swmr)
    565 
    566             if isinstance(libver, tuple):

/usr/local/lib/python3.11/dist-packages/h5py/_hl/files.py in make_fid(name, mode, userblock_size, fapl, fcpl, swmr)
    236         if swmr and swmr_support:
    237             flags |= h5f.ACC_SWMR_READ
--> 238         fid = h5f.open(name, flags, fapl=fapl)
    239     elif mode == 'r+':
    240         fid = h5f.open(name, h5f.ACC_RDWR, fapl=fapl)

h5py/_objects.pyx in h5py._objects.with_phil.wrapper()

h5py/_objects.pyx in h5py._objects.with_phil.wrapper()

h5py/h5f.pyx in h5py.h5f.open()

FileNotFoundError: [Errno 2] Unable to synchronously open file (unable to open file: name = 'best_model.weights.h5', errno = 2, error message = 'No such file or directory', flags = 0, o_flags = 0)

## === cell 18
plt.plot(history.history["accuracy"])
plt.plot(history.history["val_accuracy"])
plt.title("Model accuracy")
plt.ylabel("Accuracy")
plt.xlabel("Epoch")
plt.legend(["train", "validation"], loc="upper left")
plt.show()




## --- ERROR in cell 18, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/300272665.py in <cell line: 0>()
----> 1 plt.plot(history.history["accuracy"])
      2 plt.plot(history.history["val_accuracy"])
      3 plt.title("Model accuracy")
      4 plt.ylabel("Accuracy")
      5 plt.xlabel("Epoch")

NameError: name 'history' is not defined

## === cell 19
plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])
plt.title("Model loss")
plt.ylabel("Loss")
plt.xlabel("Epoch")
plt.legend(["train", "validation"], loc="upper left")
plt.show()




## --- ERROR in cell 19, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1032109659.py in <cell line: 0>()
----> 1 plt.plot(history.history["loss"])
      2 plt.plot(history.history["val_loss"])
      3 plt.title("Model loss")
      4 plt.ylabel("Loss")
      5 plt.xlabel("Epoch")

NameError: name 'history' is not defined

## === cell 20
num_test_images = count_data_items(test_filenames)
print(f"Number of test images: {num_test_images}")




## === cell 21
test_dataset = get_test_dataset(ordered=True)




## === cell 22
print("Computing predictions...")
test_images_ds = test_dataset.map(lambda image, idnum: image)
probabilities = model.predict(test_images_ds, verbose=1).flatten()
print("Predictions computed.")

print("Collecting test image IDs...")
test_ids_ds = test_dataset.map(lambda image, idnum: idnum).unbatch()
test_ids = [id_bytes.numpy().decode("utf-8") for id_bytes in test_ids_ds]

assert len(test_ids) == len(
    probabilities
), f"Length mismatch: {len(test_ids)} IDs vs {len(probabilities)} predictions."

print("Generating submission.csv file with pandas...")
submission_df = pd.DataFrame({"image_name": test_ids, "target": probabilities})
submission_path = "submission.csv"
submission_df.to_csv(submission_path, index=False)
print(f"{submission_path} written.")
