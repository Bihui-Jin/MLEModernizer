# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


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
keras==3.8.0
keras-core==0.1.7
keras-cv==0.9.0
keras-hub==0.18.1
keras-nlp==0.18.1
keras-tuner==1.4.7
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
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
tf_keras==2.18.0

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

# 5. Code solution

## === cell 0
import os

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "2")

try:
    from google.protobuf import message_factory as _message_factory

    def _ensure_getprototype():
        MF = _message_factory.MessageFactory
        if not hasattr(MF, "GetPrototype") and hasattr(MF, "GetMessageClass"):

            def _GetPrototype(self, descriptor):
                return self.GetMessageClass(descriptor)

            MF.GetPrototype = _GetPrototype

        try:
            inst = _message_factory.GetMessageFactory()
            if not hasattr(inst, "GetPrototype") and hasattr(inst, "GetMessageClass"):
                inst.GetPrototype = inst.GetMessageClass
        except Exception:
            pass

    _ensure_getprototype()
except Exception:
    pass

import numpy as np
import pandas as pd
import re

import tensorflow as tf
import tensorflow.keras.layers as L

print("TensorFlow:", tf.__version__)
print("Num GPUs Available:", len(tf.config.list_physical_devices("GPU")))

SEED = 42
tf.keras.utils.set_random_seed(SEED)
try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

tf.config.run_functions_eagerly(False)

try:
    tf.config.optimizer.set_jit(True)
except Exception:
    pass

try:
    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
except Exception:
    pass



## === cell 1
try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
    print("Running on TPU ", tpu.master())
except Exception:
    tpu = None

if tpu:
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.experimental.TPUStrategy(tpu)
else:
    strategy = tf.distribute.get_strategy()

print("REPLICAS: ", strategy.num_replicas_in_sync)



## === cell 2
AUTO = tf.data.experimental.AUTOTUNE

BASE_PATH = "/kaggle/input/siim-isic-melanoma-classification"
TFRECORD_PATH = os.path.join(BASE_PATH, "tfrecords")

EPOCHS = 10
BATCH_SIZE = 8 * strategy.num_replicas_in_sync
IMAGE_SIZE = [1024, 1024]

print("BASE_PATH:", BASE_PATH)
print("TFRECORD_PATH exists:", os.path.exists(TFRECORD_PATH))




## === cell 3
def append_path(pre, base_path=BASE_PATH):
    return lambda file: os.path.join(base_path, pre, file)




## === cell 4
sub = pd.read_csv(os.path.join(BASE_PATH, "sample_submission.csv"))
train = pd.read_csv(os.path.join(BASE_PATH, "train.csv"))

print("train shape:", train.shape)
print("sub shape:", sub.shape)
train.head()



## === cell 5
print("Target distribution:\n", train["target"].value_counts(dropna=False))



## === cell 6
TRAINING_FILENAMES = sorted(
    tf.io.gfile.glob(os.path.join(TFRECORD_PATH, "train*.tfrec"))
)
TEST_FILENAMES = sorted(tf.io.gfile.glob(os.path.join(TFRECORD_PATH, "test*.tfrec")))
CLASSES = [0, 1]

print("Number of training tfrecords:", len(TRAINING_FILENAMES))
print("Number of test tfrecords:", len(TEST_FILENAMES))
print("First training tfrec:", TRAINING_FILENAMES[0] if TRAINING_FILENAMES else None)

rng = np.random.RandomState(SEED)
TRAINING_FILENAMES = list(TRAINING_FILENAMES)
rng.shuffle(TRAINING_FILENAMES)

VALIDATION_FRACTION = 0.1
num_valid = max(1, int(len(TRAINING_FILENAMES) * VALIDATION_FRACTION))
VALIDATION_FILENAMES = TRAINING_FILENAMES[:num_valid]
TRAINING_FILENAMES = TRAINING_FILENAMES[num_valid:]

print(
    "Training shards:",
    len(TRAINING_FILENAMES),
    "Validation shards:",
    len(VALIDATION_FILENAMES),
)



## === cell 7
TRAINING_FILENAMES[:3], VALIDATION_FILENAMES[:3], TEST_FILENAMES[:3]




## === cell 8
@tf.function
def decode_image(image_data):
    image = tf.image.decode_jpeg(image_data, channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)  # [0,1]
    image = tf.image.resize(image, IMAGE_SIZE, method="bilinear", antialias=True)
    image.set_shape([IMAGE_SIZE[0], IMAGE_SIZE[1], 3])
    return image


@tf.function
def read_labeled_tfrecord(example):
    LABELED_TFREC_FORMAT = {
        "image": tf.io.FixedLenFeature([], tf.string),
        "target": tf.io.FixedLenFeature([], tf.int64),
    }
    example = tf.io.parse_single_example(example, LABELED_TFREC_FORMAT)
    image = decode_image(example["image"])
    label = tf.cast(example["target"], tf.int32)
    return image, label


@tf.function
def read_unlabeled_tfrecord(example):
    UNLABELED_TFREC_FORMAT = {
        "image": tf.io.FixedLenFeature([], tf.string),
        "image_name": tf.io.FixedLenFeature([], tf.string),
    }
    example = tf.io.parse_single_example(example, UNLABELED_TFREC_FORMAT)
    image = decode_image(example["image"])
    id_num = example["image_name"]
    return image, id_num


@tf.function
def data_augment(image, label):
    image = tf.image.random_flip_left_right(image)
    return image, label


def load_dataset(filenames, labeled=True, ordered=False):
    options = tf.data.Options()
    if not ordered:
        options.experimental_deterministic = False
    options.experimental_optimization.apply_default_optimizations = True
    options.experimental_optimization.map_parallelization = True
    options.experimental_optimization.parallel_batch = True

    if isinstance(filenames, (list, tuple)):
        files_ds = tf.data.Dataset.from_tensor_slices(filenames)
        cycle_len = min(16, len(filenames)) if len(filenames) > 0 else 1
    else:
        files_ds = filenames
        cycle_len = 16

    dataset = files_ds.interleave(
        lambda fn: tf.data.TFRecordDataset(fn),
        cycle_length=cycle_len,
        num_parallel_calls=AUTO,
        deterministic=ordered,
    )
    dataset = dataset.with_options(options)
    dataset = dataset.map(
        read_labeled_tfrecord if labeled else read_unlabeled_tfrecord,
        num_parallel_calls=AUTO,
        deterministic=ordered,
    )
    return dataset


def get_training_dataset():
    dataset = load_dataset(TRAINING_FILENAMES, labeled=True, ordered=False)
    dataset = dataset.map(data_augment, num_parallel_calls=AUTO, deterministic=False)
    dataset = dataset.repeat()
    dataset = dataset.shuffle(2048, seed=SEED, reshuffle_each_iteration=True)
    dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)
    try:
        dataset = dataset.apply(tf.data.experimental.prefetch_to_device("/GPU:0"))
    except Exception:
        dataset = dataset.prefetch(AUTO)
    return dataset


def get_validation_dataset(ordered=False):
    dataset = load_dataset(VALIDATION_FILENAMES, labeled=True, ordered=ordered)
    dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)
    dataset = dataset.prefetch(AUTO)
    return dataset


def get_test_dataset(ordered=False):
    dataset = load_dataset(TEST_FILENAMES, labeled=False, ordered=ordered)
    dataset = dataset.batch(BATCH_SIZE, drop_remainder=False)
    dataset = dataset.prefetch(AUTO)
    return dataset


def count_data_items(filenames):
    n = [int(re.compile(r"-([0-9]*)\.").search(fn).group(1)) for fn in filenames]
    return int(np.sum(n))


NUM_TRAINING_IMAGES = count_data_items(TRAINING_FILENAMES) if TRAINING_FILENAMES else 0
NUM_VALID_IMAGES = count_data_items(VALIDATION_FILENAMES) if VALIDATION_FILENAMES else 0
NUM_TEST_IMAGES = count_data_items(TEST_FILENAMES) if TEST_FILENAMES else 0

STEPS_PER_EPOCH = max(1, NUM_TRAINING_IMAGES // BATCH_SIZE)
VALIDATION_STEPS = max(1, NUM_VALID_IMAGES // BATCH_SIZE) if NUM_VALID_IMAGES else 1

print(
    f"Dataset: {NUM_TRAINING_IMAGES} training images, {NUM_VALID_IMAGES} validation images, {NUM_TEST_IMAGES} test images"
)
print(
    "BATCH_SIZE:",
    BATCH_SIZE,
    "STEPS_PER_EPOCH:",
    STEPS_PER_EPOCH,
    "VALIDATION_STEPS:",
    VALIDATION_STEPS,
)




## === cell 9
def build_lrfn(
    lr_start=0.00001,
    lr_max=0.0001,
    lr_min=0.000001,
    lr_rampup_epochs=20,
    lr_sustain_epochs=0,
    lr_exp_decay=0.8,
):
    lr_max = lr_max * strategy.num_replicas_in_sync

    def lrfn(epoch):
        if epoch < lr_rampup_epochs:
            lr = (lr_max - lr_start) / lr_rampup_epochs * epoch + lr_start
        elif epoch < lr_rampup_epochs + lr_sustain_epochs:
            lr = lr_max
        else:
            lr = (lr_max - lr_min) * lr_exp_decay ** (
                epoch - lr_rampup_epochs - lr_sustain_epochs
            ) + lr_min
        return lr

    return lrfn




## === cell 10
with strategy.scope():
    model = tf.keras.Sequential(
        [
            tf.keras.applications.ResNet50(
                weights="imagenet", input_shape=[*IMAGE_SIZE, 3], include_top=False
            ),
            L.GlobalAveragePooling2D(),
            L.Dense(1024, activation="relu"),
            L.Dropout(0.3),
            L.Dense(512, activation="relu"),
            L.Dropout(0.2),
            L.Dense(256, activation="relu"),
            L.Dropout(0.2),
            L.Dense(128, activation="relu"),
            L.Dropout(0.1),
            L.Dense(1, activation="sigmoid"),
        ]
    )

model.compile(
    loss=tf.keras.losses.BinaryCrossentropy(label_smoothing=0.1),
    metrics=[tf.keras.metrics.AUC(name="auc")],
    optimizer="adam",
)

model.summary()



## === cell 11
lrfn = build_lrfn()
lr_schedule = tf.keras.callbacks.LearningRateScheduler(lrfn, verbose=1)

STEPS_PER_EPOCH = max(1, NUM_TRAINING_IMAGES // BATCH_SIZE)
VALIDATION_STEPS = max(1, NUM_VALID_IMAGES // BATCH_SIZE) if NUM_VALID_IMAGES else 1



## === cell 12
checkpoint_path = "/kaggle/working/model_best.keras"
checkpoint = tf.keras.callbacks.ModelCheckpoint(
    checkpoint_path, monitor="val_loss", mode="min", save_best_only=True, verbose=1
)

callbacks = [checkpoint, lr_schedule]

print("Checkpoint path:", checkpoint_path)



## === cell 13
history = model.fit(
    get_training_dataset(),
    validation_data=get_validation_dataset(ordered=True),
    epochs=EPOCHS,
    steps_per_epoch=STEPS_PER_EPOCH,
    validation_steps=VALIDATION_STEPS,
    callbacks=callbacks,
    verbose=1,
)



## === cell 14
best_model = model
if os.path.exists(checkpoint_path):
    try:
        best_model = tf.keras.models.load_model(checkpoint_path, compile=False)
        print("Loaded best model from:", checkpoint_path)
    except Exception as e:
        print(
            "Warning: failed to load saved model; using current model weights. Error:",
            repr(e),
        )

best_model.save("/kaggle/working/resnet.keras")



## === cell 15
test_ds = get_test_dataset(ordered=True)

print("Computing predictions + collecting ids (single pass)...")
test_ids_list = []
probs_list = []
for batch_images, batch_ids in test_ds:
    batch_probs = best_model.predict_on_batch(batch_images)
    test_ids_list.append(batch_ids)
    probs_list.append(tf.reshape(batch_probs, [-1]))

test_ids_raw = tf.concat(test_ids_list, axis=0).numpy()
test_ids = np.array(
    [
        x.decode("utf-8") if isinstance(x, (bytes, bytearray)) else str(x)
        for x in test_ids_raw
    ],
    dtype="U",
)
probabilities = tf.concat(probs_list, axis=0).numpy().astype(np.float32)

print("Predictions shape:", probabilities.shape, "Test ids count:", len(test_ids))



## === cell 16
pred_df = pd.DataFrame(
    {"image_name": test_ids.astype(str), "target": probabilities.astype(np.float32)}
)
pred_df.head()



## === cell 17
subm = sub[["image_name"]].merge(pred_df, on="image_name", how="left")
subm["target"] = subm["target"].astype("float32").fillna(0.5).clip(0.0, 1.0)

submission_path = "/kaggle/working/submission.csv"
subm.to_csv(submission_path, index=False)

print("Wrote:", submission_path)
print(subm.head())
print("Submission shape:", subm.shape, "columns:", list(subm.columns))
assert list(subm.columns) == ["image_name", "target"]
assert len(subm) == len(sub)
assert subm["target"].notna().all()
