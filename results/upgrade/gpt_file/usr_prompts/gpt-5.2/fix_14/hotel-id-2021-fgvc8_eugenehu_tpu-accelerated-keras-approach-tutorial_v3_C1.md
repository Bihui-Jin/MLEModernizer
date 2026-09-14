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
Identify hotels from images.

## Metric
Mean Average Precision @ 5 (MAP@5)

## Submission Format
For each image in the test set, you must predict a space-delimited list of hotel IDs that could match that image. The first ID should be the most relevant one and the last the least relevant one. The file should contain a header and have the following format:

```
image,hotel_id
99e91ad5f2870678.jpg,36363 53586 18807 64314 60181
b5cc62ab665591a9.jpg,36363 53586 18807 64314 60181
d5664a972d5a644b.jpg,36363 53586 18807 64314 60181
```

## Dataset
**train.csv** - The training set metadata.

- `image` - The image ID.

- `chain` - An ID code for the hotel chain. A `chain` of zero (0) indicates that the hotel is either not part of a chain or the chain is not known. This field is not available for the test set. The number of hotels per chain varies widely.

- `hotel_id` - The hotel ID. The target class.

- `timestamp` - When the image was taken. Provided for the training set only.

**sample_submission.csv** - A sample submission file in the correct format.

- `image` The image ID

- `hotel_id` The hotel ID. The target class.

**train_images** - The training set contains 97000+ images from around 7700 hotels from across the globe. All of the images for each hotel chain are in a dedicated subfolder for that chain.

**test_images** - The test set images. This competition has a hidden test set: only three images are provided here as samples while the remaining 13,000 images will be available to your notebook once it is submitted.

# 2. Python version

3.9

# 3. Installed packages

geopandas==0.14.4
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

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (120 lines)
            sample_submission.csv (9757 lines)
            sample_submission.csv.zip (106.8 kB)
            test.zip (160 Bytes)
            test_images.zip (2.6 GB)
            train.csv (87799 lines)
            train.csv.zip (1.9 MB)
            train.zip (162 Bytes)
            train_images.zip (23.5 GB)
            hotel-id-2021-fgvc8/
                description.md (120 lines)
                sample_submission.csv (9757 lines)
                ... and 7 other files
                hotel-id-2021-fgvc8/
                test/
                    test/
                test_images/
                    ccc436fc41bf402f.jpg (72.8 kB)
                    fb9d48b39c614c32.jpg (91.3 kB)
                    ... and 9754 other files
                    test_images/
                train/
                    train/
                train_images/
                    0/
                        b5bd0a0a2de05bb5.jpg (73.8 kB)
                        c242bcf0719f9d61.jpg (71.9 kB)
                        ... and 18211 other files
                    1/
                        a7ad6a44813b77c8.jpg (81.1 kB)
                        9b89db65b496490d.jpg (630.0 kB)
                        ... and 1116 other files
                    ... and 87 other folders
            test/
                test/
            test_images/
                ccc436fc41bf402f.jpg (72.8 kB)
                fb9d48b39c614c32.jpg (91.3 kB)
                ... and 9754 other files
                test_images/
            train/
                train/
            train_images/
                0/
                    b5bd0a0a2de05bb5.jpg (73.8 kB)
                    c242bcf0719f9d61.jpg (71.9 kB)
                    ... and 18211 other files
                1/
                    a7ad6a44813b77c8.jpg (81.1 kB)
                    9b89db65b496490d.jpg (630.0 kB)
                    ... and 1116 other files
                ... and 87 other folders
        input/
            description.md (120 lines)
            sample_submission.csv (9757 lines)
            sample_submission.csv.zip (106.8 kB)
            test.zip (160 Bytes)
            test_images.zip (2.6 GB)
            train.csv (87799 lines)
            train.csv.zip (1.9 MB)
            train.zip (162 Bytes)
            train_images.zip (23.5 GB)
            hotel-id-2021-fgvc8/
                description.md (120 lines)
                sample_submission.csv (9757 lines)
                ... and 7 other files
                hotel-id-2021-fgvc8/
                test/
                    test/
                test_images/
                    ccc436fc41bf402f.jpg (72.8 kB)
                    fb9d48b39c614c32.jpg (91.3 kB)
                    ... and 9754 other files
                    test_images/
                train/
                    train/
                train_images/
                    0/
                        b5bd0a0a2de05bb5.jpg (73.8 kB)
                        c242bcf0719f9d61.jpg (71.9 kB)
                        ... and 18211 other files
                    1/
                        a7ad6a44813b77c8.jpg (81.1 kB)
                        9b89db65b496490d.jpg (630.0 kB)
                        ... and 1116 other files
                    ... and 87 other folders
            test/
                test/
                    test/
            test_images/
                ccc436fc41bf402f.jpg (72.8 kB)
                fb9d48b39c614c32.jpg (91.3 kB)
                ... and 9754 other files
                test_images/
                    ccc436fc41bf402f.jpg (72.8 kB)
                    fb9d48b39c614c32.jpg (91.3 kB)
                    ... and 9754 other files
                    test_images/
            train/
                train/
                    train/
            train_images/
                0/
                    b5bd0a0a2de05bb5.jpg (73.8 kB)
                    c242bcf0719f9d61.jpg (71.9 kB)
                    ... and 18211 other files
                1/
                    a7ad6a44813b77c8.jpg (81.1 kB)
                    9b89db65b496490d.jpg (630.0 kB)
                    ... and 1116 other files
                ... and 87 other folders
        working/
            hotel-id-2021-fgvc8/
                description.md (120 lines)
                sample_submission.csv (9757 lines)
                ... and 7 other files
                hotel-id-2021-fgvc8/
                test/
                    test/
                test_images/
                    ccc436fc41bf402f.jpg (72.8 kB)
                    fb9d48b39c614c32.jpg (91.3 kB)
                    ... and 9754 other files
                    test_images/
                train/
                    train/
                train_images/
                    0/
                        b5bd0a0a2de05bb5.jpg (73.8 kB)
                        c242bcf0719f9d61.jpg (71.9 kB)
                        ... and 18211 other files
                    1/
                        a7ad6a44813b77c8.jpg (81.1 kB)
                        9b89db65b496490d.jpg (630.0 kB)
                        ... and 1116 other files
                    ... and 87 other folders
```

-> data/hotel-id-2021-fgvc8/sample_submission.csv has 9756 rows and 2 columns.
The columns are: image, hotel_id

-> data/hotel-id-2021-fgvc8/train.csv has 87798 rows and 4 columns.
The columns are: image, chain, hotel_id, timestamp

-> data/sample_submission.csv has 9756 rows and 2 columns.
The columns are: image, hotel_id

-> data/train.csv has 87798 rows and 4 columns.
The columns are: image, chain, hotel_id, timestamp

-> input/hotel-id-2021-fgvc8/sample_submission.csv has 9756 rows and 2 columns.
The columns are: image, hotel_id

-> input/hotel-id-2021-fgvc8/train.csv has 87798 rows and 4 columns.
The columns are: image, chain, hotel_id, timestamp

-> (stopped after 10 files for performance)

# 5. Target score

0.0463500884061631

# 6. Current score

0.00209

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.00186) has done: 'Main bottlenecks are (1) the per-path existence check using `tf.map_fn(... .numpy() ...)` which forces slow Python execution and (2) caching full decoded/resized train/val/test image tensors to disk, which adds huge IO/serialization overhead and can easily dominate runtime. I replace the existence check with a fast vectorized `tf.io.gfile.exists` loop in Python (no TF graph/`map_fn`) and keep only lightweight caching (file-path lists) while relying on parallel decode/resize + prefetch for throughput. I also make the input pipeline more efficient but equivalent by using `num_parallel_calls=AUTO`, `deterministic=True`, adding `prefetch(AUTO)` before expensive stages where appropriate, and avoiding redundant dataset materialization; model/training/prediction logic and semantics remain unchanged.'
- What this solution (achieved 0.00245) has done: 'I fix the crash at import time caused by an incompatibility between TensorFlow 2.18 and protobuf 6.x by forcing protobuf to use the pure-Python implementation before importing TensorFlow. Then I keep your training/inference logic the same, but make one minimal, score-improving change: switch ResNet50 to ImageNet pretrained weights (still frozen) so predictions aren’t effectively random, which should move MAP@5 substantially toward your target. I also make the train/val split happen after dropping missing image paths to avoid mismatched indexing, and ensure the submission is aligned to `sample_submission.csv` and always written as `submission.csv`. All other architecture, loss, and loops remain unchanged.'
- What this solution (achieved 0.00245) has done: 'We fix the import-time crash by setting both protobuf environment variables early enough and by making the TensorFlow import robust to the protobuf 6.x / TF 2.18 incompatibility. Then we fix a path bug where `DIR="../input/hotel-id-2021-fgvc8"` may not exist in this environment by auto-selecting the first existing dataset root from the provided paths (without changing the rest of the pipeline). Finally, to improve MAP@5 toward your target with minimal semantic change, we ensure test images are predicted in the exact `sample_submission.csv` order (instead of filesystem glob order), preventing misalignment that can severely depress the score.'
- What this solution (achieved 0.00245) has done: 'You’re hitting a TensorFlow import crash caused by the protobuf 6.x runtime API change (`MessageFactory.GetPrototype`), so the pipeline never reaches training/inference or writes `submission.csv`. I fix this by forcing TensorFlow to use the C++ protobuf implementation (instead of the pure-Python one) and by importing TensorFlow first (before importing `tensorflow_hub/tfds` etc.), which resolves this specific error in TF 2.18 + protobuf 6 in Kaggle environments. After that, I keep your model/training/prediction logic the same, but ensure the test image paths are constructed from `sample_submission.csv` order (already done) and that any missing test files still produce valid 5-id strings so a valid `submission.csv` is always written. These changes are execution-unblocking and score-positive (the model can actually run), without altering your core architecture/training semantics.'
- What this solution (achieved 0.00245) has done: 'I fix the TensorFlow import crash by setting protobuf environment variables before any TensorFlow-related import so TF 2.18 can work with protobuf 6.x in this environment. Then I keep your model/training/prediction logic unchanged, but make the pipeline robust by ensuring the dataset path resolution is stable and the test predictions remain aligned to `sample_submission.csv` order (which strongly affects MAP@5). Finally, I keep the submission generation identical but ensure it always writes a valid `submission.csv` with the required columns even if some test files are missing.'
- What this solution (achieved 0.00245) has done: 'I fix the root cause preventing the whole notebook from running: TensorFlow can’t import because protobuf’s C-extension `_message` isn’t available, so we must force protobuf to use the pure-Python implementation before importing TF. Once TF imports, the downstream NameErrors (tf/strategy/preprocessing undefined and missing `image_path`) resolve automatically because earlier cells execute. I also make the submission generation robust by collecting kept indices from the batched test dataset (instead of iterating the unbatched dataset, which breaks after batching) and ensure the final `submission.csv` always has exactly the sample submission order and required columns.'
- What this solution (achieved 0.00209) has done: 'I fix the TensorFlow import crash (`MessageFactory.GetPrototype`) by setting protobuf env vars correctly before any TensorFlow-related import and by importing `google.protobuf` first so TF uses a compatible implementation in this runtime. Then I keep your model/training/prediction logic unchanged, but add a safe fallback: if TF still can’t import, we still generate a valid `submission.csv` using the most frequent hotel_ids from train (score be low, but you always get a valid file instead of a crash). This should unblock end-to-end execution and, when TF imports successfully, restore the intended pretrained ResNet50 pipeline which should move MAP@5 upward toward your target. All paths, dataset usage, and submission formatting remain the same.'

# 9. Code solution

## === cell 0
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "cpp")
os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", None)

import random
import numpy as np
import pandas as pd

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

try:
    import google.protobuf  # noqa: F401
except Exception as _e:
    print("Warning: google.protobuf import issue:", repr(_e))

TF_AVAILABLE = True
TF_IMPORT_ERROR = None
try:
    import tensorflow as tf

    tf.random.set_seed(SEED)
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception as e:
        print("Determinism setting skipped:", repr(e))
    print("TF:", tf.__version__)
except Exception as e:
    TF_AVAILABLE = False
    TF_IMPORT_ERROR = e
    print("TensorFlow import failed:", repr(e))

from sklearn import preprocessing




## === cell 1
def intialize_accel(hardware):
    """
    input:
    str: GPU or TPU for hardware accelerator

    output:
    strategy -- used later for model definition and fitting
    """
    if not TF_AVAILABLE:
        return None

    if hardware == "TPU":
        try:
            resolver = tf.distribute.cluster_resolver.TPUClusterResolver()
            tf.config.experimental_connect_to_cluster(resolver)
            tf.tpu.experimental.initialize_tpu_system(resolver)
            strategy = tf.distribute.TPUStrategy(resolver)
            print("TPU Initialized")
            print("TPU Units:", strategy.num_replicas_in_sync)
            return strategy
        except Exception as e:
            print("TPU Initialization Failed:", repr(e))
            print("Falling back to default strategy.")
            return tf.distribute.get_strategy()

    elif hardware == "GPU":
        print("Num GPUs Available: ", len(tf.config.list_physical_devices("GPU")))
        try:
            strategy = tf.distribute.MirroredStrategy()
        except Exception as e:
            print("MirroredStrategy init failed:", repr(e))
            strategy = tf.distribute.get_strategy()
        return strategy

    print("Unknown hardware option; using default strategy.")
    return tf.distribute.get_strategy()


strategy = intialize_accel("TPU")
AUTO = tf.data.AUTOTUNE if TF_AVAILABLE else None



## === cell 2
CANDIDATE_DIRS = [
    "../input/hotel-id-2021-fgvc8",
    "/kaggle/input/hotel-id-2021-fgvc8",
    "/kaggle/data/hotel-id-2021-fgvcvc8",
    "/kaggle/data/hotel-id-2021-fgvc8",
    "../input",
    "/kaggle/input",
    "/kaggle/data",
]
DIR = None
for d in CANDIDATE_DIRS:
    if os.path.exists(d) and os.path.isdir(d):
        if os.path.basename(d) == "hotel-id-2021-fgvc8":
            DIR = d
            break
        if os.path.exists(os.path.join(d, "hotel-id-2021-fgvc8")):
            DIR = os.path.join(d, "hotel-id-2021-fgvc8")
            break

if DIR is None:
    raise FileNotFoundError(
        "Could not locate dataset directory. Tried: " + ", ".join(CANDIDATE_DIRS)
    )

Train_PATH = os.path.join(DIR, "train_images")
Test_PATH = os.path.join(DIR, "test_images")

train_csv_path = os.path.join(DIR, "train.csv")
sample_sub_path = os.path.join(DIR, "sample_submission.csv")

train_df = pd.read_csv(train_csv_path)
train_df = train_df.drop_duplicates(subset=["image"]).reset_index(drop=True)

print("Using DIR:", DIR)
print("Number of unique hotel chains: ", train_df.chain.nunique())
print("Number of unique hotels: ", train_df.hotel_id.nunique())
print("Number of Training Samples: ", train_df.shape[0])
print(train_df.head())



## === cell 3
Classes = train_df.hotel_id.nunique()
Channels = 3
size = (200, 200)



## === cell 4
le = preprocessing.LabelEncoder()
train_df["label"] = le.fit_transform(train_df["hotel_id"])

train_df["image_path"] = (
    Train_PATH
    + os.sep
    + train_df["chain"].astype(str).to_numpy()
    + os.sep
    + train_df["image"].to_numpy()
)

if TF_AVAILABLE:
    paths_list = train_df["image_path"].tolist()
    exists_mask_np = np.fromiter(
        (tf.io.gfile.exists(p) for p in paths_list), dtype=bool, count=len(paths_list)
    )
    missing = int((~exists_mask_np).sum())
    if missing:
        print(f"Warning: {missing} train image paths missing; dropping them.")
    train_df = train_df.loc[exists_mask_np].reset_index(drop=True)

print(train_df[["hotel_id", "label", "image_path"]].head())

Split = int(0.9 * train_df.shape[0])




## === cell 5
def image_proces(path, labels):
    """
    Reads, decodes, resizes and scales image to [0,1].

    We already filtered missing TRAIN paths in Python; for TEST we use ignore_errors().
    """
    data = tf.io.read_file(path)
    img = tf.io.decode_jpeg(data, channels=3, dct_method="INTEGER_FAST")
    img = tf.image.resize(img, size)
    img = tf.cast(img, tf.float32) / 255.0
    return img, labels


if TF_AVAILABLE:
    _DS_OPTIONS = tf.data.Options()
    _DS_OPTIONS.deterministic = True
    try:
        _DS_OPTIONS.threading.private_threadpool_size = 16
    except Exception:
        pass


def import_image(paths, labels, cache=False, cache_path=None, ignore_errors=False):
    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))
    dataset = dataset.with_options(_DS_OPTIONS)
    dataset = dataset.map(image_proces, num_parallel_calls=AUTO, deterministic=True)
    if ignore_errors:
        dataset = dataset.apply(tf.data.experimental.ignore_errors())
    if cache:
        dataset = (
            dataset.cache(cache_path) if cache_path is not None else dataset.cache()
        )
    return dataset


def data_augment(image, labels):
    image = tf.image.random_brightness(image, max_delta=0.1)
    image = tf.image.random_contrast(image, lower=0.8, upper=1.2)
    return image, labels


sample = pd.read_csv(sample_sub_path)
sample_images = sample["image"].tolist()
Paths_Test = [os.path.join(Test_PATH, img) for img in sample_images]

if TF_AVAILABLE:
    exists_test = np.fromiter(
        (tf.io.gfile.exists(p) for p in Paths_Test), dtype=bool, count=len(Paths_Test)
    )
    if not exists_test.all():
        missing_test = int((~exists_test).sum())
        print(
            f"Warning: {missing_test} test image paths missing; they will be skipped by ignore_errors() "
            f"and filled with fallback ids in submission merge."
        )
    print("Test images (from sample_submission):", len(Paths_Test))

    dataset_Test = import_image(
        Paths_Test,
        np.arange(len(Paths_Test), dtype=np.int32),
        cache=False,
        ignore_errors=True,
    )



## === cell 6
if TF_AVAILABLE:
    print(
        "Number of Test Samples (after ignore_errors):",
        int(dataset_Test.cardinality().numpy()),
    )
else:
    print("TensorFlow unavailable; will run fallback submission generation.")



## === cell 7
if TF_AVAILABLE:
    paths = train_df["image_path"].values
    labels = train_df["label"].values.astype(np.int32)

    train_paths, val_paths = paths[:Split], paths[Split:]
    train_labels, val_labels = labels[:Split], labels[Split:]

    BATCH_SIZE = 32

    train_dataset_base = import_image(
        train_paths,
        train_labels,
        cache=False,
        ignore_errors=False,
    )
    train_dataset = train_dataset_base.map(
        data_augment, num_parallel_calls=AUTO, deterministic=True
    )
    train_dataset = train_dataset.shuffle(
        2048, seed=SEED, reshuffle_each_iteration=True
    )
    train_dataset = train_dataset.batch(BATCH_SIZE, drop_remainder=False).prefetch(AUTO)

    val_dataset = import_image(
        val_paths,
        val_labels,
        cache=False,
        ignore_errors=False,
    )
    val_dataset = val_dataset.batch(BATCH_SIZE, drop_remainder=False).prefetch(AUTO)

    print("Train batches:", tf.data.experimental.cardinality(train_dataset).numpy())
    print("Val batches:", tf.data.experimental.cardinality(val_dataset).numpy())




## === cell 8
def create_model(Base, input_shape):
    inputs = tf.keras.Input(shape=tuple(input_shape))

    norm = tf.keras.layers.Normalization()
    x = norm(inputs)

    x = Base(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(50, activation="relu", dtype="float32")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    outputs = tf.keras.layers.Dense(Classes, activation="softmax", dtype="float32")(x)

    model = tf.keras.Model(inputs, outputs)
    model._norm_layer = norm
    return model


def compile_model(model, lr):
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
    metrics = [tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")]

    model.compile(
        optimizer=optimizer, loss=loss, metrics=metrics, steps_per_execution=8
    )
    return model




## === cell 9
if TF_AVAILABLE:
    EPOCHS = 1
    VERBOSE = 1

    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_accuracy",
        factor=0.1,
        patience=3,
        mode="max",
        min_delta=0.0001,
        verbose=1,
    )

    checkpoint_filepath = "./best_model.weights.h5"
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=True,
        monitor="val_accuracy",
        mode="max",
        save_best_only=True,
        verbose=1,
    )

    callback = tf.keras.callbacks.EarlyStopping(
        monitor="val_accuracy", patience=10, mode="max", min_delta=0.0001, verbose=1
    )



## === cell 10
if TF_AVAILABLE:
    input_shape = [200, 200, Channels]

    with strategy.scope():
        Base = tf.keras.applications.ResNet50(
            weights="imagenet", include_top=False, input_shape=tuple(input_shape)
        )
        Base.trainable = False
        model = create_model(Base, input_shape)
        model = compile_model(model, lr=0.001)

    try:
        adapt_ds = (
            train_dataset_base.map(lambda x, y: x, num_parallel_calls=AUTO)
            .take(512)
            .prefetch(AUTO)
        )
        model._norm_layer.adapt(adapt_ds)
        print("Normalization layer adapted.")
    except Exception as e:
        print("Normalization adapt skipped due to:", repr(e))

    print("Fitting")
    History = model.fit(
        train_dataset,
        epochs=EPOCHS,
        callbacks=[reduce_lr, model_checkpoint_callback, callback],
        validation_data=val_dataset,
        verbose=VERBOSE,
    )



## === cell 11
if TF_AVAILABLE:
    if os.path.exists(checkpoint_filepath):
        model.load_weights(checkpoint_filepath)
        print("Loaded best weights from:", checkpoint_filepath)
    else:
        print("Best weights file not found; using last epoch weights.")

    best_model = model



## === cell 12
if TF_AVAILABLE:
    dataset_Test_batched = dataset_Test.batch(32, drop_remainder=False).prefetch(AUTO)

    kept_indices = []
    for _, idx in dataset_Test_batched:
        kept_indices.extend(idx.numpy().astype(np.int32).tolist())
    kept_indices = np.asarray(kept_indices, dtype=np.int32)

    predictions = best_model.predict(dataset_Test_batched, verbose=1)
    print("Pred shape:", predictions.shape)
    print("Kept indices:", kept_indices.shape)



## === cell 13
topk = 5

if TF_AVAILABLE:
    k = topk
    part = np.argpartition(-predictions, kth=k - 1, axis=1)[:, :k]
    row_idx = np.arange(predictions.shape[0])[:, None]
    part_scores = predictions[row_idx, part]
    order_within = np.argsort(-part_scores, axis=1)
    topk_idx = part[row_idx, order_within]

    topk_hotel_ids = le.inverse_transform(topk_idx.reshape(-1)).reshape(-1, topk)

    images_names_kept = [os.path.basename(Paths_Test[i]) for i in kept_indices]
    pred_strings = [" ".join(map(str, row)) for row in topk_hotel_ids]

    submission_partial = pd.DataFrame(
        {"image": images_names_kept, "hotel_id": pred_strings}
    )

    sample = pd.read_csv(sample_sub_path)
    submission = sample[["image"]].merge(submission_partial, on="image", how="left")

    if submission["hotel_id"].isna().any():
        fallback_ids = le.inverse_transform(np.arange(min(topk, Classes))).tolist()
        if len(fallback_ids) < topk:
            fallback_ids = (fallback_ids * (topk // len(fallback_ids) + 1))[:topk]
        fallback = " ".join(map(str, fallback_ids))
        submission["hotel_id"] = submission["hotel_id"].fillna(fallback)

    print(submission.head())
    print("Submission rows:", len(submission))

    submission.to_csv("submission.csv", index=False)
    print("Wrote submission.csv")
else:
    sample = pd.read_csv(sample_sub_path)
    top_ids = train_df["hotel_id"].value_counts().index.astype(str).tolist()[:topk]
    if len(top_ids) < topk:
        top_ids = (top_ids * (topk // len(top_ids) + 1))[:topk]
    fallback_pred = " ".join(top_ids)
    submission = pd.DataFrame(
        {"image": sample["image"].values, "hotel_id": fallback_pred}
    )
    submission.to_csv("submission.csv", index=False)
    print("TensorFlow was unavailable, wrote fallback submission.csv instead.")
    print("TF import error was:", repr(TF_IMPORT_ERROR))
