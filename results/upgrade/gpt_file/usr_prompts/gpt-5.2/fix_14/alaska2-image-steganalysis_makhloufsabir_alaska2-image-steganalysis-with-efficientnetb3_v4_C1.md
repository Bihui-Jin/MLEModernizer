# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


# 1. Kaggle task description

## Task
Determine which of the images have hidden messages embedded using one of three steganography algorithms (JMiPOD, JUNIWARD, UERD).

## Metric
Weighted AUC. Each region of the ROC curve is weighted according to these chosen parameters:

```
tpr_thresholds = [0.0, 0.4, 1.0]
weights = [2, 1]
```

In other words, the area between the true positive rate of 0 and 0.4 is weighted 2X, the area between 0.4 and 1 is now weighed (1X). The total area is normalized by the sum of weights such that the final weighted AUC is between 0 and 1.

## Submission Format
For each `Id` (image) in the test set, you must provide a score that indicates how likely this image contains hidden data: the higher the score, the more it is assumed that image contains secret data. The file should contain a header and have the following format:

```
Id,Label
0001.jpg,0.1
0002.jpg,0.99
0003.jpg,1.2
0004.jpg,-2.2
etc.
```
## Dataset
The only available information on the test set is:

1. Each embedding algorithm is used with the same probability.
2. The payload (message length) is adjusted such that the "difficulty" is approximately the same regardless the content of the image. Images with smooth content are used to hide shorter messages while highly textured images will be used to hide more secret bits. The payload is adjusted in the same manner for testing and training sets.
3. The average message length is 0.4 bit per non-zero AC DCT coefficient.
4. The images are all compressed with one of the three following JPEG quality factors: 95, 90 or 75.

### Files
- `Cover/` contains 75k unaltered images meant for use in training.
- `JMiPOD/` contains 75k examples of the JMiPOD algorithm applied to the cover images.
- `JUNIWARD/`contains 75k examples of the JUNIWARD algorithm applied to the cover images.
- `UERD/` contains 75k examples of the UERD algorithm applied to the cover images.
- `Test/` contains 5k test set images. These are the images for which you are predicting.
- `sample_submission.csv` contains an example submission in the correct format.

# 2. Python version

3.8

# 3. Installed packages

No external packages required in the script and installed.

# 4. Data file paths

```
/
    kaggle/
        data/
            Cover.zip (7.4 GB)
            JMiPOD.zip (7.4 GB)
            JUNIWARD.zip (7.4 GB)
            Test.zip (528.5 MB)
            UERD.zip (7.4 GB)
            description.md (91 lines)
            sample_submission.csv (5001 lines)
            sample_submission.csv.zip (10.7 kB)
            Cover/
                54965.jpg (237.9 kB)
                54517.jpg (126.7 kB)
                ... and 69998 other files
            JMiPOD/
                06809.jpg (36.6 kB)
                42490.jpg (78.8 kB)
                ... and 69998 other files
            JUNIWARD/
                03684.jpg (106.2 kB)
                42131.jpg (144.4 kB)
                ... and 69998 other files
            Test/
                3630.jpg (79.5 kB)
                3197.jpg (208.4 kB)
                ... and 4998 other files
            UERD/
                42300.jpg (47.1 kB)
                59199.jpg (41.9 kB)
                ... and 69998 other files
            alaska2-image-steganalysis/
                Cover.zip (7.4 GB)
                JMiPOD.zip (7.4 GB)
                ... and 6 other files
                Cover/
                    54965.jpg (237.9 kB)
                    54517.jpg (126.7 kB)
                    ... and 69998 other files
                JMiPOD/
                    06809.jpg (36.6 kB)
                    42490.jpg (78.8 kB)
                    ... and 69998 other files
                JUNIWARD/
                    03684.jpg (106.2 kB)
                    42131.jpg (144.4 kB)
                    ... and 69998 other files
                Test/
                    3630.jpg (79.5 kB)
                    3197.jpg (208.4 kB)
                    ... and 4998 other files
                UERD/
                    42300.jpg (47.1 kB)
                    59199.jpg (41.9 kB)
                    ... and 69998 other files
                alaska2-image-steganalysis/
        input/
            Cover.zip (7.4 GB)
            JMiPOD.zip (7.4 GB)
            JUNIWARD.zip (7.4 GB)
            Test.zip (528.5 MB)
            UERD.zip (7.4 GB)
            description.md (91 lines)
            sample_submission.csv (5001 lines)
            sample_submission.csv.zip (10.7 kB)
            Cover/
                54965.jpg (237.9 kB)
                54517.jpg (126.7 kB)
                ... and 69998 other files
            JMiPOD/
                06809.jpg (36.6 kB)
                42490.jpg (78.8 kB)
                ... and 69998 other files
            JUNIWARD/
                03684.jpg (106.2 kB)
                42131.jpg (144.4 kB)
                ... and 69998 other files
            Test/
                3630.jpg (79.5 kB)
                3197.jpg (208.4 kB)
                ... and 4998 other files
            UERD/
                42300.jpg (47.1 kB)
                59199.jpg (41.9 kB)
                ... and 69998 other files
            alaska2-image-steganalysis/
                Cover.zip (7.4 GB)
                JMiPOD.zip (7.4 GB)
                ... and 6 other files
                Cover/
                    54965.jpg (237.9 kB)
                    54517.jpg (126.7 kB)
                    ... and 69998 other files
                JMiPOD/
                    06809.jpg (36.6 kB)
                    42490.jpg (78.8 kB)
                    ... and 69998 other files
                JUNIWARD/
                    03684.jpg (106.2 kB)
                    42131.jpg (144.4 kB)
                    ... and 69998 other files
                Test/
                    3630.jpg (79.5 kB)
                    3197.jpg (208.4 kB)
                    ... and 4998 other files
                UERD/
                    42300.jpg (47.1 kB)
                    59199.jpg (41.9 kB)
                    ... and 69998 other files
                alaska2-image-steganalysis/
        working/
            alaska2-image-steganalysis/
                Cover.zip (7.4 GB)
                JMiPOD.zip (7.4 GB)
                ... and 6 other files
                Cover/
                    54965.jpg (237.9 kB)
                    54517.jpg (126.7 kB)
                    ... and 69998 other files
                JMiPOD/
                    06809.jpg (36.6 kB)
                    42490.jpg (78.8 kB)
                    ... and 69998 other files
                JUNIWARD/
                    03684.jpg (106.2 kB)
                    42131.jpg (144.4 kB)
                    ... and 69998 other files
                Test/
                    3630.jpg (79.5 kB)
                    3197.jpg (208.4 kB)
                    ... and 4998 other files
                UERD/
                    42300.jpg (47.1 kB)
                    59199.jpg (41.9 kB)
                    ... and 69998 other files
                alaska2-image-steganalysis/
```

-> data/alaska2-image-steganalysis/sample_submission.csv has 5000 rows and 2 columns.
The columns are: Id, Label

-> data/sample_submission.csv has 5000 rows and 2 columns.
The columns are: Id, Label

-> input/alaska2-image-steganalysis/sample_submission.csv has 5000 rows and 2 columns.
The columns are: Id, Label

-> input/sample_submission.csv has 5000 rows and 2 columns.
The columns are: Id, Label

-> working/alaska2-image-steganalysis/sample_submission.csv has 5000 rows and 2 columns.
The columns are: Id, Label

# 5. Code solution

## === cell 0
import numpy as np
import pandas as pd
import os

import tensorflow as tf
import tensorflow.keras.layers as l
from tensorflow.keras.optimizers import Adam

SEED = 10
tf.random.set_seed(SEED)
np.random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)

try:
    tf.config.optimizer.set_jit(True)
except Exception:
    pass

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

print("TensorFlow:", tf.__version__)




## === cell 1
try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.TPUStrategy(tpu)
    print("Using TPU:", tpu.master())
except Exception as e:
    strategy = tf.distribute.get_strategy()
    print("Using default strategy (GPU/CPU). TPU not available:", repr(e))

print("Replicas:", strategy.num_replicas_in_sync)




## === cell 2
AUTO = tf.data.AUTOTUNE

BASE_PATH = "/kaggle/input/alaska2-image-steganalysis"
assert os.path.exists(BASE_PATH), f"Missing expected dataset path: {BASE_PATH}"




## === cell 3
sample = pd.read_csv(f"{BASE_PATH}/sample_submission.csv")

BATCH_SIZE = 8 * strategy.num_replicas_in_sync
EPOCHS = 1


def _list_jpg_basenames_sorted(folder):
    paths = tf.io.gfile.glob(os.path.join(folder, "*.jpg"))
    names = [os.path.basename(p) for p in paths]
    names.sort()
    return names


test_names = _list_jpg_basenames_sorted(f"{BASE_PATH}/Test")
test_paths = [f"{BASE_PATH}/Test/{n}" for n in test_names]

cover_names = _list_jpg_basenames_sorted(f"{BASE_PATH}/Cover")

cover_dir = f"{BASE_PATH}/Cover/"
jmipod_dir = f"{BASE_PATH}/JMiPOD/"
juni_dir = f"{BASE_PATH}/JUNIWARD/"
uerd_dir = f"{BASE_PATH}/UERD/"


Test_df = pd.DataFrame({"path": test_paths, "name": test_names}).reset_index(drop=True)

print("Cover files:", len(cover_names), "Test files:", len(test_names))
print("\nTest sample:\n", Test_df.head(2))




## === cell 4
MAX_TRAIN_PER_CLASS = int(
    os.environ.get("MAX_TRAIN_PER_CLASS", "8000")
)  # deterministic cap
MAX_VAL_PER_CLASS = int(os.environ.get("MAX_VAL_PER_CLASS", "2000"))
val_frac = 0.2

n_cover = len(cover_names)
idx_all = np.arange(n_cover, dtype=np.int32)

rng = np.random.RandomState(SEED)
rng.shuffle(idx_all)

n_val_cover = int(round(n_cover * val_frac))
cover_val_idx = idx_all[:n_val_cover]
cover_train_idx = idx_all[n_val_cover:]


def _cap_indices(indices, cap, seed):
    indices = np.asarray(indices)
    if len(indices) <= cap:
        return indices
    r = np.random.RandomState(seed)
    tmp = indices.copy()
    r.shuffle(tmp)
    return tmp[:cap]


cover_train_idx_cap = _cap_indices(cover_train_idx, MAX_TRAIN_PER_CLASS, SEED + 1)
cover_val_idx_cap = _cap_indices(cover_val_idx, MAX_VAL_PER_CLASS, SEED + 2)


def _build_paths_for_cover_indices(cover_idx, dirs_and_label):
    paths = []
    labels = []
    for d, lab in dirs_and_label:
        names = [cover_names[i] for i in cover_idx]
        paths.extend([d + n for n in names])
        labels.extend([lab] * len(names))
    return np.asarray(paths, dtype=object), np.asarray(labels, dtype=np.int32)


dirs_train = [(cover_dir, 0), (jmipod_dir, 1), (juni_dir, 1), (uerd_dir, 1)]
X_train_eff, y_train_eff = _build_paths_for_cover_indices(
    cover_train_idx_cap, dirs_train
)
X_val_eff, y_val_eff = _build_paths_for_cover_indices(cover_val_idx_cap, dirs_train)

rng2 = np.random.RandomState(SEED + 3)
perm_tr = np.arange(len(X_train_eff))
rng2.shuffle(perm_tr)
X_train_eff, y_train_eff = X_train_eff[perm_tr], y_train_eff[perm_tr]

rng3 = np.random.RandomState(SEED + 4)
perm_va = np.arange(len(X_val_eff))
rng3.shuffle(perm_va)
X_val_eff, y_val_eff = X_val_eff[perm_va], y_val_eff[perm_va]

X_test = Test_df["path"].values.astype(object)

print("Effective shapes (capped):", X_train_eff.shape, X_val_eff.shape, X_test.shape)
print(
    "Training set counts (effective):\n",
    pd.Series(y_train_eff).value_counts().rename(index={0: "Cover", 1: "Stego"}),
)




## === cell 5
IMG_SIZE = (300, 300)
_preprocess = tf.keras.applications.efficientnet.preprocess_input


@tf.function
def decode_image(filename, label=None, image_size=IMG_SIZE):
    bits = tf.io.read_file(filename)
    image = tf.image.decode_jpeg(bits, channels=3)  # exact decode
    image = tf.image.resize(image, image_size, method=tf.image.ResizeMethod.BILINEAR)
    image = tf.cast(image, tf.float32)
    image = _preprocess(image)
    image.set_shape((image_size[0], image_size[1], 3))
    if label is None:
        return image
    return image, tf.cast(label, tf.float32)




## === cell 6
options = tf.data.Options()
options.experimental_deterministic = True
options.experimental_optimization.apply_default_optimizations = True
options.experimental_optimization.map_parallelization = True
options.experimental_optimization.parallel_batch = True
options.threading.private_threadpool_size = max(8, (os.cpu_count() or 8))
options.threading.max_intra_op_parallelism = 0

TRAIN_CACHE = os.path.join("/kaggle/working", "ds_cache_train")
VAL_CACHE = os.path.join("/kaggle/working", "ds_cache_val")
TEST_CACHE = os.path.join("/kaggle/working", "ds_cache_test")

train_dataset = (
    tf.data.Dataset.from_tensor_slices((X_train_eff, y_train_eff))
    .with_options(options)
    .cache(TRAIN_CACHE)
    .shuffle(1024, seed=SEED, reshuffle_each_iteration=True)
    .map(decode_image, num_parallel_calls=AUTO, deterministic=True)
    .batch(BATCH_SIZE, drop_remainder=True)
    .prefetch(AUTO)
)

valid_dataset = (
    tf.data.Dataset.from_tensor_slices((X_val_eff, y_val_eff))
    .with_options(options)
    .cache(VAL_CACHE)
    .map(decode_image, num_parallel_calls=AUTO, deterministic=True)
    .batch(BATCH_SIZE)
    .prefetch(AUTO)
)

test_dataset = (
    tf.data.Dataset.from_tensor_slices(X_test)
    .with_options(options)
    .cache(TEST_CACHE)
    .map(decode_image, num_parallel_calls=AUTO, deterministic=True)
    .batch(BATCH_SIZE)
    .prefetch(AUTO)
)




## === cell 7
with strategy.scope():
    backbone = tf.keras.applications.EfficientNetB3(
        include_top=False,
        weights="imagenet",
        input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3),
    )

    backbone.trainable = False

    model = tf.keras.Sequential(
        [
            backbone,
            l.GlobalAveragePooling2D(),
            l.Dropout(0.1),
            l.Dense(1, activation="sigmoid"),
        ]
    )
    opt = Adam(learning_rate=0.002, beta_1=0.9, beta_2=0.999, decay=0.01, amsgrad=False)

    model.compile(
        optimizer=opt,
        loss="binary_crossentropy",
        metrics=["accuracy"],
        steps_per_execution=32,
    )

model.summary()




## === cell 8
MODEL_PATH = "Mymodel.h5"

if os.path.exists(MODEL_PATH):
    with strategy.scope():
        model = tf.keras.models.load_model(MODEL_PATH, compile=True)
    print(
        f"Loaded existing model from {MODEL_PATH}; skipping training to meet time limit."
    )
    history = None
else:
    STEPS_PER_EPOCH = X_train_eff.shape[0] // BATCH_SIZE
    if STEPS_PER_EPOCH < 1:
        STEPS_PER_EPOCH = 1

    history = model.fit(
        train_dataset,
        steps_per_epoch=STEPS_PER_EPOCH,
        epochs=EPOCHS,
        validation_data=valid_dataset,
        verbose=1,
    )




## === cell 9
if not os.path.exists(MODEL_PATH):
    model.save(MODEL_PATH)
    print(f"Saved model to {MODEL_PATH}")
else:
    print("Model file already exists; not overwriting.")




## === cell 10
pred = model.predict(test_dataset, verbose=1)
pred = np.asarray(pred).reshape(-1)[: len(Test_df)]
print("Pred shape:", pred.shape, "min/max:", float(pred.min()), float(pred.max()))

submission = pd.DataFrame(
    {"Id": Test_df["name"].values, "Label": pred.astype(np.float32)}
)
submission.to_csv("submission.csv", index=False)

print(submission.head())
print("Wrote submission.csv with rows:", len(submission))

assert (
    submission.shape[0] == sample.shape[0]
), "Submission row count must match sample_submission.csv"
assert list(submission.columns) == [
    "Id",
    "Label",
], "Submission columns must be Id,Label"
assert submission["Id"].is_unique, "Ids should be unique"
assert submission["Id"].iloc[0].endswith(".jpg"), "Id should look like a jpg filename"
