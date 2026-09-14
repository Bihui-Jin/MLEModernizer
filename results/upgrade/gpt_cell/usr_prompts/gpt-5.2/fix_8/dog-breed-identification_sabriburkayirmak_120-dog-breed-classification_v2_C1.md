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
Given a dataset of images of dogs, predict the breed of each image.

## Metric
Multi Class Log Loss.

## Submission Format
For each image in the test set, you must predict a probability for each of the different breeds. The file should contain a header and have the following format:
```
id,affenpinscher,afghan_hound,..,yorkshire_terrier
000621fb3cbb32d8935728e48679680e,0.0083,0.0,...,0.0083
etc.
```

## Dataset Description
- `train.zip` - the training set, you are provided the breed for these dogs
- `test.zip` - the test set, you must predict the probability of each breed for each image
- `sample_submission.csv` - a sample submission file in the correct format
- `labels.csv` - the breeds for the images in the train set

# 2. Python version

3.12

# 3. Installed packages

geopandas==0.14.4
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
            description.md (169 lines)
            labels.csv (9200 lines)
            labels.csv.zip (201.6 kB)
            sample_submission.csv (1024 lines)
            sample_submission.csv.zip (32.1 kB)
            test.zip (36.4 MB)
            train.zip (324.7 MB)
            dog-breed-identification/
                description.md (169 lines)
                labels.csv (9200 lines)
                ... and 5 other files
                dog-breed-identification/
                test/
                    bca88d42e4fc84b3169b13a615f5fdbf.jpg (38.6 kB)
                    53cb3ed2547cdaf15ec7983d8325f007.jpg (32.9 kB)
                    ... and 1021 other files
                    test/
                train/
                    868decd906bb483bac17a005a3f06bc3.jpg (22.6 kB)
                    7b44341b91b48e2eafe679c00ba1a0a6.jpg (48.4 kB)
                    ... and 9197 other files
                    train/
            test/
                bca88d42e4fc84b3169b13a615f5fdbf.jpg (38.6 kB)
                53cb3ed2547cdaf15ec7983d8325f007.jpg (32.9 kB)
                ... and 1021 other files
                test/
            train/
                868decd906bb483bac17a005a3f06bc3.jpg (22.6 kB)
                7b44341b91b48e2eafe679c00ba1a0a6.jpg (48.4 kB)
                ... and 9197 other files
                train/
        input/
            description.md (169 lines)
            labels.csv (9200 lines)
            labels.csv.zip (201.6 kB)
            sample_submission.csv (1024 lines)
            sample_submission.csv.zip (32.1 kB)
            test.zip (36.4 MB)
            train.zip (324.7 MB)
            dog-breed-identification/
                description.md (169 lines)
                labels.csv (9200 lines)
                ... and 5 other files
                dog-breed-identification/
                test/
                    bca88d42e4fc84b3169b13a615f5fdbf.jpg (38.6 kB)
                    53cb3ed2547cdaf15ec7983d8325f007.jpg (32.9 kB)
                    ... and 1021 other files
                    test/
                train/
                    868decd906bb483bac17a005a3f06bc3.jpg (22.6 kB)
                    7b44341b91b48e2eafe679c00ba1a0a6.jpg (48.4 kB)
                    ... and 9197 other files
                    train/
            test/
                bca88d42e4fc84b3169b13a615f5fdbf.jpg (38.6 kB)
                53cb3ed2547cdaf15ec7983d8325f007.jpg (32.9 kB)
                ... and 1021 other files
                test/
                    bca88d42e4fc84b3169b13a615f5fdbf.jpg (38.6 kB)
                    53cb3ed2547cdaf15ec7983d8325f007.jpg (32.9 kB)
                    ... and 1021 other files
                    test/
            train/
                868decd906bb483bac17a005a3f06bc3.jpg (22.6 kB)
                7b44341b91b48e2eafe679c00ba1a0a6.jpg (48.4 kB)
                ... and 9197 other files
                train/
                    868decd906bb483bac17a005a3f06bc3.jpg (22.6 kB)
                    7b44341b91b48e2eafe679c00ba1a0a6.jpg (48.4 kB)
                    ... and 9197 other files
                    train/
        working/
            dog-breed-identification/
                description.md (169 lines)
                labels.csv (9200 lines)
                ... and 5 other files
                dog-breed-identification/
                test/
                    bca88d42e4fc84b3169b13a615f5fdbf.jpg (38.6 kB)
                    53cb3ed2547cdaf15ec7983d8325f007.jpg (32.9 kB)
                    ... and 1021 other files
                    test/
                train/
                    868decd906bb483bac17a005a3f06bc3.jpg (22.6 kB)
                    7b44341b91b48e2eafe679c00ba1a0a6.jpg (48.4 kB)
                    ... and 9197 other files
                    train/
```

-> data/dog-breed-identification/labels.csv has 9199 rows and 2 columns.
The columns are: id, breed

-> data/dog-breed-identification/sample_submission.csv has 1023 rows and 121 columns.
The columns are: id, affenpinscher, afghan_hound, african_hunting_dog, airedale, american_staffordshire_terrier, appenzeller, australian_terrier, basenji, basset, beagle, bedlington_terrier, bernese_mountain_dog, black-and-tan_coonhound, blenheim_spaniel... and 106 more columns

-> data/labels.csv has 9199 rows and 2 columns.
The columns are: id, breed

-> data/sample_submission.csv has 1023 rows and 121 columns.
The columns are: id, affenpinscher, afghan_hound, african_hunting_dog, airedale, american_staffordshire_terrier, appenzeller, australian_terrier, basenji, basset, beagle, bedlington_terrier, bernese_mountain_dog, black-and-tan_coonhound, blenheim_spaniel... and 106 more columns

-> input/dog-breed-identification/labels.csv has 9199 rows and 2 columns.
The columns are: id, breed

-> input/dog-breed-identification/sample_submission.csv has 1023 rows and 121 columns.
The columns are: id, affenpinscher, afghan_hound, african_hunting_dog, airedale, american_staffordshire_terrier, appenzeller, australian_terrier, basenji, basset, beagle, bedlington_terrier, bernese_mountain_dog, black-and-tan_coonhound, blenheim_spaniel... and 106 more columns

-> (stopped after 10 files for performance)

# 5. Target score

0.47953

# 6. Current score

0.54853

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 0.54) has done: 'Diagnosis: The crash happens during `import tensorflow as tf` in cell 0, before any of your code runs. With TensorFlow 2.18.0 and protobuf 6.x, this `AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'` is a known incompatibility caused by protobuf’s C++ implementation API changes. A minimal runtime workaround is to force protobuf to use the pure-Python implementation before TensorFlow (and anything that transitively imports protobuf internals) is imported.

Patch summary: Modify cell 0 to set `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` (and its version) at the very top, before importing TensorFlow. This avoids the incompatible C++ protobuf path and allows TensorFlow to import cleanly without changing your model/training logic.

Updated cells: Only cell 0 is changed.

Compatibility notes for cell k+1: Cell 1 expects `tf` to be successfully imported and available; this patch preserves the same `tf` symbol and does not alter any downstream interfaces.

Assumptions: The environment allows setting `os.environ` before importing TensorFlow in the same process (standard for notebooks/scripts), and using the Python protobuf implementation is acceptable for this run.'
- What this solution (achieved 0.53407) has done: 'Diagnosis: The crash happens during `import tensorflow as tf` in cell 0 and is caused by an incompatibility between TensorFlow 2.18 and `protobuf==6.33.0` (TF 2.18 expects protobuf < 5). The environment variables forcing the pure-Python protobuf implementation do not fix this, because the TensorFlow/protobuf API mismatch still triggers `MessageFactory.GetPrototype` errors. The most localized way to unblock execution is to ensure an older protobuf version is installed before importing TensorFlow, then proceed with the same imports and variables as originally intended.

Patch summary: In cell 0 only, install a compatible protobuf version (`protobuf<5`) via pip at runtime before importing TensorFlow, then keep the rest of the imports unchanged. This avoids changing any model/training logic and preserves all symbols (`cv2`, `np`, `pd`, `tf`, `K`) used in later cells.

Updated cells: (cell 0 only)

Compatibility notes for cell k+1: Cell 1 continues to work unchanged because `tf.distribute.MirroredStrategy()` import successfully once protobuf is downgraded; the `tf` symbol is still defined in cell 0.

Assumptions: The runtime allows `pip` installs (standard in Kaggle-style notebook environments) and has network/package access to fetch the older protobuf wheel; if not, a preinstalled compatible protobuf wheel is available in the environment cache.'
- What this solution (achieved 0.54488) has done: 'Diagnosis: The crash happens inside `CustomCallback.on_epoch_end` when trying to update the optimizer learning rate via `self.model.optimizer.lr`. In TensorFlow/Keras 2.18, `Adam` no longer exposes the `lr` attribute; the correct attribute is `learning_rate` (or `optimizer.learning_rate.assign(...)`). This makes `K.set_value(self.model.optimizer.lr, new_lr)` raise `AttributeError` during training.

Patch summary: Update the callback to set the learning rate through `self.model.optimizer.learning_rate`, using `.assign(new_lr)` when available and falling back to `K.set_value` for compatibility. Keep all callback logic (monitoring, patience, factor, early stop) identical, only changing the learning-rate update line(s) that currently reference the removed `lr` attribute.

Updated cells: (only cell 14)

Compatibility notes for cell k+1: No variables, outputs, or interfaces used by cell 15 are changed; `history = model.fit(...)` now complete without the callback crashing.

Assumptions: `self.model.optimizer.learning_rate` is a `tf.Variable` (standard in TF 2.18 for Adam); if it’s not assignable, `K.set_value` still work on the underlying variable/tensor.'
- What this solution (achieved 5.01369) has done: 'Diagnosis: The crash happens inside `CustomCallback.on_epoch_end` when it tries to call `K.set_value(self.model.optimizer.lr, new_lr)`. In TensorFlow/Keras 2.18, optimizers no longer expose the legacy `.lr` attribute; the correct attribute is `.learning_rate` (which may be a `tf.Variable` or a schedule). This raises `AttributeError: 'Adam' object has no attribute 'lr'` during `model.fit()`.

Patch summary: Update the callback to set the optimizer learning rate via `self.model.optimizer.learning_rate` (falling back to `.lr` only if present for older compatibility), using `assign` when possible and `K.set_value` otherwise. This keeps the training logic identical (same factor/patience/min_lr behavior) while making it compatible with TF/Keras 2.18+.

Updated cells: Only cell 4 is modified.

Compatibility notes for cell k+1: No variables or interfaces used by later cells are changed; `CustomCallback` remains the same class name and is still passed into `model.fit(...)` exactly as before.

Assumptions: `model.optimizer.learning_rate` is assignable (variable) for `Adam(1e-3)` in TF 2.18; if it is not (e.g., schedule), the code safely fall back to `K.set_value` where applicable and otherwise skip direct assignment without changing any other logic.'
- What this solution (achieved 0.55064) has done: 'Your logloss is extremely high because the model is never trained (so predictions are essentially random), and there’s also a silent label/probability column-order mismatch risk: `pd.get_dummies(train_df.breed)` produces columns in a different order than `sample_submission.csv`, which makes logloss explode even if the model learns. I (1) make the one-line fix in the learning-rate callback in cell 4 so it doesn’t reference the removed `.lr` attribute, (2) ensure one consistent class order everywhere by using the sample submission columns as the definitive class list, (3) add a minimal `model.fit(...)` step using your existing datasets/model/loss (no architectural changes), and (4) guarantee the submission columns align to the sample submission header.'
- What this solution (achieved 0.54847) has done: 'To move logloss down toward the 0.47953 target with minimal risk, I keep your exact model and training loop but fix two high-impact data pipeline issues that can depress performance: (1) ensure images are always read correctly (your current paths miss the `test/` and `train/` subfolders), and (2) feed float32 images (and float32 labels) into the model so MobileNetV2 preprocessing and the loss operate as intended. I also make the train/val split reproducible and safe (fixed `random_state`, `replace` handling) without changing the split logic, and I align generator label one-hot encoding strictly to `CLASS_NAMES` to avoid any remaining class-order surprises. These changes are directly tied to improving the actual learned predictions (not post-hoc manipulation) and should reduce logloss toward the target band.'
- What this solution (achieved 0.54853) has done: 'Your current logloss (0.54847) is worse than the 0.47953 target, so we should make a small, legitimate improvement without changing the model/training core. The biggest low-risk issue is that your `tf.data` pipeline caches the *entire* decoded image dataset in memory (`.cache()` with no filename), which can lead to memory pressure/instability and indirectly hurt effective training; switching to file-based caching preserves identical data and semantics while improving throughput/reliability. I also add a deterministic `.repeat()` plus explicit `steps_per_epoch`/`validation_steps` so training consistently covers the full dataset each epoch (instead of being affected by generator exhaustion/interleave quirks), which usually reduces logloss without changing architecture or loss. Finally, I keep submission formatting identical, just ensuring prediction-row alignment is preserved.'

# 9. Code solution

## === cell 0
import os
import sys
import subprocess

subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "protobuf<5"])

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "2")

import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import backend as K

np.random.seed(42)
tf.random.set_seed(42)



## === cell 1
strategy = tf.distribute.MirroredStrategy()
print(f"Number of devices: {strategy.num_replicas_in_sync}")



## === cell 2
PATH = "/kaggle/input/dog-breed-identification"
NUM_CHANNEL = 3
INPUT_SHAPE = 256
BATCH_SIZE = 32 * strategy.num_replicas_in_sync




## === cell 3
class ImageDatastore:

    def __init__(self, path, csv, output_shape, train_val_test, class_names=None):
        self.path = path
        self.csv = csv
        self.output_shape = output_shape
        self.train_val_test = train_val_test
        self.class_names = class_names
        self.image_paths, self.labels = self.get_files_and_labels()

    def get_files_and_labels(self):
        image_paths = [
            os.path.join(self.path, f"{img_id}.jpg") for img_id in self.csv.index
        ]

        if self.train_val_test == "test":
            labels = None
        else:
            if self.class_names is None:
                dummies = pd.get_dummies(self.csv.breed)
            else:
                dummies = pd.get_dummies(self.csv.breed).reindex(
                    columns=self.class_names, fill_value=0
                )

            labels = dummies.astype("float32").to_numpy()
        return image_paths, labels

    def __call__(self):
        if self.train_val_test == "test":
            for image_path in self.image_paths:
                image = cv2.imread(image_path)
                if image is None:
                    raise FileNotFoundError(f"Could not read image: {image_path}")
                image = cv2.resize(image, self.output_shape)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                yield image.astype("float32")
        else:
            pairs = list(zip(self.image_paths, self.labels))
            for image_path, label in pairs:
                image = cv2.imread(image_path)
                if image is None:
                    raise FileNotFoundError(f"Could not read image: {image_path}")
                image = cv2.resize(image, self.output_shape)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                yield image.astype("float32"), label




## === cell 4
class CustomCallback(tf.keras.callbacks.Callback):
    def __init__(self, monitor="loss", factor=0.5, patience=0, min_lr=0.01):
        super(CustomCallback, self).__init__()
        self.monitor = monitor
        self.factor = factor
        self.patience = patience
        self.min_lr = min_lr
        self.job = 0

    def on_train_begin(self, logs=None):
        self.wait = 0
        self.stopped_epoch = 0
        if "loss" in self.monitor:
            self.best = np.inf
        else:
            self.best = -1

    def on_epoch_end(self, epoch, logs=None):
        current = logs.get(self.monitor)
        if "loss" in self.monitor and current < self.best:
            self.best = current
            self.wait = 0
        elif "acc" in self.monitor and current > self.best:
            self.best = current
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                if self.job == 0:
                    lr = float(K.get_value(self.model.optimizer.learning_rate))
                    new_lr = lr * self.factor
                    if new_lr < self.min_lr:
                        new_lr = self.min_lr
                        self.job = 1
                    try:
                        self.model.optimizer.learning_rate.assign(new_lr)
                    except Exception:
                        K.set_value(self.model.optimizer.learning_rate, new_lr)
                    self.wait = 0
                    print(f"\nLearning rate reduced from {lr} to {new_lr}")
                elif self.job == 1:
                    self.stopped_epoch = epoch
                    self.model.stop_training = True

    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0:
            print(f"Epoch {self.stopped_epoch + 1}: early stopping")




## === cell 5
train_df = pd.read_csv(os.path.join(PATH, "labels.csv"), index_col="id")
train_df.head()



## === cell 6
NUM_CLASS = train_df.breed.nunique()



## === cell 7
val_ratio = 0.2
num_sapmle = int(len(train_df) * val_ratio / NUM_CLASS)



## === cell 8
val_df = pd.concat(
    [
        train_df[train_df.breed == lbl].sample(
            n=num_sapmle,
            random_state=42,
            replace=(len(train_df[train_df.breed == lbl]) < num_sapmle),
        )
        for lbl in train_df.breed.unique()
    ],
    axis=0,
)
val_df = val_df.sample(frac=1, random_state=42)

train_df = train_df.drop(val_df.index)



## === cell 9
sub_test_df = pd.read_csv(
    "/kaggle/input/dog-breed-identification/sample_submission.csv", index_col="id"
)
CLASS_NAMES = list(sub_test_df.columns)
train_df = train_df.copy()
val_df = val_df.copy()
train_df["breed"] = pd.Categorical(train_df["breed"], categories=CLASS_NAMES)
val_df["breed"] = pd.Categorical(val_df["breed"], categories=CLASS_NAMES)
NUM_CLASS = len(CLASS_NAMES)



## === cell 10
train_ds = ImageDatastore(
    os.path.join(PATH, "train"),
    train_df,
    (INPUT_SHAPE, INPUT_SHAPE),
    "train",
    class_names=CLASS_NAMES,
)
val_ds = ImageDatastore(
    os.path.join(PATH, "train"),
    val_df,
    (INPUT_SHAPE, INPUT_SHAPE),
    "val",
    class_names=CLASS_NAMES,
)



## === cell 11
OUTPUT_SIGNATURE = (
    tf.TensorSpec(shape=(INPUT_SHAPE, INPUT_SHAPE, NUM_CHANNEL), dtype="float32"),
    tf.TensorSpec(shape=(NUM_CLASS), dtype="float32"),
)

train = tf.data.Dataset.from_generator(
    generator=train_ds, output_signature=OUTPUT_SIGNATURE
)
train = (
    tf.data.Dataset.range(1)
    .interleave(lambda _: train, num_parallel_calls=tf.data.AUTOTUNE)
    .batch(batch_size=BATCH_SIZE, drop_remainder=True)
    .cache(os.path.join("/kaggle/working", "train_cache"))
    .repeat()
    .prefetch(buffer_size=tf.data.AUTOTUNE)
)

val = tf.data.Dataset.from_generator(
    generator=val_ds, output_signature=OUTPUT_SIGNATURE
)
val = (
    tf.data.Dataset.range(1)
    .interleave(lambda _: val, num_parallel_calls=tf.data.AUTOTUNE)
    .batch(batch_size=BATCH_SIZE, drop_remainder=True)
    .cache(os.path.join("/kaggle/working", "val_cache"))
    .repeat()
    .prefetch(buffer_size=tf.data.AUTOTUNE)
)




## === cell 12
def build_model():
    input_shape = (INPUT_SHAPE, INPUT_SHAPE, NUM_CHANNEL)

    base = tf.keras.applications.MobileNetV2(
        input_shape=input_shape, weights="imagenet", include_top=False, pooling="avg"
    )
    base.trainable = False
    base.training = False

    inputs = tf.keras.Input(shape=input_shape)

    x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
    x = base(x)

    outputs = tf.keras.layers.Dense(NUM_CLASS, activation="softmax")(x)

    model = tf.keras.Model(inputs, outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
        metrics=[tf.keras.metrics.CategoricalAccuracy(name="accuracy")],
    )
    return model




## === cell 13
with strategy.scope():
    model = build_model()

model.summary()




## === cell 14
class CustomCallback(tf.keras.callbacks.Callback):
    def __init__(self, monitor="loss", factor=0.5, patience=0, min_lr=0.01):
        super(CustomCallback, self).__init__()
        self.monitor = monitor
        self.factor = factor
        self.patience = patience
        self.min_lr = min_lr
        self.job = 0

    def on_train_begin(self, logs=None):
        self.wait = 0
        self.stopped_epoch = 0
        if "loss" in self.monitor:
            self.best = np.inf
        else:
            self.best = -1

    def on_epoch_end(self, epoch, logs=None):
        current = logs.get(self.monitor)
        if "loss" in self.monitor and current < self.best:
            self.best = current
            self.wait = 0
        elif "acc" in self.monitor and current > self.best:
            self.best = current
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                if self.job == 0:
                    opt = self.model.optimizer
                    lr_attr = getattr(opt, "learning_rate", None)
                    if lr_attr is None:
                        lr_attr = getattr(opt, "lr", None)

                    lr = float(K.get_value(lr_attr))
                    new_lr = lr * self.factor
                    if new_lr < self.min_lr:
                        new_lr = self.min_lr
                        self.job = 1

                    try:
                        lr_attr.assign(new_lr)
                    except Exception:
                        K.set_value(lr_attr, new_lr)

                    self.wait = 0
                    print(f"\nLearning rate reduced from {lr} to {new_lr}")
                elif self.job == 1:
                    self.stopped_epoch = epoch
                    self.model.stop_training = True

    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0:
            print(f"Epoch {self.stopped_epoch + 1}: early stopping")




## === cell 15
steps_per_epoch = len(train_df) // BATCH_SIZE
validation_steps = len(val_df) // BATCH_SIZE

EPOCHS = 5
history = model.fit(
    train,
    validation_data=val,
    epochs=EPOCHS,
    steps_per_epoch=steps_per_epoch,
    validation_steps=validation_steps,
    callbacks=[CustomCallback(monitor="val_loss", factor=0.5, patience=1, min_lr=1e-5)],
    verbose=1,
)



## === cell 16
sub_test_ds = ImageDatastore(
    os.path.join(PATH, "test"),
    sub_test_df,
    (INPUT_SHAPE, INPUT_SHAPE),
    "test",
    class_names=CLASS_NAMES,
)



## === cell 17
SUB_OUTPUT_SIGNATURE = tf.TensorSpec(
    shape=(INPUT_SHAPE, INPUT_SHAPE, NUM_CHANNEL), dtype="float32"
)
sub_test = tf.data.Dataset.from_generator(
    generator=sub_test_ds, output_signature=SUB_OUTPUT_SIGNATURE
)
sub_test = (
    tf.data.Dataset.range(1)
    .interleave(lambda _: sub_test, num_parallel_calls=tf.data.AUTOTUNE)
    .batch(batch_size=BATCH_SIZE, drop_remainder=False)
    .cache()
    .prefetch(buffer_size=tf.data.AUTOTUNE)
)



## === cell 18
pred = model.predict(sub_test, verbose=1)



## === cell 19
sub_test_df.iloc[:, :] = pred[:, :NUM_CLASS]



## === cell 20
sub_path = os.path.join("/kaggle", "working", "submission.csv")
sub_test_df.to_csv(sub_path)
print("Wrote:", sub_path)
sub_test_df.head()
