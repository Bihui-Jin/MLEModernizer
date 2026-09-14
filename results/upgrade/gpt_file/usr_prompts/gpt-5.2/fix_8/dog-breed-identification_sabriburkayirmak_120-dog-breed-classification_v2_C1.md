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

0.78058

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 0.55495) has done: 'I fix two runtime-breaking issues while keeping your modeling/training logic intact: the TensorFlow import crash (protobuf incompatibility) and the callback’s use of the deprecated `optimizer.lr` attribute (now `optimizer.learning_rate`). I also make the stratified validation split deterministic (seeded) to stabilize training and typically improve log loss slightly toward your target without changing the core approach. Finally, I ensure the submission is written with the correct shape/alignment and a `.csv` suffix in `/kaggle/working/submission.csv`.'
- What this solution (achieved 0.57685) has done: 'I fix the TensorFlow import crash by avoiding the protobuf “python” fallback and forcing the compatible upb runtime instead, which removes the `MessageFactory.GetPrototype` error in this environment. I also fix the learning-rate callback to handle `optimizer.learning_rate` being either a tf.Variable/schedule or a plain float/string-like, by using `optimizer.learning_rate.assign(...)` when possible (and falling back safely when not). These changes are runtime/stability fixes and should keep your core modeling/training logic intact; training now run through, allowing you to generate a valid `submission.csv`. The rest of the pipeline (data loading, split strategy, model, and submission formatting) is preserved.'
- What this solution (achieved 0.77586) has done: 'The timeout is dominated by (1) an unnecessarily heavy distribution strategy on CPU-only environments, (2) data pipeline overhead from caching full 256×256×3 uint8 tensors in RAM, and (3) running eager-mode training without XLA compilation. I keep the exact same model, losses, epoch counts, and callback semantics, but (a) automatically fall back to the default (single-device) strategy when no GPUs are present, (b) switch tf.data caching to on-disk caches (so decoding happens once without blowing RAM and stalling), and (c) enable XLA JIT and set tf.data options to reduce pipeline overhead while keeping determinism. These changes are runtime-focused and preserve the same training/evaluation logic and numerical intent (only negligible float-level differences possible).'
- What this solution (achieved 0.7698) has done: 'The immediate blocker is that TensorFlow can’t import due to a protobuf runtime incompatibility (`MessageFactory.GetPrototype`), so the first change is to force the compatible “python” protobuf implementation (and avoid setting `upb`) before importing TensorFlow. Next, I keep your exact model/training logic but fix the validation split so it is truly stratified at the intended ratio without silently under-sampling each class (your current `num_sapmle` formula makes the validation set ~1/NUM_CLASS of the intended size), which should materially improve log-loss toward the target without changing the approach. Finally, I keep your submission writing logic but add small safety checks (image read failures, stable class column alignment to the sample submission) to ensure a valid `submission.csv` is always produced.'
- What this solution (achieved 0.78058) has done: 'I fix the immediate runtime blocker by changing the protobuf environment setting to the compatible default runtime for this Kaggle image, which avoids the `MessageFactory.GetPrototype` TensorFlow import crash. Next, I correct the validation split logic to be truly stratified at the intended `val_ratio` (your current `num_sample` formula under-samples each class and makes the validation set far too small), which should materially improve multiclass log loss toward the target without changing the model or training procedure. Finally, I make the one-hot label column order consistent between train/val and the sample submission columns to prevent silent class-order mismatches that severely hurt log loss, while keeping the same architecture/loss/training loops and still writing `/kaggle/working/submission.csv`.'

# 9. Code solution

## === cell 0
import os

os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", None)

import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import backend as K

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

AUTOTUNE = tf.data.AUTOTUNE

try:
    tf.config.optimizer.set_jit(True)
except Exception:
    pass



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
gpus = tf.config.list_physical_devices("GPU")
if gpus:
    strategy = tf.distribute.MirroredStrategy()
else:
    strategy = tf.distribute.get_strategy()

print(f"Number of devices: {strategy.num_replicas_in_sync}")



## === cell 2
PATH = "/kaggle/input/dog-breed-identification"
NUM_CHANNEL = 3
INPUT_SHAPE = 256
BATCH_SIZE = 32 * strategy.num_replicas_in_sync




## === cell 3
class ImageDatastore:
    def __init__(self, path, csv, output_shape, train_val_test):
        self.path = path
        self.csv = csv
        self.output_shape = output_shape
        self.train_val_test = train_val_test
        self.image_paths, self.labels = self.get_files_and_labels()

    def get_files_and_labels(self):
        image_paths = [
            os.path.join(self.path, path) + ".jpg" for path in self.csv.index
        ]
        if self.train_val_test == "test":
            labels = ["" for _ in range(len(image_paths))]
        else:
            labels = pd.get_dummies(self.csv.breed).astype("uint8").to_numpy()
        return image_paths, labels

    def __call__(self):
        pairs = list(zip(self.image_paths, self.labels))
        for image_path, label in pairs:
            image = cv2.imread(image_path)
            if image is None:
                image = np.zeros(
                    (self.output_shape[1], self.output_shape[0], 3), dtype=np.uint8
                )

            image = cv2.resize(image, self.output_shape)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            if self.train_val_test == "test":
                yield image
            else:
                yield image, label




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

    def _get_current_lr(self):
        lr = self.model.optimizer.learning_rate
        try:
            return float(tf.keras.backend.get_value(lr))
        except Exception:
            try:
                return float(lr)
            except Exception:
                return None

    def _set_lr(self, new_lr):
        lr = self.model.optimizer.learning_rate
        try:
            if hasattr(lr, "assign"):
                lr.assign(new_lr)
                return True
        except Exception:
            pass
        try:
            self.model.optimizer.learning_rate = new_lr
            return True
        except Exception:
            return False

    def on_epoch_end(self, epoch, logs=None):
        current = logs.get(self.monitor) if logs else None
        if current is None:
            return

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
                    lr = self._get_current_lr()
                    if lr is None:
                        self.wait = 0
                        return
                    new_lr = lr * self.factor
                    if new_lr < self.min_lr:
                        new_lr = self.min_lr
                        self.job = 1
                    ok = self._set_lr(new_lr)
                    self.wait = 0
                    if ok:
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

sub_cols = pd.read_csv(
    os.path.join(PATH, "sample_submission.csv"), nrows=1
).columns.tolist()
CLASS_NAMES = [c for c in sub_cols if c != "id"]
assert len(CLASS_NAMES) == NUM_CLASS, (len(CLASS_NAMES), NUM_CLASS)



## === cell 7
val_ratio = 0.2
val_df = (
    train_df.groupby("breed", group_keys=False)
    .apply(lambda x: x.sample(frac=val_ratio, random_state=SEED))
    .sample(frac=1.0, random_state=SEED)
)
train_df = train_df.drop(val_df.index)

print("Train size:", len(train_df), "Val size:", len(val_df), "Classes:", NUM_CLASS)



## === cell 8
y_train = (
    pd.get_dummies(train_df["breed"])
    .reindex(columns=CLASS_NAMES, fill_value=0)
    .astype("uint8")
    .to_numpy()
)
y_val = (
    pd.get_dummies(val_df["breed"])
    .reindex(columns=CLASS_NAMES, fill_value=0)
    .astype("uint8")
    .to_numpy()
)

train_image_paths = [
    os.path.join(PATH, "train", idx) + ".jpg" for idx in train_df.index
]
val_image_paths = [os.path.join(PATH, "train", idx) + ".jpg" for idx in val_df.index]




## === cell 9
def _decode_resize_rgb_uint8(path):
    img_bytes = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img_bytes, channels=3)  # RGB
    img = tf.image.resize(
        img, (INPUT_SHAPE, INPUT_SHAPE), method=tf.image.ResizeMethod.BILINEAR
    )
    img = tf.clip_by_value(img, 0.0, 255.0)
    img = tf.cast(img, tf.uint8)
    return img


def _make_trainval_dataset(
    image_paths, labels, batch_size, drop_remainder, cache_path=None
):
    ds = tf.data.Dataset.from_tensor_slices((image_paths, labels))
    ds = ds.map(
        lambda p, y: (_decode_resize_rgb_uint8(p), y),
        num_parallel_calls=AUTOTUNE,
        deterministic=True,
    )
    ds = ds.batch(batch_size=batch_size, drop_remainder=drop_remainder)
    if cache_path is not None:
        ds = ds.cache(cache_path)
    ds = ds.prefetch(AUTOTUNE)

    opts = tf.data.Options()
    opts.deterministic = True
    try:
        opts.experimental_optimization.map_parallelization = True
        opts.experimental_optimization.parallel_batch = True
    except Exception:
        pass
    ds = ds.with_options(opts)
    return ds


def _make_test_dataset(image_paths, batch_size, cache_path=None):
    ds = tf.data.Dataset.from_tensor_slices(image_paths)
    ds = ds.map(
        _decode_resize_rgb_uint8, num_parallel_calls=AUTOTUNE, deterministic=True
    )
    ds = ds.batch(batch_size=batch_size, drop_remainder=False)
    if cache_path is not None:
        ds = ds.cache(cache_path)
    ds = ds.prefetch(AUTOTUNE)

    opts = tf.data.Options()
    opts.deterministic = True
    try:
        opts.experimental_optimization.map_parallelization = True
        opts.experimental_optimization.parallel_batch = True
    except Exception:
        pass
    ds = ds.with_options(opts)
    return ds


train_cache_path = os.path.join("/kaggle/working", "tfdata_cache_train")
val_cache_path = os.path.join("/kaggle/working", "tfdata_cache_val")

train = _make_trainval_dataset(
    np.array(train_image_paths, dtype=object),
    np.asarray(y_train, dtype=np.uint8),
    batch_size=BATCH_SIZE,
    drop_remainder=True,
    cache_path=train_cache_path,
)

val = _make_trainval_dataset(
    np.array(val_image_paths, dtype=object),
    np.asarray(y_val, dtype=np.uint8),
    batch_size=BATCH_SIZE,
    drop_remainder=True,
    cache_path=val_cache_path,
)




## === cell 10
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
        jit_compile=False,
    )
    return model




## === cell 11
with strategy.scope():
    model = build_model()

model.summary()



## === cell 12
history = model.fit(
    train,
    epochs=100,
    validation_data=val,
    callbacks=[CustomCallback(monitor="val_loss", factor=0.5, patience=5, min_lr=2e-6)],
)



## === cell 13
with strategy.scope():
    base = None
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model) and layer.name.lower().startswith(
            "mobilenetv2"
        ):
            base = layer
            break
    if base is not None:
        base.trainable = True

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-5),
        loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
        metrics=[tf.keras.metrics.CategoricalAccuracy(name="accuracy")],
        jit_compile=False,
    )

history_ft = model.fit(
    train,
    epochs=10,
    validation_data=val,
    callbacks=[CustomCallback(monitor="val_loss", factor=0.5, patience=3, min_lr=2e-6)],
)



## === cell 14
sub_test_df = pd.read_csv(
    "/kaggle/input/dog-breed-identification/sample_submission.csv", index_col="id"
)
sub_test_df.head()



## === cell 15
test_image_paths = [
    os.path.join(PATH, "test", idx) + ".jpg" for idx in sub_test_df.index
]

sub_cache_path = os.path.join("/kaggle/working", "tfdata_cache_test")
sub_test = _make_test_dataset(
    np.array(test_image_paths, dtype=object),
    batch_size=BATCH_SIZE,
    cache_path=sub_cache_path,
)



## === cell 16
pred = model.predict(sub_test, verbose=1)



## === cell 17
pred = np.asarray(pred)
if pred.shape[0] != len(sub_test_df):
    raise ValueError(
        f"Prediction rows ({pred.shape[0]}) != submission rows ({len(sub_test_df)})"
    )
if pred.shape[1] != sub_test_df.shape[1]:
    raise ValueError(
        f"Prediction cols ({pred.shape[1]}) != submission cols ({sub_test_df.shape[1]})"
    )

sub_test_df.iloc[:, :] = pred

out_path = os.path.join("/kaggle", "working", "submission.csv")
sub_test_df.to_csv(out_path)
print("Wrote:", out_path)
sub_test_df.head()
