# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


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

# 5. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import tensorflow as tf
import cv2
import numpy as np
import pandas as pd
from tensorflow.keras import backend as K

tf.config.optimizer.set_jit(True)

tf.random.set_seed(42)
np.random.seed(42)

from tensorflow.keras.mixed_precision import Policy, set_policy

policy = Policy("mixed_float16")
set_policy(policy)



## === cell 1
strategy = tf.distribute.get_strategy()
print(f"Number of devices: {strategy.num_replicas_in_sync}")



## === cell 2
PATH = "/kaggle/input/dog-breed-identification"
NUM_CHANNEL = 3
INPUT_SHAPE = 256
BATCH_SIZE = int(32 * strategy.num_replicas_in_sync)



## === cell 3
sample_sub = pd.read_csv(os.path.join(PATH, "sample_submission.csv"), index_col="id")
BREED_ORDER = sample_sub.columns.tolist()




## === cell 4
class ImageDatastore:
    """
    Stores only image file paths and one‑hot labels.
    Image decoding/resizing is handled later by a TensorFlow map function,
    which runs in parallel and avoids the Python‑level OpenCV bottleneck.
    """

    def __init__(self, path, csv, output_shape, train_val_test):
        self.path = path
        self.csv = csv
        self.output_shape = output_shape
        self.train_val_test = train_val_test
        self.image_paths, self.labels = self.get_files_and_labels()

    def get_files_and_labels(self):
        image_paths = [
            os.path.join(self.path, img_id) + ".jpg" for img_id in self.csv.index
        ]
        if self.train_val_test == "test":
            labels = None
        else:
            dummies = pd.get_dummies(self.csv.breed)
            dummies = dummies.reindex(columns=BREED_ORDER, fill_value=0)
            labels = dummies.astype("uint8").to_numpy()
        return image_paths, labels




## === cell 5
class CustomCallback(tf.keras.callbacks.Callback):
    def __init__(self, monitor="loss", factor=0.5, patience=10, min_lr=2e-6):
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
            self.best = np.inf  # lower is better
        else:
            self.best = -np.inf  # higher is better

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
                    lr_tensor = getattr(self.model.optimizer, "lr", None)
                    if lr_tensor is None:
                        lr_tensor = getattr(self.model.optimizer, "learning_rate", None)

                    try:
                        lr = float(K.get_value(lr_tensor))
                    except Exception:
                        lr = float(lr_tensor)

                    new_lr = max(lr * self.factor, self.min_lr)

                    try:
                        K.set_value(lr_tensor, new_lr)
                    except Exception:
                        if hasattr(self.model.optimizer, "lr"):
                            self.model.optimizer.lr = new_lr
                        if hasattr(self.model.optimizer, "learning_rate"):
                            self.model.optimizer.learning_rate = new_lr

                    self.job = 1 if new_lr <= self.min_lr else 0
                    self.wait = 0
                    print(f"\nLearning rate reduced to {new_lr}")
                elif self.job == 1:
                    self.stopped_epoch = epoch
                    self.model.stop_training = True

    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0:
            print(f"Epoch {self.stopped_epoch + 1}: early stopping")




## === cell 6
train_df = pd.read_csv(os.path.join(PATH, "labels.csv"), index_col="id")
train_df.head()



## === cell 7
NUM_CLASS = train_df.breed.nunique()



## === cell 8
val_ratio = 0.2
num_sample = int(len(train_df) * val_ratio / NUM_CLASS)

val_df = pd.concat(
    [
        train_df[train_df.breed == lbl].sample(num_sample, random_state=42)
        for lbl in train_df.breed.unique()
    ],
    axis=0,
)
val_df = val_df.sample(frac=1, random_state=42)

train_df = train_df.drop(val_df.index)



## === cell 9
train_ds = ImageDatastore(
    os.path.join(PATH, "train"), train_df, (INPUT_SHAPE, INPUT_SHAPE), "train"
)
val_ds = ImageDatastore(
    os.path.join(PATH, "train"), val_df, (INPUT_SHAPE, INPUT_SHAPE), "val"
)




## === cell 10
def _process_path(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=NUM_CHANNEL)
    img = tf.image.resize(
        img, [INPUT_SHAPE, INPUT_SHAPE], method=tf.image.ResizeMethod.BILINEAR
    )
    img = tf.cast(tf.clip_by_value(img, 0, 255), tf.uint8)
    return img


train_labels_tf = tf.constant(train_ds.labels)
val_labels_tf = tf.constant(val_ds.labels)

train = tf.data.Dataset.from_tensor_slices((train_ds.image_paths, train_labels_tf))
train = train.map(
    lambda p, l: (_process_path(p), l), num_parallel_calls=tf.data.AUTOTUNE
)
train = (
    train.shuffle(buffer_size=10000, seed=42, reshuffle_each_iteration=True)
    .batch(BATCH_SIZE, drop_remainder=True)
    .cache()
    .prefetch(tf.data.AUTOTUNE)
)

val = tf.data.Dataset.from_tensor_slices((val_ds.image_paths, val_labels_tf))
val = val.map(lambda p, l: (_process_path(p), l), num_parallel_calls=tf.data.AUTOTUNE)
val = val.batch(BATCH_SIZE, drop_remainder=True).cache().prefetch(tf.data.AUTOTUNE)




## === cell 11
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




## === cell 12
with strategy.scope():
    model = build_model()

model.summary()



## === cell 13
history = model.fit(
    train,
    epochs=100,
    validation_data=val,
    callbacks=[
        CustomCallback(monitor="val_loss", factor=0.5, patience=10, min_lr=2e-6)
    ],
)

base = model.get_layer("mobilenetv2_1.00_224")
base.trainable = True

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-4),
    loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
    metrics=[tf.keras.metrics.CategoricalAccuracy(name="accuracy")],
)

fine_tune_history = model.fit(
    train,
    epochs=20,
    validation_data=val,
    callbacks=[CustomCallback(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6)],
)



## === cell 14
sub_test_df = pd.read_csv(os.path.join(PATH, "sample_submission.csv"), index_col="id")



## === cell 15
sub_test_ds = ImageDatastore(
    os.path.join(PATH, "test"), sub_test_df, (INPUT_SHAPE, INPUT_SHAPE), "test"
)



## === cell 16
sub_test = tf.data.Dataset.from_tensor_slices(sub_test_ds.image_paths)
sub_test = sub_test.map(lambda p: _process_path(p), num_parallel_calls=tf.data.AUTOTUNE)
sub_test = (
    sub_test.batch(BATCH_SIZE, drop_remainder=False).cache().prefetch(tf.data.AUTOTUNE)
)



## === cell 17
pred = model.predict(sub_test, verbose=0)



## === cell 18
sub_test_df.iloc[:] = pred
submission_path = os.path.join("/kaggle", "working", "submission.csv")
sub_test_df.to_csv(submission_path)
print(f"Submission written to {submission_path}")
sub_test_df.head()
