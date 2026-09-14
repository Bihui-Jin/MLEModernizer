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

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "3")

import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import backend as K

SEED = 42
np.random.seed(SEED)
tf.keras.utils.set_random_seed(SEED)



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
            os.path.join(self.path, path) + ".jpg" for path in self.csv.index
        ]
        if self.train_val_test == "test":
            labels = ["" for _ in range(len(image_paths))]
        else:
            if self.class_names is None:
                labels = pd.get_dummies(self.csv.breed).astype("uint8").to_numpy()
            else:
                labels = (
                    pd.get_dummies(self.csv.breed)
                    .reindex(columns=self.class_names, fill_value=0)
                    .astype("uint8")
                    .to_numpy()
                )
        return image_paths, labels

    def __call__(self):
        pairs = list(zip(self.image_paths, self.labels))
        for image_path, label in pairs:
            image = cv2.imread(image_path)
            if image is None:
                raise FileNotFoundError(f"Could not read image: {image_path}")
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
        self.best_weights = None

    def on_train_begin(self, logs=None):
        self.wait = 0
        self.stopped_epoch = 0
        if "loss" in self.monitor:
            self.best = np.inf
        else:
            self.best = -1

    def _get_lr(self):
        lr_obj = self.model.optimizer.learning_rate
        try:
            return float(K.get_value(lr_obj))
        except Exception:
            try:
                return float(tf.keras.backend.get_value(lr_obj))
            except Exception:
                return float(lr_obj)

    def _set_lr(self, new_lr: float):
        lr_obj = self.model.optimizer.learning_rate
        try:
            if hasattr(lr_obj, "assign"):
                lr_obj.assign(new_lr)
            else:
                self.model.optimizer.learning_rate = new_lr
        except Exception:
            try:
                K.set_value(lr_obj, new_lr)
            except Exception:
                self.model.optimizer.learning_rate = new_lr

    def on_epoch_end(self, epoch, logs=None):
        current = logs.get(self.monitor)
        if current is None:
            return

        improved = False
        if "loss" in self.monitor and current < self.best:
            improved = True
        elif "acc" in self.monitor and current > self.best:
            improved = True

        if improved:
            self.best_weights = self.model.get_weights()
            self.best = current
            self.wait = 0
            return

        self.wait += 1
        if self.wait < self.patience:
            return

        if self.job == 0:
            if self.best_weights is not None:
                self.model.set_weights(self.best_weights)

            lr = self._get_lr()
            new_lr = lr * self.factor
            if new_lr < self.min_lr:
                new_lr = self.min_lr
                self.job = 1

            self._set_lr(new_lr)
            self.wait = 0
            print(
                f"\nLearning rate reduced from {'{:.3g}'.format(lr)} to {'{:.3g}'.format(new_lr)}"
            )
        elif self.job == 1:
            self.stopped_epoch = epoch
            self.model.stop_training = True

    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0:
            print(f"Epoch {self.stopped_epoch + 1}: early stopping")




## === cell 5
sub_template_df = pd.read_csv(
    os.path.join(PATH, "sample_submission.csv"), index_col="id"
)
CLASS_NAMES = list(sub_template_df.columns)
NUM_CLASS = len(CLASS_NAMES)

train_df = pd.read_csv(os.path.join(PATH, "labels.csv"), index_col="id")
train_df.head()



## === cell 6
val_ratio = 0.2
num_sample = max(1, int(len(train_df) * val_ratio / NUM_CLASS))



## === cell 7
val_df = pd.concat(
    [
        train_df[train_df.breed == lbl].sample(n=num_sample, random_state=SEED)
        for lbl in train_df.breed.unique()
    ],
    axis=0,
)
val_df = val_df.sample(frac=1, random_state=SEED)
train_df = train_df.drop(val_df.index)



## === cell 8
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



## === cell 9
data_augmentation = tf.keras.Sequential(
    [
        tf.keras.layers.RandomFlip(),
        tf.keras.layers.RandomRotation(0.2),
        tf.keras.layers.RandomZoom(0.2),
    ]
)



## === cell 10
OUTPUT_SIGNATURE = (
    tf.TensorSpec(shape=(INPUT_SHAPE, INPUT_SHAPE, NUM_CHANNEL), dtype="uint8"),
    tf.TensorSpec(shape=(NUM_CLASS,), dtype="uint8"),
)


def _cast_train(X, y):
    return tf.cast(X, tf.float32), tf.cast(y, tf.float32)


train = tf.data.Dataset.from_generator(
    generator=train_ds, output_signature=OUTPUT_SIGNATURE
)
train = (
    tf.data.Dataset.range(1)
    .interleave(lambda _: train, num_parallel_calls=tf.data.AUTOTUNE)
    .batch(batch_size=BATCH_SIZE, drop_remainder=True)
    .map(
        lambda X, y: (data_augmentation(X, training=True), y),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    .map(_cast_train, num_parallel_calls=tf.data.AUTOTUNE)
    .cache()
    .prefetch(buffer_size=tf.data.AUTOTUNE)
)

val = tf.data.Dataset.from_generator(
    generator=val_ds, output_signature=OUTPUT_SIGNATURE
)
val = (
    tf.data.Dataset.range(1)
    .interleave(lambda _: val, num_parallel_calls=tf.data.AUTOTUNE)
    .batch(batch_size=BATCH_SIZE, drop_remainder=True)
    .map(_cast_train, num_parallel_calls=tf.data.AUTOTUNE)
    .cache()
    .prefetch(buffer_size=tf.data.AUTOTUNE)
)




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

    x = tf.keras.layers.Dense(512)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.Dropout(0.5)(x)

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



## === cell 14
sub_test_df = sub_template_df.copy()



## === cell 15
sub_test_ds = ImageDatastore(
    os.path.join(PATH, "test"), sub_test_df, (INPUT_SHAPE, INPUT_SHAPE), "test"
)



## === cell 16
SUB_OUTPUT_SIGNATURE = tf.TensorSpec(
    shape=(INPUT_SHAPE, INPUT_SHAPE, NUM_CHANNEL), dtype="uint8"
)


def _cast_test(X):
    return tf.cast(X, tf.float32)


sub_test = tf.data.Dataset.from_generator(
    generator=sub_test_ds, output_signature=SUB_OUTPUT_SIGNATURE
)
sub_test = (
    tf.data.Dataset.range(1)
    .interleave(lambda _: sub_test, num_parallel_calls=tf.data.AUTOTUNE)
    .batch(batch_size=BATCH_SIZE, drop_remainder=False)
    .map(_cast_test, num_parallel_calls=tf.data.AUTOTUNE)
    .cache()
    .prefetch(buffer_size=tf.data.AUTOTUNE)
)



## === cell 17
pred = model.predict(sub_test, verbose=1)



## === cell 18
pred = pred[: len(sub_test_df)]
sub_test_df.iloc[:, :] = pred

out_path = os.path.join("/kaggle", "working", "submission.csv")
sub_test_df.to_csv(out_path)
print("Wrote:", out_path)
sub_test_df.head()
