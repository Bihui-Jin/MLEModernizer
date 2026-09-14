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

try:
    from google.protobuf.message_factory import MessageFactory

    if not hasattr(MessageFactory, "GetPrototype"):

        def GetPrototype(self, descriptor):
            return self.GetMessageClass(descriptor)

        MessageFactory.GetPrototype = GetPrototype
except Exception:
    pass

import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.mixed_precision import set_global_policy

tf.config.optimizer.set_jit(True)

set_global_policy("mixed_float16")
tf.random.set_seed(42)
np.random.seed(42)



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
    """
    Stores file paths and one‑hot labels.
    The actual image reading is performed inside the tf.data pipeline
    using TensorFlow ops (no OpenCV), preserving the original
    CV2‑based preprocessing (resize + BGR→RGB conversion).
    """

    def __init__(self, base_path, csv, output_shape, train_val_test):
        self.base_path = base_path
        self.csv = csv
        self.output_shape = output_shape
        self.train_val_test = train_val_test
        self.image_paths, self.labels = self._gather_paths_and_labels()

    def _gather_paths_and_labels(self):
        image_paths = [
            os.path.join(self.base_path, img_id) + ".jpg" for img_id in self.csv.index
        ]
        if self.train_val_test == "test":
            labels = np.zeros((len(image_paths), 0), dtype="uint8")
        else:
            labels = pd.get_dummies(self.csv.breed).astype("uint8").to_numpy()
        return image_paths, labels




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

    def on_epoch_end(self, epoch, logs=None):
        current = logs.get(self.monitor)
        if "loss" in self.monitor and current < self.best:
            self.best_weights = self.model.get_weights()
            self.best = current
            self.wait = 0
        elif "acc" in self.monitor and current > self.best:
            self.best_weights = self.model.get_weights()
            self.best = current
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                if self.job == 0:
                    self.model.set_weights(self.best_weights)
                    lr_tensor = self.model.optimizer.learning_rate
                    lr = K.get_value(lr_tensor)
                    new_lr = lr * self.factor
                    if new_lr < self.min_lr:
                        new_lr = self.min_lr
                        self.job = 1
                    if hasattr(lr_tensor, "assign"):
                        lr_tensor.assign(new_lr)
                    else:
                        self.model.optimizer.learning_rate = new_lr
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
        train_df[train_df.breed == lbl].sample(num_sapmle)
        for lbl in train_df.breed.unique()
    ],
    axis=0,
)
val_df = val_df.sample(frac=1)

train_df = train_df.drop(val_df.index)



## === cell 9
train_ds = ImageDatastore(
    os.path.join(PATH, "train"), train_df, (INPUT_SHAPE, INPUT_SHAPE), "train"
)
val_ds = ImageDatastore(
    os.path.join(PATH, "train"), val_df, (INPUT_SHAPE, INPUT_SHAPE), "val"
)




## === cell 10
def load_image(path, label):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=NUM_CHANNEL)  # uint8, RGB
    img = tf.image.resize(img, [INPUT_SHAPE, INPUT_SHAPE], method="bilinear")
    img = tf.cast(img, tf.uint8)  # back to uint8
    return img, label


def load_image_no_label(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=NUM_CHANNEL)
    img = tf.image.resize(img, [INPUT_SHAPE, INPUT_SHAPE], method="bilinear")
    img = tf.cast(img, tf.uint8)
    return img


data_augmentation = tf.keras.Sequential(
    [
        tf.keras.layers.RandomFlip(),
        tf.keras.layers.RandomRotation(0.2),
        tf.keras.layers.RandomZoom(0.2),
    ]
)

AUTOTUNE = tf.data.AUTOTUNE

train = tf.data.Dataset.from_tensor_slices((train_ds.image_paths, train_ds.labels))
train = (
    train.map(load_image, num_parallel_calls=AUTOTUNE)
    .cache()
    .map(
        lambda X, y: (data_augmentation(X, training=True), y),
        num_parallel_calls=AUTOTUNE,
    )
    .batch(BATCH_SIZE, drop_remainder=True)
    .prefetch(AUTOTUNE)
)

val = tf.data.Dataset.from_tensor_slices((val_ds.image_paths, val_ds.labels))
val = (
    val.map(load_image, num_parallel_calls=AUTOTUNE)
    .cache()
    .batch(BATCH_SIZE, drop_remainder=True)
    .prefetch(AUTOTUNE)
)




## === cell 11
def build_model():
    input_shape = (INPUT_SHAPE, INPUT_SHAPE, NUM_CHANNEL)

    base = tf.keras.applications.MobileNetV2(
        input_shape=input_shape, weights="imagenet", include_top=False, pooling="avg"
    )
    base.trainable = False

    inputs = tf.keras.Input(shape=input_shape)
    x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
    x = base(x)
    x = tf.keras.layers.Dense(512)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.Dropout(0.0)(x)
    outputs = tf.keras.layers.Dense(NUM_CLASS, activation="softmax")(x)

    model = tf.keras.Model(inputs, outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
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
    epochs=15,  # extended training for better convergence
    validation_data=val,
    callbacks=[
        CustomCallback(monitor="val_loss", factor=0.5, patience=10, min_lr=2e-6)
    ],
)



## === cell 14
with strategy.scope():
    base_layer = None
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model):
            base_layer = layer
            break
    if base_layer is not None:
        base_layer.trainable = True

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-5),
        loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
        metrics=[tf.keras.metrics.CategoricalAccuracy(name="accuracy")],
    )

    fine_tune_history = model.fit(
        train,
        epochs=10,  # longer fine‑tuning phase
        validation_data=val,
        callbacks=[
            CustomCallback(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6)
        ],
    )



## === cell 15
sub_test_df = pd.read_csv(
    "/kaggle/input/dog-breed-identification/sample_submission.csv", index_col="id"
)



## === cell 16
sub_test_ds = ImageDatastore(
    os.path.join(PATH, "test"), sub_test_df, (INPUT_SHAPE, INPUT_SHAPE), "test"
)



## === cell 17
sub_test = tf.data.Dataset.from_tensor_slices(sub_test_ds.image_paths)
sub_test = (
    sub_test.map(load_image_no_label, num_parallel_calls=AUTOTUNE)
    .batch(BATCH_SIZE, drop_remainder=False)
    .prefetch(AUTOTUNE)
)



## === cell 18
pred = model.predict(sub_test, verbose=0)



## === cell 19
sub_test_df.iloc[:, 1:] = pred



## === cell 20
output_path = os.path.join("/kaggle/working", "submission.csv")
sub_test_df.to_csv(output_path)
sub_test_df.head()
