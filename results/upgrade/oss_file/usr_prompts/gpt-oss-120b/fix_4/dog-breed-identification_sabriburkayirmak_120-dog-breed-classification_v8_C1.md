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

0.68346

# 6. Current score

0.78183

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 0.75706) has done: 'I fix the protobuf import error by setting the protocol‑buffers implementation before loading TensorFlow, and correct the learning‑rate update in the custom callback to use `optimizer.learning_rate` (the proper attribute in TF 2.x). These changes remove the runtime crashes and let the training proceed, which should bring the validation loss closer to the target score while keeping the original model architecture unchanged.'
- What this solution (achieved 0.75916) has done: 'The fix updates the learning‑rate handling in the custom callback to use TensorFlow’s `assign` (avoiding the string‑attribute error), lowers the initial Adam learning rate for better generalisation, and correctly writes predictions into the submission DataFrame without overwriting the ID column. All changes retain the original model architecture and training flow while addressing the runtime crash and ensuring a valid `.csv` submission file.'
- What this solution (achieved 0.78183) has done: 'I reduced the dropout rate in the model to lessen under‑fitting, then added a two‑stage training: first train with the MobileNetV2 base frozen, then unfreeze the base, re‑compile with a smaller learning‑rate and continue fine‑tuning. This modest adjustment should lower the validation log‑loss toward the target while keeping the original architecture and data pipeline intact, and it still writes a proper `.csv` submission file.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import backend as K



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

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
            labels = ["" for i in range(len(image_paths))]
        else:
            labels = pd.get_dummies(self.csv.breed).astype("uint8").to_numpy()
        return image_paths, labels

    def __call__(self):
        pairs = list(zip(self.image_paths, self.labels))
        for image_path, label in pairs:
            image = cv2.imread(image_path)
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
                    if isinstance(lr_tensor, tf.Variable):
                        lr = lr_tensor.numpy()
                    else:
                        lr = float(lr_tensor)
                    new_lr = lr * self.factor
                    if new_lr < self.min_lr:
                        new_lr = self.min_lr
                        self.job = 1
                    if isinstance(lr_tensor, tf.Variable):
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
data_augmentation = tf.keras.Sequential(
    [
        tf.keras.layers.RandomFlip(),
        tf.keras.layers.RandomRotation(0.2),
        tf.keras.layers.RandomZoom(0.2),
    ]
)



## === cell 11
OUTPUT_SIGNATURE = (
    tf.TensorSpec(shape=(INPUT_SHAPE, INPUT_SHAPE, NUM_CHANNEL), dtype="uint8"),
    tf.TensorSpec(shape=(NUM_CLASS,), dtype="uint8"),
)

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
    .cache()
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
    x = tf.keras.layers.Dense(512)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(NUM_CLASS, activation="softmax")(x)

    model = tf.keras.Model(inputs, outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),  # initial LR
        loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
        metrics=[tf.keras.metrics.CategoricalAccuracy(name="accuracy")],
    )
    return model




## === cell 13
with strategy.scope():
    model = build_model()
model.summary()



## === cell 14
history = model.fit(
    train,
    epochs=100,
    validation_data=val,
    callbacks=[
        CustomCallback(monitor="val_loss", factor=0.5, patience=10, min_lr=2e-6)
    ],
)



## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_56/2153569748.py in <cell line: 0>()
----> 1 history = model.fit(
      2     train,
      3     epochs=100,
      4     validation_data=val,
      5     callbacks=[

/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py in error_handler(*args, **kwargs)
    120             # To get the full stack trace, call:
    121             # `keras.config.disable_traceback_filtering()`
--> 122             raise e.with_traceback(filtered_tb) from None
    123         finally:
    124             del filtered_tb

/tmp/ipykernel_56/4152818415.py in on_epoch_end(self, epoch, logs)
     36                         lr = lr_tensor.numpy()
     37                     else:
---> 38                         lr = float(lr_tensor)
     39                     new_lr = lr * self.factor
     40                     if new_lr < self.min_lr:

TypeError: float() argument must be a string or a real number, not 'MirroredVariable'

## === cell 15
base_layer = None
for layer in model.layers:
    if isinstance(layer, tf.keras.Model):
        base_layer = layer
        break
if base_layer is not None:
    base_layer.trainable = True

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-5),  # smaller LR for fine‑tuning
    loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
    metrics=[tf.keras.metrics.CategoricalAccuracy(name="accuracy")],
)



## === cell 16
fine_tune_history = model.fit(
    train,
    epochs=30,
    validation_data=val,
    callbacks=[CustomCallback(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6)],
)



## --- ERROR in cell 16, traceback:
---------------------------------------------------------------------------
RuntimeError                              Traceback (most recent call last)
/tmp/ipykernel_56/1443433864.py in <cell line: 0>()
----> 1 fine_tune_history = model.fit(
      2     train,
      3     epochs=30,
      4     validation_data=val,
      5     callbacks=[CustomCallback(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6)],

/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py in error_handler(*args, **kwargs)
    120             # To get the full stack trace, call:
    121             # `keras.config.disable_traceback_filtering()`
--> 122             raise e.with_traceback(filtered_tb) from None
    123         finally:
    124             del filtered_tb

/usr/local/lib/python3.11/dist-packages/tensorflow/python/distribute/distribute_lib.py in _wrong_strategy_scope(strategy, context)
    301         (strategy,))
    302   else:
--> 303     raise RuntimeError(
    304         "Mixing different tf.distribute.Strategy objects: %s is not %s" %
    305         (context.strategy, strategy))

RuntimeError: Mixing different tf.distribute.Strategy objects: <tensorflow.python.distribute.mirrored_strategy.MirroredStrategy object at 0x7fcd1e18b190> is not <tensorflow.python.distribute.distribute_lib._DefaultDistributionStrategy object at 0x7fcd1db97250>

## === cell 17
sub_test_df = pd.read_csv(
    "/kaggle/input/dog-breed-identification/sample_submission.csv", index_col="id"
)



## === cell 18
sub_test_ds = ImageDatastore(
    os.path.join(PATH, "test"), sub_test_df, (INPUT_SHAPE, INPUT_SHAPE), "test"
)



## === cell 19
SUB_OUTPUT_SIGNATURE = tf.TensorSpec(
    shape=(INPUT_SHAPE, INPUT_SHAPE, NUM_CHANNEL), dtype="uint8"
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



## === cell 20
pred = model.predict(sub_test)



## === cell 21
sub_test_df.iloc[:, :] = pred



## === cell 22
sub_test_df.to_csv(os.path.join("/kaggle", "working", "submission.csv"))
sub_test_df.head()
