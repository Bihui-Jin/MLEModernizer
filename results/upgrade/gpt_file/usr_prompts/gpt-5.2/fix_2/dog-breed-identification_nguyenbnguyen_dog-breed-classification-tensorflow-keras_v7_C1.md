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

3.13

# 3. Installed packages

No external packages required in the script and installed.

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

1.581792533337866

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")



## === cell 1
from pathlib import Path
import pandas as pd

pd.set_option("display.max_columns", None)

KAGGLE_DATA_DIR = Path("/kaggle/input/dog-breed-identification")

labels_df = pd.read_csv(KAGGLE_DATA_DIR / "labels.csv")

filenames = [
    str(KAGGLE_DATA_DIR / f"train/{filename}.jpg") for filename in labels_df["id"]
]



## === cell 2
labels = labels_df["breed"].to_numpy()
len(labels) == len(filenames)



## === cell 3
filenames[:5]



## === cell 4
len(filenames)



## === cell 5
pass



## === cell 6
import matplotlib.pyplot as plt

import numpy as np
from IPython.display import Image




## === cell 7
unique_breeds = np.unique(labels)
unique_breeds[:10]



## === cell 8
labels_df.head()



## === cell 9
labels_df.describe()



## === cell 10
try:
    labels_df["breed"].value_counts(ascending=True).plot.barh(figsize=(20, 30))
    plt.tight_layout()
    plt.show()
except Exception as e:
    print("Plot skipped due to environment limitation:", repr(e))



## === cell 11
example_dog_breed_name = labels_df[
    labels_df["id"] == "0021f9ceb3235effd7fcde7f7538ed62"
]["breed"].values[0]

print(f"{example_dog_breed_name}")

example_dog_breed = Image(
    KAGGLE_DATA_DIR / "train/0021f9ceb3235effd7fcde7f7538ed62.jpg"
)
example_dog_breed



## === cell 12
image = plt.imread(filenames[0])
image.shape



## === cell 13
image[:1]



## === cell 14
pass



## === cell 15
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer

import tensorflow as tf

IMG_WIDTH = 224
IMG_HEIGHT = IMG_WIDTH
IMG_CHANNELS = 3

BATCH_SIZE = 32


@tf.autograph.experimental.do_not_convert
def process_image(image_path: str):
    """
    Take an image file path and turns the image into a Tensor
    """
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, size=[IMG_WIDTH, IMG_HEIGHT])
    return image


@tf.autograph.experimental.do_not_convert
def get_image_label(image_path: str, label):
    """
    Takes an image file path name and the associated label,
    processes the image and returns a tuple (image, label)
    """
    image = process_image(image_path)
    return image, label


def create_data_batches(
    X, y=None, batch_size=BATCH_SIZE, valid_data=False, test_data=False
):
    """
    Creates batches of data out of image (X) and label (y) pairs.
    Shuffles the data if it's training data but doesn't shuffle if it's validation data.
    Also accepts test data as input (no labels).
    """
    if test_data:
        print("Creating test data batches...")
        data = tf.data.Dataset.from_tensor_slices(
            (tf.constant(X))
        )  # only filepaths (no labels)
        data_batch = (
            data.map(process_image, num_parallel_calls=tf.data.AUTOTUNE)
            .batch(batch_size)
            .prefetch(tf.data.AUTOTUNE)
        )
        return data_batch

    if valid_data:
        print("Creating validation data batches...")
        data = tf.data.Dataset.from_tensor_slices((tf.constant(X), tf.constant(y)))
        data_batch = (
            data.map(get_image_label, num_parallel_calls=tf.data.AUTOTUNE)
            .batch(batch_size)
            .prefetch(tf.data.AUTOTUNE)
        )
        return data_batch

    print("Creating training data batches...")
    data = tf.data.Dataset.from_tensor_slices((tf.constant(X), tf.constant(y)))
    data_batch = (
        data.shuffle(buffer_size=len(X))
        .map(get_image_label, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )
    return data_batch


random_state = 42
random_seed = random_state

tf.random.set_seed(random_seed)
np.random.seed(random_seed)

print("TF version:", tf.__version__)

if tf.config.list_physical_devices("GPU"):
    print("GPU enabled")
else:
    print("GPU is not available, switch to CPU")

lb = LabelBinarizer()
encoded_labels = lb.fit_transform(labels)

print(f"{encoded_labels[:1] = }")

y_class = labels_df["breed"].to_numpy()

X_train, X_test, y_train, y_test = train_test_split(
    filenames,
    encoded_labels,
    test_size=0.1,
    random_state=random_state,
    stratify=y_class,
)

y_train_class_idx = np.argmax(y_train, axis=1)
X_train, X_valid, y_train, y_valid = train_test_split(
    X_train, y_train, test_size=0.1, random_state=7, stratify=y_train_class_idx
)

train_data = create_data_batches(X_train, y_train)
valid_data = create_data_batches(X_valid, y_valid, valid_data=True)
test_data_ = create_data_batches(X_test, y_test, valid_data=True)



## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 16
train_data.element_spec, valid_data.element_spec




## === cell 17
def show_25_images(images, labels):
    """
    Displays a plot of 25 images and their labels from a data batch.
    """
    plt.figure(figsize=(15, 10))
    n = min(25, len(images))
    for i in range(n):
        ax = plt.subplot(5, 5, i + 1)
        plt.imshow(images[i])
        plt.title(unique_breeds[np.argmax(labels[i])])
        plt.axis("off")
    plt.tight_layout()




## === cell 18
try:
    train_images, train_labels = next(train_data.as_numpy_iterator())
    show_25_images(train_images, train_labels)
    plt.show()
except Exception as e:
    print("Image grid skipped:", repr(e))



## === cell 19
try:
    valid_images, valid_labels = next(valid_data.as_numpy_iterator())
    show_25_images(valid_images, valid_labels)
    plt.show()
except Exception as e:
    print("Image grid skipped:", repr(e))



## === cell 20
try:
    test_images, test_labels = next(test_data_.as_numpy_iterator())
    show_25_images(test_images, test_labels)
    plt.show()
except Exception as e:
    print("Image grid skipped:", repr(e))



## === cell 21
pass



## === cell 22
import datetime
import tensorflow_hub as hub
from tensorflow.keras import layers


def save_model(model, file_name, path_folder="./", include_datetime=True):
    """
    Saves a given model in a models directory and appends a suffix (string).
    """
    suffix = None
    if include_datetime:
        suffix = f'{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}'

    if suffix:
        full_file_name = f"{suffix}-{file_name}.keras"
    else:
        full_file_name = f"{file_name}.keras"

    model_path = Path(path_folder) / full_file_name
    print(f"Saving model to: {model_path}...")
    model.save(model_path)
    return model_path


def load_model(model_path):
    """
    Loads a saved model from a specified path.
    """
    print(f"Loading saved model from: {model_path}")
    model = tf.keras.models.load_model(model_path)
    return model


def MobileNetV2(
    input_shape=(IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS),
    num_classes=len(unique_breeds),
    base_model_trainable=False,
    data_augmentation=None,
):
    hub_layer = hub.KerasLayer(
        "https://www.kaggle.com/models/google/mobilenet-v2/TensorFlow1/035-224-feature-vector/2",
        trainable=base_model_trainable,
    )

    inputs = layers.Input(shape=input_shape)
    x = inputs

    if data_augmentation is not None:
        x = data_augmentation(x)

    x = hub_layer(x)
    outputs = layers.Dense(num_classes)(x)
    return tf.keras.Model(inputs=inputs, outputs=outputs)


def create_model(learning_rate=1e-3):
    model = MobileNetV2()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )
    return model


def data_augmenter():
    """
    Create a Sequential model composed of augmentation layers.
    """
    data_augmentation = tf.keras.models.Sequential()
    data_augmentation.add(layers.RandomFlip("horizontal"))
    data_augmentation.add(layers.RandomRotation(0.2))
    return data_augmentation


lr = 1e-2
model = create_model(learning_rate=lr)

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss", factor=0.3, patience=2, min_lr=1e-7
)

val_acc_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_accuracy",
    restore_best_weights=True,
    patience=10,
    start_from_epoch=5,
)

model.summary(show_trainable=True)



## --- ERROR in cell 22, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3635857084.py in <cell line: 0>()
     78 
     79 lr = 1e-2
---> 80 model = create_model(learning_rate=lr)
     81 
     82 reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(

/tmp/ipykernel_11/3635857084.py in create_model(learning_rate)
     58 # Fix: don't instantiate the model as a default argument (it runs at definition time).
     59 def create_model(learning_rate=1e-3):
---> 60     model = MobileNetV2()
     61     model.compile(
     62         optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),

/tmp/ipykernel_11/3635857084.py in MobileNetV2(input_shape, num_classes, base_model_trainable, data_augmentation)
     51         x = data_augmentation(x)
     52 
---> 53     x = hub_layer(x)
     54     outputs = layers.Dense(num_classes)(x)
     55     return tf.keras.Model(inputs=inputs, outputs=outputs)

/usr/local/lib/python3.11/dist-packages/tf_keras/src/utils/traceback_utils.py in error_handler(*args, **kwargs)
     68             # To get the full stack trace, call:
     69             # `tf.debugging.disable_traceback_filtering()`
---> 70             raise e.with_traceback(filtered_tb) from None
     71         finally:
     72             del filtered_tb

/usr/local/lib/python3.11/dist-packages/tensorflow_hub/keras_layer.py in call(self, inputs, training)
    240     # or else Keras' global `learning_phase`, which might actually be a tensor.
    241     if not self._has_training_argument:
--> 242       result = f()
    243     else:
    244       if self.trainable:

TypeError: Exception encountered when calling layer 'keras_layer' (type KerasLayer).

Binding inputs to tf.function failed due to `too many positional arguments`. Received args: (<KerasTensor shape=(None, 224, 224, 3), dtype=float32, sparse=False, name=keras_tensor>,) and kwargs: {} for signature: () -> Dict[['default', TensorSpec(shape=(None, 1280), dtype=tf.float32, name=None)]].
Fallback to flat signature also failed due to: pruned(images): expected argument #0(zero-based) to be a Tensor; got KerasTensor (<KerasTensor shape=(None, 224, 224, 3), dtype=float32, sparse=False, name=keras_tensor>).

Call arguments received by layer 'keras_layer' (type KerasLayer):
  • inputs=<KerasTensor shape=(None, 224, 224, 3), dtype=float32, sparse=False, name=keras_tensor>
  • training=None

## === cell 23
pass



## === cell 24
pass



## === cell 25
epochs = 20

history = model.fit(
    train_data,
    validation_data=valid_data,
    callbacks=[reduce_lr, val_acc_stopping],
    epochs=epochs,
)



## --- ERROR in cell 25, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3160199677.py in <cell line: 0>()
      1 epochs = 20
      2 
----> 3 history = model.fit(
      4     train_data,
      5     validation_data=valid_data,

NameError: name 'model' is not defined

## === cell 26
pass



## === cell 27
acc = [0.0] + history.history["accuracy"]
val_acc = [0.0] + history.history["val_accuracy"]

loss = history.history["loss"]
val_loss = history.history["val_loss"]

plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, label="Training Accuracy")
plt.plot(val_acc, label="Validation Accuracy")
plt.legend(loc="lower right")
plt.ylabel("Accuracy")
plt.ylim([min(plt.ylim()), 1])
plt.title("Training and Validation Accuracy")

plt.subplot(2, 1, 2)
plt.plot(loss, label="Training Loss")
plt.plot(val_loss, label="Validation Loss")
plt.legend(loc="upper right")
plt.ylabel("Cross Entropy")
plt.ylim([0, max(1.0, max(val_loss) if len(val_loss) else 1.0)])
plt.title("Training and Validation Loss")
plt.xlabel("epoch")
plt.show()



## --- ERROR in cell 27, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3506780988.py in <cell line: 0>()
----> 1 acc = [0.0] + history.history["accuracy"]
      2 val_acc = [0.0] + history.history["val_accuracy"]
      3 
      4 loss = history.history["loss"]
      5 val_loss = history.history["val_loss"]

NameError: name 'history' is not defined

## === cell 28
hub_layer = None
for lyr in model.layers:
    if isinstance(lyr, hub.KerasLayer):
        hub_layer = lyr
        break

if hub_layer is None:
    raise RuntimeError("Could not find hub.KerasLayer inside the model to fine-tune.")

hub_layer.trainable = True
model.summary(show_trainable=True)



## --- ERROR in cell 28, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2111683724.py in <cell line: 0>()
      2 # The original code used model.layers[-2] which may point to the hub layer, but be explicit.
      3 hub_layer = None
----> 4 for lyr in model.layers:
      5     if isinstance(lyr, hub.KerasLayer):
      6         hub_layer = lyr

NameError: name 'model' is not defined

## === cell 29
loss_function = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
optimizer = tf.keras.optimizers.Adam(lr * 1e-3)
metrics = [tf.keras.metrics.CategoricalAccuracy(name="accuracy", dtype=np.float32)]

model.compile(optimizer=optimizer, loss=loss_function, metrics=metrics)



## --- ERROR in cell 29, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2353691805.py in <cell line: 0>()
      4 
      5 # Re-compile after changing trainable flags
----> 6 model.compile(optimizer=optimizer, loss=loss_function, metrics=metrics)
      7 

NameError: name 'model' is not defined

## === cell 30
fine_tune_epochs = 50
total_epochs = epochs + fine_tune_epochs

history_fine = model.fit(
    train_data,
    validation_data=valid_data,
    epochs=total_epochs,
    callbacks=[reduce_lr, val_acc_stopping],
    initial_epoch=history.epoch[-1],
)



## --- ERROR in cell 30, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3082078555.py in <cell line: 0>()
      2 total_epochs = epochs + fine_tune_epochs
      3 
----> 4 history_fine = model.fit(
      5     train_data,
      6     validation_data=valid_data,

NameError: name 'model' is not defined

## === cell 31
acc += history_fine.history["accuracy"]
val_acc += history_fine.history["val_accuracy"]

loss += history_fine.history["loss"]
val_loss += history_fine.history["val_loss"]



## --- ERROR in cell 31, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/805784071.py in <cell line: 0>()
----> 1 acc += history_fine.history["accuracy"]
      2 val_acc += history_fine.history["val_accuracy"]
      3 
      4 loss += history_fine.history["loss"]
      5 val_loss += history_fine.history["val_loss"]

NameError: name 'acc' is not defined

## === cell 32
plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, label="Training Accuracy")
plt.plot(val_acc, label="Validation Accuracy")
plt.ylim([0, 1])
plt.plot([epochs - 1, epochs - 1], plt.ylim(), label="Start Fine Tuning")
plt.legend(loc="lower right")
plt.title("Training and Validation Accuracy")

plt.subplot(2, 1, 2)
plt.plot(loss, label="Training Loss")
plt.plot(val_loss, label="Validation Loss")
plt.ylim([0, max(1.0, max(val_loss) if len(val_loss) else 1.0)])
plt.plot([epochs - 1, epochs - 1], plt.ylim(), label="Start Fine Tuning")
plt.legend(loc="upper right")
plt.title("Training and Validation Loss")
plt.xlabel("epoch")
plt.show()




## --- ERROR in cell 32, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2317997520.py in <cell line: 0>()
      1 plt.figure(figsize=(8, 8))
      2 plt.subplot(2, 1, 1)
----> 3 plt.plot(acc, label="Training Accuracy")
      4 plt.plot(val_acc, label="Validation Accuracy")
      5 plt.ylim([0, 1])

NameError: name 'acc' is not defined

## === cell 33
def get_pred_label(prediction_probabilities):
    """
    Turns an array of prediction probabilities into a label.
    """
    return unique_breeds[np.argmax(prediction_probabilities)]


preds = model.predict(test_data_, verbose=0)

index = 0
print(
    f"Max value (probability of prediction): {np.max(tf.nn.softmax(preds[index]).numpy())}"
)
print("Sum:", float(np.sum(tf.nn.softmax(preds[index]).numpy())))
print("Max index:", int(np.argmax(preds[index])))
print("Predicted label:", unique_breeds[np.argmax(preds[index])])
print("Actual label:", unique_breeds[np.argmax(y_test[index])])

pred_label = get_pred_label(preds[7])
print(f"{pred_label = }")




## --- ERROR in cell 33, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1273890189.py in <cell line: 0>()
      6 
      7 
----> 8 preds = model.predict(test_data_, verbose=0)
      9 
     10 index = 0

NameError: name 'model' is not defined

## === cell 34
def unbatchify(data):
    """
    Takes a batched dataset of (image, label) Tensors and returns separate arrays
    of images and labels.
    """
    images = []
    labels_ = []

    for image, label in data.unbatch().as_numpy_iterator():
        images.append(image)
        labels_.append(unique_breeds[np.argmax(label)])

    return images, labels_


test_images, test_labels = unbatchify(test_data_)
test_images[0], test_labels[0]




## === cell 35
def plot_pred(prediction_probabilities, labels, images, n=1):
    """
    View the prediction, ground truth and image for sample n
    """
    pred_prob, true_label, image = prediction_probabilities[n], labels[n], images[n]

    pred_label = get_pred_label(pred_prob)

    plt.imshow(image)
    plt.xticks([])
    plt.yticks([])

    color = "green" if pred_label == true_label else "red"
    plt.title(
        "{} {:2.0f}% {}".format(pred_label, np.max(pred_prob) * 100, true_label),
        color=color,
    )


plot_pred(
    prediction_probabilities=tf.nn.softmax(preds, axis=1).numpy(),
    labels=test_labels,
    images=test_images,
    n=10,
)
plt.show()




## --- ERROR in cell 35, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1057690906.py in <cell line: 0>()
     19 
     20 plot_pred(
---> 21     prediction_probabilities=tf.nn.softmax(preds, axis=1).numpy(),
     22     labels=test_labels,
     23     images=test_images,

NameError: name 'preds' is not defined

## === cell 36
def plot_pred_conf(prediction_probabilities, labels, n=1):
    """
    Plus the top 10 highest prediction confidences along with the truth label for sample n.
    """
    pred_prob, true_label = prediction_probabilities[n], labels[n]
    pred_label = get_pred_label(pred_prob)

    top_10_pred_indexes = pred_prob.argsort()[-10:][::-1]
    top_10_pred_values = pred_prob[top_10_pred_indexes]
    top_10_pred_labels = unique_breeds[top_10_pred_indexes]

    top_10_plot = plt.barh(
        np.arange(len(top_10_pred_labels)), top_10_pred_values, color="grey"
    )
    plt.gca().invert_yaxis()
    plt.yticks(np.arange(len(top_10_pred_labels)), labels=top_10_pred_labels)

    if np.isin(true_label, top_10_pred_labels):
        top_10_plot[np.argmax(top_10_pred_labels == true_label)].set_color("green")


prediction_probabilities = tf.nn.softmax(preds, axis=1).numpy()
plot_pred_conf(
    prediction_probabilities=prediction_probabilities, labels=test_labels, n=42
)
plt.show()



## --- ERROR in cell 36, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/540749921.py in <cell line: 0>()
     20 
     21 
---> 22 prediction_probabilities = tf.nn.softmax(preds, axis=1).numpy()
     23 plot_pred_conf(
     24     prediction_probabilities=prediction_probabilities, labels=test_labels, n=42

NameError: name 'preds' is not defined

## === cell 37
i_multiplier = 30
num_rows = 3
num_cols = 2
num_images = num_rows * num_cols
plt.figure(figsize=(10 * num_cols, 5 * num_rows))
for i in range(num_images):
    plt.subplot(num_rows, 2 * num_cols, 2 * i + 1)
    plot_pred(
        prediction_probabilities=prediction_probabilities,
        labels=test_labels,
        images=test_images,
        n=i + i_multiplier,
    )
    plt.subplot(num_rows, 2 * num_cols, 2 * i + 2)
    plot_pred_conf(
        prediction_probabilities=prediction_probabilities,
        labels=test_labels,
        n=i + i_multiplier,
    )
plt.tight_layout(h_pad=1.0)
plt.show()



## --- ERROR in cell 37, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/36736674.py in <cell line: 0>()
      7     plt.subplot(num_rows, 2 * num_cols, 2 * i + 1)
      8     plot_pred(
----> 9         prediction_probabilities=prediction_probabilities,
     10         labels=test_labels,
     11         images=test_images,

NameError: name 'prediction_probabilities' is not defined

## === cell 38
import seaborn as sns
from sklearn.metrics import confusion_matrix

cf_matrix = confusion_matrix(
    test_labels, [get_pred_label(pred) for pred in prediction_probabilities]
)

plt.figure(figsize=(30, 15))
ax = sns.heatmap(
    cf_matrix,
    cmap="Reds",
    linewidths=1,
    xticklabels=lb.classes_,
    yticklabels=lb.classes_,
)
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
ax.xaxis.tick_top()
plt.tight_layout()
plt.xticks(rotation=90)
plt.show()



## --- ERROR in cell 38, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/38176160.py in <cell line: 0>()
      3 
      4 cf_matrix = confusion_matrix(
----> 5     test_labels, [get_pred_label(pred) for pred in prediction_probabilities]
      6 )
      7 

NameError: name 'prediction_probabilities' is not defined

## === cell 39
pass



## === cell 40
save_model(model, file_name="mobilenetv2-Adam", include_datetime=False)



## --- ERROR in cell 40, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/851370407.py in <cell line: 0>()
----> 1 save_model(model, file_name="mobilenetv2-Adam", include_datetime=False)
      2 

NameError: name 'model' is not defined

## === cell 41
pass



## === cell 42
submitted_data_dir = KAGGLE_DATA_DIR / "test"
submitted_img_names = sorted(
    [str(p) for p in submitted_data_dir.iterdir() if p.suffix.lower() == ".jpg"]
)
submitted_img_names[:2]



## === cell 43
submitted_data = create_data_batches(submitted_img_names, test_data=True)
next(submitted_data.as_numpy_iterator())[0][0][0]



## === cell 44
next(iter(sorted(submitted_data_dir.iterdir()))).stem



## === cell 45
sample_sub = pd.read_csv(KAGGLE_DATA_DIR / "sample_submission.csv")
sample_ids = sample_sub["id"].tolist()
class_cols = [c for c in sample_sub.columns if c != "id"]

id_to_path = {Path(p).stem: p for p in submitted_img_names}
ordered_paths = [id_to_path[_id] for _id in sample_ids]

submitted_data_ordered = create_data_batches(ordered_paths, test_data=True)

submit_logits = model.predict(submitted_data_ordered, verbose=1)
submit_preds = tf.nn.softmax(submit_logits, axis=1).numpy()

model_class_order = list(lb.classes_)
col_index = {c: i for i, c in enumerate(model_class_order)}
reordered = np.zeros((submit_preds.shape[0], len(class_cols)), dtype=np.float32)
for j, c in enumerate(class_cols):
    reordered[:, j] = submit_preds[:, col_index[c]]

out_df = pd.DataFrame(reordered, columns=class_cols)
out_df.insert(0, "id", sample_ids)
out_df.head()



## --- ERROR in cell 45, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4127744542.py in <cell line: 0>()
     11 submitted_data_ordered = create_data_batches(ordered_paths, test_data=True)
     12 
---> 13 submit_logits = model.predict(submitted_data_ordered, verbose=1)
     14 submit_preds = tf.nn.softmax(submit_logits, axis=1).numpy()
     15 

NameError: name 'model' is not defined

## === cell 46
out_df.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", out_df.shape)
print(out_df.head())

## --- ERROR in cell 46, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2385065931.py in <cell line: 0>()
----> 1 out_df.to_csv("submission.csv", index=False)
      2 print("Wrote submission.csv with shape:", out_df.shape)
      3 print(out_df.head())

NameError: name 'out_df' is not defined
