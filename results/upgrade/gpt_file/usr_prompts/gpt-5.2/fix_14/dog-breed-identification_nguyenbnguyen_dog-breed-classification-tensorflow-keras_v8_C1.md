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

# 5. Code solution

## === cell 0
import os, sys
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

try:
    import tensorflow as tf  # noqa: F401
except Exception as e:
    os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
    os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "2")
    try:
        import tensorflow as tf  # noqa: F401
    except Exception as e2:
        raise RuntimeError(
            "TensorFlow failed to import. Fallback to pure-Python protobuf also failed."
        ) from e2




## === cell 1
import multiprocessing

try:
    _cpu = multiprocessing.cpu_count()
    tf.config.threading.set_intra_op_parallelism_threads(max(1, _cpu))
    tf.config.threading.set_inter_op_parallelism_threads(max(1, min(4, _cpu)))
except Exception:
    pass




## === cell 2
import pandas as pd

pd.set_option("display.max_columns", None)

_CANDIDATE_DIRS = [
    Path("/kaggle/input/dog-breed-identification"),
    Path("/kaggle/input/dog-breed-identification/dog-breed-identification"),
]
KAGGLE_DATA_DIR = next(
    (p for p in _CANDIDATE_DIRS if (p / "labels.csv").exists()), _CANDIDATE_DIRS[0]
)

labels_df = pd.read_csv(KAGGLE_DATA_DIR / "labels.csv")
filenames = [
    str(KAGGLE_DATA_DIR / f"train/{filename}.jpg") for filename in labels_df["id"]
]




## === cell 3
labels = labels_df["breed"].to_numpy()
len(labels) == len(filenames)




## === cell 4
filenames[:5]




## === cell 5
len(filenames)




## === cell 6
import random
import numpy as np

random_state = 42
random.seed(random_state)
np.random.seed(random_state)
tf.random.set_seed(random_state)

try:
    tf.config.experimental.enable_op_determinism(True)
except Exception:
    pass




## === cell 7
import matplotlib.pyplot as plt

try:
    from IPython.display import Image  # noqa: F401
except Exception:
    Image = None  # type: ignore




## === cell 8
unique_breeds = np.unique(labels)
unique_breeds[:10]




## === cell 9
labels_df.head()




## === cell 10
labels_df.describe()




## === cell 11
if False:
    labels_df["breed"].value_counts(ascending=True).plot.barh(figsize=(20, 30))
    plt.tight_layout()
    plt.show()




## === cell 12
if False and Image is not None:
    example_dog_breed_name = labels_df[
        labels_df["id"] == "0021f9ceb3235effd7fcde7f7538ed62"
    ]["breed"].values[0]
    print(f"{example_dog_breed_name}")
    example_dog_breed = Image(
        KAGGLE_DATA_DIR / "train/0021f9ceb3235effd7fcde7f7538ed62.jpg"
    )
    example_dog_breed




## === cell 13
if False:
    image = plt.imread(filenames[0])
    image.shape




## === cell 14
if False:
    image[:1]




## === cell 15
print("TF version:", tf.__version__)
if tf.config.list_physical_devices("GPU"):
    print("GPU enabled")
else:
    print("GPU is not available, switch to CPU")




## === cell 16
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer

IMG_WIDTH = 224
IMG_HEIGHT = IMG_WIDTH
IMG_CHANNELS = 3
BATCH_SIZE = 32

_DATASET_OPTIONS = tf.data.Options()
_DATASET_OPTIONS.experimental_deterministic = True
try:
    _DATASET_OPTIONS.experimental_optimization.apply_default_optimizations = True
    _DATASET_OPTIONS.experimental_optimization.map_parallelization = True
    _DATASET_OPTIONS.experimental_optimization.parallel_batch = True
    _DATASET_OPTIONS.experimental_optimization.map_and_batch_fusion = True
except Exception:
    pass


@tf.function(reduce_retracing=True)
def process_image(image_path: tf.Tensor):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.image.resize(
        image,
        size=[IMG_WIDTH, IMG_HEIGHT],
        method=tf.image.ResizeMethod.BILINEAR,
        antialias=False,
    )
    image.set_shape([IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS])
    return image


@tf.function(reduce_retracing=True)
def get_image_label(image_path: tf.Tensor, label: tf.Tensor):
    image = process_image(image_path)
    return image, label


def create_data_batches(
    X, y=None, batch_size=BATCH_SIZE, valid_data=False, test_data=False
):
    if test_data:
        print("Creating test data batches...")
        data = tf.data.Dataset.from_tensor_slices(X).with_options(_DATASET_OPTIONS)
        data = data.map(
            process_image, num_parallel_calls=tf.data.AUTOTUNE, deterministic=True
        )
        data = data.apply(tf.data.experimental.ignore_errors())
        data = data.cache()  # in-memory cache of decoded/resized images
        data = data.batch(batch_size, drop_remainder=False)
        return data.prefetch(tf.data.AUTOTUNE)

    if valid_data:
        print("Creating validation data batches...")
        data = tf.data.Dataset.from_tensor_slices((X, y)).with_options(_DATASET_OPTIONS)
        data = data.map(
            get_image_label, num_parallel_calls=tf.data.AUTOTUNE, deterministic=True
        )
        data = data.apply(tf.data.experimental.ignore_errors())
        data = data.cache()  # in-memory cache of decoded/resized images+labels
        data = data.batch(batch_size, drop_remainder=False)
        return data.prefetch(tf.data.AUTOTUNE)

    print("Creating training data batches...")
    data = tf.data.Dataset.from_tensor_slices((X, y)).with_options(_DATASET_OPTIONS)
    data = data.map(
        get_image_label, num_parallel_calls=tf.data.AUTOTUNE, deterministic=True
    )
    data = data.apply(tf.data.experimental.ignore_errors())
    data = data.cache()  # cache decoded/resized images+labels once

    shuffle_buf = min(int(len(filenames)), 2048)
    data = data.shuffle(
        buffer_size=shuffle_buf, seed=random_state, reshuffle_each_iteration=True
    )
    data = data.batch(batch_size, drop_remainder=False)
    return data.prefetch(tf.data.AUTOTUNE)


lb = LabelBinarizer()
encoded_labels = lb.fit_transform(labels)
print(f"{encoded_labels[:1] = }")

strat = np.argmax(encoded_labels, axis=1)
X_train, X_test, y_train, y_test = train_test_split(
    filenames, encoded_labels, test_size=0.1, random_state=random_state, stratify=strat
)
X_train, X_valid, y_train, y_valid = train_test_split(
    X_train, y_train, test_size=0.1, random_state=7, stratify=np.argmax(y_train, axis=1)
)

train_data = create_data_batches(X_train, y_train)
valid_data = create_data_batches(X_valid, y_valid, valid_data=True)

train_steps = int(np.ceil(len(X_train) / BATCH_SIZE))
valid_steps = int(np.ceil(len(X_valid) / BATCH_SIZE))

train_data_rep = train_data.repeat()
valid_data_rep = valid_data.repeat()




## === cell 17
train_data.element_spec, valid_data.element_spec




## === cell 18
def show_25_images(images, labels):
    plt.figure(figsize=(15, 10))
    for i in range(25):
        ax = plt.subplot(5, 5, i + 1)
        plt.imshow(images[i])
        plt.title(unique_breeds[np.argmax(labels[i])])
        plt.axis("off")
    plt.tight_layout()




## === cell 19
if False:
    train_images, train_labels = next(train_data.as_numpy_iterator())
    show_25_images(train_images, train_labels)




## === cell 20
if False:
    valid_images, valid_labels = next(valid_data.as_numpy_iterator())
    show_25_images(valid_images, valid_labels)




## === cell 21
if False:
    test_data_ = create_data_batches(X_test, y_test, valid_data=True)
    test_images, test_labels = next(test_data_.as_numpy_iterator())
    show_25_images(test_images, test_labels)




## === cell 22
import datetime

try:
    import tensorflow_hub as hub  # noqa: F401
except Exception:
    hub = None  # type: ignore

from tensorflow.keras import layers


def save_model(model, file_name, path_folder="./", include_datetime=True):
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
    print(f"Loading saved model from: {model_path}")
    model = tf.keras.models.load_model(model_path)
    return model


def MobileNetV2(
    input_shape=(IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS),
    num_classes=len(unique_breeds),
    base_model_trainable=False,
    data_augmentation=None,
):
    base_model = tf.keras.applications.mobilenet_v2.MobileNetV2(
        input_shape=input_shape, include_top=False, weights="imagenet", pooling="max"
    )
    base_model.trainable = base_model_trainable

    inputs = layers.Input(input_shape)
    if data_augmentation:
        X = data_augmentation(inputs)
    else:
        X = inputs
    X = base_model(X, training=base_model_trainable)
    outputs = layers.Dense(num_classes)(X)
    return tf.keras.Model(inputs=inputs, outputs=outputs)


def create_model(model=None, learning_rate=1e-3, steps_per_execution=512):
    if model is None:
        model = MobileNetV2()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
        jit_compile=True,
        steps_per_execution=steps_per_execution,
    )
    return model


def data_augmenter():
    data_augmentation = tf.keras.models.Sequential()
    data_augmentation.add(layers.RandomFlip("horizontal"))
    data_augmentation.add(layers.RandomZoom(0.5))
    return data_augmentation


lr = 3e-1
_spe = int(min(512, max(1, train_steps)))
model = create_model(
    MobileNetV2(data_augmentation=data_augmenter()),
    learning_rate=lr,
    steps_per_execution=_spe,
)

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss", factor=0.3, patience=3, min_lr=1e-9
)
val_acc_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_accuracy", restore_best_weights=True, patience=5, start_from_epoch=3
)
model.summary(show_trainable=True)




## === cell 23
train_images, train_labels = next(iter(train_data.take(1)))
_tmp_logits = model(train_images[:2], training=False)
_tmp_probs = tf.nn.softmax(_tmp_logits, axis=1)
print("Softmax sum (first sample):", float(tf.reduce_sum(_tmp_probs[0])))




## === cell 24
epochs = 20

history = model.fit(
    train_data_rep,
    validation_data=valid_data_rep,
    callbacks=[reduce_lr, val_acc_stopping],
    epochs=epochs,
    steps_per_epoch=train_steps,
    validation_steps=valid_steps,
)




## === cell 25
print(
    "Last epoch:", history.epoch[-1], "Last val_loss:", history.history["val_loss"][-1]
)




## === cell 26
if False:
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
    plt.ylim([0, 1.0])
    plt.title("Training and Validation Loss")
    plt.xlabel("epoch")
    plt.show()




## === cell 27
base_model = model.layers[-2]
base_model.trainable = True
model.summary(show_trainable=True)




## === cell 28
loss_function = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
optimizer = tf.keras.optimizers.Adam(learning_rate=lr * 1e-3)
metrics = [tf.keras.metrics.CategoricalAccuracy(name="accuracy", dtype=np.float32)]

model.compile(
    optimizer=optimizer,
    loss=loss_function,
    metrics=metrics,
    jit_compile=True,
    steps_per_execution=_spe,
)




## === cell 29
fine_tune_epochs = 50
total_epochs = epochs + fine_tune_epochs

history_fine = model.fit(
    train_data_rep,
    validation_data=valid_data_rep,
    epochs=total_epochs,
    callbacks=[reduce_lr, val_acc_stopping],
    initial_epoch=history.epoch[-1],
    steps_per_epoch=train_steps,
    validation_steps=valid_steps,
)




## === cell 30
if False:
    acc = [0.0] + history.history["accuracy"]
    val_acc = [0.0] + history.history["val_accuracy"]

    loss = history.history["loss"]
    val_loss = history.history["val_loss"]

    acc += history_fine.history["accuracy"]
    val_acc += history_fine.history["val_accuracy"]
    loss += history_fine.history["loss"]
    val_loss += history_fine.history["val_loss"]

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
    plt.ylim([0, 1.0])
    plt.plot([epochs - 1, epochs - 1], plt.ylim(), label="Start Fine Tuning")
    plt.legend(loc="upper right")
    plt.title("Training and Validation Loss")
    plt.xlabel("epoch")
    plt.show()




## === cell 31
def get_pred_label(prediction_probabilities):
    return unique_breeds[np.argmax(prediction_probabilities)]


test_eval_data = create_data_batches(X_test, test_data=True)
preds = model.predict(test_eval_data, verbose=0)

index = 0
print(f"Max value (probability of prediction): {np.max(tf.nn.softmax(preds[index]))}")
print("Sum:", np.sum(tf.nn.softmax(preds[index])))
print("Max index:", np.argmax(preds[index]))
print("Predicted label:", unique_breeds[np.argmax(preds[index])])
print("Actual label:", unique_breeds[np.argmax(y_test[index])])

pred_label = get_pred_label(preds[7])
print(f"{pred_label = }")




## === cell 32
def unbatchify(data):
    images = []
    labels_out = []
    for image, label in data.unbatch().as_numpy_iterator():
        images.append(image)
        labels_out.append(unique_breeds[np.argmax(label)])
    return images, labels_out


if False:
    test_data_ = create_data_batches(X_test, y_test, valid_data=True)
    test_images, test_labels = unbatchify(test_data_)
    test_images[0], test_labels[0]




## === cell 33
def plot_pred(prediction_probabilities, labels, images, n=1):
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


if False:
    plot_pred(
        prediction_probabilities=tf.nn.softmax(preds, axis=1),
        labels=test_labels,
        images=test_images,
        n=10,
    )




## === cell 34
def plot_pred_conf(prediction_probabilities, labels, n=1):
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


if False:
    prediction_probabilities = tf.nn.softmax(preds, axis=1).numpy()
    plot_pred_conf(
        prediction_probabilities=prediction_probabilities, labels=test_labels, n=42
    )




## === cell 35
if False:
    prediction_probabilities = tf.nn.softmax(preds, axis=1).numpy()
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




## === cell 36
if False:
    import seaborn as sns
    from sklearn.metrics import confusion_matrix

    prediction_probabilities = tf.nn.softmax(preds, axis=1).numpy()
    test_data_ = create_data_batches(X_test, y_test, valid_data=True)
    test_images, test_labels = unbatchify(test_data_)
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




## === cell 37
if False:
    _ = save_model(model, file_name="mobilenetv2-Adam", include_datetime=False)




## === cell 38
sample_sub_path = KAGGLE_DATA_DIR / "sample_submission.csv"
sample_sub = pd.read_csv(sample_sub_path)
breed_columns = [c for c in sample_sub.columns if c != "id"]

submitted_data_dir = KAGGLE_DATA_DIR / "test"
submitted_img_paths = sorted([str(p) for p in submitted_data_dir.glob("*.jpg")])
submitted_ids = [Path(p).stem for p in submitted_img_paths]

model_class_order = list(unique_breeds)
print(
    "Classes in model:",
    len(model_class_order),
    "Classes in sample:",
    len(breed_columns),
)

submitted_data = create_data_batches(submitted_img_paths, test_data=True)




## === cell 39
submit_logits = model.predict(submitted_data, verbose=1)
submit_probs = tf.nn.softmax(submit_logits, axis=1).numpy()

if model_class_order != breed_columns:
    model_idx = {cls: i for i, cls in enumerate(model_class_order)}
    reorder_idx = [model_idx[c] for c in breed_columns]
    submit_probs = submit_probs[:, reorder_idx]

out_df = pd.DataFrame(submit_probs, columns=breed_columns)
out_df.insert(0, "id", submitted_ids)
out_df = out_df.set_index("id").reindex(sample_sub["id"]).reset_index()
out_df.head()




## === cell 40
out_df.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", out_df.shape)
print("Columns:", out_df.columns[:5].tolist(), "...", out_df.columns[-3:].tolist())
print("First id:", out_df.loc[0, "id"])
