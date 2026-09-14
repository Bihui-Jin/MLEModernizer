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
import os

os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", None)

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "4")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "2")



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
SKIP_PLOTS = True

if not SKIP_PLOTS:
    try:
        labels_df["breed"].value_counts(ascending=True).plot.barh(figsize=(20, 30))
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print("Plot skipped due to environment limitation:", repr(e))



## === cell 11
if not SKIP_PLOTS:
    example_dog_breed_name = labels_df[
        labels_df["id"] == "0021f9ceb3235effd7fcde7f7538ed62"
    ]["breed"].values[0]
    print(f"{example_dog_breed_name}")

    example_dog_breed = Image(
        KAGGLE_DATA_DIR / "train/0021f9ceb3235effd7fcde7f7538ed62.jpg"
    )
    example_dog_breed



## === cell 12
if not SKIP_PLOTS:
    image = plt.imread(filenames[0])
    image.shape



## === cell 13
if not SKIP_PLOTS:
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

tf.config.run_functions_eagerly(False)
try:
    tf.config.optimizer.set_jit(True)
except Exception:
    pass

try:
    tf.keras.utils.set_random_seed(42)
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

CACHE_ROOT = Path("/kaggle/working/tfdata_cache")
CACHE_ROOT.mkdir(parents=True, exist_ok=True)

DATASET_OPTIONS = tf.data.Options()
DATASET_OPTIONS.deterministic = True


@tf.function
def process_image(image_path: tf.Tensor):
    """
    Take an image file path and turns the image into a Tensor
    """
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3, dct_method="INTEGER_FAST")
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(
        image, size=[IMG_WIDTH, IMG_HEIGHT], method="bilinear", antialias=False
    )
    image.set_shape([IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS])
    return image


@tf.function
def get_image_label(image_path: tf.Tensor, label: tf.Tensor):
    """
    Takes an image file path name and the associated label,
    processes the image and returns a tuple (image, label)
    """
    image = process_image(image_path)
    return image, label


def create_data_batches(
    X,
    y=None,
    batch_size=BATCH_SIZE,
    valid_data=False,
    test_data=False,
    cache_name=None,
):
    """
    Creates batches of data out of image (X) and label (y) pairs.
    Shuffles the data if it's training data but doesn't shuffle if it's validation data.
    Also accepts test data as input (no labels).
    """
    shuffle_buffer = min(8192, len(X))

    cache_path = None
    if cache_name:
        cache_path = str(CACHE_ROOT / cache_name)

    if test_data:
        print("Creating test data batches...")
        data = tf.data.Dataset.from_tensor_slices(X)
        data = data.with_options(DATASET_OPTIONS)
        data = data.map(
            process_image, num_parallel_calls=tf.data.AUTOTUNE, deterministic=True
        )
        data = data.cache(cache_path)
        data_batch = data.batch(batch_size, drop_remainder=False).prefetch(
            tf.data.AUTOTUNE
        )
        return data_batch

    if valid_data:
        print("Creating validation data batches...")
        data = tf.data.Dataset.from_tensor_slices((X, y))
        data = data.with_options(DATASET_OPTIONS)
        data = data.map(
            get_image_label, num_parallel_calls=tf.data.AUTOTUNE, deterministic=True
        )
        data = data.cache(cache_path)
        data_batch = data.batch(batch_size, drop_remainder=False).prefetch(
            tf.data.AUTOTUNE
        )
        return data_batch

    print("Creating training data batches...")
    data = tf.data.Dataset.from_tensor_slices((X, y))
    data = data.shuffle(
        buffer_size=shuffle_buffer, seed=42, reshuffle_each_iteration=True
    )
    data = data.with_options(DATASET_OPTIONS)
    data = data.map(
        get_image_label, num_parallel_calls=tf.data.AUTOTUNE, deterministic=True
    )
    data = data.cache(cache_path)
    data_batch = data.batch(batch_size, drop_remainder=False).prefetch(tf.data.AUTOTUNE)
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

train_data = create_data_batches(X_train, y_train, cache_name="train.cache")
valid_data = create_data_batches(
    X_valid, y_valid, valid_data=True, cache_name="valid.cache"
)
test_data_ = create_data_batches(
    X_test, y_test, valid_data=True, cache_name="holdout.cache"
)



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
if not SKIP_PLOTS:
    try:
        train_images, train_labels = next(train_data.as_numpy_iterator())
        show_25_images(train_images, train_labels)
        plt.show()
    except Exception as e:
        print("Image grid skipped:", repr(e))



## === cell 19
if not SKIP_PLOTS:
    try:
        valid_images, valid_labels = next(valid_data.as_numpy_iterator())
        show_25_images(valid_images, valid_labels)
        plt.show()
    except Exception as e:
        print("Image grid skipped:", repr(e))



## === cell 20
if not SKIP_PLOTS:
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
    base = tf.keras.applications.MobileNetV2(
        input_shape=input_shape, include_top=False, weights="imagenet"
    )
    base.trainable = base_model_trainable

    inputs = layers.Input(shape=input_shape)
    x = inputs
    if data_augmentation is not None:
        x = data_augmentation(x)

    x = tf.keras.applications.mobilenet_v2.preprocess_input(x * 255.0)
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
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



## === cell 26
pass



## === cell 27
acc = [0.0] + history.history["accuracy"]
val_acc = [0.0] + history.history["val_accuracy"]

loss = history.history["loss"]
val_loss = history.history["val_loss"]

if not SKIP_PLOTS:
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



## === cell 28
base_model = None
for lyr in model.layers:
    if isinstance(lyr, tf.keras.Model) and lyr.name.startswith("mobilenetv2"):
        base_model = lyr
        break

if base_model is None:
    for lyr in model.layers:
        if "MobileNetV2" in lyr.__class__.__name__:
            base_model = lyr
            break

if base_model is None:
    raise RuntimeError(
        "Could not find the MobileNetV2 base model inside the model to fine-tune."
    )

base_model.trainable = True
model.summary(show_trainable=True)



## === cell 29
loss_function = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
optimizer = tf.keras.optimizers.Adam(lr * 1e-3)
metrics = [tf.keras.metrics.CategoricalAccuracy(name="accuracy", dtype=np.float32)]

model.compile(optimizer=optimizer, loss=loss_function, metrics=metrics)



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



## === cell 31
acc += history_fine.history["accuracy"]
val_acc += history_fine.history["val_accuracy"]

loss += history_fine.history["loss"]
val_loss += history_fine.history["val_loss"]



## === cell 32
if not SKIP_PLOTS:
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


if not SKIP_PLOTS:
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


if not SKIP_PLOTS:
    plot_pred(
        prediction_probabilities=tf.nn.softmax(preds, axis=1).numpy(),
        labels=test_labels,
        images=test_images,
        n=10,
    )
    plt.show()




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


if not SKIP_PLOTS:
    prediction_probabilities = tf.nn.softmax(preds, axis=1).numpy()
    plot_pred_conf(
        prediction_probabilities=prediction_probabilities, labels=test_labels, n=42
    )
    plt.show()



## === cell 37
if not SKIP_PLOTS:
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



## === cell 38
if not SKIP_PLOTS:
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



## === cell 39
pass



## === cell 40
save_model(model, file_name="mobilenetv2-Adam", include_datetime=False)



## === cell 41
pass



## === cell 42
submitted_data_dir = KAGGLE_DATA_DIR / "test"
submitted_img_names = sorted(
    [str(p) for p in submitted_data_dir.iterdir() if p.suffix.lower() == ".jpg"]
)
submitted_img_names[:2]



## === cell 43
submitted_data = create_data_batches(
    submitted_img_names, test_data=True, cache_name="submit_unordered.cache"
)
next(submitted_data.as_numpy_iterator())[0][0][0]



## === cell 44
next(iter(sorted(submitted_data_dir.iterdir()))).stem



## === cell 45
sample_sub = pd.read_csv(KAGGLE_DATA_DIR / "sample_submission.csv")
sample_ids = sample_sub["id"].tolist()
class_cols = [c for c in sample_sub.columns if c != "id"]

id_to_path = {Path(p).stem: p for p in submitted_img_names}
missing = [i for i in sample_ids if i not in id_to_path]
if missing:
    raise RuntimeError(
        f"Missing {len(missing)} test images referenced in sample_submission. First few: {missing[:5]}"
    )

ordered_paths = [id_to_path[_id] for _id in sample_ids]
submitted_data_ordered = create_data_batches(
    ordered_paths, test_data=True, cache_name="submit_ordered.cache"
)

submit_logits = model.predict(submitted_data_ordered, verbose=1)
submit_preds = tf.nn.softmax(submit_logits, axis=1).numpy()

model_class_order = list(lb.classes_)
col_index = {c: i for i, c in enumerate(model_class_order)}

reordered_idx = np.array([col_index[c] for c in class_cols], dtype=np.int32)
reordered = submit_preds[:, reordered_idx].astype(np.float32, copy=False)

out_df = pd.DataFrame(reordered, columns=class_cols)
out_df.insert(0, "id", sample_ids)
out_df.head()



## === cell 46
out_df.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", out_df.shape)
print(out_df.head())
