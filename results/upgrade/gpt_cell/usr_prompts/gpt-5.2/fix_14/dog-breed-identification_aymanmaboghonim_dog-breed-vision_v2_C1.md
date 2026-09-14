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

3.9

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
import sys
import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", None)
os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", None)

os.environ.setdefault("TFHUB_CACHE_DIR", "/kaggle/working/tfhub_cache")

import tensorflow as tf
import tensorflow_hub as hub
import tf_keras

from sklearn.model_selection import train_test_split

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)
try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

try:
    tf.config.threading.set_inter_op_parallelism_threads(2)
    tf.config.threading.set_intra_op_parallelism_threads(0)
except Exception:
    pass

AUTOTUNE = tf.data.AUTOTUNE

DATASET_OPTIONS = tf.data.Options()
try:
    DATASET_OPTIONS.deterministic = True
except Exception:
    pass

CACHE_DIR = "/kaggle/working/tfdata_cache"
os.makedirs(CACHE_DIR, exist_ok=True)



## === cell 1
print("TF version:", tf.__version__)
print(
    "GPU",
    (
        "available (YESS!!!!)"
        if tf.config.list_physical_devices("GPU")
        else "not available :("
    ),
)



## === cell 2
labels_csv = pd.read_csv("../input/dog-breed-identification/labels.csv")



## === cell 3
if False:
    from pandas_profiling import ProfileReport  # noqa: F401

    Labels_Profile = ProfileReport(labels_csv, progress_bar=False)
    Labels_Profile.to_notebook_iframe()



## === cell 4
if False:
    labels_csv["breed"].value_counts().plot.bar(figsize=(20, 10))



## === cell 5
filenames = [
    "../input/dog-breed-identification/train/" + fname + ".jpg"
    for fname in labels_csv["id"]
]
filenames[:5]



## === cell 6
if len(os.listdir("../input/dog-breed-identification/train/")) == len(filenames):
    print("Filenames match actual amount of files!")
else:
    print("Filenames do not match actual amount of files, check the target directory.")



## === cell 7
if False:
    from IPython.display import Image  # noqa: F401

    Image(filenames[42])



## === cell 8
labels = labels_csv["breed"].to_numpy()
labels[:10]



## === cell 9
if len(labels) == len(filenames):
    print("Number of labels matches number of filenames!")
else:
    print(
        "Number of labels does not match number of filenames, check data directories."
    )



## === cell 10
unique_breeds = np.unique(labels)
len(unique_breeds)



## === cell 11
breed_to_index = {b: i for i, b in enumerate(unique_breeds)}
y_indices = np.fromiter(
    (breed_to_index[b] for b in labels), dtype=np.int32, count=len(labels)
)
boolean_labels = np.eye(len(unique_breeds), dtype=bool)[y_indices]
boolean_labels[:2]



## === cell 12
X = filenames
y = boolean_labels



## === cell 13
NUM_IMAGES = 1000

X_train, X_val, y_train, y_val = train_test_split(
    X[:NUM_IMAGES], y[:NUM_IMAGES], test_size=0.2, random_state=SEED
)

len(X_train), len(y_train), len(X_val), len(y_val)



## === cell 14
IMG_SIZE = 224


def process_image(image_path):
    """
    Takes an image file path and turns it into a Tensor.
    """
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, size=[IMG_SIZE, IMG_SIZE])
    return image




## === cell 15
def get_image_label(image_path, label):
    """
    Takes an image file path name and the associated label,
    processes the image and returns a tuple of (image, label).
    """
    image = process_image(image_path)
    return image, label




## === cell 16
BATCH_SIZE = 32

SHUFFLE_BUFFER = 2048


def create_data_batches(
    x,
    y=None,
    batch_size=BATCH_SIZE,
    valid_data=False,
    test_data=False,
    cache_id="default",
):
    """
    Creates batches of data out of image (x) and label (y) pairs.
    Shuffles the data if it's training data but doesn't shuffle it if it's validation data.
    Also accepts test data as input (no labels).
    """
    if test_data:
        print("Creating test data batches...")
        data = tf.data.Dataset.from_tensor_slices(x)
        data = data.map(process_image, num_parallel_calls=AUTOTUNE)
        data = data.with_options(DATASET_OPTIONS)
        data_batch = data.batch(batch_size, drop_remainder=False).prefetch(AUTOTUNE)
        return data_batch

    elif valid_data:
        print("Creating validation data batches...")
        data = tf.data.Dataset.from_tensor_slices((x, y))
        data = data.map(get_image_label, num_parallel_calls=AUTOTUNE)

        cache_path = os.path.join(CACHE_DIR, f"val_{cache_id}.cache")
        data = data.cache(cache_path)

        data = data.with_options(DATASET_OPTIONS)
        data_batch = data.batch(batch_size, drop_remainder=False).prefetch(AUTOTUNE)
        return data_batch

    else:
        print("Creating training data batches...")
        data = tf.data.Dataset.from_tensor_slices((x, y))
        data = data.shuffle(
            buffer_size=min(len(x), SHUFFLE_BUFFER),
            seed=SEED,
            reshuffle_each_iteration=True,
        )
        data = data.map(get_image_label, num_parallel_calls=AUTOTUNE)

        cache_path = os.path.join(CACHE_DIR, f"train_{cache_id}.cache")
        data = data.cache(cache_path)

        data = data.with_options(DATASET_OPTIONS)
        data_batch = data.batch(batch_size, drop_remainder=True).prefetch(AUTOTUNE)
        return data_batch




## === cell 17
train_data = create_data_batches(X_train, y_train, cache_id="subset")
val_data = create_data_batches(X_val, y_val, valid_data=True, cache_id="subset")



## === cell 18
train_data.element_spec, val_data.element_spec




## === cell 19
def show_25_images(images, labels):
    """
    Displays 25 images from a data batch.
    """
    plt.figure(figsize=(10, 10))
    for i in range(25):
        ax = plt.subplot(5, 5, i + 1)
        plt.imshow(images[i])
        plt.title(unique_breeds[labels[i].argmax()])
        plt.axis("off")




## === cell 20
if False:
    train_images, train_labels = next(train_data.as_numpy_iterator())
    show_25_images(train_images, train_labels)



## === cell 21
INPUT_SHAPE = [None, IMG_SIZE, IMG_SIZE, 3]
OUTPUT_SHAPE = len(unique_breeds)

MODEL_URL = "https://tfhub.dev/google/imagenet/mobilenet_v2_130_224/feature_vector/4"




## === cell 22
def create_model(
    input_shape=INPUT_SHAPE, output_shape=OUTPUT_SHAPE, model_url=MODEL_URL
):
    print("Building model with:", MODEL_URL)

    model = tf_keras.Sequential(
        [
            hub.KerasLayer(model_url, name="hub_layer"),
            tf_keras.layers.Dense(
                units=output_shape, activation="softmax", name="predictions"
            ),
        ],
        name="hub_classifier",
    )

    model.compile(
        loss=tf_keras.losses.CategoricalCrossentropy(),
        optimizer=tf_keras.optimizers.Adam(),
        metrics=["accuracy"],
    )

    model.build(input_shape)
    return model


model = create_model()
model.summary()



## === cell 23
pass




## === cell 24
def create_tensorboard_callback():
    class _NoOpCallback(tf_keras.callbacks.Callback):
        pass

    return _NoOpCallback()




## === cell 25
early_stopping = tf_keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=3)



## === cell 26
print(
    "GPU",
    (
        "available (YESS!!!!)"
        if tf.config.list_physical_devices("GPU")
        else "not available :("
    ),
)



## === cell 27
NUM_EPOCHS = 100


def train_model():
    """
    Trains a given model and returns the trained version.
    """
    model = create_model()
    tensorboard = create_tensorboard_callback()

    steps_per_epoch = (
        len(X_train) // BATCH_SIZE
    )  # drop_remainder=True in train pipeline
    validation_steps = int(np.ceil(len(X_val) / BATCH_SIZE))

    model.fit(
        x=train_data,
        epochs=NUM_EPOCHS,
        validation_data=val_data,
        validation_freq=1,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
        callbacks=[tensorboard, early_stopping],
        verbose=2,
    )
    return model




## === cell 28
model = None



## === cell 29
pass



## === cell 30
predictions = None



## === cell 31
pass



## === cell 32
pass




## === cell 33
def get_pred_label(prediction_probabilities):
    """
    Turns an array of prediction probabilities into a label.
    """
    return unique_breeds[np.argmax(prediction_probabilities)]




## === cell 34
def unbatchify(data):
    """
    Takes a batched dataset of (image, label) Tensors and returns separate arrays
    of images and labels.
    """
    images = []
    labels_out = []
    for image, label in data.unbatch().as_numpy_iterator():
        images.append(image)
        labels_out.append(unique_breeds[np.argmax(label)])
    return images, labels_out




## === cell 35
def plot_pred(prediction_probabilities, labels, images, n=1):
    """
    View the prediction, ground truth label and image for sample n.
    """
    pred_prob, true_label, image = prediction_probabilities[n], labels[n], images[n]
    pred_label = get_pred_label(pred_prob)

    plt.imshow(image)
    plt.xticks([])
    plt.yticks([])

    color = "green" if pred_label == true_label else "red"
    plt.title(
        "{} {:2.0f}% ({})".format(pred_label, np.max(pred_prob) * 100, true_label),
        color=color,
    )




## === cell 36
pass




## === cell 37
def plot_pred_conf(prediction_probabilities, labels, n=1):
    """
    Plots the top 10 highest prediction confidences along with
    the truth label for sample n.
    """
    pred_prob, true_label = prediction_probabilities[n], labels[n]
    pred_label = get_pred_label(pred_prob)

    top_10_pred_indexes = pred_prob.argsort()[-10:][::-1]
    top_10_pred_values = pred_prob[top_10_pred_indexes]
    top_10_pred_labels = unique_breeds[top_10_pred_indexes]

    top_plot = plt.bar(
        np.arange(len(top_10_pred_labels)), top_10_pred_values, color="grey"
    )
    plt.xticks(
        np.arange(len(top_10_pred_labels)),
        labels=top_10_pred_labels,
        rotation="vertical",
    )

    if np.isin(true_label, top_10_pred_labels):
        top_plot[np.argmax(top_10_pred_labels == true_label)].set_color("green")




## === cell 38
pass



## === cell 39
pass



## === cell 40
full_data = create_data_batches(X, y, cache_id="full")



## === cell 41
full_model = create_model()



## === cell 42
full_model_tensorboard = create_tensorboard_callback()
full_model_early_stopping = tf_keras.callbacks.EarlyStopping(
    monitor="accuracy", patience=3
)



## === cell 43
pass



## === cell 44
full_steps_per_epoch = len(X) // BATCH_SIZE  # drop_remainder=True
full_model.fit(
    x=full_data,
    epochs=NUM_EPOCHS,
    steps_per_epoch=full_steps_per_epoch,
    callbacks=[full_model_tensorboard, full_model_early_stopping],
    verbose=2,
)




## === cell 45
def save_model(model, suffix=None):
    """
    Saves a given model in a models directory and appends a suffix (str)
    for clarity and reuse.
    """
    modeldir = os.path.join("./", datetime.datetime.now().strftime("%Y%m%d-%H%M%s"))
    model_path = modeldir + "-" + suffix + ".h5"
    print(f"Saving model to: {model_path}...")
    model.save(model_path)
    return model_path




## === cell 46
def load_model(model_path):
    """
    Loads a saved model from a specified path.
    """
    print(f"Loading saved model from: {model_path}")
    model = tf_keras.models.load_model(
        model_path, custom_objects={"KerasLayer": hub.KerasLayer}
    )
    return model




## === cell 47
saved_model_path = None



## === cell 48
loaded_full_model = full_model



## === cell 49
test_path = "../input/dog-breed-identification/test/"
test_files_sorted = sorted(os.listdir(test_path))
test_filenames = [os.path.join(test_path, fname) for fname in test_files_sorted]
test_filenames[:5]



## === cell 50
len(test_filenames)



## === cell 51
test_data = create_data_batches(test_filenames, test_data=True)



## === cell 52
test_predictions = loaded_full_model.predict(test_data, verbose=0)



## === cell 53
test_predictions[:10]



## === cell 54
preds_df = pd.DataFrame(columns=["id"] + list(unique_breeds))
preds_df.head()



## === cell 55
test_ids = [os.path.splitext(fname)[0] for fname in test_files_sorted]
preds_df["id"] = test_ids
preds_df.head()



## === cell 56
preds_df[list(unique_breeds)] = test_predictions
preds_df.head()



## === cell 57
preds_df.to_csv("./full_submission_1_mobilienetV2_adam.csv", index=False)
