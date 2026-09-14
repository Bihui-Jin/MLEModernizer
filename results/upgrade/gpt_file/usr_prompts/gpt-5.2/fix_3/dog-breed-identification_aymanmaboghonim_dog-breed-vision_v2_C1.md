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
import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
import tensorflow_hub as hub

from IPython.display import Image
from sklearn.model_selection import train_test_split


SEED = 42
tf.random.set_seed(SEED)
np.random.seed(SEED)

print("TF version:", tf.__version__)
print(
    "GPU",
    (
        "available (YESS!!!!)"
        if tf.config.list_physical_devices("GPU")
        else "not available :("
    ),
)

try:
    tf.config.optimizer.set_jit(False)  # keep numerics stable vs. XLA changes
except Exception:
    pass



## === cell 1
labels_csv = pd.read_csv("../input/dog-breed-identification/labels.csv")
labels_csv.head()



## === cell 2
print("labels_csv shape:", labels_csv.shape)
print(labels_csv["breed"].nunique(), "unique breeds")



## === cell 3
print("Skipped breed distribution plot for performance.")



## === cell 4
filenames = [
    "../input/dog-breed-identification/train/" + fname + ".jpg"
    for fname in labels_csv["id"]
]
filenames[:5]



## === cell 5
if len(os.listdir("../input/dog-breed-identification/train/")) == len(filenames):
    print("Filenames match actual amount of files!")
else:
    print("Filenames do not match actual amount of files, check the target directory.")



## === cell 6
print("Skipped sample image display for performance.")



## === cell 7
labels = labels_csv["breed"].to_numpy()
labels[:10]



## === cell 8
if len(labels) == len(filenames):
    print("Number of labels matches number of filenames!")
else:
    print(
        "Number of labels does not match number of filenames, check data directories."
    )



## === cell 9
unique_breeds = np.unique(labels)
len(unique_breeds), unique_breeds[:5]



## === cell 10
labels_col = labels[:, None]  # (N,1)
unique_row = unique_breeds[None, :]  # (1,C)
boolean_labels = (labels_col == unique_row).astype(np.float32)  # (N,C)
boolean_labels[:2]



## === cell 11
X = filenames
y = boolean_labels



## === cell 12
NUM_IMAGES = 1000

X_train, X_val, y_train, y_val = train_test_split(
    X[:NUM_IMAGES], y[:NUM_IMAGES], test_size=0.2, random_state=42
)

len(X_train), len(y_train), len(X_val), len(y_val)



## === cell 13
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




## === cell 14
def get_image_label(image_path, label):
    """
    Takes an image file path name and the associated label,
    processes the image and returns a tuple of (image, label).
    """
    image = process_image(image_path)
    return image, label




## === cell 15
BATCH_SIZE = 32


def create_data_batches(
    x, y=None, batch_size=BATCH_SIZE, valid_data=False, test_data=False
):
    """
    Creates batches of data out of image (x) and label (y) pairs.
    Shuffles the data if it's training data but doesn't shuffle it if it's validation data.
    Also accepts test data as input (no labels).
    """
    options = tf.data.Options()
    options.deterministic = True
    options.threading.private_threadpool_size = 0  # let TF choose
    options.autotune.enabled = True

    if test_data:
        print("Creating test data batches...")
        data = tf.data.Dataset.from_tensor_slices((tf.constant(x)))  # only filepaths
        data = data.with_options(options)
        data = data.map(process_image, num_parallel_calls=tf.data.AUTOTUNE)
        data = data.cache()  # safe: deterministic pure function of filepath
        data_batch = data.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        return data_batch

    elif valid_data:
        print("Creating validation data batches...")
        data = tf.data.Dataset.from_tensor_slices((tf.constant(x), tf.constant(y)))
        data = data.with_options(options)
        data = data.map(get_image_label, num_parallel_calls=tf.data.AUTOTUNE)
        data = data.cache()
        data_batch = data.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        return data_batch

    else:
        print("Creating training data batches...")
        data = tf.data.Dataset.from_tensor_slices((tf.constant(x), tf.constant(y)))
        data = data.with_options(options)
        buffer = min(len(x), 2048)
        data = data.shuffle(
            buffer_size=buffer, seed=SEED, reshuffle_each_iteration=True
        )
        data = data.map(get_image_label, num_parallel_calls=tf.data.AUTOTUNE)
        data = data.cache()
        data_batch = data.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        return data_batch




## === cell 16
train_data = create_data_batches(X_train, y_train)
val_data = create_data_batches(X_val, y_val, valid_data=True)



## === cell 17
train_data.element_spec, val_data.element_spec



## === cell 18
#     """
#     Displays 25 images from a data batch.
print("Skipped training batch visualization for performance.")



## === cell 19
print("Skipped fetching a batch for plotting.")



## === cell 20
INPUT_SHAPE = [None, IMG_SIZE, IMG_SIZE, 3]  # batch, height, width, colour channel
OUTPUT_SHAPE = len(unique_breeds)  # number of unique labels
MODEL_URL = "https://tfhub.dev/google/imagenet/mobilenet_v2_130_224/classification/4"




## === cell 21
class HubModuleLayer(tf.keras.layers.Layer):
    def __init__(self, model_url, trainable=False, **kwargs):
        super().__init__(**kwargs)
        self.model_url = model_url
        self.trainable = trainable
        self._hub_module = None

    def build(self, input_shape):
        self._hub_module = hub.load(self.model_url)
        super().build(input_shape)

    def call(self, inputs):
        x = tf.cast(inputs, tf.float32)
        out = self._hub_module(x)
        return out


def create_model(
    input_shape=INPUT_SHAPE, output_shape=OUTPUT_SHAPE, model_url=MODEL_URL
):
    print("Building model with:", model_url)

    model = tf.keras.Sequential(
        [
            tf.keras.layers.InputLayer(input_shape=input_shape[1:]),
            HubModuleLayer(model_url=model_url, name="hub_module"),
            tf.keras.layers.Dense(units=output_shape, activation="softmax"),
        ]
    )

    model.compile(
        loss=tf.keras.losses.CategoricalCrossentropy(),
        optimizer=tf.keras.optimizers.Adam(),
        metrics=["accuracy"],
    )
    return model




## === cell 22
model = create_model()
model.summary()



## === cell 23
print("TensorBoard extension skipped (script-safe).")




## === cell 24
def create_tensorboard_callback():
    logdir = os.path.join("./logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    return tf.keras.callbacks.TensorBoard(logdir, write_graph=False, profile_batch=0)




## === cell 25
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_accuracy", patience=3, restore_best_weights=True
)



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
    tensorboard = None

    model.fit(
        x=train_data,
        epochs=NUM_EPOCHS,
        validation_data=val_data,
        validation_freq=1,
        callbacks=(
            [early_stopping] if tensorboard is None else [tensorboard, early_stopping]
        ),
        verbose=2,
    )
    return model




## === cell 28
print("Skipped initial small training run to avoid redundant compute.")



## === cell 29
print("TensorBoard magic skipped (script-safe). Logs (if any) are in ./logs")



## === cell 30
print("Skipped validation predictions for performance.")



## === cell 31
print("Skipped.")



## === cell 32
print("Skipped.")




## === cell 33
def get_pred_label(prediction_probabilities):
    """
    Turns an array of prediction probabilities into a label.
    """
    return unique_breeds[np.argmax(prediction_probabilities)]


print("Skipped prediction label demo.")




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


print("Skipped unbatchify/validation image extraction.")




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
print("Skipped plot_pred for performance.")




## === cell 37
def plot_pred_conf(prediction_probabilities, labels, n=1):
    """
    Plots the top 10 highest prediction confidences along with the truth label for sample n.
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
print("Skipped plot_pred_conf for performance.")



## === cell 39
print("Skipped combined prediction plots for performance.")



## === cell 40
full_data = create_data_batches(X, y)



## === cell 41
full_model = create_model()
full_model.summary()



## === cell 42
full_model_tensorboard = None
full_model_early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="accuracy", patience=3, restore_best_weights=True
)



## === cell 43
print("TensorBoard magic skipped (script-safe).")



## === cell 44
full_model.fit(
    x=full_data,
    epochs=NUM_EPOCHS,
    callbacks=(
        [full_model_early_stopping]
        if full_model_tensorboard is None
        else [full_model_tensorboard, full_model_early_stopping]
    ),
    verbose=2,
)




## === cell 45
def save_model(model, suffix=None):
    """
    Saves a given model in a models directory and appends a suffix (str) for clarity and reuse.
    """
    modeldir = os.path.join("./", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    model_path = modeldir + "-" + (suffix if suffix else "model") + ".h5"
    print(f"Saving model to: {model_path}...")
    model.save(model_path)
    return model_path




## === cell 46
def load_model(model_path):
    """
    Loads a saved model from a specified path.
    """
    print(f"Loading saved model from: {model_path}")
    model = tf.keras.models.load_model(
        model_path, custom_objects={"HubModuleLayer": HubModuleLayer}, compile=True
    )
    return model




## === cell 47
loaded_full_model = full_model
print("Skipped model save/load; using trained full_model directly.")



## === cell 48
test_path = "../input/dog-breed-identification/test/"
test_filenames = [
    os.path.join(test_path, fname)
    for fname in os.listdir(test_path)
    if fname.lower().endswith(".jpg")
]
test_filenames[:5]



## === cell 49
len(test_filenames)



## === cell 50
print("Skipped unordered test batching to avoid duplicate prediction work.")



## === cell 51
print("Skipped unordered test predictions.")



## === cell 52
print("Skipped.")



## === cell 53
sample_sub = pd.read_csv("../input/dog-breed-identification/sample_submission.csv")
breed_cols = list(sample_sub.columns[1:])  # exact breed column order required by Kaggle

unique_breeds_list = list(unique_breeds)
idx_map = [unique_breeds_list.index(b) for b in breed_cols]

preds_df = sample_sub.copy()



## === cell 54
test_ids = preds_df["id"].tolist()
ordered_test_filenames = [os.path.join(test_path, f"{id_}.jpg") for id_ in test_ids]

missing = [p for p in ordered_test_filenames if not tf.io.gfile.exists(p)]
print("Missing test files:", len(missing))

ordered_test_data = create_data_batches(ordered_test_filenames, test_data=True)



## === cell 55
ordered_test_predictions = loaded_full_model.predict(ordered_test_data, verbose=1)

ordered_test_predictions = ordered_test_predictions[:, idx_map]

preds_df[breed_cols] = ordered_test_predictions
preds_df.head()



## === cell 56
submission_path = "./full_submission_1_mobilienetV2_adam.csv"
preds_df.to_csv(submission_path, index=False)
print("Wrote submission:", submission_path, "shape:", preds_df.shape)
print(
    "Columns match sample_submission:",
    list(preds_df.columns) == list(sample_sub.columns),
)
