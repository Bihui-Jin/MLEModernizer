# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


# 1. Kaggle task description

## Task
Given a dataset of images of dogs and cats, predict if an image is a dog or a cat.

## Metric
Log loss.

## Submission Format
For each image in the test set, you must submit a probability that image is a dog. The file should have a header and be in the following format:

```
id,label
1,0.5
2,0.5
3,0.5
...
```

## Dataset
The train folder contains 25,000 images of dogs and cats. Each image in this folder has the label as part of the filename. The test folder contains 12,500 images, named according to a numeric id.

# 2. Python version

3.10

# 3. Installed packages

geopandas==0.14.4
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
protobuf==6.33.0
pytorch-ignite==0.5.3
pytorch-lightning==2.5.5
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
torch==2.6.0+cu124
torchao==0.10.0
torchaudio==2.6.0+cu124
torchdata==0.11.0
torchinfo==1.8.0
torchmetrics==1.8.2
torchsummary==1.5.1
torchtune==0.6.1
torchvision==0.21.0+cu124

# 4. Data file paths

```
/
    kaggle/
        data/
            dogs-vs-cats-redux-kernels-edition/
                description.md (50 lines)
                sample_submission.csv (2501 lines)
                ... and 2 other files
                test/
                    1332.jpg (22.2 kB)
                    617.jpg (22.9 kB)
                    ... and 2498 other files
                train/
                    dog.10425.jpg (33.5 kB)
                    cat.2462.jpg (18.8 kB)
                    ... and 22498 other files
        input/
            dogs-vs-cats-redux-kernels-edition/
                description.md (50 lines)
                sample_submission.csv (2501 lines)
                ... and 2 other files
                test/
                    1332.jpg (22.2 kB)
                    617.jpg (22.9 kB)
                    ... and 2498 other files
                train/
                    dog.10425.jpg (33.5 kB)
                    cat.2462.jpg (18.8 kB)
                    ... and 22498 other files
            test/
                test/
                    1332.jpg (22.2 kB)
                    617.jpg (22.9 kB)
                    ... and 2498 other files
            train/
                train/
                    dog.10425.jpg (33.5 kB)
                    cat.2462.jpg (18.8 kB)
                    ... and 22498 other files
        working/
            dogs-vs-cats-redux-kernels-edition/
                description.md (50 lines)
                sample_submission.csv (2501 lines)
                ... and 2 other files
                test/
                    1332.jpg (22.2 kB)
                    617.jpg (22.9 kB)
                    ... and 2498 other files
                train/
                    dog.10425.jpg (33.5 kB)
                    cat.2462.jpg (18.8 kB)
                    ... and 22498 other files
```

-> data/dogs-vs-cats-redux-kernels-edition/sample_submission.csv has 2500 rows and 2 columns.
Here is some information about the columns:
id (int64) has range: 1.00 - 2000.00, 0 nan values
label (float64) has 1 unique values: [0.5]

-> input/dogs-vs-cats-redux-kernels-edition/sample_submission.csv has 2500 rows and 2 columns.
Here is some information about the columns:
id (int64) has range: 1.00 - 2000.00, 0 nan values
label (float64) has 1 unique values: [0.5]

-> working/dogs-vs-cats-redux-kernels-edition/sample_submission.csv has 2500 rows and 2 columns.
Here is some information about the columns:
id (int64) has range: 1.00 - 2000.00, 0 nan values
label (float64) has 1 unique values: [0.5]

# 5. Code solution

## === cell 0
import os
import google.protobuf.message_factory as _mf

if not hasattr(_mf.MessageFactory, "GetPrototype"):

    def _GetPrototype(self, descriptor):
        return self.GetMessageClass(descriptor)

    _mf.MessageFactory.GetPrototype = _GetPrototype

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import numpy as np
import glob
import tensorflow as tf
import matplotlib.pyplot as plt

tf.config.optimizer.set_jit(True)

from tensorflow.keras import mixed_precision

mixed_precision.set_global_policy("mixed_float16")

AUTOTUNE = tf.data.AUTOTUNE

base_path = "/kaggle/working"
CACHE_DIR = os.path.join(base_path, "tf_cache")
os.makedirs(CACHE_DIR, exist_ok=True)

tf.random.set_seed(42)




## === cell 1
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices("GPU")))




## === cell 2
import zipfile

base_path = "/kaggle/working"

if not os.path.isdir(os.path.join(base_path, "train")):
    zip_train = zipfile.ZipFile(
        "/kaggle/input/dogs-vs-cats-redux-kernels-edition/train.zip", "r"
    )
    zip_train.extractall(base_path)
    zip_train.close()

if not os.path.isdir(os.path.join(base_path, "test")):
    zip_test = zipfile.ZipFile(
        "/kaggle/input/dogs-vs-cats-redux-kernels-edition/test.zip", "r"
    )
    zip_test.extractall(base_path)
    zip_test.close()




## === cell 3
possible_root = None
for entry in os.listdir(base_path):
    candidate = os.path.join(base_path, entry)
    if os.path.isdir(candidate) and os.path.isdir(os.path.join(candidate, "train")):
        possible_root = candidate
        break
if possible_root is None:
    possible_root = base_path

train_dir = os.path.join(possible_root, "train")
test_dir = os.path.join(possible_root, "test")


def get_path(path, ext):
    """Return list of files with given extension under path (recursive)."""
    return glob.glob(os.path.join(path, f"**/*.{ext}"), recursive=True)


def label_from_path(path):
    """Return 1 for dog, 0 for cat based on filename."""
    return 1 if os.path.basename(path).startswith("dog") else 0




## === cell 4
data_list = get_path(train_dir, "jpg")
labels = [label_from_path(p) for p in data_list]
print("dogs:", sum(labels), "cats:", len(labels) - sum(labels))




## === cell 5
split_ratio = 0.8
rng = np.random.default_rng(seed=42)
indices = rng.permutation(len(data_list))
train_idx = indices[: int(len(data_list) * split_ratio)]
val_idx = indices[int(len(data_list) * split_ratio) :]

train_data = [data_list[i] for i in train_idx]
train_label = [labels[i] for i in train_idx]
val_data = [data_list[i] for i in val_idx]
val_label = [labels[i] for i in val_idx]




## === cell 6
img_size = 224


def preprocess_image(image):
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [img_size, img_size])
    return image


def load_and_preprocess_image(path):
    path = tf.cast(path, tf.string)
    image = tf.io.read_file(path)
    return preprocess_image(image)


def load_and_preprocess_from_path_label(path, label):
    image = load_and_preprocess_image(path)
    label_one = tf.one_hot(tf.cast(label, tf.int32), 2)
    return image, label_one


ds_train = tf.data.Dataset.from_tensor_slices(
    (tf.constant(train_data, dtype=tf.string), tf.constant(train_label, dtype=tf.int32))
)
ds_val = tf.data.Dataset.from_tensor_slices(
    (tf.constant(val_data, dtype=tf.string), tf.constant(val_label, dtype=tf.int32))
)

options = tf.data.Options()
options.experimental_deterministic = False
ds_train = ds_train.with_options(options)
ds_val = ds_val.with_options(options)

ds_train = ds_train.map(
    load_and_preprocess_from_path_label, num_parallel_calls=AUTOTUNE
)
ds_val = ds_val.map(load_and_preprocess_from_path_label, num_parallel_calls=AUTOTUNE)

ds_train = ds_train.cache()
ds_val = ds_val.cache()




## === cell 7
batch_size = 128
dsb_train = ds_train.shuffle(1000).batch(batch_size).prefetch(AUTOTUNE)
dsb_val = ds_val.batch(batch_size).prefetch(AUTOTUNE)




## === cell 8
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras import layers, Model

img_augmentation = tf.keras.Sequential(
    [
        layers.RandomRotation(factor=0.15),
        layers.RandomTranslation(height_factor=0.1, width_factor=0.1),
        layers.RandomFlip(),
        layers.RandomContrast(factor=0.1),
    ],
    name="img_augmentation",
)




## === cell 9
def build_model(num_classes, backbone_trainable=False, lr=1e-2):
    inputs = layers.Input(shape=(img_size, img_size, 3))
    x = img_augmentation(inputs)
    base_model = EfficientNetB0(include_top=False, input_tensor=x, weights="imagenet")
    base_model.trainable = backbone_trainable

    x = layers.GlobalAveragePooling2D(name="avg_pool")(base_model.output)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2, name="top_dropout")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="pred")(x)

    model = Model(inputs, outputs, name="EfficientNet")
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    model.compile(
        optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"]
    )
    return model, base_model




## === cell 10
model, base_model = build_model(num_classes=2, backbone_trainable=False, lr=1e-2)




## === cell 11
epochs_initial = 10  # unchanged
hist = model.fit(dsb_train, epochs=epochs_initial, validation_data=dsb_val, verbose=2)




## === cell 12
base_model.trainable = True
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

epochs_finetune = 20  # unchanged
hist_fine = model.fit(
    dsb_train, epochs=epochs_finetune, validation_data=dsb_val, verbose=2
)




## === cell 13
def plot_hist(history, title="Training"):
    plt.plot(history.history["accuracy"], label="train")
    plt.plot(history.history["val_accuracy"], label="val")
    plt.title(title)
    plt.ylabel("Accuracy")
    plt.xlabel("Epoch")
    plt.legend()
    plt.show()


plot_hist(hist, "Initial Training")
plot_hist(hist_fine, "Fine‑tuning")




## === cell 14
test_paths_all = get_path(test_dir, "jpg")
test_list = []
id_list = []
for p in test_paths_all:
    filename = os.path.splitext(os.path.basename(p))[0]
    if filename.isdigit():
        test_list.append(p)
        id_list.append(int(filename))

sorted_pairs = sorted(zip(test_list, id_list), key=lambda x: x[1])
if sorted_pairs:
    test_list, id_list = zip(*sorted_pairs)
else:
    test_list, id_list = [], []

ds_test = tf.data.Dataset.from_tensor_slices(
    (
        tf.constant(list(test_list), dtype=tf.string),
        tf.constant(list(id_list), dtype=tf.int32),
    )
)


def test_map(image_path, img_id):
    img = load_and_preprocess_image(image_path)
    return img, tf.cast(img_id, tf.int32)


ds_test = ds_test.map(test_map, num_parallel_calls=AUTOTUNE)
ds_test = ds_test.cache()
dsb_test = ds_test.batch(batch_size).prefetch(AUTOTUNE)




## === cell 15
submission = {"id": [], "label": []}
dog_probability = lambda probs: probs[1]  # index 1 corresponds to "dog"

for batch_images, batch_ids in dsb_test:
    probs = model.predict(batch_images, verbose=0)
    submission["id"].extend(batch_ids.numpy().tolist())
    submission["label"].extend([dog_probability(p) for p in probs])




## === cell 16
import pandas as pd

candidate_paths = glob.glob(
    os.path.join(base_path, "**/sample_submission*.csv"), recursive=True
)
if not candidate_paths:
    candidate_paths = glob.glob(
        os.path.join("/kaggle/input", "**/sample_submission*.csv"), recursive=True
    )
if not candidate_paths:
    raise FileNotFoundError("sample_submission.csv not found in any expected location.")

sample_path = candidate_paths[0]
sample_df = pd.read_csv(sample_path)

pred_df = pd.DataFrame({"id": submission["id"], "label": submission["label"]})
submission_df = sample_df.drop(columns=["label"]).merge(pred_df, on="id", how="left")
submission_df["label"] = submission_df["label"].fillna(0.5)

submission_path = os.path.join(base_path, "submission.csv")
submission_df.to_csv(submission_path, index=False)
print(f"Submission written to {submission_path}")
