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
import datetime
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split

np.random.seed(42)
random.seed(42)
tf.random.set_seed(42)




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
DATA_ROOT = "../input/dog-breed-identification"
LABELS_PATH = os.path.join(DATA_ROOT, "labels.csv")
SAMPLE_SUB_PATH = os.path.join(DATA_ROOT, "sample_submission.csv")
TRAIN_IMG_ROOT = os.path.join(DATA_ROOT, "train")
TEST_IMG_ROOT = os.path.join(DATA_ROOT, "test")

labels_df = pd.read_csv(LABELS_PATH)
sample_sub = pd.read_csv(SAMPLE_SUB_PATH)  # needed for column order




## === cell 3
train_filenames = [
    os.path.join(TRAIN_IMG_ROOT, f"{fid}.jpg") for fid in labels_df["id"]
]
if len(os.listdir(TRAIN_IMG_ROOT)) == len(train_filenames):
    print("Filenames match actual amount of files!")
else:
    print("Mismatch between file list and directory contents.")




## === cell 4
unique_breeds = np.sort(labels_df["breed"].unique())
breed_to_idx = {b: i for i, b in enumerate(unique_breeds)}
y_indices = labels_df["breed"].map(breed_to_idx).values
y_onehot = tf.keras.utils.to_categorical(y_indices, num_classes=len(unique_breeds))




## === cell 5
NUM_IMAGES = 1500  # smaller prototype subset for faster prototyping
X_subset = np.array(train_filenames[:NUM_IMAGES])
y_subset_onehot = y_onehot[:NUM_IMAGES]
y_subset_indices = y_indices[:NUM_IMAGES]

X_train, X_val, y_train, y_val = train_test_split(
    X_subset,
    y_subset_onehot,
    test_size=0.2,
    random_state=42,
    stratify=y_subset_indices,
)

print("Train/val sizes:", len(X_train), len(X_val))




## === cell 6
IMG_SIZE = 224
BATCH_SIZE = 128  # larger batch reduces number of steps per epoch


def process_image(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
    return img


def load_batch(paths, labels=None, cache=True, cache_file=None):
    """
    Build a tf.data pipeline.
    - If `cache` is True, the dataset is cached.
      * In‑memory cache when `cache_file` is None.
      * Disk cache when a filename is supplied via `cache_file`.
    - Caching after pairing images with labels ensures both are stored.
    """
    ds = tf.data.Dataset.from_tensor_slices(paths).map(
        process_image, num_parallel_calls=tf.data.AUTOTUNE
    )
    if labels is not None:
        label_ds = tf.data.Dataset.from_tensor_slices(labels)
        ds = tf.data.Dataset.zip((ds, label_ds))
    if cache:
        if cache_file:
            ds = ds.cache(cache_file)  # disk‑based cache
        else:
            ds = ds.cache()  # in‑memory cache
    ds = ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return ds


train_data = load_batch(X_train, y_train)  # cache enabled for training
val_data = load_batch(X_val, y_val)  # cache also safe for validation




## === cell 7
def build_model(num_classes):
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        include_top=False,
        weights="imagenet",
        pooling="avg",
    )
    base_model.trainable = False
    inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = base_model(inputs, training=False)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
    model = tf.keras.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.CategoricalCrossentropy(),
        metrics=["accuracy"],
    )
    return model


model = build_model(len(unique_breeds))
model.summary()




## === cell 8
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor="val_accuracy", patience=3, restore_best_weights=True
)

model.fit(
    train_data,
    epochs=30,
    validation_data=val_data,
    callbacks=[early_stop],
    verbose=2,
)




## === cell 9
CACHE_DIR = "./train_cache"
os.makedirs(CACHE_DIR, exist_ok=True)
CACHE_FILE = os.path.join(CACHE_DIR, "full_train_cache")

full_train_data = load_batch(
    np.array(train_filenames),
    y_onehot,
    cache=True,  # enable caching to avoid re‑reading images each epoch
    cache_file=CACHE_FILE,  # use disk cache to keep memory usage modest
)
model.fit(full_train_data, epochs=5, callbacks=[early_stop], verbose=2)




## === cell 10
def save_model(model, suffix=None):
    model_dir = os.path.join("./", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, f"model{('-' + suffix) if suffix else ''}.h5")
    model.save(model_path)
    print(f"Model saved to {model_path}")
    return model_path


model_path = save_model(model, suffix="mobilenetV2")




## === cell 11
test_filenames = [
    os.path.join(TEST_IMG_ROOT, f) for f in sorted(os.listdir(TEST_IMG_ROOT))
]

test_data = load_batch(test_filenames, labels=None, cache=False)  # no cache for test




## === cell 12
test_predictions = model.predict(test_data, verbose=2)




## === cell 13
submission = pd.DataFrame(test_predictions, columns=unique_breeds)
submission.insert(
    0,
    "id",
    [os.path.splitext(os.path.basename(p))[0] for p in test_filenames],
)

submission = submission[sample_sub.columns]

submission_path = "./full_submission_mobilenetV2.csv"
submission.to_csv(submission_path, index=False)
print(f"Submission saved to {submission_path} with shape {submission.shape}")
