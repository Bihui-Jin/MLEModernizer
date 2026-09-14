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
Create a classifier to predict whether an image contains a cactus.

## Metric
Area under the ROC curve.

## Submission Format
For each ID in the test set, you must predict a probability for the `has_cactus` variable. The file should contain a header and have the following format:

```
id,has_cactus
000940378805c44108d287872b2f04ce.jpg,0.5
0017242f54ececa4512b4d7937d1e21e.jpg,0.5
001ee6d8564003107853118ab87df407.jpg,0.5
etc.
```

## Dataset
This dataset contains a large number of 32 x 32 thumbnail images containing aerial photos of a cactus. The file name of an image corresponds to its `id`.

- **train/** - the training set images
- **test/** - the test set images (you must predict the labels of these)
- **train.csv** - the training set labels, indicates whether the image has a cactus (`has_cactus = 1`)
- **sample_submission.csv** - a sample submission file in the correct format

# 2. Python version

3.7

# 3. Installed packages



# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (56 lines)
            sample_submission.csv (3326 lines)
            sample_submission.csv.zip (67.3 kB)
            test.zip (3.5 MB)
            train.csv (14176 lines)
            train.csv.zip (285.6 kB)
            train.zip (15.0 MB)
            aerial-cactus-identification/
                description.md (56 lines)
                sample_submission.csv (3326 lines)
                ... and 5 other files
                aerial-cactus-identification/
                test/
                    76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                    f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                    ... and 3323 other files
                    test/
                train/
                    775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                    65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                    ... and 14173 other files
                    train/
            test/
                76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                ... and 3323 other files
                test/
            train/
                775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                ... and 14173 other files
                train/
        input/
            description.md (56 lines)
            sample_submission.csv (3326 lines)
            sample_submission.csv.zip (67.3 kB)
            test.zip (3.5 MB)
            train.csv (14176 lines)
            train.csv.zip (285.6 kB)
            train.zip (15.0 MB)
            aerial-cactus-identification/
                description.md (56 lines)
                sample_submission.csv (3326 lines)
                ... and 5 other files
                aerial-cactus-identification/
                test/
                    76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                    f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                    ... and 3323 other files
                    test/
                train/
                    775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                    65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                    ... and 14173 other files
                    train/
            test/
                76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                ... and 3323 other files
                test/
                    76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                    f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                    ... and 3323 other files
                    test/
            train/
                775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                ... and 14173 other files
                train/
                    775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                    65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                    ... and 14173 other files
                    train/
        working/
            aerial-cactus-identification/
                description.md (56 lines)
                sample_submission.csv (3326 lines)
                ... and 5 other files
                aerial-cactus-identification/
                test/
                    76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                    f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                    ... and 3323 other files
                    test/
                train/
                    775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                    65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                    ... and 14173 other files
                    train/
```

-> data/aerial-cactus-identification/sample_submission.csv has 3325 rows and 2 columns.
The columns are: id, has_cactus

-> data/aerial-cactus-identification/train.csv has 14175 rows and 2 columns.
The columns are: id, has_cactus

-> data/sample_submission.csv has 3325 rows and 2 columns.
The columns are: id, has_cactus

-> data/train.csv has 14175 rows and 2 columns.
The columns are: id, has_cactus

-> input/aerial-cactus-identification/sample_submission.csv has 3325 rows and 2 columns.
The columns are: id, has_cactus

-> input/aerial-cactus-identification/train.csv has 14175 rows and 2 columns.
The columns are: id, has_cactus

-> (stopped after 10 files for performance)

# 5. Target score

0.9935

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

import numpy as np
import pandas as pd

from PIL import Image
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras import layers

BASE_DIR = "../input/aerial-cactus-identification"
if not os.path.exists(BASE_DIR):
    BASE_DIR = "../input"

print("Using BASE_DIR:", BASE_DIR)
print("Top-level ../input contents:", os.listdir("../input")[:20])

np.random.seed(42)
tf.random.set_seed(42)



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
TRAIN_CSV = os.path.join(BASE_DIR, "train.csv")
SAMPLE_SUB = os.path.join(BASE_DIR, "sample_submission.csv")

train_dir_candidates = [
    os.path.join(BASE_DIR, "train", "train"),
    os.path.join(BASE_DIR, "train"),
]
test_dir_candidates = [
    os.path.join(BASE_DIR, "test", "test"),
    os.path.join(BASE_DIR, "test"),
]


def pick_existing_dir_with_csv_match(cands, csv_path):
    """Pick the first candidate directory that exists AND contains at least one file from csv 'id' column."""
    df_local = pd.read_csv(csv_path)
    ids = df_local["id"].values[:50]  # small probe
    existing = [d for d in cands if os.path.isdir(d)]
    if not existing:
        raise FileNotFoundError(f"No directory found among candidates: {cands}")
    for d in existing:
        if all(
            os.path.isfile(os.path.join(d, ids[i])) for i in range(min(5, len(ids)))
        ):
            return d
    best_d, best_hits = existing[0], -1
    for d in existing:
        hits = sum(os.path.isfile(os.path.join(d, x)) for x in ids)
        if hits > best_hits:
            best_d, best_hits = d, hits
    return best_d


def pick_existing_dir(cands):
    for d in cands:
        if os.path.isdir(d):
            return d
    raise FileNotFoundError(f"No directory found among candidates: {cands}")


data_dir = pick_existing_dir_with_csv_match(train_dir_candidates, TRAIN_CSV)
test_dir = pick_existing_dir(test_dir_candidates)

print("TRAIN_CSV:", TRAIN_CSV)
print("SAMPLE_SUB:", SAMPLE_SUB)
print("data_dir:", data_dir)
print("test_dir:", test_dir)



## === cell 2
df = pd.read_csv(TRAIN_CSV)
df.head(10)



## === cell 3
print("Test dir listing (first 10):", sorted(os.listdir(test_dir))[:10])
print("Train dir listing (first 10):", sorted(os.listdir(data_dir))[:10])



## === cell 4
filename = df["id"].iloc[0]
path = os.path.join(data_dir, filename)
print("Example train image path:", path)
image_pil = Image.open(path)
image_pil.size



## === cell 5
image = np.array(image_pil)
plt.figure(figsize=(2, 2))
plt.title(f"label={df['has_cactus'].iloc[0]}")
plt.imshow(image)
plt.axis("off")
plt.show()



## === cell 6
print("Positive rate:", float(np.mean(df["has_cactus"])))
print("Pos/Total:", int(np.sum(df["has_cactus"])), "/", len(df["has_cactus"]))
print("Image shape/min/max:", image.shape, np.min(image), np.max(image))




## === cell 7
def get_data(pathtuple):
    path, label = pathtuple
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Missing image file: {path}")
    image_pil = Image.open(path)
    image = np.array(image_pil).astype(np.float32) / 255.0
    label = tf.keras.utils.to_categorical(label, 2).astype(np.float32)
    return image, label




## === cell 8
train_filenames = []
missing_train = 0
for i, fname in enumerate(df["id"].values):
    full_path = os.path.join(data_dir, fname)
    if not os.path.isfile(full_path):
        missing_train += 1
        continue
    train_filenames.append((full_path, int(df["has_cactus"].iloc[i])))

test_filenames = [f for f in os.listdir(test_dir) if f.lower().endswith(".jpg")]
test_filenames = sorted(test_filenames)

test_arr = []
for testfilename in test_filenames:
    path = os.path.join(test_dir, testfilename)
    image_pil = Image.open(path)
    image = np.array(image_pil).astype(np.float32) / 255.0
    test_arr.append(image)

test_data = np.array(test_arr, dtype=np.float32)

print("Train samples:", len(train_filenames), "(missing skipped:", missing_train, ")")
print("Test samples:", len(test_filenames))
print("Test data shape:", test_data.shape)




## === cell 9
def make_batch(batch_paths):
    batch_images = []
    batch_labels = []
    for pathtuple in batch_paths:
        img, lbl = get_data(pathtuple)
        batch_images.append(img)
        batch_labels.append(lbl)
    return np.array(batch_images, dtype=np.float32), np.array(
        batch_labels, dtype=np.float32
    )




## === cell 10
batch_size = 32


def data_gen(data_paths, is_training=True):
    global_step = 0
    steps_per_epoch = max(
        1, len(data_paths) // batch_size
    )  # prevent 0 which can crash training/progressbar
    while True:
        step = global_step % steps_per_epoch
        if step == 0 and is_training:
            np.random.shuffle(data_paths)
        sl = slice(step * batch_size, (step + 1) * batch_size)
        batch = data_paths[sl]
        if len(batch) == 0:
            batch = data_paths[:batch_size]
        images, labels = make_batch(batch)
        global_step += 1
        yield images, labels




## === cell 11
gen = data_gen(train_filenames)
for i, (img, lbl) in enumerate(gen):
    if i >= 3:
        break
    plt.figure(figsize=(2, 2))
    plt.title(f"batch={i}, label={int(np.argmax(lbl[0]))}")
    plt.imshow(img[0])
    plt.axis("off")
    plt.show()



## === cell 12
num_epochs = 20
learning_rate = 0.001
num_classes = 2
input_shape = (32, 32, 3)



## === cell 13
inputs = layers.Input(input_shape)
net = layers.Conv2D(64, (3, 3), padding="same")(inputs)
net = layers.Conv2D(64, (3, 3), padding="same")(net)
net = layers.Conv2D(64, (3, 3), padding="same")(net)
net = layers.BatchNormalization()(net)
net = layers.Activation("relu")(net)
net = layers.MaxPooling2D(pool_size=(2, 2))(net)

net = layers.Conv2D(128, (3, 3), padding="same")(net)
net = layers.Conv2D(128, (3, 3), padding="same")(net)
net = layers.Conv2D(128, (3, 3), padding="same")(net)
net = layers.BatchNormalization()(net)
net = layers.Activation("relu")(net)
net = layers.MaxPooling2D(pool_size=(2, 2))(net)
net = layers.Dropout(0.25)(net)

net = layers.Conv2D(256, (3, 3), padding="same")(net)
net = layers.Conv2D(256, (3, 3), padding="same")(net)
net = layers.Conv2D(256, (3, 3), padding="same")(net)
net = layers.BatchNormalization()(net)
net = layers.Activation("relu")(net)
net = layers.MaxPooling2D(pool_size=(2, 2))(net)
net = layers.Dropout(0.25)(net)

net = layers.Conv2D(512, (3, 3), padding="same")(net)
net = layers.Conv2D(512, (3, 3), padding="same")(net)
net = layers.Conv2D(512, (3, 3), padding="same")(net)
net = layers.BatchNormalization()(net)
net = layers.Activation("relu")(net)
net = layers.MaxPooling2D(pool_size=(2, 2))(net)
net = layers.Dropout(0.25)(net)

net = layers.Conv2D(512, (3, 3), padding="same")(net)
net = layers.Conv2D(512, (3, 3), padding="same")(net)
net = layers.Conv2D(512, (3, 3), padding="same")(net)
net = layers.BatchNormalization()(net)
net = layers.Activation("relu")(net)
net = layers.MaxPooling2D(pool_size=(2, 2))(net)
net = layers.Dropout(0.25)(net)

net = layers.Flatten()(net)
net = layers.Dense(512)(net)
net = layers.Activation("relu")(net)
net = layers.Dropout(0.5)(net)
net = layers.Dense(num_classes)(net)
net = layers.Activation("softmax")(net)

model = tf.keras.Model(inputs=inputs, outputs=net)



## === cell 14
model.compile(
    loss="categorical_crossentropy",
    optimizer=tf.keras.optimizers.Adam(learning_rate),
    metrics=["accuracy"],
)

model.summary()



## === cell 15
steps_per_epoch = max(1, len(train_filenames) // batch_size)

history = model.fit(
    data_gen(train_filenames, is_training=True),
    steps_per_epoch=steps_per_epoch,
    epochs=num_epochs,
    verbose=1,
)



## === cell 16
acc_key = (
    "accuracy"
    if "accuracy" in history.history
    else ("acc" if "acc" in history.history else None)
)
loss_key = "loss"

if acc_key is not None:
    plt.plot(history.history[acc_key])
    plt.title("model accuracy")
    plt.ylabel("accuracy")
    plt.xlabel("epoch")
    plt.show()

plt.plot(history.history[loss_key])
plt.title("model loss")
plt.ylabel("loss")
plt.xlabel("epoch")
plt.show()



## === cell 17
if test_data.shape[0] == 0:
    raise RuntimeError(f"No test images found/loaded from: {test_dir}")

pred_probs = model.predict(test_data, batch_size=256, verbose=1)
has_cactus_prob = pred_probs[:, 1].astype(np.float32)

print("Pred probs shape:", pred_probs.shape)
print(
    "Submission prob stats:",
    float(has_cactus_prob.min()),
    float(has_cactus_prob.max()),
    float(has_cactus_prob.mean()),
)



## --- ERROR in cell 17, traceback:
---------------------------------------------------------------------------
RuntimeError                              Traceback (most recent call last)
/tmp/ipykernel_11/374101946.py in <cell line: 0>()
      2 # (can happen if test_data has 0 rows due to a path issue).
      3 if test_data.shape[0] == 0:
----> 4     raise RuntimeError(f"No test images found/loaded from: {test_dir}")
      5 
      6 pred_probs = model.predict(test_data, batch_size=256, verbose=1)

RuntimeError: No test images found/loaded from: ../input/aerial-cactus-identification/test/test

## === cell 18
submission = pd.DataFrame({"id": test_filenames, "has_cactus": has_cactus_prob})
submission_path = "submission.csv"
submission.to_csv(submission_path, index=False)

print("Wrote:", submission_path)
print(submission.head())
print("Rows:", len(submission), "Expected:", len(test_filenames))

## --- ERROR in cell 18, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2635626192.py in <cell line: 0>()
----> 1 submission = pd.DataFrame({"id": test_filenames, "has_cactus": has_cactus_prob})
      2 submission_path = "submission.csv"
      3 submission.to_csv(submission_path, index=False)
      4 
      5 print("Wrote:", submission_path)

NameError: name 'has_cactus_prob' is not defined
