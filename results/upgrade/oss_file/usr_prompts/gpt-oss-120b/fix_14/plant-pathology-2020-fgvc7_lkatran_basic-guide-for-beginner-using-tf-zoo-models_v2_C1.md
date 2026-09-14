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
Detect apple diseases from images.

## Metric
Mean column-wise ROC AUC.

## Submission Format
For each image_id in the test set, you must predict a probability for each target variable. The file should contain a header and have the following format:

```
image_id,
test_0,0.25,0.25,0.25,0.25
test_1,0.25,0.25,0.25,0.25
test_2,0.25,0.25,0.25,0.25
etc.
```

## Dataset
Given a photo of an apple leaf, can you accurately assess its health? This competition will challenge you to distinguish between leaves which are healthy, those which are infected with apple rust, those that have apple scab, and those with more than one disease.

**train.csv**

- `image_id`: the foreign key
- combinations: one of the target labels
- healthy: one of the target labels
- rust: one of the target labels
- scab: one of the target labels

**images**

A folder containing the train and test images, in jpg format.

**test.csv**

- `image_id`: the foreign key

**sample_submission.csv**

- `image_id`: the foreign key
- combinations: one of the target labels
- healthy: one of the target labels
- rust: one of the target labels
- scab: one of the target labels

# 2. Python version

3.8

# 3. Installed packages

No external packages required in the script and installed.

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (94 lines)
            images.zip (397.8 MB)
            sample_submission.csv (184 lines)
            sample_submission.csv.zip (682 Bytes)
            test.csv (184 lines)
            test.csv.zip (542 Bytes)
            train.csv (1639 lines)
            train.csv.zip (4.6 kB)
            images/
                Train_370.jpg (133.2 kB)
                Test_59.jpg (220.5 kB)
                ... and 1819 other files
            plant-pathology-2020-fgvc7/
                description.md (94 lines)
                images.zip (397.8 MB)
                ... and 6 other files
                images/
                    Train_370.jpg (133.2 kB)
                    Test_59.jpg (220.5 kB)
                    ... and 1819 other files
                plant-pathology-2020-fgvc7/
        input/
            description.md (94 lines)
            images.zip (397.8 MB)
            sample_submission.csv (184 lines)
            sample_submission.csv.zip (682 Bytes)
            test.csv (184 lines)
            test.csv.zip (542 Bytes)
            train.csv (1639 lines)
            train.csv.zip (4.6 kB)
            images/
                Train_370.jpg (133.2 kB)
                Test_59.jpg (220.5 kB)
                ... and 1819 other files
            plant-pathology-2020-fgvc7/
                description.md (94 lines)
                images.zip (397.8 MB)
                ... and 6 other files
                images/
                    Train_370.jpg (133.2 kB)
                    Test_59.jpg (220.5 kB)
                    ... and 1819 other files
                plant-pathology-2020-fgvc7/
        working/
            plant-pathology-2020-fgvc7/
                description.md (94 lines)
                images.zip (397.8 MB)
                ... and 6 other files
                images/
                    Train_370.jpg (133.2 kB)
                    Test_59.jpg (220.5 kB)
                    ... and 1819 other files
                plant-pathology-2020-fgvc7/
```

-> data/plant-pathology-2020-fgvc7/sample_submission.csv has 183 rows and 5 columns.
The columns are: image_id, healthy, multiple_diseases, rust, scab

-> data/plant-pathology-2020-fgvc7/test.csv has 183 rows and 1 columns.
The columns are: image_id

-> data/plant-pathology-2020-fgvc7/train.csv has 1638 rows and 5 columns.
The columns are: image_id, healthy, multiple_diseases, rust, scab

-> data/sample_submission.csv has 183 rows and 5 columns.
The columns are: image_id, healthy, multiple_diseases, rust, scab

-> data/test.csv has 183 rows and 1 columns.
The columns are: image_id

-> data/train.csv has 1638 rows and 5 columns.
The columns are: image_id, healthy, multiple_diseases, rust, scab

-> (stopped after 10 files for performance)

# 5. Target score

0.7953659417082056

# 6. Current score

0.59702

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.48708) has done: 'I fixed the runtime errors by removing the incompatible call to `datagen.fit`, switched to the modern `model.fit` API, and corrected the model for the multi‑label task (using sigmoid activation and binary cross‑entropy loss with an AUC metric). These changes allow the script to run end‑to‑end and produce a proper `submission.csv`, while the updated loss/activation should raise the ROC‑AUC score toward the target.'
- What this solution (achieved 0.59702) has done: 'I speed up the pipeline by (1) limiting the image‑loading workers to a modest number and reusing a single constant, (2) adding TensorFlow deterministic seeds, (3) caching the training and validation datasets in memory to avoid repeated transfers each epoch, and (4) keeping all model architecture, training loops, and callbacks unchanged. These changes cut unnecessary overhead while preserving exact training behavior and final predictions.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd
from datetime import datetime as dt
from tqdm import tqdm
import concurrent.futures

os.environ["TF_DETERMINISTIC_OPS"] = "1"
import tensorflow as tf

tf.random.set_seed(42)

if tf.config.list_physical_devices("GPU"):
    from tensorflow.keras.mixed_precision import experimental as mixed_precision

    mixed_precision.set_policy("mixed_float16")
    for gpu in tf.config.experimental.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(gpu, True)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization, Dropout, Dense
from tensorflow.keras.applications import DenseNet201
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from tensorflow.keras.metrics import AUC

import matplotlib.pyplot as plt
from PIL import Image

submission = pd.read_csv(
    "/kaggle/input/plant-pathology-2020-fgvc7/sample_submission.csv"
)
train = pd.read_csv("/kaggle/input/plant-pathology-2020-fgvc7/train.csv")
test = pd.read_csv("/kaggle/input/plant-pathology-2020-fgvc7/test.csv")

MAX_WORKERS = 4




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
def _load_and_preprocess(im_id, base_path):
    """Read, resize, and convert an image to uint8 (RGB) using Pillow."""
    im_path = os.path.join(base_path, f"{im_id}.jpg")
    with Image.open(im_path) as img:
        img = img.convert("RGB")
        img = img.resize((224, 224), Image.Resampling.BILINEAR)
        return np.array(img, dtype=np.uint8)  # keep as uint8; convert later


train_cache_path = "train_img.npy"
if os.path.exists(train_cache_path):
    train_img = np.load(train_cache_path)
else:
    img_dir = "/kaggle/input/plant-pathology-2020-fgvc7/images"
    n_train = len(train)
    train_img = np.empty((n_train, 224, 224, 3), dtype=np.uint8)
    with concurrent.futures.ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        for idx, img in enumerate(
            tqdm(
                executor.map(
                    _load_and_preprocess, train["image_id"], [img_dir] * n_train
                ),
                total=n_train,
                desc="Loading train images",
            )
        ):
            train_img[idx] = img
    np.save(train_cache_path, train_img)

test_cache_path = "test_img.npy"
if os.path.exists(test_cache_path):
    test_img = np.load(test_cache_path)
else:
    img_dir = "/kaggle/input/plant-pathology-2020-fgvc7/images"
    n_test = len(test)
    test_img = np.empty((n_test, 224, 224, 3), dtype=np.uint8)
    with concurrent.futures.ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        for idx, img in enumerate(
            tqdm(
                executor.map(
                    _load_and_preprocess, test["image_id"], [img_dir] * n_test
                ),
                total=n_test,
                desc="Loading test images",
            )
        ):
            test_img[idx] = img
    np.save(test_cache_path, test_img)




## === cell 2
train_label = train.loc[:, "healthy":"scab"]  # 4 columns
train_img = train_img.astype(np.float32) / 255.0
test_img = test_img.astype(np.float32) / 255.0
train_label = np.array(train_label, dtype="float32")

X_train, X_val, y_train, y_val = train_test_split(
    train_img, train_label, test_size=0.1, random_state=42
)

batch_size = 32  # reasonable batch size for memory

train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train))
train_ds = (
    train_ds.cache()  # cache in memory to avoid recomputation
    .shuffle(buffer_size=len(X_train), seed=42)
    .batch(batch_size)
    .prefetch(tf.data.AUTOTUNE)
)

val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val))
val_ds = val_ds.cache().batch(batch_size).prefetch(tf.data.AUTOTUNE)




## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/513157913.py in <cell line: 0>()
      4 train_label = np.array(train_label, dtype="float32")
      5 
----> 6 X_train, X_val, y_train, y_val = train_test_split(
      7     train_img, train_label, test_size=0.1, random_state=42
      8 )

NameError: name 'train_test_split' is not defined

## === cell 3
tf.keras.backend.clear_session()

base_model = DenseNet201(
    include_top=False, weights="imagenet", input_shape=(224, 224, 3), pooling="avg"
)
base_model.trainable = False

model = Sequential(
    [
        base_model,
        BatchNormalization(),
        Dropout(0.5),
        Dense(128, activation="relu"),
        Dense(4, activation="sigmoid"),  # 4 target columns
    ]
)

reduce_lr = ReduceLROnPlateau(
    monitor="val_auc", factor=0.1, patience=2, cooldown=2, min_lr=1e-7, verbose=1
)
early_stop = EarlyStopping(monitor="val_auc", patience=5, restore_best_weights=True)
checkpoint = ModelCheckpoint(
    filepath="best_model.h5", monitor="val_auc", save_best_only=True, verbose=0
)

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=[AUC(name="auc")])




## === cell 4
start = dt.now()
history = model.fit(
    train_ds,
    epochs=12,  # modest number of epochs for speed
    validation_data=val_ds,
    callbacks=[reduce_lr, early_stop, checkpoint],
    verbose=2,
)
print(f"Training time: {dt.now() - start}. Epochs run: {len(history.epoch)}")




## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1887225806.py in <cell line: 0>()
      1 start = dt.now()
      2 history = model.fit(
----> 3     train_ds,
      4     epochs=12,  # modest number of epochs for speed
      5     validation_data=val_ds,

NameError: name 'train_ds' is not defined

## === cell 5
def plot_metric(his, metric_name, title):
    epochs = len(his.epoch)
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(np.arange(epochs), his.history[metric_name], label=f"train_{metric_name}")
    plt.plot(
        np.arange(epochs), his.history[f"val_{metric_name}"], label=f"val_{metric_name}"
    )
    plt.title(title)
    plt.xlabel("Epoch #")
    plt.ylabel(metric_name.capitalize())
    plt.legend(loc="lower right")
    plt.show()


plot_metric(history, "loss", "Training & Validation Loss")
plot_metric(history, "auc", "Training & Validation AUC")




## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/854303675.py in <cell line: 0>()
     14 
     15 
---> 16 plot_metric(history, "loss", "Training & Validation Loss")
     17 plot_metric(history, "auc", "Training & Validation AUC")
     18 

NameError: name 'history' is not defined

## === cell 6
y_pred = model.predict(test_img, batch_size=32)

submission.loc[:, "healthy":"scab"] = y_pred

submission.to_csv("submission.csv", index=False)
