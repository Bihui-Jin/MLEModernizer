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
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
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

0.82782

# 6. Current score

0.93479

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.7265) has done: 'I fix the environment/runtime blockers first: the protobuf/tensorflow import error, the unnecessary `efficientnet` pip install, and the unauthenticated `KaggleDatasets().get_gcs_path()` call. Then I switch the image path construction to the provided local `/kaggle/input/.../images` folder so `train_paths/test_paths` are defined and the `tf.data` pipelines can build correctly. Finally, I ensure the model output/labels match (softmax + categorical_crossentropy requires one-hot, which the CSV already provides) and write a valid `submission.csv` with the exact required column names and `.csv` suffix.'
- What this solution (achieved 0.53472) has done: 'I fix the immediate runtime blocker caused by the protobuf/TensorFlow incompatibility by removing the forced `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` override and importing TensorFlow cleanly. Then I keep your exact model/training pipeline but make the image size match the pretrained ResNet152V2’s expected resolution (224) to materially improve ROC AUC toward the target without changing the core approach. Finally, I keep the submission writing logic but add a small safety alignment to ensure row order and column names match the competition’s required format and that a valid `submission.csv` is always produced.'
- What this solution (achieved 0.58011) has done: 'I fix the immediate runtime crash (`MessageFactory.GetPrototype`) by pinning TensorFlow’s protobuf implementation to the pure-Python backend before importing TensorFlow, which is the most reliable way to avoid this TF/protobuf incompatibility in Kaggle environments. Then I keep your exact model/training/inference pipeline, but add a small, score-positive preprocessing fix: use the proper `tf.keras.applications.resnet_v2.preprocess_input` for ResNet152V2 instead of simple `/255.0` scaling (this preserves architecture/training loop while aligning inputs to the pretrained weights). Finally, I make the submission writing a bit more robust by explicitly aligning columns to the sample submission order and ensuring float dtype, while still producing `submission.csv` in the working directory.'
- What this solution (achieved 0.47919) has done: 'I fix the immediate runtime crash by removing the `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` override, which is what triggers the `MessageFactory.GetPrototype` incompatibility in this environment. I keep your model/training loop intact, but make the input pipeline deterministic/stable by enabling deterministic ops and (score-positive) adding the proper ResNet152V2 `preprocess_input` (already present) while ensuring file paths are correct. I also make submission writing robust by strictly matching the sample submission column order and guaranteeing a float output with the required `.csv` suffix. These changes are minimal, unblock end-to-end execution, and should move AUC upward from the current broken run.'
- What this solution (achieved 0.47919) has done: 'The crash happens before any training because TensorFlow is importing with an incompatible protobuf backend, producing `MessageFactory.GetPrototype` errors. The smallest reliable fix in Kaggle is to force protobuf to use the pure-Python implementation **before** importing TensorFlow, and to keep the rest of your pipeline (data, model, training, submission formatting) unchanged. I also add a tiny safety check to ensure `steps_per_epoch` is never zero (can happen if batch size exceeds dataset size on some accelerators), which prevents a runtime error without affecting semantics. With TF importing correctly again, the script run end-to-end and write a valid `submission.csv` with the required columns/order.'
- What this solution (achieved 0.58847) has done: 'I fix the immediate runtime crash by removing the protobuf “python” backend override that triggers the `MessageFactory.GetPrototype` error in this Kaggle TensorFlow/protobuf combination, letting TensorFlow import normally. Then I keep your exact model/data/training logic, but add a minimal, score-positive stability fix: split a small validation set from the training data and use it for monitoring without changing the architecture or loss (this helps prevent the model from training “blind” and typically improves AUC toward your target). Finally, I make submission generation more robust by guaranteeing correct column order and exact row alignment with `test.csv`, always writing `submission.csv` in the working directory.'
- What this solution (achieved 0.56553) has done: 'I fix the immediate runtime crash (`MessageFactory.GetPrototype`) by forcing protobuf to use the pure-Python backend **before** importing TensorFlow, which is the most reliable workaround for this TF/protobuf combo in Kaggle. I keep your data pipeline, model architecture (ResNet152V2 + GAP + Dense softmax), and training loop the same, only adding a small, score-positive stability step: label smoothing in the loss (same cross-entropy objective, better calibration for mean ROC AUC). Finally, I keep submission generation identical in semantics but make it robust to column order by strictly matching `sample_submission.csv` columns and ensuring row alignment with `test.csv`, writing `submission.csv` to the working directory.'
- What this solution (achieved 0.56553) has done: 'I fix the TensorFlow/protobuf import crash by removing the forced pure-Python protobuf backend override, which is what triggers the `MessageFactory.GetPrototype` error in this environment. Then I keep your exact data pipeline/model/training logic intact, only adding a small safety fallback to set a non-TPU strategy cleanly. Finally, I keep the same submission generation but ensure the output columns exactly match `sample_submission.csv` order and that the file is always written as `submission.csv`.'
- What this solution (achieved 0.56553) has done: 'I fix the immediate runtime crash by setting protobuf to use the pure-Python implementation **before** importing TensorFlow, which is the reliable workaround for the `MessageFactory.GetPrototype` error in this Kaggle environment. I keep your data pipeline, model (ResNet152V2 + GAP + Dense softmax), training loop, and loss semantics unchanged, only adjusting the import-time environment variables to unblock execution. I also add small safety checks around image paths and submission column ordering to guarantee the output CSV matches `sample_submission.csv` exactly and is written successfully as `submission.csv`. These changes are intended to be score-neutral (or slightly positive due to actually training/inferencing successfully) while restoring end-to-end runs.'
- What this solution (achieved 0.93479) has done: 'The crash happens before any training because forcing `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` makes TensorFlow 2.18 hit the `MessageFactory.GetPrototype` protobuf incompatibility in this environment. I remove that override and instead keep a safe import path that works with the installed TensorFlow/protobuf combo, while leaving your model/data/training logic intact. I also add a small, score-positive-but-still-core-consistent fix by freezing the pretrained ResNet base for the initial training (same architecture and loop, fewer overfitting issues on this small dataset), which should move AUC upward toward your target. Submission writing remain the same but with strict column ordering to exactly match `sample_submission.csv`.'
- What this solution (achieved 0.93479) has done: 'I fix the immediate runtime crash by forcing protobuf to use the upb (C++) backend before importing TensorFlow, which avoids the `MessageFactory.GetPrototype` error in this TF/protobuf combo. I keep your data pipeline, model architecture (ResNet152V2 + GAP + Dense softmax), training loop, and loss exactly the same so the score behavior stays close to your current 0.93479 (already above target, so we avoid any score-changing tweaks). I also add a small, score-neutral safety check to ensure the image root path resolves correctly whether the dataset is mounted under `/kaggle/input/plant-pathology-2020-fgvc7` or nested, and keep the submission column order aligned to `sample_submission.csv`. The script run end-to-end and always write a valid `submission.csv`.'

# 9. Code solution

## === cell 0
import os
import random
import numpy as np
import pandas as pd

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "upb"
os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", None)

import tensorflow as tf
import tensorflow.keras.layers as L
from tensorflow.keras.applications import ResNet152V2
from tensorflow.keras.applications.resnet_v2 import (
    preprocess_input as resnetv2_preprocess,
)
import matplotlib.pyplot as plt

SEED = 123
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

print("TF:", tf.__version__)
print("Eager:", tf.executing_eagerly())



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
AUTO = tf.data.experimental.AUTOTUNE

try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
    print("Running on TPU", tpu.master())
except Exception:
    tpu = None

if tpu:
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.TPUStrategy(tpu)
else:
    strategy = tf.distribute.get_strategy()

print("REPLICAS:", strategy.num_replicas_in_sync)



## === cell 2
candidate_roots = [
    "/kaggle/input/plant-pathology-2020-fgvc7",
    "/kaggle/input/plant-pathology-2020-fgvc7/plant-pathology-2020-fgvc7",
]
path = None
for r in candidate_roots:
    if os.path.exists(os.path.join(r, "train.csv")) and os.path.exists(
        os.path.join(r, "images")
    ):
        path = r
        break
if path is None:
    path = "/kaggle/input/plant-pathology-2020-fgvc7/"

img_path = os.path.join(path, "images")

sample_img_fp = os.path.join(img_path, "Train_0.jpg")
if not os.path.exists(sample_img_fp):
    raise FileNotFoundError(f"Expected image not found at: {sample_img_fp}")

img = plt.imread(sample_img_fp)
print(img.shape)
plt.imshow(img)
plt.axis("off")



## === cell 3
train = pd.read_csv(os.path.join(path, "train.csv"))
test = pd.read_csv(os.path.join(path, "test.csv"))
sub = pd.read_csv(os.path.join(path, "sample_submission.csv"))

label_cols = ["healthy", "multiple_diseases", "rust", "scab"]

train_paths = (
    train["image_id"].apply(lambda x: os.path.join(img_path, f"{x}.jpg")).values
)
test_paths = test["image_id"].apply(lambda x: os.path.join(img_path, f"{x}.jpg")).values
train_labels = train[label_cols].values.astype(np.float32)

print("Train:", train.shape, "Test:", test.shape, "Sub:", sub.shape)
print("Example train path exists:", train_paths[0], os.path.exists(train_paths[0]))
print("Label cols:", label_cols)

expected_sub_cols = ["image_id"] + label_cols
if list(sub.columns) != expected_sub_cols:
    sub = sub[expected_sub_cols]



## === cell 4
nb_classes = 4
BATCH_SIZE = 8 * strategy.num_replicas_in_sync
img_size = 224
EPOCHS = 5

print("BATCH_SIZE:", BATCH_SIZE, "img_size:", img_size, "EPOCHS:", EPOCHS)




## === cell 5
def decode_image(filename, label=None, image_size=(img_size, img_size)):
    bits = tf.io.read_file(filename)
    image = tf.image.decode_jpeg(bits, channels=3)
    image = tf.image.resize(image, image_size)
    image = tf.cast(image, tf.float32)
    image = resnetv2_preprocess(image)
    if label is None:
        return image
    return image, label


def data_augment(image, label=None, seed=2020):
    image = tf.image.random_flip_left_right(image, seed=seed)
    image = tf.image.random_flip_up_down(image, seed=seed)
    if label is None:
        return image
    return image, label




## === cell 6
n = len(train_paths)
idx = np.arange(n)
rng = np.random.RandomState(SEED)
rng.shuffle(idx)

val_frac = 0.10
val_n = max(1, int(n * val_frac))
val_idx = idx[:val_n]
tr_idx = idx[val_n:]

train_paths_tr = train_paths[tr_idx]
train_labels_tr = train_labels[tr_idx]
train_paths_val = train_paths[val_idx]
train_labels_val = train_labels[val_idx]

print("Train split:", train_paths_tr.shape[0], "Val split:", train_paths_val.shape[0])

train_dataset = (
    tf.data.Dataset.from_tensor_slices((train_paths_tr, train_labels_tr))
    .map(decode_image, num_parallel_calls=AUTO, deterministic=True)
    .map(data_augment, num_parallel_calls=AUTO, deterministic=True)
    .shuffle(512, seed=SEED, reshuffle_each_iteration=True)
    .repeat()
    .batch(BATCH_SIZE)
    .prefetch(AUTO)
)

val_dataset = (
    tf.data.Dataset.from_tensor_slices((train_paths_val, train_labels_val))
    .map(decode_image, num_parallel_calls=AUTO, deterministic=True)
    .batch(BATCH_SIZE)
    .prefetch(AUTO)
)

x_batch, y_batch = next(iter(train_dataset))
print("Train batch:", x_batch.shape, y_batch.shape, y_batch.dtype)



## === cell 7
test_dataset = (
    tf.data.Dataset.from_tensor_slices(test_paths)
    .map(decode_image, num_parallel_calls=AUTO, deterministic=True)
    .batch(BATCH_SIZE)
    .prefetch(AUTO)
)

x_test_batch = next(iter(test_dataset))
print("Test batch:", x_test_batch.shape)




## === cell 8
def get_model3(num_classes=nb_classes):
    base = ResNet152V2(
        input_shape=(img_size, img_size, 3),
        weights="imagenet",
        include_top=False,
    )
    base.trainable = False

    model = tf.keras.Sequential(
        [
            base,
            L.GlobalAveragePooling2D(),
            L.Dense(num_classes, activation="softmax"),
        ]
    )
    return model




## === cell 9
with strategy.scope():
    model3 = get_model3(num_classes=train_labels.shape[1])

model3.compile(
    optimizer="adam",
    loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.05),
    metrics=["categorical_accuracy"],
)
model3.summary()



## === cell 10
steps_per_epoch = max(1, train_labels_tr.shape[0] // BATCH_SIZE)
val_steps = max(1, int(np.ceil(train_labels_val.shape[0] / BATCH_SIZE)))
print("steps_per_epoch:", steps_per_epoch, "val_steps:", val_steps)

model3.fit(
    train_dataset,
    steps_per_epoch=steps_per_epoch,
    epochs=EPOCHS,
    validation_data=val_dataset,
    validation_steps=val_steps,
    verbose=1,
)



## === cell 11
probs3 = model3.predict(test_dataset, verbose=1)

assert probs3.shape[0] == len(test), (probs3.shape, len(test))
assert probs3.shape[1] == len(label_cols), (probs3.shape, len(label_cols))

sub_out = pd.DataFrame({"image_id": test["image_id"].values})
sub_out[label_cols] = probs3.astype(np.float32)

sub_out = sub_out.set_index("image_id").loc[test["image_id"]].reset_index()
sub_out = sub_out[expected_sub_cols]
sub_out.to_csv("submission.csv", index=False)

print(sub_out.head())
print("Wrote:", os.path.abspath("submission.csv"))



## === cell 12
for dirname, _, filenames in os.walk("./"):
    for filename in filenames:
        if filename.endswith(".csv"):
            print(os.path.join(dirname, filename))
