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

3.9

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

0.9723192955112148

# 6. Current score

0.51244

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.46902) has done: 'I speed up the run by extracting EfficientNetB0 features once (the base network is frozen, so this gives the same representations as feeding images through it each epoch). The classifier is then trained on these pre‑computed vectors, eliminating the heavy CNN work during every training step. I also remove the data‑augmentation step (it only adds extra compute) and keep all seeds, strategy handling, and the exact model architecture for the classifier, preserving the original logic and predictions.'
- What this solution (achieved 0.48674) has done: 'I removed the mixed‑precision setup that was causing a protobuf import error and changed the classifier to use a sigmoid output with binary cross‑entropy loss, which matches the multi‑label nature of the task and raises the ROC‑AUC. The rest of the pipeline is kept unchanged, and the script now writes a proper `submission.csv`.'
- What this solution (achieved 0.50973) has done: 'Implemented a protobuf compatibility fix, added a small hidden dense layer with dropout to strengthen the classifier, switched to AUC metric, and extended training epochs with a modest early‑stopping patience. These changes resolve the import error and raise the validation ROC‑AUC toward the target while keeping the original workflow intact.'
- What this solution (achieved 0.50634) has done: 'Implemented a robust fix that avoids the protobuf import error by wrapping TensorFlow imports in a safe try‑except block. If TensorFlow cannot be loaded, the script now falls back to a lightweight Scikit‑Learn pipeline that reads and resizes images with Pillow, flattens them, applies a modest PCA reduction, and trains a One‑Vs‑Rest Logistic Regression model. The rest of the workflow (train/validation split, metric calculation, and CSV submission) remains unchanged, ensuring a valid `submission.csv` is produced while preserving the original logic when TensorFlow is available. This change resolves the runtime crash and keeps the scoring pipeline functional.'
- What this solution (achieved 0.51244) has done: 'Implemented a more powerful sklearn fallback to boost AUC when TensorFlow cannot be loaded. The change replaces the simple Logistic Regression OVR model with a PCA + One‑Vs‑Rest RandomForest classifier (200 PCA components, 300 trees). This improves representational power while keeping the original pipeline structure and all non‑TF guards unchanged.'

# 9. Code solution

## === cell 0
import os, random, math, re, sys

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import numpy as np
import pandas as pd

USE_TF = True
try:
    import tensorflow as tf
    from tensorflow.keras import layers, Model
    from tensorflow.keras.applications import EfficientNetB0
    from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input, Dropout
except Exception as e:
    print("TensorFlow import failed:", e)
    USE_TF = False

from sklearn.model_selection import train_test_split

if USE_TF:
    tf.config.optimizer.set_jit(True)
    tf.random.set_seed(2020)
    np.random.seed(2020)
    print("TensorFlow version:", tf.__version__)



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.experimental.TPUStrategy(tpu)
except Exception:
    strategy = tf.distribute.get_strategy() if USE_TF else None
print("Using strategy:", strategy.__class__.__name__ if strategy else "None")

AUTO = tf.data.experimental.AUTOTUNE if USE_TF else None
BATCH_SIZE = 128 * (strategy.num_replicas_in_sync if strategy else 1)
img_size = 224
DATA_ROOT = "/kaggle/input/plant-pathology-2020-fgvc7"


def format_path(st):
    return os.path.join(DATA_ROOT, "images", f"{st}.jpg")




## === cell 2
train_df = pd.read_csv(os.path.join(DATA_ROOT, "train.csv"))
test_df = pd.read_csv(os.path.join(DATA_ROOT, "test.csv"))
sample_submission = pd.read_csv(os.path.join(DATA_ROOT, "sample_submission.csv"))

label_cols = ["healthy", "multiple_diseases", "rust", "scab"]
train_labels = train_df[label_cols].values.astype(np.float32)

train_paths = train_df["image_id"].apply(format_path).values
test_paths = test_df["image_id"].apply(format_path).values

train_paths, valid_paths, train_labels, valid_labels = train_test_split(
    train_paths,
    train_labels,
    test_size=0.15,
    random_state=2020,
    stratify=train_labels.argmax(axis=1),
)



## === cell 3
if USE_TF:

    def decode_image(filename, label=None, image_size=(img_size, img_size)):
        bits = tf.io.read_file(filename)
        img = tf.image.decode_jpeg(bits, channels=3)
        img = tf.cast(img, tf.float32) / 255.0
        img = tf.image.resize(img, image_size)
        return (img, label) if label is not None else img

    train_dataset_img = (
        tf.data.Dataset.from_tensor_slices((train_paths, train_labels))
        .map(decode_image, num_parallel_calls=AUTO)
        .cache()
        .batch(BATCH_SIZE)
        .prefetch(AUTO)
    )

    valid_dataset_img = (
        tf.data.Dataset.from_tensor_slices((valid_paths, valid_labels))
        .map(decode_image, num_parallel_calls=AUTO)
        .cache()
        .batch(BATCH_SIZE)
        .prefetch(AUTO)
    )

    test_dataset_img = (
        tf.data.Dataset.from_tensor_slices(test_paths)
        .map(decode_image, num_parallel_calls=AUTO)
        .cache()
        .batch(BATCH_SIZE)
        .prefetch(AUTO)
    )
else:
    from PIL import Image

    def load_and_preprocess(paths):
        imgs = []
        for p in paths:
            img = Image.open(p).convert("RGB")
            img = img.resize((img_size, img_size))
            arr = np.asarray(img, dtype=np.float32) / 255.0
            imgs.append(arr)
        return np.stack(imgs)

    train_images = load_and_preprocess(train_paths)
    valid_images = load_and_preprocess(valid_paths)
    test_images = load_and_preprocess(test_paths)



## === cell 4
if USE_TF:
    with strategy.scope():
        base = EfficientNetB0(
            weights="imagenet", include_top=False, input_shape=(img_size, img_size, 3)
        )
        base.trainable = False
        feature_extractor = tf.keras.Sequential([base, GlobalAveragePooling2D()])

        train_features = feature_extractor.predict(train_dataset_img, verbose=0)
        valid_features = feature_extractor.predict(valid_dataset_img, verbose=0)
        test_features = feature_extractor.predict(test_dataset_img, verbose=0)

        inputs = Input(shape=train_features.shape[1:])
        x = Dense(256, activation="relu")(inputs)
        x = Dropout(0.2)(x)
        outputs = Dense(len(label_cols), activation="sigmoid", dtype="float32")(x)

        classifier = Model(inputs, outputs)
        classifier.compile(
            optimizer="adam",
            loss="binary_crossentropy",
            metrics=[tf.keras.metrics.AUC(multi_label=True, name="auc")],
        )

        train_feat_ds = (
            tf.data.Dataset.from_tensor_slices((train_features, train_labels))
            .shuffle(1024)
            .batch(BATCH_SIZE)
            .prefetch(AUTO)
        )
        valid_feat_ds = (
            tf.data.Dataset.from_tensor_slices((valid_features, valid_labels))
            .batch(BATCH_SIZE)
            .prefetch(AUTO)
        )
else:
    from sklearn.decomposition import PCA
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.multiclass import OneVsRestClassifier
    from sklearn.metrics import roc_auc_score

    N_train = train_images.shape[0]
    N_valid = valid_images.shape[0]

    train_flat = train_images.reshape(N_train, -1)
    valid_flat = valid_images.reshape(N_valid, -1)

    pca = PCA(n_components=200, random_state=2020)
    train_feat = pca.fit_transform(train_flat)
    valid_feat = pca.transform(valid_flat)

    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        random_state=2020,
        n_jobs=-1,
        class_weight="balanced",
    )
    ovr = OneVsRestClassifier(rf)
    ovr.fit(train_feat, train_labels)

    classifier = ovr
    pca_fitted = pca



## === cell 5
if USE_TF:
    classifier.fit(
        train_feat_ds,
        epochs=30,
        validation_data=valid_feat_ds,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                patience=5,
                restore_best_weights=True,
                monitor="val_auc",
                mode="max",
            )
        ],
        verbose=2,
    )
else:
    valid_pred = classifier.predict_proba(valid_feat)
    val_auc = roc_auc_score(valid_labels, valid_pred, average="macro")
    print(f"Validation AUC (sklearn fallback): {val_auc:.5f}")



## === cell 6
if USE_TF:
    test_feat_ds = tf.data.Dataset.from_tensor_slices(test_features).batch(BATCH_SIZE)
    probabilities = classifier.predict(test_feat_ds, verbose=1)
else:
    test_flat = test_images.reshape(test_images.shape[0], -1)
    test_feat = pca_fitted.transform(test_flat)
    probabilities = classifier.predict_proba(test_feat)

submission = sample_submission.copy()
submission.loc[:, label_cols] = probabilities
submission_path = "submission.csv"
submission.to_csv(submission_path, index=False)
print(f"Submission written to {submission_path}")
