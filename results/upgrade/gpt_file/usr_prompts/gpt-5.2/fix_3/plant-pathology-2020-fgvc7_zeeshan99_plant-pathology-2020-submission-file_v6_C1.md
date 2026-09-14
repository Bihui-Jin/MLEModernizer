# Goal

I want you to improve my Kaggle competition solution to increase the score toward a target. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (evaluation score improvement); avoid unrelated refactors or stylistic edits.
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

geopandas==0.14.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
protobuf==6.33.0
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

0.83351

# 6. Current score

0.9911

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.50706) has done: 'I fix the environment-breaking import error by patching the protobuf `MessageFactory.GetPrototype` incompatibility that can occur with TF + protobuf 6, so TensorFlow can import cleanly. Then I remove the hard dependency on a missing external TPU model file by falling back to a small, standard TensorFlow image model trained on the provided `train.csv`/images and used to predict on `test.csv`, preserving the same “predict probabilities for 4 classes” semantics. I also correct submission formatting to exactly match `sample_submission.csv` (must include `image_id` and the 4 label columns, with `.csv` suffix). Finally, I ensure image paths/filenames match the dataset layout and that prediction order aligns with `test.csv`.'
- What this solution (achieved 0.9911) has done: 'Your current score is far below the target, so the smallest safe way to improve toward 0.83351 is to keep your EfficientNetB0 core but fix two high-impact training mismatches: use the correct image preprocessing for EfficientNet (instead of plain `/255`) and train all available data more effectively via K-fold out-of-fold averaging (same model, same loss, same pipeline semantics, just repeated training). I also ensure the train/val split is multilabel-safe (not forcing `idxmax` “single-label” stratification), and I average fold predictions to improve AUC stability without changing the submission format. These changes are directly tied to better ROC AUC and should move the score significantly upward while preserving your overall approach and staying within runtime limits.'

# 9. Code solution

## === cell 0
import os
import random
import numpy as np
import pandas as pd

try:
    from google.protobuf import message_factory as _message_factory

    if hasattr(_message_factory, "MessageFactory") and not hasattr(
        _message_factory.MessageFactory, "GetPrototype"
    ):

        def _get_prototype(self, descriptor):
            return self.GetMessageClass(descriptor)

        _message_factory.MessageFactory.GetPrototype = _get_prototype
except Exception:
    pass

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

print("TF version:", tf.__version__)
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass




## === cell 1
BASE_INPUT = "/kaggle/input/plant-pathology-2020-fgvc7"
if not os.path.exists(BASE_INPUT):
    BASE_INPUT = "/kaggle/data/plant-pathology-2020-fgvc7"

TRAIN_CSV = os.path.join(BASE_INPUT, "train.csv")
TEST_CSV = os.path.join(BASE_INPUT, "test.csv")
SAMPLE_SUB = os.path.join(BASE_INPUT, "sample_submission.csv")
IMG_DIR = os.path.join(BASE_INPUT, "images")

assert os.path.exists(TRAIN_CSV), f"Missing {TRAIN_CSV}"
assert os.path.exists(TEST_CSV), f"Missing {TEST_CSV}"
assert os.path.exists(SAMPLE_SUB), f"Missing {SAMPLE_SUB}"
assert os.path.isdir(IMG_DIR), f"Missing {IMG_DIR}"

targets = ["healthy", "multiple_diseases", "rust", "scab"]

train_df = pd.read_csv(TRAIN_CSV)
test_df = pd.read_csv(TEST_CSV)
sample_df = pd.read_csv(SAMPLE_SUB)

for c in ["image_id"] + targets:
    assert c in train_df.columns, f"train.csv missing column {c}"
assert "image_id" in test_df.columns, "test.csv missing image_id"
for c in ["image_id"] + targets:
    assert c in sample_df.columns, f"sample_submission.csv missing column {c}"

train_df = train_df.copy()
test_df = test_df.copy()
train_df["filename"] = train_df["image_id"].astype(str) + ".jpg"
test_df["filename"] = test_df["image_id"].astype(str) + ".jpg"

missing_train = sum(
    ~train_df["filename"]
    .head(20)
    .apply(lambda x: os.path.exists(os.path.join(IMG_DIR, x)))
)
missing_test = sum(
    ~test_df["filename"]
    .head(20)
    .apply(lambda x: os.path.exists(os.path.join(IMG_DIR, x)))
)
print("Missing (first 20) train files:", missing_train, "test files:", missing_test)

for c in targets:
    train_df[c] = train_df[c].astype("float32")




## === cell 2
def processAndWriteDf(
    df: pd.DataFrame, out_path: str = "./submission.csv"
) -> pd.DataFrame:
    """
    Submission must include image_id + 4 target columns.
    Ensure correct column order and valid probabilities.
    """
    needed = ["image_id"] + targets
    for c in needed:
        if c not in df.columns:
            raise ValueError(f"Submission dataframe missing column: {c}")

    sub = df[needed].copy()
    for c in targets:
        sub[c] = sub[c].astype("float32").clip(0.0, 1.0)

    sub = sub.merge(sample_df[["image_id"]], on="image_id", how="right")
    sub = sub[needed]

    sub.to_csv(out_path, index=False)
    print(f"Wrote submission: {out_path} shape={sub.shape}")
    return sub




## === cell 3
def build_model(input_shape=(224, 224, 3), n_classes=4):
    base = tf.keras.applications.EfficientNetB0(
        include_top=False, weights="imagenet", input_shape=input_shape, pooling="avg"
    )
    x = tf.keras.layers.Dropout(0.2)(base.output)
    out = tf.keras.layers.Dense(n_classes, activation="sigmoid")(x)
    model = tf.keras.Model(inputs=base.input, outputs=out)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss="binary_crossentropy",
        metrics=[
            tf.keras.metrics.AUC(
                curve="ROC", multi_label=True, num_labels=n_classes, name="auc"
            )
        ],
    )
    return model




## === cell 4
def getPredictionFromTPUModel():
    """
    Improve score toward target by:
    - Using EfficientNetB0's correct preprocessing (instead of plain /255).
      This directly improves feature alignment with ImageNet weights -> better AUC.
    - Using small K-fold training (same model, same loss) and averaging predictions.
      This reduces variance and typically boosts mean column-wise ROC AUC.
    """
    candidate_paths = [
        "/kaggle/input/plant-pathology-2020-tpu/my_modelv5.h5",
        "../input/plant-pathology-2020-tpu/my_modelv5.h5",
    ]
    model = None
    for p in candidate_paths:
        if os.path.exists(p):
            print("Loading existing model:", p)
            model = tf.keras.models.load_model(p, compile=False)
            break

    img_size = (224, 224)
    batch_size = 16

    preprocess_fn = tf.keras.applications.efficientnet.preprocess_input

    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_fn,
        rotation_range=15,
        width_shift_range=0.05,
        height_shift_range=0.05,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode="nearest",
    )
    val_datagen = ImageDataGenerator(preprocessing_function=preprocess_fn)
    test_datagen = ImageDataGenerator(preprocessing_function=preprocess_fn)

    test_gen = test_datagen.flow_from_dataframe(
        test_df,
        directory=IMG_DIR,
        x_col="filename",
        y_col=None,
        target_size=img_size,
        batch_size=batch_size,
        class_mode=None,
        shuffle=False,
    )

    if model is not None:
        p = model.predict(test_gen, verbose=1)
        p_df = pd.DataFrame(p, columns=targets)
        result = pd.concat([test_df[["image_id"]].copy(), p_df], axis=1)
        return result

    from sklearn.model_selection import StratifiedKFold

    y_key = train_df[targets].astype(int).astype(str).agg("".join, axis=1).values

    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)

    test_pred_accum = np.zeros((len(test_df), len(targets)), dtype=np.float32)

    for fold, (tr_idx, va_idx) in enumerate(skf.split(train_df, y_key), start=1):
        print(f"\n=== Fold {fold}/3 ===")
        train_part = train_df.iloc[tr_idx].reset_index(drop=True)
        val_part = train_df.iloc[va_idx].reset_index(drop=True)

        train_gen = train_datagen.flow_from_dataframe(
            train_part,
            directory=IMG_DIR,
            x_col="filename",
            y_col=targets,
            target_size=img_size,
            batch_size=batch_size,
            class_mode="raw",
            shuffle=True,
            seed=SEED
            + fold,  # slight fold-specific shuffle while remaining deterministic
        )
        val_gen = val_datagen.flow_from_dataframe(
            val_part,
            directory=IMG_DIR,
            x_col="filename",
            y_col=targets,
            target_size=img_size,
            batch_size=batch_size,
            class_mode="raw",
            shuffle=False,
        )

        fold_model = build_model(input_shape=img_size + (3,), n_classes=len(targets))

        fold_model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=5,
            verbose=1,
        )

        p = fold_model.predict(test_gen, verbose=1)
        test_pred_accum += p.astype(np.float32)

        tf.keras.backend.clear_session()

    test_pred_mean = test_pred_accum / 3.0
    p_df = pd.DataFrame(test_pred_mean, columns=targets)
    result = pd.concat([test_df[["image_id"]].copy(), p_df], axis=1)
    return result




## === cell 5
isTPU = True

if isTPU:
    df_pred = getPredictionFromTPUModel()
    print(df_pred.head(5))
    submission = processAndWriteDf(df_pred, out_path="./submission.csv")
    print(submission.head())
else:
    df = pd.read_csv("../input/notebook45bc751087/submission.csv")
    submission = processAndWriteDf(df, out_path="./submission.csv")
    print(submission.head())
