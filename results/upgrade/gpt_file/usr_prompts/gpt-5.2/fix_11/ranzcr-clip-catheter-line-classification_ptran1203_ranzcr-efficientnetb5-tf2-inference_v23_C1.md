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
Detect the presence and position of catheters and lines on chest x-rays.

## Metric
Area under the ROC curve for each label, with the final score being the average of the individual AUCs of each predicted column.

## Submission Format
For each ID in the test set, you must predict a probability for all target variables. The file should contain a header and have the following format:
```
StudyInstanceUID,ETT - Abnormal,ETT - Borderline,ETT - Normal,NGT - Abnormal,NGT - Borderline,NGT - Incompletely Imaged,NGT - Normal,CVC - Abnormal,CVC - Borderline,CVC - Normal,Swan Ganz Catheter Present
1.2.826.0.1.3680043.8.498.62451881164053375557257228990443168843,0,0,0,0,0,0,0,0,0,0,0
1.2.826.0.1.3680043.8.498.83721761279899623084220697845011427274,0,0,0,0,0,0,0,0,0,0,0
1.2.826.0.1.3680043.8.498.12732270010839808189235995393981377825,0,0,0,0,0,0,0,0,0,0,0
1.2.826.0.1.3680043.8.498.11769539755086084996287023095028033598,0,0,0,0,0,0,0,0,0,0,0
1.2.826.0.1.3680043.8.498.87838627504097587943394933987052577153,0,0,0,0,0,0,0,0,0,0,0
1.2.826.0.1.3680043.8.498.53211840524738036417560823327351887819,0,0,0,0,0,0,0,0,0,0,0
1.2.826.0.1.3680043.8.498.93555795394184819372299157360228027866,0,0,0,0,0,0,0,0,0,0,0
1.2.826.0.1.3680043.8.498.52241894131170494723503100795076463919,0,0,0,0,0,0,0,0,0,0,0
1.2.826.0.1.3680043.8.498.36500167484503936720548852591033878284,0,0,0,0,0,0,0,0,0,0,0
1.2.826.0.1.3680043.8.498.86199852603457900780565655267977637728,0,0,0,0,0,0,0,0,0,0,0
```

## Dataset
`train.csv` contains image IDs, binary labels, and patient IDs.

TFRecords are available for both train and test.

We've also included `train_annotations.csv`. These are segmentation annotations for training samples that have them. They are included solely as additional information for competitors.

- train.csv - contains image IDs, binary labels, and patient IDs.
- sample_submission.csv - a sample submission file in the correct format
- test - test images
- train - training images

### Columns
- `StudyInstanceUID` - unique ID for each image
- `ETT - Abnormal` - endotracheal tube placement abnormal
- `ETT - Borderline` - endotracheal tube placement borderline abnormal
- `ETT - Normal` - endotracheal tube placement normal
- `NGT - Abnormal` - nasogastric tube placement abnormal
- `NGT - Borderline` - nasogastric tube placement borderline abnormal
- `NGT - Incompletely Imaged` - nasogastric tube placement inconclusive due to imaging
- `NGT - Normal` - nasogastric tube placement borderline normal
- `CVC - Abnormal` - central venous catheter placement abnormal
- `CVC - Borderline` - central venous catheter placement borderline abnormal
- `CVC - Normal` - central venous catheter placement normal
- `Swan Ganz Catheter Present`
- `PatientID` - unique ID for each patient in the dataset

# 2. Python version

3.9

# 3. Installed packages

geopandas==0.14.4
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
numpy==1.26.4
opencv-python==4.12.0.88
opencv-python-headless==4.12.0.88
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
            description.md (172 lines)
            sample_submission.csv (3010 lines)
            sample_submission.csv.zip (64.2 kB)
            test.zip (642.8 MB)
            train.csv (27075 lines)
            train.csv.zip (798.6 kB)
            train.zip (5.8 GB)
            train_annotations.csv (16262 lines)
            train_annotations.csv.zip (1.4 MB)
            ranzcr-clip-catheter-line-classification/
                description.md (172 lines)
                sample_submission.csv (3010 lines)
                ... and 7 other files
                ranzcr-clip-catheter-line-classification/
                test/
                    1.2.826.0.1.3680043.8.498.57477258980718966370268402373568359767.jpg (201.5 kB)
                    1.2.826.0.1.3680043.8.498.70369997506092680321332747830211112877.jpg (176.2 kB)
                    ... and 3007 other files
                    test/
                train/
                    1.2.826.0.1.3680043.8.498.22695482406757766723043400617436591679.jpg (306.8 kB)
                    1.2.826.0.1.3680043.8.498.67485046061813043086989609654834659261.jpg (315.8 kB)
                    ... and 27072 other files
                    train/
            test/
                1.2.826.0.1.3680043.8.498.57477258980718966370268402373568359767.jpg (201.5 kB)
                1.2.826.0.1.3680043.8.498.70369997506092680321332747830211112877.jpg (176.2 kB)
                ... and 3007 other files
                test/
            train/
                1.2.826.0.1.3680043.8.498.22695482406757766723043400617436591679.jpg (306.8 kB)
                1.2.826.0.1.3680043.8.498.67485046061813043086989609654834659261.jpg (315.8 kB)
                ... and 27072 other files
                train/
        input/
            description.md (172 lines)
            sample_submission.csv (3010 lines)
            sample_submission.csv.zip (64.2 kB)
            test.zip (642.8 MB)
            train.csv (27075 lines)
            train.csv.zip (798.6 kB)
            train.zip (5.8 GB)
            train_annotations.csv (16262 lines)
            train_annotations.csv.zip (1.4 MB)
            ranzcr-clip-catheter-line-classification/
                description.md (172 lines)
                sample_submission.csv (3010 lines)
                ... and 7 other files
                ranzcr-clip-catheter-line-classification/
                test/
                    1.2.826.0.1.3680043.8.498.57477258980718966370268402373568359767.jpg (201.5 kB)
                    1.2.826.0.1.3680043.8.498.70369997506092680321332747830211112877.jpg (176.2 kB)
                    ... and 3007 other files
                    test/
                train/
                    1.2.826.0.1.3680043.8.498.22695482406757766723043400617436591679.jpg (306.8 kB)
                    1.2.826.0.1.3680043.8.498.67485046061813043086989609654834659261.jpg (315.8 kB)
                    ... and 27072 other files
                    train/
            test/
                1.2.826.0.1.3680043.8.498.57477258980718966370268402373568359767.jpg (201.5 kB)
                1.2.826.0.1.3680043.8.498.70369997506092680321332747830211112877.jpg (176.2 kB)
                ... and 3007 other files
                test/
                    1.2.826.0.1.3680043.8.498.57477258980718966370268402373568359767.jpg (201.5 kB)
                    1.2.826.0.1.3680043.8.498.70369997506092680321332747830211112877.jpg (176.2 kB)
                    ... and 3007 other files
                    test/
            train/
                1.2.826.0.1.3680043.8.498.22695482406757766723043400617436591679.jpg (306.8 kB)
                1.2.826.0.1.3680043.8.498.67485046061813043086989609654834659261.jpg (315.8 kB)
                ... and 27072 other files
                train/
                    1.2.826.0.1.3680043.8.498.22695482406757766723043400617436591679.jpg (306.8 kB)
                    1.2.826.0.1.3680043.8.498.67485046061813043086989609654834659261.jpg (315.8 kB)
                    ... and 27072 other files
                    train/
        working/
            ranzcr-clip-catheter-line-classification/
                description.md (172 lines)
                sample_submission.csv (3010 lines)
                ... and 7 other files
                ranzcr-clip-catheter-line-classification/
                test/
                    1.2.826.0.1.3680043.8.498.57477258980718966370268402373568359767.jpg (201.5 kB)
                    1.2.826.0.1.3680043.8.498.70369997506092680321332747830211112877.jpg (176.2 kB)
                    ... and 3007 other files
                    test/
                train/
                    1.2.826.0.1.3680043.8.498.22695482406757766723043400617436591679.jpg (306.8 kB)
                    1.2.826.0.1.3680043.8.498.67485046061813043086989609654834659261.jpg (315.8 kB)
                    ... and 27072 other files
                    train/
```

-> data/ranzcr-clip-catheter-line-classification/sample_submission.csv has 3009 rows and 10 columns.
The columns are: StudyInstanceUID, ETT - Abnormal, ETT - Borderline, ETT - Normal, NGT - Abnormal, NGT - Borderline, NGT - Incompletely Imaged, NGT - Normal, CVC - Abnormal, CVC - Borderline

-> data/ranzcr-clip-catheter-line-classification/train.csv has 27074 rows and 13 columns.
The columns are: StudyInstanceUID, ETT - Abnormal, ETT - Borderline, ETT - Normal, NGT - Abnormal, NGT - Borderline, NGT - Incompletely Imaged, NGT - Normal, CVC - Abnormal, CVC - Borderline, CVC - Normal, Swan Ganz Catheter Present, PatientID

-> data/ranzcr-clip-catheter-line-classification/train_annotations.csv has 16261 rows and 3 columns.
The columns are: StudyInstanceUID, label, data

-> data/sample_submission.csv has 3009 rows and 10 columns.
The columns are: StudyInstanceUID, ETT - Abnormal, ETT - Borderline, ETT - Normal, NGT - Abnormal, NGT - Borderline, NGT - Incompletely Imaged, NGT - Normal, CVC - Abnormal, CVC - Borderline

-> data/train.csv has 27074 rows and 13 columns.
The columns are: StudyInstanceUID, ETT - Abnormal, ETT - Borderline, ETT - Normal, NGT - Abnormal, NGT - Borderline, NGT - Incompletely Imaged, NGT - Normal, CVC - Abnormal, CVC - Borderline, CVC - Normal, Swan Ganz Catheter Present, PatientID

-> data/train_annotations.csv has 16261 rows and 3 columns.
The columns are: StudyInstanceUID, label, data

-> (stopped after 10 files for performance)

# 5. Target score

0.9258020787998772

# 6. Current score

0.49338

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.53908) has done: 'I fix the protobuf/TensorFlow import crash by forcing TensorFlow to use the pure-Python protobuf implementation (a common compatibility issue with protobuf 6.x) before importing TensorFlow. Then I fix the inference-time error by removing the invalid `.numpy()` call on the already-numpy output of `predict_on_batch`. Finally, I make submission generation robust to the provided `sample_submission.csv` (which may have missing target columns in your copy) by reindexing to the required 11 target columns and filling missing columns with 0.5, ensuring a valid `submission.csv` is always written.'
- What this solution (achieved 0.48283) has done: 'I fix the TensorFlow/protobuf crash by forcing the compatible pure-Python protobuf implementation *and* disabling the C++ implementation explicitly (this is the root cause of the `MessageFactory.GetPrototype` error with protobuf 6.x). Then I correct a small but score-relevant logic bug in the model input shape (width/height were swapped), which can materially hurt predictions while keeping the same architecture and weights. Finally, I make submission generation match the competition’s required 11-label format by starting from the provided `sample_submission.csv` and filling all target columns deterministically, ensuring the script always writes a valid `submission.csv`.'
- What this solution (achieved 0.54456) has done: 'Your current notebook never yields a Kaggle score mainly because the environment is trying to `pip install protobuf<5` and restart the process, which is typically blocked/unreliable in Kaggle-style runtimes and can prevent `submission.csv` from being produced. I remove that self-restarting install logic and instead force TensorFlow to use the pure-Python protobuf runtime (the minimal change that fixes the protobuf 6.x crash without downloads). Next, I add a safe fallback for missing external weights: if the provided `.h5` isn’t present, we load ImageNet weights (same architecture/training loop) so predictions are non-random and score moves up toward your target. Finally, I keep your submission-column robustness, but also ensure test prediction row order matches `sample_submission.csv` exactly to avoid silent UID misalignment.'
- What this solution (achieved 0.50736) has done: 'I fix the TensorFlow/protobuf crash by forcing protobuf to use the pure-Python implementation and explicitly disabling the C++ one, and I do it before importing TensorFlow (this is the direct cause of the `MessageFactory.GetPrototype` error with protobuf 6.x). I also ensure we only use the dataset copy that actually has the full 11-label `sample_submission.csv` (your top-level `data/sample_submission.csv` appears to have only 10 columns), so the written submission always matches the competition format. These changes are execution/stability fixes and should not alter the core model logic; they also avoid the silent score hit from submitting with missing/incorrect columns. The rest of your pipeline (EfficientNet backbone, preprocessing, inference loop) remains unchanged.'
- What this solution (achieved 0.45695) has done: 'I fix the TensorFlow/protobuf crash by moving the protobuf environment-variable setup to the very top of the script and importing `google.protobuf` once (before TensorFlow) to ensure the pure-Python implementation is actually used in this runtime. I also add a safe fallback that pins TF to CPU-only execution if GPU init triggers the same protobuf symbol path (this is score-neutral but stabilizes execution). Finally, I keep your model and inference logic unchanged, but make the data directory resolution and sample-submission selection deterministic so the submission always has all 11 required columns and the row order exactly matches `sample_submission.csv`.'
- What this solution (achieved 0.51049) has done: 'I fix the TensorFlow/protobuf crash by removing the brittle protobuf env overrides and instead using the Kaggle-stable workaround of forcing TensorFlow to use the legacy pure-Python protobuf implementation via `TF_USE_LEGACY_KERAS=1` plus safe GPU-memory-growth (this is execution-critical). Then I correct the dataset path resolution so it always points at the competition root that contains `train/` and `test/`, avoiding accidental selection of `/kaggle/data` which can silently break file discovery. Finally, I keep your exact model/inference core logic but make submission creation always match the full 11-label format by starting from the competition’s `sample_submission.csv` (the one with all 11 columns) and reindexing columns deterministically.'
- What this solution (achieved 0.49338) has done: 'I fix the TensorFlow/protobuf crash by forcing the pure-Python protobuf implementation before importing TensorFlow (this directly addresses the `MessageFactory.GetPrototype` AttributeError in your environment). I also switch to a Kaggle-stable Keras mode by keeping legacy Keras enabled and avoiding any pip-installs/restarts, which is execution-critical and score-neutral. Finally, I keep your exact model/inference logic but make the dataset root and sample-submission selection deterministic (prefer the competition folder copy) to prevent silent label-column mismatches that can hurt score and/or invalidate submissions.'

# 9. Code solution

## === cell 0
import os
import sys

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "2")

os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import tensorflow as tf
import pandas as pd
import numpy as np

tf.random.set_seed(42)
np.random.seed(42)

try:
    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
except Exception as e:
    print("GPU memory-growth setup skipped:", repr(e))

W = H = 338
N_CLASSES = 11
autotune = tf.data.AUTOTUNE

target_cols = [
    "CVC - Abnormal",
    "CVC - Borderline",
    "CVC - Normal",
    "ETT - Abnormal",
    "ETT - Borderline",
    "ETT - Normal",
    "NGT - Abnormal",
    "NGT - Borderline",
    "NGT - Incompletely Imaged",
    "NGT - Normal",
    "Swan Ganz Catheter Present",
]

mean = tf.constant([0.485, 0.456, 0.406], dtype=tf.float32)
std = tf.constant([0.229, 0.224, 0.225], dtype=tf.float32)

DATA_DIR_CANDIDATES = [
    "/kaggle/input/ranzcr-clip-catheter-line-classification",
    "/kaggle/data/ranzcr-clip-catheter-line-classification",
    "../input/ranzcr-clip-catheter-line-classification",
    "/kaggle/input",
    "/kaggle/data",
]
DATA_DIR = None
for p in DATA_DIR_CANDIDATES:
    if os.path.isdir(os.path.join(p, "train")) and os.path.isdir(
        os.path.join(p, "test")
    ):
        DATA_DIR = p
        break
if DATA_DIR is None:
    for p in DATA_DIR_CANDIDATES:
        nested = os.path.join(p, "ranzcr-clip-catheter-line-classification")
        if os.path.isdir(os.path.join(nested, "train")) and os.path.isdir(
            os.path.join(nested, "test")
        ):
            DATA_DIR = nested
            break

assert (
    DATA_DIR is not None
), f"Could not find dataset directory among: {DATA_DIR_CANDIDATES}"
test_img_dir = os.path.join(DATA_DIR, "test")
assert os.path.isdir(test_img_dir), f"Test image directory not found: {test_img_dir}"

weight_dir = "../input/cassava2020weights"
if not os.path.isdir(weight_dir):
    weight_dir = "../input"  # harmless fallback; load will fail gracefully

model_map = {
    "efficientb3": [
        tf.keras.applications.EfficientNetB3,
        os.path.join(weight_dir, "ranzcr_efficientb3.h5"),
    ],
    "efficientb5": [
        tf.keras.applications.EfficientNetB5,
        os.path.join(weight_dir, "ranzcr_efficientb5.h5"),
    ],
    "efficientb7": [
        tf.keras.applications.EfficientNetB7,
        os.path.join(weight_dir, "ranzcr_efficientb7.h5"),
    ],
}

print("Python:", sys.version)
print("TensorFlow:", tf.__version__)
print("Data dir:", DATA_DIR)
print("Test dir:", test_img_dir)
print("Weights dir:", weight_dir)




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
def _uid_from_path(path):
    fname = tf.strings.split(path, os.sep)[-1]
    uid = tf.strings.regex_replace(fname, r"\.jpg$", "")
    return uid


def parse_jpeg(path):
    img_bytes = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img_bytes, channels=3)
    img = tf.image.resize(img, (H, W))
    uid = _uid_from_path(path)
    return img, uid


def preprocess(images, labels):
    images = tf.cast(images, tf.float32) / 255.0
    images = (images - mean) / std
    return images, labels


test_paths = sorted(
    [
        os.path.join(test_img_dir, f)
        for f in os.listdir(test_img_dir)
        if f.lower().endswith(".jpg")
    ]
)
assert len(test_paths) > 0, "No test JPGs found."

test_data = tf.data.Dataset.from_tensor_slices(test_paths)
test_data = test_data.map(lambda p: parse_jpeg(p), num_parallel_calls=autotune)
test_data = test_data.batch(16)
test_data = test_data.map(preprocess, num_parallel_calls=autotune)
test_data = test_data.prefetch(autotune)

print("Number of test images:", len(test_paths))




## === cell 2
def get_model(
    base_model,
    baseline_weight=None,
    init_weight=None,
    lr=0.001,
    optimizer=tf.optimizers.Adam,
):
    if init_weight and os.path.isfile(init_weight):
        chosen_backbone_weights = (
            baseline_weight  # usually None; we'll load full model weights next
        )
    else:
        chosen_backbone_weights = "imagenet"

    base_model = base_model(
        include_top=False,
        input_shape=(H, W, 3),
        pooling="avg",
        weights=chosen_backbone_weights,
    )
    base_out = base_model.output
    out = tf.keras.layers.Dropout(0.3)(base_out)
    out = tf.keras.layers.Dense(N_CLASSES, activation="sigmoid")(out)
    model = tf.keras.models.Model(inputs=base_model.input, outputs=out)

    model.compile(
        optimizer=optimizer(learning_rate=lr),
        loss="binary_crossentropy",
        metrics=[tf.keras.metrics.AUC()],
    )

    if init_weight:
        if os.path.isfile(init_weight):
            try:
                model.load_weights(init_weight)
                print(f"Weight loaded from {init_weight}")
            except Exception as e:
                print(f"Load weight from {init_weight} failed, {e}")
        else:
            print(
                f"Custom weight file not found at {init_weight}; using ImageNet backbone weights."
            )
    return model


base_model_fn, weight_path = model_map["efficientb5"]
model = get_model(base_model_fn, init_weight=weight_path)



## === cell 3
image_ids = []
preds = []

for images, image_id in test_data:
    batch_ids = [x.decode("utf-8") for x in image_id.numpy().tolist()]
    image_ids.extend(batch_ids)

    batch_pred = model.predict_on_batch(images)
    preds.append(batch_pred)

preds = np.concatenate(preds, axis=0)
pred_df = pd.DataFrame(preds, columns=target_cols)
pred_df.insert(0, "StudyInstanceUID", image_ids)

candidate_sample_paths = [
    os.path.join(DATA_DIR, "sample_submission.csv"),
    "/kaggle/input/ranzcr-clip-catheter-line-classification/sample_submission.csv",
    "/kaggle/data/ranzcr-clip-catheter-line-classification/sample_submission.csv",
    "/kaggle/input/sample_submission.csv",
    "/kaggle/data/sample_submission.csv",
]
sample_sub = None
used_path = None
for p in candidate_sample_paths:
    if os.path.isfile(p):
        tmp = pd.read_csv(p)
        if ("StudyInstanceUID" in tmp.columns) and all(
            c in tmp.columns for c in target_cols
        ):
            sample_sub = tmp
            used_path = p
            break

if sample_sub is None:
    for p in candidate_sample_paths:
        if os.path.isfile(p):
            sample_sub = pd.read_csv(p)
            used_path = p
            break

assert (
    sample_sub is not None
), "Could not find any sample_submission.csv in expected locations."

sub = sample_sub[["StudyInstanceUID"]].merge(pred_df, on="StudyInstanceUID", how="left")

for c in target_cols:
    if c not in sub.columns:
        sub[c] = 0.5
    sub[c] = sub[c].astype("float32").fillna(0.5).clip(0.0, 1.0)

sub = sub[["StudyInstanceUID"] + target_cols]

sub.to_csv("submission.csv", index=False)

print("Used sample_submission:", used_path)
print("Wrote submission.csv with shape:", sub.shape)
print("Columns:", list(sub.columns))
print(sub.head())
