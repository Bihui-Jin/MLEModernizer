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

0.9339675346749576

# 6. Current score

0.50907

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.49312) has done: 'I fix the TensorFlow/protobuf import crash by removing the forced pure-Python protobuf setting and, if needed, pinning protobuf to the C++ implementation via environment variables before importing TensorFlow. I also fix the inference loop error by treating `predict_on_batch` output as a NumPy array (no `.numpy()`), which unblocks submission creation. Finally, I correct a couple of small but score-relevant correctness issues without changing the model logic: swap `(W,H,3)` to `(H,W,3)` for input shape consistency and ensure the submission columns exactly match `target_cols` (adding any missing columns, and ordering them properly). The script run end-to-end and write `submission.csv`.'
- What this solution (achieved 0.51184) has done: 'I fix the TensorFlow/protobuf crash by forcing the pure-Python protobuf implementation *before* importing TensorFlow (this avoids the `MessageFactory.GetPrototype` mismatch seen with TF 2.18 + protobuf 6). Then I correct a score-killing submission-format issue: your `sample_submission.csv` in this environment has only 9 target columns, so we must build the submission from `train.csv`’s 11 label columns and align predictions to those exact columns. Finally, I keep your existing EfficientNetB5+Dropout+Dense(sigmoid) logic, but ensure the model always outputs the correct number/order of classes by deriving `target_cols` from `train.csv` and setting `N_CLASSES` accordingly.'
- What this solution (achieved 0.52224) has done: 'I fix the TensorFlow/protobuf import crash by switching to the C++ protobuf runtime (the pure-Python setting is what triggers the `MessageFactory.GetPrototype` failure in this TF 2.18 + protobuf 6 environment). Then I fix a score-killing correctness issue: your `sample_submission.csv` in this environment is missing some required target columns, so we must build the submission columns from `train.csv` (all 11 labels) and ensure the output CSV has those exact columns in the correct order. Finally, I keep your existing EfficientNetB5 + Dropout + Dense(sigmoid) inference logic unchanged, but make the test UID list come from actual test images (not the truncated sample submission), ensuring predictions cover the full test set and align to the correct IDs.'
- What this solution (achieved 0.49231) has done: 'I fix the TensorFlow/protobuf crash by forcing the pure-Python protobuf runtime *before* importing TensorFlow (this is the common stable workaround for TF 2.18 + protobuf 6 `MessageFactory.GetPrototype` failures). Then I correct the submission schema to always match the competition’s required 11 target columns (your environment’s `sample_submission.csv` is missing columns, which severely hurts AUC when Kaggle fills missing ones as 0). Finally, I keep your EfficientNetB5 + Dropout + Dense(sigmoid) logic unchanged, but ensure the test IDs come from the actual test folder and that predictions are aligned and ordered to those exact required columns before writing `submission.csv`.'
- What this solution (achieved 0.49275) has done: 'I fix the TensorFlow import crash by switching the protobuf runtime setting to the pure-Python implementation before importing TensorFlow (the current `"cpp"` setting fails because `_message` is unavailable in this environment). Then I make the script robust to Kaggle’s inconsistent `sample_submission.csv` by always deriving the 11 required target columns from `train.csv` and writing those exact columns in that order. Finally, I keep your existing EfficientNetB5 + Dropout + Dense(sigmoid) inference logic unchanged, but add a safe fallback to use the sample submission IDs only if the test folder path is missing so the notebook always produces `submission.csv` end-to-end.'
- What this solution (achieved 0.50907) has done: 'I fix the TensorFlow import crash by switching the protobuf runtime to the pure-Python implementation (your current `"cpp"` setting fails because `google.protobuf.pyext._message` is not available in this environment). Then, with TensorFlow importing cleanly, the downstream `NameError`s disappear because `tf/np/TEST_IMG_DIR` actually be defined. Finally, I ensure the submission schema always matches the competition’s 11 target columns derived from `train.csv` (even if the provided `sample_submission.csv` is missing some columns), and still write a valid `submission.csv`.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", None)

import tensorflow as tf
import pandas as pd
import numpy as np

W = H = 338
AUTOTUNE = tf.data.AUTOTUNE

DATA_DIR = "../input/ranzcr-clip-catheter-line-classification"
if not os.path.isdir(DATA_DIR):
    alt = "/kaggle/data/ranzcr-clip-catheter-line-classification"
    if os.path.isdir(alt):
        DATA_DIR = alt

TEST_IMG_DIR = os.path.join(DATA_DIR, "test")
TRAIN_CSV = os.path.join(DATA_DIR, "train.csv")
SAMPLE_SUB = os.path.join(DATA_DIR, "sample_submission.csv")

weight_dir = "../input/cassava2020weights"
if not os.path.isdir(weight_dir):
    alt_w = "/kaggle/data/cassava2020weights"
    if os.path.isdir(alt_w):
        weight_dir = alt_w

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

train_df = pd.read_csv(TRAIN_CSV)

target_cols = [
    c for c in train_df.columns if c not in ["StudyInstanceUID", "PatientID"]
]
N_CLASSES = len(target_cols)

print("DATA_DIR =", DATA_DIR)
print("TEST_IMG_DIR exists:", os.path.isdir(TEST_IMG_DIR))
print("Using N_CLASSES =", N_CLASSES)
print("Target columns:", target_cols)




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
def decode_and_resize_jpeg(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, (H, W), method="bilinear")
    return img


def parse_path(path):
    img = decode_and_resize_jpeg(path)
    uid = tf.strings.regex_replace(tf.strings.split(path, os.sep)[-1], r"\.jpg$", "")
    return img, uid


def preprocess(images, labels):
    images = tf.cast(images, tf.float32) / 255.0
    return images, labels


def get_model(
    base_model,
    baseline_weight=None,
    init_weight=None,
    lr=0.001,
    optimizer=tf.optimizers.Adam,
):
    base_model = base_model(
        include_top=False,
        input_shape=(H, W, 3),
        pooling="avg",
        weights=baseline_weight,
    )
    base_out = base_model.output
    out = tf.keras.layers.Dropout(0.3)(base_out)
    out = tf.keras.layers.Dense(N_CLASSES, activation="sigmoid")(out)
    model = tf.keras.models.Model(inputs=base_model.input, outputs=out)

    model.compile(
        optimizer=optimizer(learning_rate=lr),
        loss="binary_crossentropy",
        metrics=[tf.keras.metrics.AUC(name="auc")],
    )

    if init_weight:
        if tf.io.gfile.exists(init_weight):
            try:
                model.load_weights(init_weight)
                print(f"Weight loaded from {init_weight}")
            except Exception as e:
                print(f"Load weight from {init_weight} failed, {e}")
        else:
            print(
                f"Weight file not found at {init_weight}; proceeding with randomly initialized head."
            )
    return model




## === cell 2
test_uids = None
if os.path.isdir(TEST_IMG_DIR):
    test_files = sorted(
        [f for f in os.listdir(TEST_IMG_DIR) if f.lower().endswith(".jpg")]
    )
    test_uids = [os.path.splitext(f)[0] for f in test_files]
    print("Found test images in folder:", len(test_uids))
else:
    print(
        f"WARNING: test image folder not found at {TEST_IMG_DIR}. Falling back to sample_submission IDs."
    )
    ss = pd.read_csv(SAMPLE_SUB)
    test_uids = ss["StudyInstanceUID"].astype(str).tolist()
    print("Found test IDs in sample_submission:", len(test_uids))

test_paths = tf.constant(
    [os.path.join(TEST_IMG_DIR, f"{uid}.jpg") for uid in test_uids]
)

test_data = tf.data.Dataset.from_tensor_slices(test_paths)
test_data = test_data.map(parse_path, num_parallel_calls=AUTOTUNE)
test_data = test_data.batch(16)
test_data = test_data.map(preprocess, num_parallel_calls=AUTOTUNE)
test_data = test_data.prefetch(AUTOTUNE)

base_mode, weight_path = model_map["efficientb5"]
model = get_model(base_mode, init_weight=weight_path)



## === cell 3
preds = np.zeros((len(test_uids), N_CLASSES), dtype=np.float32)

row = 0
uid_out = []
for images, uids in test_data:
    batch_preds = model.predict_on_batch(images)  # numpy array
    bs = int(batch_preds.shape[0])
    preds[row : row + bs] = batch_preds
    uid_out.extend([u.decode("utf-8") for u in uids.numpy().tolist()])
    row += bs

n = min(len(uid_out), preds.shape[0])
uid_out = uid_out[:n]
preds = preds[:n]

out_df = pd.DataFrame({"StudyInstanceUID": np.array(uid_out, dtype=object)})

for i, c in enumerate(target_cols):
    out_df[c] = preds[:, i]

required_cols = target_cols[:]
for c in required_cols:
    if c not in out_df.columns:
        out_df[c] = 0.0

out_df = out_df[["StudyInstanceUID"] + required_cols]
out_df.to_csv("submission.csv", index=False)

print("Wrote submission.csv:", out_df.shape)
print(out_df.columns.tolist())
print(out_df.head())
