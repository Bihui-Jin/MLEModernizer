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

0.9234351883856996

# 6. Current score

0.51552

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.51303) has done: 'I fix the protobuf-related crash by removing the forced `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION="python"` override, which is what triggers the `MessageFactory.GetPrototype` error with your installed protobuf/TensorFlow versions. I also fix the inference loop error by not calling `.numpy()` on the output of `predict_on_batch()` (it already returns a NumPy array). Finally, I ensure the model input shape uses `(H, W, 3)` (not swapped) to avoid subtle shape mismatches, and I make the submission columns match exactly the provided `sample_submission.csv` (which in this environment contains only 9 target columns, not 11), so a valid submission is always written.'
- What this solution (achieved 0.48547) has done: 'You’re crashing before any inference because TensorFlow 2.18 + protobuf 6.x is incompatible with your current protobuf runtime (it triggers `MessageFactory.GetPrototype` during TF import). I fix this by pinning protobuf to the pure-Python implementation *before* importing TensorFlow, which avoids the missing `GetPrototype` attribute in this environment. I also make the sample-submission/target-column handling robust: if the provided `sample_submission.csv` is missing some expected columns, we add them (filled with 0.5) so the submission matches the competition format. Core model/inference logic is unchanged; these edits are just to make the notebook run end-to-end and produce a valid `submission.csv`.'
- What this solution (achieved 0.46769) has done: 'I fix the TensorFlow/protobuf crash by removing the forced `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION="python"` override (it’s what triggers the `MessageFactory.GetPrototype` import error with TF 2.18 + protobuf 6.x here). I also fix a score-killing label/column mismatch by using the target columns exactly as provided by this environment’s `sample_submission.csv` (it contains fewer columns than the full competition schema), so predictions align with the evaluated labels. Finally, I make test file ordering deterministic and ensure the submission rows align 1:1 with the sample submission (same UIDs/order), producing a valid `submission.csv`.'
- What this solution (achieved 0.48477) has done: 'You’re crashing at TensorFlow import due to an incompatibility between TensorFlow 2.18 and protobuf 6.x in this environment; setting `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` before importing TensorFlow reliably avoids the `MessageFactory.GetPrototype` error. Your current score is also far below target because the weights directory/path is incorrect for this competition setup, so the model likely ran with random weights; I switch `weight_dir` to the competition dataset folder and fix the one wrong filename mapping for EfficientNetB7. Finally, I make the train/test/sample_submission path resolution robust to both Kaggle `/kaggle/input/...` and relative `../input/...` layouts while keeping the same model/inference logic and ensuring `submission.csv` matches the provided `sample_submission.csv` columns/order.'
- What this solution (achieved 0.48041) has done: 'I fix the TensorFlow/protobuf import crash by forcing the pure-Python protobuf implementation *before* importing TensorFlow (this is required with TF 2.18 + protobuf 6.x). Then I fix the missing weights error by making weight loading optional: if the `.h5` files are not present in this dataset, the model fall back to ImageNet weights (same architecture/inference loop, just avoids a hard crash). Finally, I make the submission always match the provided `sample_submission.csv` columns/order and ensure `submission.csv` is written successfully.'
- What this solution (achieved 0.51593) has done: 'The crash happens before any training/inference because TensorFlow 2.18 is incompatible with protobuf 6.x when forcing the pure-Python protobuf implementation, so I remove that environment override to restore a working TF import. Your low score is also consistent with predicting only the 9 columns present in this environment’s `sample_submission.csv`; the competition expects 11 targets, so I build the submission using the full 11-label schema and add any missing columns (filled from model outputs when available, otherwise safe defaults). Finally, I keep your model/inference core intact (same EfficientNet, preprocessing, loop), but ensure submission row order matches `sample_submission.csv` and always writes a valid `submission.csv`.'
- What this solution (achieved 0.51552) has done: 'You’re crashing on TensorFlow import due to the known TensorFlow 2.18 + protobuf 6.x incompatibility unless protobuf is forced to use the pure-Python implementation. I restore that environment setting (but do it safely/early, before importing TensorFlow) to eliminate the `MessageFactory.GetPrototype` error and allow the notebook to run end-to-end. Then I keep your model/inference logic intact, but make the submission schema consistent with the environment’s `sample_submission.csv` by predicting only those columns (and default-filling any missing predictions) so the file is valid and your AUC isn’t dragged down by dummy columns. This should both fix runtime and improve score substantially toward the target.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import tensorflow as tf
import pandas as pd
import numpy as np

W = H = 338
AUTOTUNE = tf.data.AUTOTUNE


def _resolve_data_dir():
    candidates = [
        "../input/ranzcr-clip-catheter-line-classification",
        "/kaggle/input/ranzcr-clip-catheter-line-classification",
        "../input/ranzcr-clip-catheter-line-classification/ranzcr-clip-catheter-line-classification",
        "/kaggle/input/ranzcr-clip-catheter-line-classification/ranzcr-clip-catheter-line-classification",
        "/kaggle/data/ranzcr-clip-catheter-line-classification",
        "/kaggle/data/ranzcr-clip-catheter-line-classification/ranzcr-clip-catheter-line-classification",
        "data/ranzcr-clip-catheter-line-classification",
        "/kaggle/data/ranzcr-clip-catheter-line-classification/ranzcr-clip-catheter-line-classification",
    ]
    for p in candidates:
        if os.path.isdir(p):
            return p
    return "../input/ranzcr-clip-catheter-line-classification"


DATA_DIR = _resolve_data_dir()
TEST_DIR = os.path.join(DATA_DIR, "test")

weight_dir = DATA_DIR

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

print("TF version:", tf.__version__)
print("Resolved DATA_DIR:", DATA_DIR)
print("Test dir exists:", os.path.isdir(TEST_DIR))
print("Weights existence:")
for k, (_, wp) in model_map.items():
    print(f"  {k}: {wp} -> {os.path.exists(wp)}")

FULL_TARGET_COLS = [
    "ETT - Abnormal",
    "ETT - Borderline",
    "ETT - Normal",
    "NGT - Abnormal",
    "NGT - Borderline",
    "NGT - Incompletely Imaged",
    "NGT - Normal",
    "CVC - Abnormal",
    "CVC - Borderline",
    "CVC - Normal",
    "Swan Ganz Catheter Present",
]




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
def triple_image(image):
    return tf.concat([image] * 3, axis=-1)


def decode_and_resize(path):
    img_bytes = tf.io.read_file(path)
    image = tf.image.decode_jpeg(img_bytes, channels=1)
    image = tf.image.resize(image, (H, W))
    image = triple_image(image)  # (H, W, 3)
    image = tf.cast(image, tf.float32)
    return image


def preprocess_for_efficientnet(image):
    return tf.keras.applications.efficientnet.preprocess_input(image)


def make_test_dataset(test_dir, batch_size=16):
    test_files = sorted(tf.io.gfile.glob(os.path.join(test_dir, "*.jpg")))
    if len(test_files) == 0:
        raise FileNotFoundError(f"No .jpg files found under: {test_dir}")

    uids = [os.path.splitext(os.path.basename(p))[0] for p in test_files]

    ds = tf.data.Dataset.from_tensor_slices((test_files, uids))
    ds = ds.map(
        lambda p, uid: (preprocess_for_efficientnet(decode_and_resize(p)), uid),
        num_parallel_calls=AUTOTUNE,
    )
    ds = ds.batch(batch_size).prefetch(AUTOTUNE)
    return ds, uids


def get_model(
    base_model_fn,
    n_classes,
    baseline_weight=None,
    init_weight=None,
    lr=0.001,
    optimizer=tf.optimizers.Adam,
):
    base_model = base_model_fn(
        include_top=False, input_shape=(H, W, 3), pooling="avg", weights=baseline_weight
    )
    base_out = base_model.output
    out = tf.keras.layers.Dropout(0.3)(base_out)
    out = tf.keras.layers.Dense(n_classes, activation="sigmoid")(out)
    model = tf.keras.models.Model(inputs=base_model.input, outputs=out)

    model.compile(
        optimizer=optimizer(learning_rate=lr),
        loss="binary_crossentropy",
        metrics=[tf.keras.metrics.AUC()],
    )

    if init_weight:
        if os.path.exists(init_weight):
            model.load_weights(init_weight)
            print(f"Weight loaded from {init_weight}")
        else:
            print(
                f"WARNING: weight file not found, using baseline weights={baseline_weight}. Missing: {init_weight}"
            )

    return model




## === cell 2
test_data, test_uids = make_test_dataset(TEST_DIR, batch_size=16)

sample_path = os.path.join(DATA_DIR, "sample_submission.csv")
if not os.path.exists(sample_path):
    sample_path_alt = (
        "/kaggle/input/ranzcr-clip-catheter-line-classification/sample_submission.csv"
    )
    if os.path.exists(sample_path_alt):
        sample_path = sample_path_alt
    else:
        sample_path_alt2 = "/kaggle/data/ranzcr-clip-catheter-line-classification/sample_submission.csv"
        if os.path.exists(sample_path_alt2):
            sample_path = sample_path_alt2

sub = pd.read_csv(sample_path)
if "StudyInstanceUID" not in sub.columns:
    raise ValueError("sample_submission.csv missing required column: StudyInstanceUID")

target_cols = [c for c in sub.columns if c != "StudyInstanceUID"]
N_CLASSES = len(target_cols)

print("Using N_CLASSES =", N_CLASSES)
print("Target columns (from sample_submission.csv):", target_cols)

base_model_fn, weight_path = model_map["efficientb5"]
model = get_model(
    base_model_fn,
    n_classes=N_CLASSES,
    baseline_weight="imagenet",
    init_weight=weight_path,
)

preds = []
image_ids = []

for batch_images, batch_uids in test_data:
    batch_pred = model.predict_on_batch(batch_images)  # numpy array
    preds.append(batch_pred)

    batch_uids_np = batch_uids.numpy()
    image_ids.extend([uid.decode("utf-8") for uid in batch_uids_np])

preds = np.concatenate(preds, axis=0)

pred_df = pd.DataFrame(preds, columns=target_cols)
pred_df.insert(0, "StudyInstanceUID", image_ids)

sub_out = sub[["StudyInstanceUID"] + target_cols].merge(
    pred_df, on="StudyInstanceUID", how="left", suffixes=("_base", "")
)

for c in target_cols:
    base_col = f"{c}_base"
    if base_col in sub_out.columns:
        sub_out[c] = sub_out[c].fillna(sub_out[base_col])
        sub_out.drop(columns=[base_col], inplace=True)

sub_out[target_cols] = sub_out[target_cols].fillna(0.5).astype(np.float32)

sub_out.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", sub_out.shape)
print("Columns:", list(sub_out.columns))
print(sub_out.head())
