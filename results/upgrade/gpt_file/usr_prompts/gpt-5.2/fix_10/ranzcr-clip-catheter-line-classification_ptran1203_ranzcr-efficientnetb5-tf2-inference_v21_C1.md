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

0.9334153454286166

# 6. Current score

0.52573

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.49403) has done: 'I fix the TensorFlow/protobuf initialization crash by removing the forced pure-Python protobuf environment override (it conflicts with protobuf 6.x in this Kaggle image). Then I fix the prediction loop runtime error by treating `predict_on_batch` output as a NumPy array (no `.numpy()`), keeping the model and preprocessing unchanged. Finally, I ensure the submission column set/order exactly matches the provided `sample_submission.csv` (including any missing target columns), so the pipeline always writes a valid `submission.csv` end-to-end.'
- What this solution (achieved 0.50084) has done: 'You’re crashing at import time due to an incompatibility between TensorFlow 2.18 and protobuf 6.x that shows up as `MessageFactory.GetPrototype` missing; the minimal fix is to force the pure-Python protobuf implementation *before* importing TensorFlow. Next, your declared `N_CLASSES=11` doesn’t match the competition’s `sample_submission.csv` in this environment (it has only 10 target columns), which both hurts score and risks shape/column misalignment; we derive `target_cols` directly from `sample_submission.csv` and set `N_CLASSES` accordingly while keeping the same model and preprocessing. Finally, we ensure model input shape uses `(H, W, 3)` and keep the submission columns exactly in `sample_submission` order so the CSV is always valid.'
- What this solution (achieved 0.50094) has done: 'We fix the import-time crash by removing the protobuf environment override that is triggering the `MessageFactory.GetPrototype` error under TensorFlow 2.18 + protobuf 6.x in this environment. Then we make the code robust to the dataset mismatch where `sample_submission.csv` here contains only 10 target columns by always building predictions and the submission strictly from that file’s columns (already the intent, we just keep it consistent). Finally, we keep the exact same model/prediction logic, but add a safe fallback to generate a valid submission even if the external weights path is missing, so the notebook always finishes and writes `submission.csv`.'
- What this solution (achieved 0.50356) has done: 'We fix the import-time crash caused by the TensorFlow 2.18 + protobuf 6.x incompatibility by forcing the pure-Python protobuf implementation **before** importing TensorFlow (this is the minimal, standard workaround for the `MessageFactory.GetPrototype` error). Then we correct a silent but very impactful data/label mismatch: your `sample_submission.csv` in this environment has only 10 target columns, so we build predictions for exactly those columns and also guard against the common “missing columns” case by adding any absent expected targets as zeros to keep the submission valid. Finally, we keep your EfficientNet model and preprocessing intact but ensure inference is deterministic and always produces a correctly ordered `submission.csv`.'
- What this solution (achieved 0.50504) has done: 'I fix the import-time crash coming from the TensorFlow 2.18 + protobuf 6.x incompatibility by patching the missing `MessageFactory.GetPrototype` symbol before importing TensorFlow (keeping your existing pure-Python protobuf setting). Then I correct the dataset/label mismatch that is holding your score near random: this environment’s `sample_submission.csv` is missing 2 target columns, so we always build the full 11-label target list from `train.csv` (the ground-truth label columns) and create those missing columns in the submission, ensuring the model outputs 11 probabilities in the correct order. Finally, I keep your model architecture and preprocessing intact and still write a valid `submission.csv` end-to-end with the required columns.'
- What this solution (achieved 0.49918) has done: 'We fix the protobuf/TensorFlow import crash by applying a safe compatibility shim that works whether `MessageFactory` is a class or an instance in protobuf 6.x, and do it before importing TensorFlow. Then we make the submission schema match the competition requirement by ensuring we always output all 11 label columns (the two missing from this environment’s `sample_submission.csv` are added), while keeping the same model, preprocessing, and prediction loop. Finally, we add a small guard to pick an available weight file (so we don’t silently run untrained if a mapped file is missing), which should improve score toward your target without changing the core modeling approach.'
- What this solution (achieved 0.48773) has done: 'We fix the protobuf/TensorFlow import crash by making the `MessageFactory.GetPrototype` shim handle the protobuf 6.x case where `MessageFactory` can be an *instance* (not just a class), which is why your current patch still raises. Then we keep your model/inference logic the same but correct the submission schema to the competition-required 11 targets: we always use the 11 label columns from `train.csv` for `N_CLASSES` and submission columns, rather than the incomplete `sample_submission.csv` in this environment. Finally, we keep the same EfficientNet + weights loading behavior and ensure a valid `submission.csv` is written with all required columns in the correct order.'
- What this solution (achieved 0.52573) has done: 'We fix the import-time crash by removing the protobuf pure-Python override and instead applying a tiny compatibility shim that safely adds `MessageFactory.GetPrototype` when missing (the current override is what triggers the error in this TF 2.18 + protobuf 6.x image). Then we correct the test image path root to the actual provided dataset location so the pipeline can read images and run inference end-to-end. Finally, we keep your model, preprocessing, and inference loop intact, but we build the submission columns to include all 11 competition targets (adding the two missing columns to the local sample_submission schema) and ensure the output CSV is valid and correctly ordered.'

# 9. Code solution

## === cell 0
import os
import sys

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", None)

try:
    import google.protobuf.message_factory as _mf

    def _install_getprototype_if_missing(factory_obj):
        if factory_obj is None:
            return
        if hasattr(factory_obj, "GetPrototype"):
            return
        if not hasattr(factory_obj, "GetMessageClass"):
            return

        def _GetPrototype(self, descriptor):
            return self.GetMessageClass(descriptor)

        try:
            if isinstance(factory_obj, type):
                setattr(factory_obj, "GetPrototype", _GetPrototype)
            else:
                try:
                    setattr(type(factory_obj), "GetPrototype", _GetPrototype)
                except Exception:
                    pass
                try:
                    setattr(
                        factory_obj,
                        "GetPrototype",
                        _GetPrototype.__get__(factory_obj, type(factory_obj)),
                    )
                except Exception:
                    pass
        except Exception:
            pass

    _install_getprototype_if_missing(getattr(_mf, "MessageFactory", None))
    _install_getprototype_if_missing(getattr(_mf, "message_factory", None))
except Exception:
    pass

import tensorflow as tf
import pandas as pd
import numpy as np

W = H = 338
autotune = tf.data.AUTOTUNE

DATA_ROOT = "/kaggle/data/ranzcr-clip-catheter-line-classification"
TEST_DIR = os.path.join(DATA_ROOT, "test")
SAMPLE_SUB_PATH = os.path.join(DATA_ROOT, "sample_submission.csv")
TRAIN_CSV_PATH = os.path.join(DATA_ROOT, "train.csv")

sample_sub = pd.read_csv(SAMPLE_SUB_PATH)
if "StudyInstanceUID" not in sample_sub.columns:
    raise ValueError("sample_submission.csv must contain 'StudyInstanceUID'")

train_df = pd.read_csv(TRAIN_CSV_PATH)

target_cols = [
    c for c in train_df.columns if c not in ("StudyInstanceUID", "PatientID")
]
N_CLASSES = len(target_cols)

mean = tf.constant([0.485, 0.456, 0.406], dtype=tf.float32)
std = tf.constant([0.229, 0.224, 0.225], dtype=tf.float32)

weight_dir = "../input/cassava2020weights"
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

print("TensorFlow:", tf.__version__)
print("Test dir exists:", os.path.isdir(TEST_DIR))
print("Sample submission exists:", os.path.isfile(SAMPLE_SUB_PATH))
print("Train CSV exists:", os.path.isfile(TRAIN_CSV_PATH))
print("Detected N_CLASSES from train.csv labels:", N_CLASSES)
print("Target columns:", target_cols)
print(
    "Sample-sub target cols (may be incomplete in this environment):",
    [c for c in sample_sub.columns if c != "StudyInstanceUID"],
)




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
def triple_image(image):
    return tf.concat([image] * 3, axis=-1)


def decode_and_resize_jpeg(path):
    img_bytes = tf.io.read_file(path)
    image = tf.image.decode_jpeg(img_bytes, channels=1)
    image = tf.image.resize(image, (H, W), method="bilinear")
    image = triple_image(image)
    return image


def parse_from_path(path):
    image = decode_and_resize_jpeg(path)
    image_id = tf.strings.regex_replace(
        tf.strings.split(path, os.sep)[-1], r"\.jpg$", ""
    )
    return image, image_id


def preprocess(images, labels):
    images = tf.cast(images, tf.float32) / 255.0
    images = (images - mean) / std
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
        metrics=[tf.keras.metrics.AUC()],
    )

    if init_weight and tf.io.gfile.exists(init_weight):
        try:
            model.load_weights(init_weight)
            print(f"Weight loaded from {init_weight}")
        except Exception as e:
            print(f"Load weight from {init_weight} failed: {e}")
    else:
        print(
            f"Weight file not found (or not provided): {init_weight}. Proceeding without loading weights."
        )
    return model




## === cell 2
test_ids = sample_sub["StudyInstanceUID"].astype(str).tolist()
test_paths = [os.path.join(TEST_DIR, f"{uid}.jpg") for uid in test_ids]

for p in test_paths[:3]:
    if not tf.io.gfile.exists(p):
        raise FileNotFoundError(f"Expected test image not found: {p}")

test_ds = tf.data.Dataset.from_tensor_slices(test_paths)
test_ds = test_ds.map(parse_from_path, num_parallel_calls=autotune)
test_ds = test_ds.batch(16)
test_ds = test_ds.map(preprocess, num_parallel_calls=autotune)
test_ds = test_ds.prefetch(1)

base_mode, weight_path = model_map["efficientb5"]
if not tf.io.gfile.exists(weight_path):
    for _k, (_bm, _wp) in model_map.items():
        if tf.io.gfile.exists(_wp):
            base_mode, weight_path = _bm, _wp
            print(f"Preferred weights missing; falling back to {_k} weights at: {_wp}")
            break

model = get_model(base_mode, init_weight=weight_path)



## === cell 3
preds = np.zeros((len(test_ids), N_CLASSES), dtype=np.float32)
offset = 0

for batch_images, batch_ids in test_ds:
    batch_pred = model.predict_on_batch(batch_images)  # numpy array
    bs = int(batch_pred.shape[0])
    preds[offset : offset + bs] = batch_pred
    offset += bs

if offset != len(test_ids):
    raise RuntimeError(
        f"Prediction rows mismatch: got {offset}, expected {len(test_ids)}"
    )

submission = pd.DataFrame({"StudyInstanceUID": test_ids})
for j, col in enumerate(target_cols):
    submission[col] = preds[:, j]

required_cols = ["StudyInstanceUID"] + target_cols
for c in required_cols:
    if c not in submission.columns:
        submission[c] = 0.0
submission = submission[required_cols]

submission.to_csv("submission.csv", index=False)

print("Wrote submission.csv with shape:", submission.shape)
print(submission.head())
print("Columns:", list(submission.columns))
