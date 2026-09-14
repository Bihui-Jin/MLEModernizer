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

0.8883446571951024

# 6. Current score

0.48899

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.48602) has done: 'I remove the protobuf environment override that is forcing the pure-Python protobuf backend, because with protobuf 6.x it triggers the `MessageFactory.GetPrototype` AttributeError before TensorFlow can run. I also fix the prediction loop to handle `predict_on_batch` consistently (it already returns a NumPy array in TF 2.18), removing the erroneous `.numpy()` call. Finally, I make the data paths robust to your provided `/kaggle/data/...` layout while keeping the same model and preprocessing, and ensure the submission contains all 11 required target columns in the correct order and is written to `submission.csv`.'
- What this solution (achieved 0.51149) has done: 'The crash happens before any of your code runs because TensorFlow 2.18 is importing `tensorflow` which in turn touches `protobuf` APIs that are incompatible with this environment’s protobuf build (the `MessageFactory.GetPrototype` AttributeError). The minimal reliable fix is to force the pure-Python protobuf implementation **before** importing TensorFlow (and do not pop it), which avoids that specific compiled-protobuf mismatch. I also fix a small but real bug in your EfficientNetB7 mapping (it points to the B5 weights) and correct the EfficientNet input_shape order from `(W, H, 3)` to `(H, W, 3)`; this keeps the same architecture intent but prevents silent shape mismatch/poor resizing semantics that can hurt score. Finally, I keep your submission-column completion logic but make it robust to the provided sample submission missing some target columns, ensuring the output always has all 11 required columns in the correct order.'
- What this solution (achieved 0.50656) has done: 'I fix the TensorFlow import crash by setting `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` (the current `"cpp"` setting is what triggers the `_message` import error with protobuf 6.x in this environment). Because cell 0 currently fails, all later NameErrors are just cascading; once TF imports, the existing dataset/model/prediction logic can run. I also make submission-column handling robust to the provided `sample_submission.csv` missing some required targets by adding any missing columns and enforcing the exact 11-column target order. These changes are execution-critical and score-neutral (they don’t change the model or inference logic beyond allowing it to run and output a valid `submission.csv`).'
- What this solution (achieved 0.50371) has done: 'I fix the TensorFlow/protobuf import crash by setting the protobuf env vars *before* TensorFlow import and by additionally forcing the pure-Python protobuf backend at runtime (this avoids the `MessageFactory.GetPrototype` failure with protobuf 6.x). Then I keep your exact model/dataset/inference logic intact, but make submission generation strictly match the competition’s required 11 target columns even when the provided `sample_submission.csv` is missing some columns (as in your environment). These changes are execution-critical and should also improve your score versus the current ~0.50 by ensuring the model actually runs and produces real predictions instead of fallbacks.'
- What this solution (achieved 0.51045) has done: 'The runtime failure is happening before your code executes because TensorFlow 2.18 + protobuf 6.x can crash on import with `MessageFactory.GetPrototype` in some Kaggle images. I fix this by forcing the pure-Python protobuf backend *and* proactively importing protobuf before TensorFlow (so TF sees the correct implementation), which is the minimal execution-unblocking change. Then I keep your model/dataset/inference logic intact, but make the submission column handling robust: your provided `sample_submission.csv` is missing 2 required target columns, so we must add them and enforce the exact 11-label order to avoid a low/invalid score. These changes are primarily correctness/stability and should also improve AUC versus the current ~0.50 by ensuring the submission contains the full required target set.'
- What this solution (achieved 0.50888) has done: 'We fix the TensorFlow import crash caused by the protobuf 6.x `MessageFactory.GetPrototype` incompatibility by pinning protobuf to the pure-Python backend *and* injecting a small compatibility shim **before** importing TensorFlow. Then we keep your model/dataset/prediction logic the same, but make the submission-column handling fully robust: your provided `sample_submission.csv` is missing required target columns, so we always create and output all 11 labels in the exact required order. These changes are execution-critical (unblocks runtime) and correctness-critical (valid submission schema), and should also improve score versus ~0.51 by ensuring the submission is properly aligned and complete.'
- What this solution (achieved 0.50309) has done: 'I fix the protobuf compatibility shim so it works with protobuf 6.x by patching the correct class (`google.protobuf.message_factory.MessageFactory`) rather than the removed/changed symbol you were importing; this removes the import-time `GetPrototype` crash and lets TensorFlow load. Then I keep your model/dataset/inference logic unchanged, but make one correctness fix to ensure the submission always contains all 11 required target columns in the exact order even when the provided `sample_submission.csv` is missing some of them. Finally, I keep the existing merge-by-UID alignment and ensure `submission.csv` is always written successfully.'
- What this solution (achieved 0.53656) has done: 'The immediate blocker is that the protobuf shim is patching the wrong method signature: TF triggers a call to `message_factory.GetPrototype(factory_instance, descriptor)` (module-level), but we only added `MessageFactory.GetPrototype` (class method), so import still fails. I add a minimal, protobuf-6-safe shim that defines both `google.protobuf.message_factory.GetPrototype` (module-level function) and `MessageFactory.GetPrototype` (method) when missing, before importing TensorFlow. Then the rest of your pipeline (dataset creation, EfficientNet model load, prediction loop, and submission creation) run unchanged and should improve AUC substantially versus ~0.50 because it produce real predictions instead of failing/degenerating.'
- What this solution (achieved 0.47852) has done: 'I fix the TensorFlow/protobuf import crash by extending your existing protobuf shim so it also handles the `MessageFactory` **instance** missing `GetPrototype` (the actual error you’re seeing), while keeping the rest of your pipeline unchanged. This unblocks execution so the model can run and generate real predictions instead of failing early, which should move your AUC score upward toward the target. I also keep your robust submission-column completion logic and ensure the output CSV always contains all 11 required target columns in the correct order. No model architecture, preprocessing, or inference loop logic is changed beyond the import-compatibility fix.'
- What this solution (achieved 0.51385) has done: 'The current crash happens before TensorFlow can import because protobuf 6.x removed `MessageFactory.GetPrototype`, and your shim doesn’t cover the exact call pattern TensorFlow uses (it can call `factory.GetPrototype(descriptor)` on an instance). I replace the fragile partial shim with a minimal, protobuf-6-safe compatibility patch that (a) adds `GetPrototype` to the `MessageFactory` class and (b) wraps `MessageFactory()` to ensure instances also have `GetPrototype`, all before importing TensorFlow. Then I keep your model/dataset/inference logic unchanged, but add a small deterministic seed setup and ensure the submission always contains all 11 required target columns in the exact required order (even if the provided sample submission is missing some). This should both fix the runtime error and restore real predictions, improving AUC toward your target.'
- What this solution (achieved 0.49197) has done: 'I fix the TensorFlow/protobuf import crash by simplifying and hardening the protobuf-6 compatibility handling: instead of patching `MessageFactory` (which can still produce instances without `GetPrototype` in TF’s call path), we force the pure-Python protobuf backend and add a minimal module-level `GetPrototype` fallback that always works for TF’s usage. This is an execution-unblocking change and should be score-improving versus the current behavior (which never reaches real inference due to the crash). I keep your model, preprocessing, and prediction logic unchanged, and I also keep the robust submission-column completion so the output always has all 11 required target columns in the right order. The script write `submission.csv` in the working directory.'
- What this solution (achieved 0.48899) has done: 'I fix the TensorFlow/protobuf import crash by extending the existing protobuf-6 shim to also add `GetPrototype` to the **MessageFactory class** (TensorFlow calls it as an instance method, not only as a module-level function). This is the minimal execution-unblocking change and preserves all model/data logic exactly as-is. I also keep the robust submission-column completion (since your sample submission is missing required target columns in this environment) so the generated `submission.csv` always has all 11 targets in the correct order. No training/inference logic is altered beyond making the environment import successfully so real predictions can be produced, which should improve AUC toward the target versus the current ~0.49.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "2")

import google.protobuf  # noqa: F401
from google.protobuf import message_factory as _message_factory  # noqa: E402

if not hasattr(_message_factory, "GetPrototype"):

    def _GetPrototype(factory, descriptor):
        if factory is not None and hasattr(factory, "GetMessageClass"):
            return factory.GetMessageClass(descriptor)
        if hasattr(_message_factory, "GetMessageClass"):
            return _message_factory.GetMessageClass(descriptor)
        raise AttributeError(
            "protobuf message_factory has no GetPrototype/GetMessageClass"
        )

    setattr(_message_factory, "GetPrototype", _GetPrototype)

try:
    MessageFactoryCls = _message_factory.MessageFactory
    if not hasattr(MessageFactoryCls, "GetPrototype"):

        def _MF_GetPrototype(self, descriptor):
            if hasattr(self, "GetMessageClass"):
                return self.GetMessageClass(descriptor)
            return _message_factory.GetPrototype(self, descriptor)

        setattr(MessageFactoryCls, "GetPrototype", _MF_GetPrototype)
except Exception as _e:
    print("WARNING: Could not patch MessageFactory.GetPrototype:", _e)

import tensorflow as tf  # noqa: E402
import pandas as pd  # noqa: E402
import numpy as np  # noqa: E402

tf.random.set_seed(42)
np.random.seed(42)

W = H = 456
N_CLASSES = 11
autotune = tf.data.AUTOTUNE

CANDIDATE_DATA_DIRS = [
    "/kaggle/input/ranzcr-clip-catheter-line-classification",
    "/kaggle/data/ranzcr-clip-catheter-line-classification",
    "/kaggle/data/input/ranzcr-clip-catheter-line-classification",
    "../input/ranzcr-clip-catheter-line-classification",
]
DATA_DIR = None
for d in CANDIDATE_DATA_DIRS:
    if tf.io.gfile.exists(d):
        DATA_DIR = d
        break
if DATA_DIR is None:
    raise FileNotFoundError(
        "Could not find competition data directory. Tried: "
        + ", ".join(CANDIDATE_DATA_DIRS)
    )

TEST_IMG_DIR = os.path.join(DATA_DIR, "test")
SAMPLE_SUB_PATH = os.path.join(DATA_DIR, "sample_submission.csv")

target_cols = [
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

weight_dir_candidates = [
    "/kaggle/input/cassava2020weights",
    "/kaggle/data/cassava2020weights",
    "/kaggle/data/input/cassava2020weights",
    "../input/cassava2020weights",
]
weight_dir = None
for d in weight_dir_candidates:
    if tf.io.gfile.exists(d):
        weight_dir = d
        break

model_map = {
    "efficientb3": [
        tf.keras.applications.EfficientNetB3,
        (
            os.path.join(weight_dir, "ranzcr_efficientb3.h5")
            if weight_dir
            else "ranzcr_efficientb3.h5"
        ),
    ],
    "efficientb5": [
        tf.keras.applications.EfficientNetB5,
        (
            os.path.join(weight_dir, "ranzcr_efficientb5.h5")
            if weight_dir
            else "ranzcr_efficientb5.h5"
        ),
    ],
    "efficientb7": [
        tf.keras.applications.EfficientNetB7,
        (
            os.path.join(weight_dir, "ranzcr_efficientb7.h5")
            if weight_dir
            else "ranzcr_efficientb7.h5"
        ),
    ],
}

mean = tf.constant([0.485, 0.456, 0.406], dtype=tf.float32)
std = tf.constant([0.229, 0.224, 0.225], dtype=tf.float32)

print("TensorFlow:", tf.__version__)
print("DATA_DIR:", DATA_DIR)
print("TEST_IMG_DIR exists:", tf.io.gfile.exists(TEST_IMG_DIR))
print("SAMPLE_SUB_PATH exists:", tf.io.gfile.exists(SAMPLE_SUB_PATH))
print("weight_dir:", weight_dir)




## === cell 1
def triple_image(image):
    return tf.concat([image] * 3, axis=-1)


def decode_and_resize_from_path(path):
    img_bytes = tf.io.read_file(path)
    image = tf.image.decode_jpeg(img_bytes, channels=1)  # source is grayscale
    image = tf.image.resize(image, (H, W), method="bilinear")
    image = triple_image(image)

    image = tf.cast(image, tf.float32) / 255.0
    image = (image - mean) / std
    return image


def get_uid_from_path(path):
    uid = tf.strings.regex_replace(tf.strings.split(path, os.sep)[-1], r"\.jpg$", "")
    return uid


def make_test_dataset(test_img_dir, batch_size=16):
    files = tf.io.gfile.glob(os.path.join(test_img_dir, "*.jpg"))
    files = sorted(files)
    if len(files) == 0:
        raise FileNotFoundError(f"No .jpg files found in: {test_img_dir}")
    paths = tf.constant(files)

    ds = tf.data.Dataset.from_tensor_slices(paths)
    ds = ds.map(
        lambda p: (decode_and_resize_from_path(p), get_uid_from_path(p)),
        num_parallel_calls=autotune,
    )
    ds = ds.batch(batch_size)
    ds = ds.prefetch(1)
    return ds, files


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

    if init_weight:
        if tf.io.gfile.exists(init_weight):
            try:
                model.load_weights(init_weight)
                print(f"Weight loaded from {init_weight}")
            except Exception as e:
                print(f"Load weight from {init_weight} failed, {e}")
        else:
            print(
                f"WARNING: Weight path not found: {init_weight}. "
                f"Model will use baseline weights={baseline_weight}."
            )
    return model




## === cell 2
test_data, test_files = make_test_dataset(TEST_IMG_DIR, batch_size=16)

sample_sub = pd.read_csv(SAMPLE_SUB_PATH)

for c in ["StudyInstanceUID"] + target_cols:
    if c not in sample_sub.columns:
        sample_sub[c] = 0.0
sample_sub = sample_sub[["StudyInstanceUID"] + target_cols]

base_mode, weight_path = model_map["efficientb5"]
model = get_model(base_mode, init_weight=weight_path)



## === cell 3
preds = []
image_ids = []

for batch_images, batch_uids in test_data:
    batch_pred = model.predict_on_batch(batch_images)
    preds.append(batch_pred)
    image_ids.extend([uid.decode("utf-8") for uid in batch_uids.numpy()])

preds = np.concatenate(preds, axis=0)

sub = pd.DataFrame(preds, columns=target_cols)
sub.insert(0, "StudyInstanceUID", image_ids)

sub = sample_sub[["StudyInstanceUID"]].merge(sub, on="StudyInstanceUID", how="left")
sub[target_cols] = sub[target_cols].fillna(0.5)

sub = sub[["StudyInstanceUID"] + target_cols]
sub.to_csv("submission.csv", index=False)

print("Wrote submission.csv with shape:", sub.shape)
print("Columns:", list(sub.columns))
print(sub.head())
