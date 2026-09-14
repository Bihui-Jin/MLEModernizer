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
keras==3.8.0
keras-core==0.1.7
keras-cv==0.9.0
keras-hub==0.18.1
keras-nlp==0.18.1
keras-tuner==1.4.7
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
tf_keras==2.18.0

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

0.9003152108948389

# 6. Current score

0.76329

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.65192) has done: 'I fix the environment import crash by avoiding the standalone `keras` import that triggers the protobuf `MessageFactory` error in this Kaggle image, and rely on `tf.keras` instead. Because the referenced external pretrained model file does not exist in your provided `/kaggle/input` paths, I replace the missing `load_model()` step with a minimal, standard `tf.keras` CNN trained on the provided `train/` JPEGs and `train.csv`, keeping the same multi-label sigmoid + binary crossentropy semantics. I also correct the submission pipeline to match `sample_submission.csv`’s exact columns (your sample has 10 columns), filling only those targets and guaranteeing a valid `submission.csv` is written. Finally, I make the image reading robust (handle missing/failed reads) and use a batched `tf.data` input for test prediction so it finishes within the time limit.'
- What this solution (achieved 0.65623) has done: 'I fix the immediate runtime crash caused by the protobuf `MessageFactory.GetPrototype` incompatibility by forcing TensorFlow to use the pure-Python protobuf implementation before importing TensorFlow. This is a minimal environment fix and keeps your existing CNN training/inference pipeline unchanged. I also add a safe fallback to locate the dataset under either `/kaggle/input/ranzcr-clip-catheter-line-classification` or `/kaggle/data/...` (without changing I/O semantics) so it runs in the provided filesystem. Finally, I keep the submission columns exactly matching `sample_submission.csv` and ensure `submission.csv` is always written.'
- What this solution (achieved 0.65519) has done: 'I fix the protobuf/TensorFlow import crash by ensuring protobuf is pinned to the pure-Python implementation *before* any TensorFlow-related modules load, and by force-reloading protobuf if it was already imported by the notebook kernel. Then I keep your existing CNN training/inference pipeline unchanged, only adding a safe fallback to ensure TF doesn’t touch incompatible compiled protobuf bindings. Finally, I keep the submission format identical to `sample_submission.csv` and always write `submission.csv` successfully.'
- What this solution (achieved 0.65269) has done: 'I fix the TensorFlow/protobuf crash by forcing the pure-Python protobuf implementation and patching the missing `MessageFactory.GetPrototype` method (TensorFlow 2.18 expects it, but protobuf 6 provides `GetMessageClass`). This is a minimal environment/runtime fix that keeps your CNN training/inference logic unchanged. I also update the dataset path search to include the exact directories present in your filesystem (`/kaggle/input/.../ranzcr-clip-catheter-line-classification` and `/kaggle/data/.../ranzcr-clip-catheter-line-classification`) so it reliably finds the CSVs and image folders. The rest of the pipeline (patient split, tf.data, model, submission formatting) remains the same and write a valid `submission.csv`.'
- What this solution (achieved 0.66077) has done: 'I make two minimal changes that tend to improve AUC without changing your model architecture or training loop structure: (1) switch the loss to `BinaryCrossentropy(label_smoothing=0.01)` to reduce overconfidence and improve ranking, and (2) add `tf.keras.metrics.AUC(multi_label=True)` so you can verify training is moving in the right direction (this doesn’t affect predictions). I also ensure the submission schema matches the *true* competition format (11 target columns) by always starting from `sample_submission.csv` and filling any missing target columns with model predictions when available, otherwise a safe constant (so the CSV is always valid). These are small, low-risk edits intended to move your 0.65269 upward toward 0.9003 while keeping runtime under the limit and preserving the same CNN pipeline.'
- What this solution (achieved 0.71261) has done: 'Your current score (0.66077) is far below the target (0.9003), so we should make small, legitimate changes that improve AUC without changing the core CNN/training loop. The biggest low-risk gain here is fixing the label set mismatch: your `sample_submission.csv` in this environment has only 9 targets, while `train.csv` has 11; training on fewer labels and submitting fewer columns hurts both learning and evaluation validity. I keep your exact model and training approach, but (1) always train/predict all 11 true target columns from `train.csv`, (2) write a submission that includes all 11 targets (creating missing columns if the provided sample is truncated), and (3) use simple per-image standardization (not a new model) to improve ranking stability for AUC.'
- What this solution (achieved 0.75193) has done: 'The timeout is dominated by repeatedly decoding/resizing/standardizing ~27k JPEGs on CPU for 6 epochs, plus the expensive multi-label AUC metric computed every epoch. To keep the exact model/training logic intact, the main speedups are: (1) cache the decoded+resized+standardized images to disk once (so later epochs reuse them), (2) remove the redundant `tf.cond(True, ...)` and keep a pure-TF decode path, (3) use a single parallelized preprocessing pipeline with `ignore_errors()` to avoid stalls, and (4) compute the AUC metric only at validation time via a callback (same metric semantics, but avoids per-step training AUC overhead). These changes preserve the architecture, loss, optimizer, epochs, and data split, while reducing the per-epoch input pipeline cost drastically and cutting metric overhead.'
- What this solution (achieved 0.73718) has done: 'Your current score (0.75193) is well below the target (0.9003), so we should make small, legitimate changes that tend to improve AUC without changing the CNN architecture or the overall training loop. The biggest likely gain with minimal risk is to fix the input standardization mismatch caused by random brightness/contrast being applied *after* per-image standardization; we re-standardize after augmentation so the model sees a more consistent distribution (better ranking/AUC). We also add a tiny amount of L2 weight decay in the optimizer (doesn’t change the model, just regularization) to improve generalization, and ensure test-time uses deterministic inference while keeping the same epochs, data split, and loss semantics. Submission writing remains identical, still producing a valid `submission.csv` with all 11 target columns.'
- What this solution (achieved 0.52697) has done: 'Your current score (0.73718) is well below the target (0.9003), so we should make small, legitimate changes that tend to improve AUC without changing your CNN architecture or training loop. The biggest low-risk fix is to align the submission schema to the true competition (11 target columns) by using `train.csv`’s target list as the source of truth, and only using `sample_submission.csv` for test IDs—your provided sample is truncated to 9 targets, which can invalidate scoring. Next, we add a minimal class-imbalance aware weighting inside the existing binary crossentropy (same loss family, same sigmoid outputs) to improve ranking for rare labels and typically lift mean AUC. Finally, we keep everything deterministic and keep epochs/optimizer/model unchanged, only adding the loss weighting and robust submission column handling.'
- What this solution (achieved 0.75962) has done: 'I fix the training crash by correcting the custom weighted BCE implementation: `BinaryCrossentropy(reduction=NONE)` returns a per-sample loss (shape `[B]`), so multiplying it by a per-label weight matrix causes the broadcast error; we instead compute per-label BCE manually to keep shape `[B, L]` and then apply your existing positive-class weights. This keeps the same CNN architecture, sigmoid outputs, and weighted BCE semantics, but makes it executable. I also make the validation AUC callback robust and ensure the submission always contains all 11 required target columns in the correct order, even if the provided `sample_submission.csv` is truncated. These changes are directly tied to correctness and should restore (and likely improve) the score relative to the broken run.'
- What this solution (achieved 0.76042) has done: 'We keep your exact CNN, epochs, tf.data pipeline structure, and weighted BCE core logic, but make two small, AUC-relevant fixes. First, we compute `pos_rate/pos_weight` from the full `train_df` (not just the training split) to reduce split-noise in the loss weighting, which typically improves ranking stability without changing model semantics. Second, we add lightweight test-time augmentation (horizontal flip) and average the two predictions; this does not change training or architecture and often yields a modest AUC lift toward your target. Everything else (paths, caching, submission columns/order, determinism settings, runtime budget) stays the same and still writes a valid `submission.csv`.'
- What this solution (achieved 0.76329) has done: 'We keep your exact CNN, epochs, weighted BCE, and tf.data structure, but make two minimal AUC-relevant fixes to move upward toward 0.9003. First, we switch the weighted BCE to a numerically-stable logits-based formulation (same loss semantics, but avoids saturation/clipping artifacts that can hurt ranking/AUC). Second, we add a tiny multi-scale test-time augmentation (average predictions at the original size and a slightly larger resize), which often gives a modest AUC lift without changing training or architecture. Submission writing still force the full 11 required columns in the correct order and write `submission.csv` end-to-end.'
- What this solution (achieved 0.76329) has done: 'Your current score (0.76329) is below the target (0.9003), so we should make a small, legitimate change that tends to improve mean AUC without changing your CNN, epochs, optimizer, or the overall training/prediction flow. The biggest low-risk issue is that you’re training on a patient-level split but not using those OOF predictions to calibrate the loss-weighted logits into better-ranked probabilities; adding a simple per-label temperature scaling fit on the validation set can improve AUC ranking/calibration while keeping the same model outputs and semantics (still probabilities). I keep the same training exactly, then fit 11 scalar temperatures (one per label) on the validation predictions to minimize log-loss, and apply that transform to test predictions (including your existing TTA). This is fast, deterministic, and typically nudges AUC upward without changing core architecture/loop. The submission schema/order remains exactly `StudyInstanceUID` + 11 targets and still writes `submission.csv`.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION"] = "2"

import sys
import importlib

if "google.protobuf" in sys.modules:
    for k in list(sys.modules.keys()):
        if k.startswith("google.protobuf"):
            del sys.modules[k]
    importlib.invalidate_caches()

from google.protobuf import message_factory as _message_factory  # noqa: E402

if not hasattr(_message_factory.MessageFactory, "GetPrototype"):

    def _GetPrototype(self, descriptor):
        return self.GetMessageClass(descriptor)

    _message_factory.MessageFactory.GetPrototype = _GetPrototype

import random
import numpy as np
import pandas as pd

import tensorflow as tf
import cv2

SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

try:
    tf.config.threading.set_intra_op_parallelism_threads(0)
    tf.config.threading.set_inter_op_parallelism_threads(0)
except Exception:
    pass

print("TF:", tf.__version__)
print("Num GPUs:", len(tf.config.list_physical_devices("GPU")))



## === cell 1
CANDIDATE_DATA_DIRS = [
    "/kaggle/input/ranzcr-clip-catheter-line-classification/ranzcr-clip-catheter-line-classification",
    "/kaggle/input/ranzcr-clip-catheter-line-classification",
    "/kaggle/data/ranzcr-clip-catheter-line-classification",
]
DATA_DIR = None
for d in CANDIDATE_DATA_DIRS:
    if os.path.exists(d):
        if os.path.exists(os.path.join(d, "train.csv")) and os.path.exists(
            os.path.join(d, "sample_submission.csv")
        ):
            DATA_DIR = d
            break

if DATA_DIR is None:
    raise FileNotFoundError(f"Could not find dataset dir. Tried: {CANDIDATE_DATA_DIRS}")

TRAIN_CSV = os.path.join(DATA_DIR, "train.csv")
SAMPLE_SUB = os.path.join(DATA_DIR, "sample_submission.csv")
TRAIN_IMG_DIR = os.path.join(DATA_DIR, "train")
TEST_IMG_DIR = os.path.join(DATA_DIR, "test")

assert os.path.exists(TRAIN_CSV), f"Missing: {TRAIN_CSV}"
assert os.path.exists(SAMPLE_SUB), f"Missing: {SAMPLE_SUB}"
assert os.path.isdir(TRAIN_IMG_DIR), f"Missing dir: {TRAIN_IMG_DIR}"
assert os.path.isdir(TEST_IMG_DIR), f"Missing dir: {TEST_IMG_DIR}"

train_df = pd.read_csv(TRAIN_CSV)
sub = pd.read_csv(SAMPLE_SUB)

print("DATA_DIR:", DATA_DIR)
print("train_df:", train_df.shape)
print("sub:", sub.shape)
print("sample submission columns:", list(sub.columns))

ALL_TARGET_COLS = [
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
missing_train_targets = [c for c in ALL_TARGET_COLS if c not in train_df.columns]
if missing_train_targets:
    raise ValueError(f"Targets missing in train.csv: {missing_train_targets}")

TARGET_COLS = ALL_TARGET_COLS
print("Using TARGET_COLS (train/predict):", TARGET_COLS)

if "StudyInstanceUID" not in sub.columns:
    raise ValueError("sample_submission.csv must contain StudyInstanceUID")
missing_in_sub = [c for c in TARGET_COLS if c not in sub.columns]
if missing_in_sub:
    print("WARNING: sample_submission missing target columns:", missing_in_sub)
    for c in missing_in_sub:
        sub[c] = 0.0  # will be overwritten by predictions

sub = sub[["StudyInstanceUID"] + [c for c in sub.columns if c != "StudyInstanceUID"]]



## === cell 2
IMG_SIZE = 260
BATCH_SIZE = 16
AUTOTUNE = tf.data.AUTOTUNE

EPOCHS = 6  # keep identical training length


def read_image_cv2(path, img_size=IMG_SIZE):
    """Reads jpg with cv2 (BGR), converts to RGB float32 and standardizes. Robust to read failures."""
    img = cv2.imread(path)
    if img is None:
        img = np.zeros((img_size, img_size, 3), dtype=np.uint8)
    else:
        img = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_AREA)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img = img.astype(np.float32) / 255.0

    m = float(img.mean())
    s = float(img.std())
    img = (img - m) / (s + 1e-6)

    return img


def make_path(uid, root_dir):
    return os.path.join(root_dir, f"{uid}.jpg")


patients = train_df["PatientID"].astype(str).values
unique_patients = np.unique(patients)
rng = np.random.default_rng(SEED)
rng.shuffle(unique_patients)
val_frac = 0.1
n_val = int(len(unique_patients) * val_frac)
val_patients = set(unique_patients[:n_val])

is_val = train_df["PatientID"].astype(str).isin(val_patients).values
trn_df = train_df.loc[~is_val].reset_index(drop=True)
val_df = train_df.loc[is_val].reset_index(drop=True)

print("Train/Val:", trn_df.shape, val_df.shape)




## === cell 3
@tf.function
def tf_decode_resize_standardize(path):
    img_bytes = tf.io.read_file(path)
    img = tf.io.decode_jpeg(img_bytes, channels=3)  # RGB
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE], method=tf.image.ResizeMethod.AREA)
    img = tf.cast(img, tf.float32) / 255.0
    mean = tf.reduce_mean(img)
    std = tf.math.reduce_std(img)
    img = (img - mean) / (std + 1e-6)
    img.set_shape([IMG_SIZE, IMG_SIZE, 3])
    return img


@tf.function
def tf_standardize(img):
    mean = tf.reduce_mean(img)
    std = tf.math.reduce_std(img)
    img = (img - mean) / (std + 1e-6)
    img.set_shape([IMG_SIZE, IMG_SIZE, 3])
    return img


def tf_load_train_example(uid, y):
    path = tf.strings.join(
        [tf.constant(TRAIN_IMG_DIR), tf.constant("/"), uid, tf.constant(".jpg")]
    )
    img = tf_decode_resize_standardize(path)
    y = tf.cast(y, tf.float32)
    y.set_shape([len(TARGET_COLS)])
    return img, y


CACHE_DIR = "/kaggle/working/tf_cache_ranzcr"
os.makedirs(CACHE_DIR, exist_ok=True)
TRAIN_CACHE = os.path.join(CACHE_DIR, f"train_img_{IMG_SIZE}.cache")
VAL_CACHE = os.path.join(CACHE_DIR, f"val_img_{IMG_SIZE}.cache")


def make_ds(df, training=True, cache_path=None):
    uids = df["StudyInstanceUID"].astype(str).values
    y = df[TARGET_COLS].astype(np.float32).values
    ds = tf.data.Dataset.from_tensor_slices((uids, y))

    options = tf.data.Options()
    options.experimental_deterministic = True  # preserve reproducibility
    ds = ds.with_options(options)

    if training:
        ds = ds.shuffle(min(len(df), 4096), seed=SEED, reshuffle_each_iteration=True)

    ds = ds.map(tf_load_train_example, num_parallel_calls=AUTOTUNE)
    ds = ds.apply(tf.data.experimental.ignore_errors())

    if cache_path is not None:
        ds = ds.cache(cache_path)
    else:
        ds = ds.cache()

    if training:

        def aug(img, y):
            img = tf.image.random_flip_left_right(img, seed=SEED)
            img = tf.image.random_brightness(img, max_delta=0.08, seed=SEED)
            img = tf.image.random_contrast(img, lower=0.9, upper=1.1, seed=SEED)
            img = tf_standardize(img)
            return img, y

        ds = ds.map(aug, num_parallel_calls=AUTOTUNE)

    ds = ds.batch(BATCH_SIZE, drop_remainder=False).prefetch(AUTOTUNE)
    return ds


train_ds = make_ds(trn_df, training=True, cache_path=TRAIN_CACHE)
val_ds = make_ds(val_df, training=False, cache_path=VAL_CACHE)



## === cell 4
inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
x = tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu")(inputs)
x = tf.keras.layers.MaxPooling2D()(x)
x = tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu")(x)
x = tf.keras.layers.MaxPooling2D()(x)
x = tf.keras.layers.Conv2D(128, 3, padding="same", activation="relu")(x)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dropout(0.2)(x)
outputs = tf.keras.layers.Dense(len(TARGET_COLS), activation="sigmoid")(x)

model = tf.keras.Model(inputs, outputs)

pos_rate = train_df[TARGET_COLS].mean(axis=0).astype(np.float32).values
pos_weight = (1.0 - pos_rate) / (pos_rate + 1e-6)
pos_weight = np.clip(pos_weight, 1.0, 10.0).astype(np.float32)
pos_weight_tf = tf.constant(pos_weight, dtype=tf.float32)

label_smoothing = 0.01


@tf.function
def weighted_bce(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    if label_smoothing and label_smoothing > 0.0:
        y_true = y_true * (1.0 - label_smoothing) + 0.5 * label_smoothing

    eps = tf.constant(1e-7, dtype=tf.float32)
    y_pred = tf.clip_by_value(y_pred, eps, 1.0 - eps)
    logits = tf.math.log(y_pred) - tf.math.log(1.0 - y_pred)  # logit(p)

    per_elem = tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true, logits=logits)
    weights = 1.0 + y_true * (pos_weight_tf - 1.0)
    per_elem = per_elem * weights
    return tf.reduce_mean(per_elem)


try:
    optimizer = tf.keras.optimizers.AdamW(learning_rate=1e-3, weight_decay=1e-5)
except Exception:
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

model.compile(
    optimizer=optimizer,
    loss=weighted_bce,
)

model.summary()
print("Per-label pos_rate:", dict(zip(TARGET_COLS, pos_rate.round(4))))
print("Per-label pos_weight (clipped):", dict(zip(TARGET_COLS, pos_weight.round(3))))




## === cell 5
class ValAUC(tf.keras.callbacks.Callback):
    def __init__(self, val_ds):
        super().__init__()
        self.val_ds = val_ds
        self.metric = tf.keras.metrics.AUC(
            multi_label=True, num_thresholds=200, name="val_auc"
        )

    def on_epoch_end(self, epoch, logs=None):
        self.metric.reset_state()
        for xb, yb in self.val_ds:
            ypred = self.model(xb, training=False)
            self.metric.update_state(yb, ypred)
        auc = float(self.metric.result().numpy())
        if logs is not None:
            logs["val_auc"] = auc
        print(f"\nval_auc: {auc:.6f}")


history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    verbose=1,
    callbacks=[ValAUC(val_ds)],
)




## === cell 6
def _collect_val_preds(model, val_ds):
    ys = []
    ps = []
    for xb, yb in val_ds:
        pb = model(xb, training=False).numpy()
        ys.append(yb.numpy())
        ps.append(pb)
    y = np.concatenate(ys, axis=0).astype(np.float32)
    p = np.concatenate(ps, axis=0).astype(np.float32)
    p = np.clip(p, 1e-6, 1.0 - 1e-6)
    return y, p


def _fit_temperature_per_label(y, p, n_iter=60):
    """
    Fit temperature T per label by minimizing BCE on validation:
      q = sigmoid(logit(p) / T)
    This keeps outputs as probabilities and usually improves AUC a bit.
    """
    logits = np.log(p) - np.log(1.0 - p)
    L = y.shape[1]
    Ts = np.ones((L,), dtype=np.float32)

    for j in range(L):
        yj = tf.constant(y[:, j : j + 1], dtype=tf.float32)
        lj = tf.constant(logits[:, j : j + 1], dtype=tf.float32)
        t = tf.Variable(1.0, dtype=tf.float32)

        log_t = tf.Variable(0.0, dtype=tf.float32)
        opt = tf.keras.optimizers.Adam(learning_rate=0.05)

        for _ in range(n_iter):
            with tf.GradientTape() as tape:
                t_pos = tf.exp(log_t)
                q = tf.sigmoid(lj / t_pos)
                loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(yj, q))
            grads = tape.gradient(loss, [log_t])
            opt.apply_gradients(zip(grads, [log_t]))

        Ts[j] = float(tf.exp(log_t).numpy())

    Ts = np.clip(Ts, 0.5, 3.0).astype(np.float32)
    return Ts


val_y, val_p = _collect_val_preds(model, val_ds)
temps = _fit_temperature_per_label(val_y, val_p, n_iter=50)
print("Fitted per-label temperatures:", dict(zip(TARGET_COLS, np.round(temps, 3))))


def apply_temperature(p, temps):
    p = np.clip(p, 1e-7, 1.0 - 1e-7).astype(np.float32)
    logits = np.log(p) - np.log(1.0 - p)
    logits = logits / temps.reshape(1, -1)
    out = 1.0 / (1.0 + np.exp(-logits))
    return np.clip(out, 0.0, 1.0).astype(np.float32)




## === cell 7
def tf_load_test_image(uid):
    path = tf.strings.join(
        [tf.constant(TEST_IMG_DIR), tf.constant("/"), uid, tf.constant(".jpg")]
    )
    img = tf_decode_resize_standardize(path)
    return img


@tf.function
def tta_avg_predict(batch_imgs):
    p1 = model(batch_imgs, training=False)
    p2 = model(tf.image.flip_left_right(batch_imgs), training=False)
    p_base = (p1 + p2) * 0.5

    bigger = tf.image.resize(
        batch_imgs, [IMG_SIZE + 24, IMG_SIZE + 24], method=tf.image.ResizeMethod.AREA
    )
    bigger = tf.image.resize(
        bigger, [IMG_SIZE, IMG_SIZE], method=tf.image.ResizeMethod.AREA
    )
    p3 = model(bigger, training=False)
    p4 = model(tf.image.flip_left_right(bigger), training=False)
    p_big = (p3 + p4) * 0.5

    return (p_base + p_big) * 0.5


test_uids = sub["StudyInstanceUID"].astype(str).values
test_ds = tf.data.Dataset.from_tensor_slices(test_uids)

options = tf.data.Options()
options.experimental_deterministic = True
test_ds = test_ds.with_options(options)

TEST_CACHE = os.path.join(CACHE_DIR, f"test_img_{IMG_SIZE}.cache")

test_ds = (
    test_ds.map(tf_load_test_image, num_parallel_calls=AUTOTUNE)
    .apply(tf.data.experimental.ignore_errors())
    .cache(TEST_CACHE)
    .batch(BATCH_SIZE)
    .prefetch(AUTOTUNE)
)

pred_chunks = []
for xb in test_ds:
    pred_chunks.append(tta_avg_predict(xb).numpy())

pred = np.concatenate(pred_chunks, axis=0)
pred = np.clip(pred, 0.0, 1.0)

pred = apply_temperature(pred, temps)

pred_df = pd.DataFrame(pred, columns=TARGET_COLS)

submission = sub.copy()
for c in TARGET_COLS:
    submission[c] = pred_df[c].values

submission = submission[["StudyInstanceUID"] + TARGET_COLS]

assert submission.shape[0] == sub.shape[0]
out_path = "submission.csv"
submission.to_csv(out_path, index=False)
print("Wrote:", out_path)
print("submission shape:", submission.shape)
print(submission.head())
