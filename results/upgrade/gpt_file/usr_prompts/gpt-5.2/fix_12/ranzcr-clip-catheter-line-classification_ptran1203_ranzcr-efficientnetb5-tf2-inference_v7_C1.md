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

0.6936111762907413

# 6. Current score

0.49159

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.4577) has done: 'I fix the protobuf/TensorFlow crash by removing the forced pure-Python protobuf implementation (it breaks with protobuf 6.x) so TF can import cleanly. Then I fix the prediction loop error by treating `predict_on_batch` output as a NumPy array (it’s already NumPy in TF 2.18), removing the invalid `.numpy()` call. Finally, I make the I/O paths robust for your provided `/kaggle/input/...` layout, and ensure the submission file contains all 11 required target columns (the provided sample has only 10) in the correct order with a `.csv` suffix.'
- What this solution (achieved 0.50178) has done: 'You’re hitting a protobuf 6.x vs older-generated-protos incompatibility during the TensorFlow import, which triggers the `MessageFactory.GetPrototype` error before any model code can run. I add a small, safe protobuf “compat shim” that defines `GetPrototype` in terms of `GetMessageClass` when missing, so TF can import cleanly without downgrading packages. Then I fix a correctness issue that can hurt score: `EfficientNetB3` expects `input_shape=(H, W, 3)` but the code used `(W, H, 3)`; this swaps dimensions and can degrade predictions. Finally, I keep the rest of the pipeline intact and ensure the submission contains all 11 required target columns in the correct order and writes `submission.csv`.'
- What this solution (achieved 0.4781) has done: 'I keep your inference-only pipeline and EfficientNetB3 architecture intact, and make two minimal changes that typically move AUC upward for this competition: (1) use the exact EfficientNetB3 `preprocess_input` instead of manual mean/std (mismatch can noticeably hurt logits), and (2) run test-time augmentation via a simple left-right flip and average the two predictions (this preserves the same model and loss, but improves ranking/AUC without extra training). I also ensure the dataset decoding outputs float tensors correctly for `preprocess_input` and keep the submission column order at 11 targets. These changes are small, safe, and should improve your score toward 0.6936 without changing the core approach.'
- What this solution (achieved 0.44956) has done: 'I make two minimal inference-time adjustments that usually improve mean AUC for this competition without changing your model/training logic: switch the EfficientNet preprocessing call to the correct B3 variant, and apply test-time augmentation with both left-right and up-down flips (averaged) to improve ranking stability. I also keep the submission aligned to the sample submission ordering (so IDs exactly match Kaggle’s expected row order) and add a safety check that all test image files exist for those IDs to avoid silent misalignment. These changes are small, keep your architecture and weights intact, and are directly aimed at nudging your score upward toward the 0.6936 target.'
- What this solution (achieved 0.47419) has done: 'Your current score gap to the target is large (0.44956 vs 0.6936), and the biggest low-risk issue in the provided code is a preprocessing/model-family mismatch: you are using `EfficientNetB3` (v1) but applying `efficientnet_v2.preprocess_input`, which significantly degrade ranking/AUC. I switch preprocessing to the correct `tf.keras.applications.efficientnet.preprocess_input` while keeping the same model, weights, inference loop, and TTA strategy. I also add a small safety clamp to keep predictions strictly within (0,1) to avoid any rare numerical edge issues with sigmoid+float ops. Everything else (paths, column order, TTA averaging, submission writing) stays the same to keep changes minimal and stable.'
- What this solution (achieved 0.46082) has done: 'The biggest remaining score limiter (given your fixed EfficientNetB3 preprocessing and TTA) is likely train/test distribution mismatch: these X-rays are usually single-channel and the weight file is typically trained on 1-channel replicated input but with histogram/contrast normalization. To move the AUC upward toward your target with minimal change and identical model/inference semantics, I add a very light, deterministic per-image contrast normalization (standardize to zero-mean/unit-std) before EfficientNet preprocessing; this often improves ranking without changing architecture or training loops. I also make the resizing method match common EfficientNet pipelines (`bicubic`) and ensure dtype/range are stable. Everything else (model, weights, TTA averaging, submission formatting/ordering) stays the same.'
- What this solution (achieved 0.45751) has done: 'I keep your EfficientNetB3 + loaded weights + sigmoid head and the same 4-way flip TTA, but fix two small inference-time issues that can materially affect AUC without changing the modeling approach. First, your current per-image standardization happens on the already triplicated 3-channel tensor; I standardize on the original single channel before replication so the normalization is consistent and not channel-dependent. Second, I ensure the template submission (row order) remains the one from `sample_submission.csv` by not overwriting `StudyInstanceUID` with the iterated IDs (we instead assert alignment and only fill prediction columns); this avoids any silent ordering mismatch that can heavily depress score even when predictions are good. These are minimal, deterministic changes aimed at improving ranking/AUC toward your 0.6936 target while preserving core logic.'
- What this solution (achieved 0.50522) has done: 'You’re still far below the target (0.4575 vs 0.6936), so we should make a small, low-risk inference-time change that typically increases AUC without changing your model or training logic. The biggest likely limiter in your current preprocessing is the extra per-image standardization before `tf.keras.applications.efficientnet.preprocess_input`, which can distort the intensity distribution that your provided B3 weights were trained on. I remove that standardization (keeping everything else: 1-channel decode, replication to 3 channels, correct EfficientNetB3 preprocess, and the same 4-flip TTA), and I keep the strict ID alignment checks and 11-column submission formatting unchanged. This should move the score upward toward your target while staying within minimal-change constraints.'
- What this solution (achieved 0.45017) has done: 'Your current score (0.50522) is well below the target (0.69361), so we should make a small, legitimate inference-time change that tends to improve mean AUC without changing your model, weights, or training approach. The most likely issue is that applying vertical-flip TTA on chest X-rays is distribution-shifting (anatomically implausible) and can degrade ranking; we keep TTA but restrict it to only the horizontal flip (and original), averaging the two predictions. This is a minimal change to your inference loop that preserves architecture/loss/weights and should move the score upward toward the target band. Everything else (EfficientNetB3 preprocessing, ID alignment checks, and 11-column submission formatting) is kept the same.'
- What this solution (achieved 0.49159) has done: 'We keep your EfficientNetB3 model/weights and inference loop intact, but fix a likely major score drag: you are currently using ImageNet backbone weights *and* then loading your task-specific weights on top, which can partially overwrite/mismatch layers depending on how the `.h5` was saved. The minimal, low-risk change is to build the backbone with `weights=None` when you intend to load a full set of trained weights, and to load weights with `by_name=True, skip_mismatch=True` to avoid silent shape/name issues that can harm predictions. This preserves the same architecture, loss, preprocessing, and TTA strategy, but should improve the effective use of your provided competition weights and move AUC upward toward your target. The submission writing, column order, and ID alignment checks remain unchanged to ensure a valid `.csv`.'

# 9. Code solution

## === cell 0
import os

try:
    from google.protobuf import message_factory as _message_factory

    if hasattr(_message_factory, "MessageFactory") and not hasattr(
        _message_factory.MessageFactory, "GetPrototype"
    ):

        def _GetPrototype(self, descriptor):
            return self.GetMessageClass(descriptor)

        _message_factory.MessageFactory.GetPrototype = _GetPrototype
except Exception:
    pass

import tensorflow as tf
import pandas as pd
import numpy as np

W = H = 224
N_CLASSES = 11
autotune = tf.data.AUTOTUNE

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

DATA_ROOT = "/kaggle/input/ranzcr-clip-catheter-line-classification"
if not os.path.isdir(DATA_ROOT):
    DATA_ROOT = "../input/ranzcr-clip-catheter-line-classification"

TEST_DIR = os.path.join(DATA_ROOT, "test")
SAMPLE_SUB_PATH = os.path.join(DATA_ROOT, "sample_submission.csv")

weight_path = "../input/cassava2020weights/ranzcr_efficientb3.h5"
if not os.path.isfile(weight_path):
    alt_weight_path = "/kaggle/input/cassava2020weights/ranzcr_efficientb3.h5"
    if os.path.isfile(alt_weight_path):
        weight_path = alt_weight_path

print("TF version:", tf.__version__)
print(
    "Test dir exists:",
    os.path.isdir(TEST_DIR),
    "num files:",
    len(os.listdir(TEST_DIR)) if os.path.isdir(TEST_DIR) else 0,
)
print("Sample submission exists:", os.path.isfile(SAMPLE_SUB_PATH))
print("Weight exists:", os.path.isfile(weight_path))




## === cell 1
def triple_image(image):
    return tf.concat([image] * 3, axis=-1)


def decode_and_resize_from_path(path):
    img_bytes = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img_bytes, channels=1)
    img = tf.image.resize(img, (H, W), method="bicubic")
    return img  # 1-channel


def preprocess_for_effnet(image, image_id):
    image = tf.cast(image, tf.float32)  # [0..255], shape (H,W,1)
    image = triple_image(image)  # (H,W,3)
    image = tf.keras.applications.efficientnet.preprocess_input(image)
    return image, image_id


def make_test_dataset(test_dir, batch_size=16):
    sub = pd.read_csv(SAMPLE_SUB_PATH)
    ids = sub["StudyInstanceUID"].astype(str).values
    paths = np.array([os.path.join(test_dir, f"{uid}.jpg") for uid in ids])

    missing = [p for p in paths if not os.path.isfile(p)]
    if len(missing) > 0:
        raise FileNotFoundError(
            f"Missing {len(missing)} test images (showing up to 5): {missing[:5]}"
        )

    ds = tf.data.Dataset.from_tensor_slices((paths, ids))
    ds = ds.map(
        lambda p, uid: (decode_and_resize_from_path(p), uid),
        num_parallel_calls=autotune,
    )
    ds = ds.map(preprocess_for_effnet, num_parallel_calls=autotune)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(autotune)
    return ds, sub


def get_model(
    baseline_weight=None, init_weight=None, lr=0.001, optimizer=tf.optimizers.Adam
):
    if init_weight:
        effective_baseline = None
    else:
        effective_baseline = baseline_weight

    base_model = tf.keras.applications.EfficientNetB3(
        include_top=False,
        input_shape=(H, W, 3),
        pooling="avg",
        weights=effective_baseline,
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
        try:
            model.load_weights(init_weight, by_name=True, skip_mismatch=True)
            print(
                f"Weight loaded from {init_weight} (by_name=True, skip_mismatch=True)"
            )
        except TypeError:
            model.load_weights(init_weight)
            print(f"Weight loaded from {init_weight} (default load_weights)")
        except Exception as e:
            print(f"Load weight from {init_weight} failed: {e}")
    return model




## === cell 2
test_data, submission_template = make_test_dataset(TEST_DIR, batch_size=16)

baseline = "imagenet"
model = get_model(baseline_weight=baseline, init_weight=weight_path, lr=0.001)

preds = []
image_ids = []

for batch_imgs, batch_ids in test_data:
    p0 = model.predict_on_batch(batch_imgs)
    p1 = model.predict_on_batch(tf.image.flip_left_right(batch_imgs))

    p = (np.asarray(p0) + np.asarray(p1)) / 2.0
    p = np.clip(p, 1e-7, 1.0 - 1e-7)

    preds.append(p)

    batch_ids_np = batch_ids.numpy()
    image_ids.extend(
        [
            x.decode("utf-8") if isinstance(x, (bytes, bytearray)) else str(x)
            for x in batch_ids_np
        ]
    )

preds = np.concatenate(preds, axis=0)

sub = submission_template.copy()

for c in target_cols:
    if c not in sub.columns:
        sub[c] = 0.0

if preds.shape[1] != len(target_cols):
    raise ValueError(
        f"Model output shape {preds.shape} does not match expected {len(target_cols)} classes."
    )

template_ids = sub["StudyInstanceUID"].astype(str).tolist()
if template_ids != image_ids:
    raise ValueError(
        "StudyInstanceUID order mismatch between sample_submission and dataset iteration. "
        "Refusing to write a potentially misaligned submission."
    )

sub[target_cols] = preds.astype(np.float32)
sub = sub[["StudyInstanceUID"] + target_cols]
sub.to_csv("submission.csv", index=False)

print("Wrote submission.csv with shape:", sub.shape)
print(sub.head())
print("Columns:", list(sub.columns))
