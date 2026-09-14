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
Classify each cassava image into four disease categories or a fifth category indicating a healthy leaf.

## Metric
Categorization accuracy.

## Submission Format
```
image_id,label
1000471002.jpg,4
1000840542.jpg,4
etc.
```

## Dataset
**[train/test]_images** the image files.

**train.csv**

- `image_id` the image file name.

- `label` the ID code for the disease.

**sample_submission.csv** A properly formatted sample submission, given the disclosed test set content.

- `image_id` the image file name.

- `label` the predicted ID code for the disease.

**[train/test]_tfrecords** the image files in tfrecord format.

**label_num_to_disease_map.json** The mapping between each disease code and the real disease name.

# 2. Python version

2.7

# 3. Installed packages

No external packages required in the script and installed.

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (124 lines)
            label_num_to_disease_map.json (1 lines)
            sample_submission.csv (2677 lines)
            sample_submission.csv.zip (13.4 kB)
            test.zip (160 Bytes)
            test_images.zip (319.5 MB)
            test_tfrecords.zip (451.9 MB)
            train.csv (18722 lines)
            train.csv.zip (100.0 kB)
            train.zip (162 Bytes)
            train_images.zip (2.2 GB)
            train_tfrecords.zip (3.2 GB)
            cassava-leaf-disease-classification/
                description.md (124 lines)
                label_num_to_disease_map.json (1 lines)
                ... and 10 other files
                cassava-leaf-disease-classification/
                test_images/
                    2574872277.jpg (183.5 kB)
                    1449210447.jpg (100.8 kB)
                    ... and 2674 other files
                    test_images/
                test_tfrecords/
                    ld_test00-1338.tfrec (225.9 MB)
                    ld_test01-1338.tfrec (226.2 MB)
                train_images/
                    478676678.jpg (90.6 kB)
                    2315755156.jpg (59.5 kB)
                    ... and 18719 other files
                    train_images/
                train_tfrecords/
                    ld_train00-1338.tfrec (227.2 MB)
                    ld_train01-1338.tfrec (227.0 MB)
                    ... and 12 other files
            test_images/
                2574872277.jpg (183.5 kB)
                1449210447.jpg (100.8 kB)
                ... and 2674 other files
                test_images/
            test_tfrecords/
                ld_test00-1338.tfrec (225.9 MB)
                ld_test01-1338.tfrec (226.2 MB)
            train_images/
                478676678.jpg (90.6 kB)
                2315755156.jpg (59.5 kB)
                ... and 18719 other files
                train_images/
            train_tfrecords/
                ld_train00-1338.tfrec (227.2 MB)
                ld_train01-1338.tfrec (227.0 MB)
                ... and 12 other files
        input/
            description.md (124 lines)
            label_num_to_disease_map.json (1 lines)
            sample_submission.csv (2677 lines)
            sample_submission.csv.zip (13.4 kB)
            test.zip (160 Bytes)
            test_images.zip (319.5 MB)
            test_tfrecords.zip (451.9 MB)
            train.csv (18722 lines)
            train.csv.zip (100.0 kB)
            train.zip (162 Bytes)
            train_images.zip (2.2 GB)
            train_tfrecords.zip (3.2 GB)
            cassava-leaf-disease-classification/
                description.md (124 lines)
                label_num_to_disease_map.json (1 lines)
                ... and 10 other files
                cassava-leaf-disease-classification/
                test_images/
                    2574872277.jpg (183.5 kB)
                    1449210447.jpg (100.8 kB)
                    ... and 2674 other files
                    test_images/
                test_tfrecords/
                    ld_test00-1338.tfrec (225.9 MB)
                    ld_test01-1338.tfrec (226.2 MB)
                train_images/
                    478676678.jpg (90.6 kB)
                    2315755156.jpg (59.5 kB)
                    ... and 18719 other files
                    train_images/
                train_tfrecords/
                    ld_train00-1338.tfrec (227.2 MB)
                    ld_train01-1338.tfrec (227.0 MB)
                    ... and 12 other files
            test_images/
                2574872277.jpg (183.5 kB)
                1449210447.jpg (100.8 kB)
                ... and 2674 other files
                test_images/
                    2574872277.jpg (183.5 kB)
                    1449210447.jpg (100.8 kB)
                    ... and 2674 other files
                    test_images/
            test_tfrecords/
                ld_test00-1338.tfrec (225.9 MB)
                ld_test01-1338.tfrec (226.2 MB)
            train_images/
                478676678.jpg (90.6 kB)
                2315755156.jpg (59.5 kB)
                ... and 18719 other files
                train_images/
                    478676678.jpg (90.6 kB)
                    2315755156.jpg (59.5 kB)
                    ... and 18719 other files
                    train_images/
            train_tfrecords/
                ld_train00-1338.tfrec (227.2 MB)
                ld_train01-1338.tfrec (227.0 MB)
                ... and 12 other files
        working/
            cassava-leaf-disease-classification/
                description.md (124 lines)
                label_num_to_disease_map.json (1 lines)
                ... and 10 other files
                cassava-leaf-disease-classification/
                test_images/
                    2574872277.jpg (183.5 kB)
                    1449210447.jpg (100.8 kB)
                    ... and 2674 other files
                    test_images/
                test_tfrecords/
                    ld_test00-1338.tfrec (225.9 MB)
                    ld_test01-1338.tfrec (226.2 MB)
                train_images/
                    478676678.jpg (90.6 kB)
                    2315755156.jpg (59.5 kB)
                    ... and 18719 other files
                    train_images/
                train_tfrecords/
                    ld_train00-1338.tfrec (227.2 MB)
                    ld_train01-1338.tfrec (227.0 MB)
                    ... and 12 other files
```

-> data/cassava-leaf-disease-classification/label_num_to_disease_map.json has auto-generated json schema:
{
  "$schema": "http://json-schema.org/schema#",
  "type": "object",
  "properties": {
    "0": {
      "type": "string"
    },
    "1": {
      "type": "string"
    },
    "2": {
      "type": "string"
    },
    "3": {
      "type": "string"
    },
    "4": {
      "type": "string"
    }
  },
  "required": [
    "0",
    "1",
    "2",
    "3",
    "4"
  ]
}

-> data/cassava-leaf-disease-classification/sample_submission.csv has 2676 rows and 2 columns.
The columns are: image_id, label

-> data/cassava-leaf-disease-classification/train.csv has 18721 rows and 2 columns.
The columns are: image_id, label

-> data/label_num_to_disease_map.json has auto-generated json schema:
{
  "$schema": "http://json-schema.org/schema#",
  "type": "object",
  "properties": {
    "0": {
      "type": "string"
    },
    "1": {
      "type": "string"
    },
    "2": {
      "type": "string"
    },
    "3": {
      "type": "string"
    },
    "4": {
      "type": "string"
    }
  },
  "required": [
    "0",
    "1",
    "2",
    "3",
    "4"
  ]
}

-> data/sample_submission.csv has 2676 rows and 2 columns.
The columns are: image_id, label

-> data/train.csv has 18721 rows and 2 columns.
The columns are: image_id, label

-> input/cassava-leaf-disease-classification/label_num_to_disease_map.json has auto-generated json schema:
{
  "$schema": "http://json-schema.org/schema#",
  "type": "object",
  "properties": {
    "0": {
      "type": "string"
    },
    "1": {
      "type": "string"
    },
    "2": {
      "type": "string"
    },
    "3": {
      "type": "string"
    },
    "4": {
      "type": "string"
    }
  },
  "required": [
    "0",
    "1",
    "2",
    "3",
    "4"
  ]
}

-> (stopped after 10 files for performance)

# 5. Target score

0.8785131459655485

# 6. Current score

0.18946

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.63901) has done: 'I remove the expensive in-memory `.cache()` of decoded/resized images (it can exceed RAM and/or thrash, causing the observed timeout) and keep fast streaming with parallel `map` + `prefetch`. I also ensure the input pipeline is fully static-shaped and uses deterministic, parallel decoding/resize to maximize throughput without changing any pixels, labels, shuffling, model, loss, or training loop semantics. Finally, I add `steps_per_execution` to reduce Python overhead inside `model.fit` while keeping identical optimization steps and results (up to negligible FP differences). Paths, architecture, epochs, batch size, and split logic remain unchanged.'
- What this solution (achieved 0.63453) has done: 'I remove the two environment settings that force the slow pure-Python protobuf implementation (they significantly slow TFRecord/data pipeline + graph construction) and instead keep TensorFlow on its default fast C++ protobuf backend. I also add tf.data optimizations that are provably semantics-preserving: set a fixed `deterministic=True` on `map`, use `cache()` for validation/test (no reshuffle, so correctness is identical), and avoid expensive full-dataset shuffle buffer by capping it without changing the shuffle semantics materially (still uniform-ish, but to preserve semantics strictly we keep the full buffer). Finally, I ensure the input pipeline uses `num_parallel_calls=AUTOTUNE` everywhere and keep determinism and seeds intact.'
- What this solution (achieved 0.61099) has done: 'The crash happens before any training because TensorFlow/protobuf is incompatible in this Kaggle Python 2.7 environment, triggering the `MessageFactory.GetPrototype` AttributeError during `import tensorflow`. The minimal fix is to avoid using TensorFlow entirely and switch to a pure-Python baseline that still produces a valid `submission.csv` with the correct columns and ordering. To move accuracy up from ~0.63 toward the 0.878 target without introducing external dependencies, the safest approach is to use a stratified frequency-based prior estimated from `train.csv` and predict the most likely class for all test images (a common strong baseline for this dataset due to class imbalance). All file paths remain the same resolution logic, and the script always write `/kaggle/working/submission.csv`.'
- What this solution (achieved 0.13453) has done: 'Your current code always predicts the single majority class, which is a stable but low-accuracy baseline (~0.61). To move the accuracy upward toward the 0.8785 target without changing the “no-TensorFlow, pure-Python” core approach, I switch to a tiny image-based classifier that uses only the standard library + NumPy/Pandas: compute per-class average RGB color from a subset of training images, then predict each test image by nearest class mean in color space. This keeps runtime under the limit by sampling a fixed number of train images per class and using downscaled thumbnails, and it preserves submission format and file paths. If any image read fails, it falls back to the majority label to ensure a valid CSV is always produced.'
- What this solution (achieved 0.18946) has done: 'Your current score (0.13453) is far below the target (0.8785), and the main reason is that the “mean RGB” heuristic is too weak; the smallest legitimate way to push accuracy up without changing the overall non-TensorFlow, pure-Python approach is to compute a slightly richer but still cheap image feature. I keep the exact same workflow (compute per-class prototypes from sampled train images, then nearest-prototype prediction for test), but replace the 3D mean-RGB feature with a compact color-histogram feature (per-channel bins) which captures more discriminative information while staying fast. I also make sampling deterministic and ensure we read images from directories first (no zip scanning), keeping the same submission writing and paths. This should substantially increase accuracy toward the target while preserving the same training/prediction semantics (prototype-from-train, nearest-class at test).'

# 9. Code solution

## === cell 0
from __future__ import print_function

import os
import io
import zipfile
import numpy as np
import pandas as pd

np.random.seed(42)

print(
    "Python OK; running non-TensorFlow fallback (pure stdlib + numpy/pandas) to avoid TF/protobuf crash."
)




## === cell 1
def _resolve_data_dir():
    candidates = [
        "/kaggle/input/cassava-leaf-disease-classification",
        "/kaggle/data/cassava-leaf-disease-classification",
        "../input/cassava-leaf-disease-classification",
        "../data/cassava-leaf-disease-classification",
        "/kaggle/input",
        "/kaggle/data",
    ]
    for c in candidates[:4]:
        if os.path.exists(c) and os.path.exists(os.path.join(c, "train.csv")):
            return c
    for base in candidates[4:]:
        comp = os.path.join(base, "cassava-leaf-disease-classification")
        if os.path.exists(comp) and os.path.exists(os.path.join(comp, "train.csv")):
            return comp
    raise OSError(
        "Could not find cassava-leaf-disease-classification data directory in known locations."
    )


DATA_DIR = _resolve_data_dir()
train_csv_path = os.path.join(DATA_DIR, "train.csv")
sample_path = os.path.join(DATA_DIR, "sample_submission.csv")

train_df = pd.read_csv(train_csv_path)
sample_sub = pd.read_csv(sample_path)

print("DATA_DIR:", DATA_DIR)
print("Train rows:", len(train_df), "Test rows:", len(sample_sub))
print("Train label counts:\n", train_df["label"].value_counts().sort_index())




## === cell 2
def _find_images_dir(split):
    candidates = [
        os.path.join(DATA_DIR, split),
        os.path.join(DATA_DIR, "cassava-leaf-disease-classification", split),
        os.path.join("/kaggle/input/cassava-leaf-disease-classification", split),
        os.path.join("/kaggle/data/cassava-leaf-disease-classification", split),
    ]
    for p in candidates:
        if os.path.exists(p) and os.path.isdir(p):
            return p
    return None


def _find_images_zip(split):
    candidates = [
        os.path.join(DATA_DIR, split),
        os.path.join(DATA_DIR, "cassava-leaf-disease-classification", split),
        os.path.join("/kaggle/input/cassava-leaf-disease-classification", split),
        os.path.join("/kaggle/data/cassava-leaf-disease-classification", split),
        os.path.join(
            os.path.dirname(DATA_DIR), split
        ),  # e.g. /kaggle/input/test_images.zip
        os.path.join(
            os.path.dirname(DATA_DIR), "cassava-leaf-disease-classification", split
        ),
    ]
    for p in candidates:
        if os.path.exists(p) and os.path.isfile(p):
            return p
    return None


def _try_import_pil():
    try:
        from PIL import Image  # noqa: F401

        return True
    except Exception:
        return False


_HAS_PIL = _try_import_pil()
if _HAS_PIL:
    from PIL import Image


def _color_hist_feature_from_image(im_rgb, bins=16):
    arr = np.asarray(im_rgb, dtype=np.uint8).reshape(-1, 3)
    feat = []
    for c in range(3):
        h, _ = np.histogram(arr[:, c], bins=bins, range=(0, 256))
        feat.append(h.astype(np.float32))
    feat = np.concatenate(feat, axis=0)  # (3*bins,)
    s = float(feat.sum())
    if s > 0:
        feat /= s  # normalize to be robust to brightness/size; semantics still "prototype distance"
    return feat


def _feature_from_path(path, thumb=96, bins=16):
    if not _HAS_PIL:
        return None
    try:
        with Image.open(path) as im:
            im = im.convert("RGB")
            im = im.resize((thumb, thumb))
            return _color_hist_feature_from_image(im, bins=bins)
    except Exception:
        return None


def _feature_from_zip(zf, member, thumb=96, bins=16):
    if not _HAS_PIL:
        return None
    try:
        data = zf.read(member)
        with Image.open(io.BytesIO(data)) as im:
            im = im.convert("RGB")
            im = im.resize((thumb, thumb))
            return _color_hist_feature_from_image(im, bins=bins)
    except Exception:
        return None


label_counts = train_df["label"].value_counts()
majority_label = int(label_counts.idxmax())
print("Majority label:", majority_label)

train_images_dir = _find_images_dir("train_images")
test_images_dir = _find_images_dir("test_images")
train_images_zip = _find_images_zip("train_images.zip")
test_images_zip = _find_images_zip("test_images.zip")

print("PIL available:", _HAS_PIL)
print("train_images_dir:", train_images_dir)
print("test_images_dir:", test_images_dir)
print("train_images_zip:", train_images_zip)
print("test_images_zip:", test_images_zip)

if not _HAS_PIL:
    results_new = sample_sub[["image_id"]].copy()
    results_new["label"] = majority_label
    out_path = "/kaggle/working/submission.csv"
    results_new.to_csv(out_path, index=False)
    print(
        "PIL unavailable; wrote majority-label submission to:",
        out_path,
        "rows:",
        len(results_new),
    )
else:
    PER_CLASS_SAMPLES = 400  # modest increase to stabilize prototypes; still small (<=2000 images total)
    THUMB = 96
    BINS = 16

    class_means = {}
    labels = sorted(train_df["label"].unique().tolist())

    zf_train = None
    if train_images_dir is None and train_images_zip is not None:
        zf_train = zipfile.ZipFile(train_images_zip, "r")

    for lab in labels:
        df_lab = train_df[train_df["label"] == lab]
        n = min(PER_CLASS_SAMPLES, len(df_lab))
        df_s = df_lab.sample(n=n, random_state=42)
        feats = []
        for img_id in df_s["image_id"].values:
            v = None
            if train_images_dir is not None:
                p = os.path.join(train_images_dir, img_id)
                v = _feature_from_path(p, thumb=THUMB, bins=BINS)
            elif zf_train is not None:
                member = img_id
                if "train_images/" + img_id in zf_train.NameToInfo:
                    member = "train_images/" + img_id
                v = _feature_from_zip(zf_train, member, thumb=THUMB, bins=BINS)
            if v is not None:
                feats.append(v)
        if len(feats) == 0:
            class_means[lab] = np.ones((3 * BINS,), dtype=np.float32) / float(3 * BINS)
            print(
                "Warning: no readable images for label",
                lab,
                "- using uniform histogram.",
            )
        else:
            class_means[lab] = np.vstack(feats).mean(axis=0).astype(np.float32)
            s = float(class_means[lab].sum())
            if s > 0:
                class_means[lab] /= s
        print(
            "Label",
            lab,
            "prototype feature len:",
            len(class_means[lab]),
            "from",
            len(feats),
            "images",
        )

    if zf_train is not None:
        zf_train.close()

    zf_test = None
    if test_images_dir is None and test_images_zip is not None:
        zf_test = zipfile.ZipFile(test_images_zip, "r")

    means_mat = np.vstack([class_means[lab] for lab in labels])  # (C,F)

    pred_labels = []
    unreadable = 0
    for img_id in sample_sub["image_id"].values:
        v = None
        if test_images_dir is not None:
            p = os.path.join(test_images_dir, img_id)
            v = _feature_from_path(p, thumb=THUMB, bins=BINS)
        elif zf_test is not None:
            member = img_id
            if "test_images/" + img_id in zf_test.NameToInfo:
                member = "test_images/" + img_id
            v = _feature_from_zip(zf_test, member, thumb=THUMB, bins=BINS)

        if v is None:
            pred_labels.append(majority_label)
            unreadable += 1
            continue

        d = ((means_mat - v.reshape(1, -1)) ** 2).sum(axis=1)
        pred_labels.append(int(labels[int(np.argmin(d))]))

    if zf_test is not None:
        zf_test.close()

    results_new = sample_sub[["image_id"]].copy()
    results_new["label"] = pred_labels

    out_path = "/kaggle/working/submission.csv"
    results_new.to_csv(out_path, index=False)

    print("Wrote:", out_path, "rows:", len(results_new))
    print("Unreadable test images (fell back to majority):", unreadable)
    print(results_new.head())
    print("Submission columns:", list(results_new.columns))
    print("Label value counts:\n", results_new["label"].value_counts().sort_index())
