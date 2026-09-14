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

0.8819885161680266

# 6. Current score

0.28849

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.13752) has done: 'I fix the environment/runtime issues that currently prevent any submission from being written: (1) remove the failing TensorFlow/Keras import in the first cell that triggers the protobuf `MessageFactory.GetPrototype` error, and (2) replace the missing external SavedModel dependencies with a small, in-notebook baseline CNN that keeps the same “predict class label per image_id” semantics and produces a valid `submission.csv`. I also fix Python-2.7 incompatibilities by avoiding Keras/TensorFlow code paths and using only stdlib + numpy/pandas with direct JPEG reading via PIL. Finally, I ensure the submission rows align exactly to `sample_submission.csv` ordering and write `/kaggle/working/submission.csv` with the required columns.'
- What this solution (achieved 0.28849) has done: 'I speed up the two dominant bottlenecks: (1) iterating `train_df` with `iterrows()` and repeatedly doing expensive Python/Pandas work, and (2) slow JPEG decode/resize + feature extraction done thousands of times in pure Python loops. The refactor keeps the exact same feature definitions and nearest-prototype classification, but makes execution faster by using `itertuples()` + pre-built arrays for labels/paths, converting computations to float32 where it is algebraically identical, reducing intermediate allocations in `_image_feature`, and adding safe on-disk caching of extracted features (so repeated runs or notebook re-exec don’t redo work). I also avoid repeatedly listing directories and remove unused crop-related generators from hot paths (left intact but not invoked). No sampling, no early stopping, no changes to algorithmic semantics.'

# 9. Code solution

## === cell 0
import os
import sys
import numpy as np
import pandas as pd

try:
    from PIL import Image
except Exception as e:
    raise RuntimeError(
        "PIL (Pillow) is required to read JPEGs in this fallback pipeline: %r" % e
    )

RNG = np.random.RandomState(123)

try:
    Image.MAX_IMAGE_PIXELS = None
except Exception:
    pass




## === cell 1
def for_loop(num_iters, body, initial_args):
    for i in range(num_iters):
        if i == 0:
            outputs = body(*initial_args)
        else:
            outputs = body(*outputs)
    return outputs




## === cell 2
def acc_gambler(y_true, y_pred):
    raise NotImplementedError(
        "TensorFlow-dependent function not used in this execution path."
    )


def loss_gambler(label_smoothing=0.0):
    def loss_gamb(y_true, y_pred):
        raise NotImplementedError(
            "TensorFlow-dependent function not used in this execution path."
        )

    return loss_gamb




## === cell 3
def random_crop(img, random_crop_size):
    assert img.shape[2] == 3
    height, width = img.shape[0], img.shape[1]
    dy, dx = random_crop_size
    if width < dx or height < dy:
        return img
    x = RNG.randint(0, width - dx + 1)
    y = RNG.randint(0, height - dy + 1)
    return img[y : (y + dy), x : (x + dx), :]


def crop_generator(batches, crop_length):
    while True:
        batch_x = next(batches)
        batch_crops = np.zeros(
            (batch_x.shape[0], crop_length, crop_length, 3), dtype=np.float32
        )
        for i in range(batch_x.shape[0]):
            batch_crops[i] = random_crop(batch_x[i], (crop_length, crop_length))
        yield batch_crops




## === cell 4
sample_path = "/kaggle/input/cassava-leaf-disease-classification/sample_submission.csv"
train_csv_path = "/kaggle/input/cassava-leaf-disease-classification/train.csv"
train_dir = "/kaggle/input/cassava-leaf-disease-classification/train_images"
test_dir = "/kaggle/input/cassava-leaf-disease-classification/test_images"

if not os.path.exists(sample_path):
    alt = "/kaggle/data/cassava-leaf-disease-classification/sample_submission.csv"
    if os.path.exists(alt):
        sample_path = alt

if not os.path.exists(train_csv_path):
    alt = "/kaggle/data/cassava-leaf-disease-classification/train.csv"
    if os.path.exists(alt):
        train_csv_path = alt

if not os.path.isdir(test_dir):
    alt = "/kaggle/data/cassava-leaf-disease-classification/test_images"
    if os.path.isdir(alt):
        test_dir = alt

if not os.path.isdir(train_dir):
    alt = "/kaggle/data/cassava-leaf-disease-classification/train_images"
    if os.path.isdir(alt):
        train_dir = alt

sub_df = pd.read_csv(sample_path)
test_df = sub_df[["image_id"]].copy()
train_df = pd.read_csv(train_csv_path)

test_dir_ok = os.path.isdir(test_dir)
print("Loaded:")
print(" sample_submission:", sub_df.shape, "cols:", list(sub_df.columns))
print(" train.csv:", train_df.shape, "cols:", list(train_df.columns))
print(" test images dir exists:", test_dir_ok)




## === cell 5


def _load_image_rgb(path, size=(224, 224)):
    im = Image.open(path).convert("RGB")
    if size is not None:
        im = im.resize(size, Image.BILINEAR)
    arr = np.asarray(im, dtype=np.float32) * (1.0 / 255.0)
    return arr


def _rgb_to_hsv_np(rgb):
    r = rgb[..., 0]
    g = rgb[..., 1]
    b = rgb[..., 2]
    cmax = np.maximum(np.maximum(r, g), b)
    cmin = np.minimum(np.minimum(r, g), b)
    delta = cmax - cmin

    h = np.zeros_like(cmax, dtype=np.float32)
    mask = delta > 1e-8

    inv = 1.0 / (delta + 1e-12)
    dr = ((g - b) * inv) % 6.0
    dg = ((b - r) * inv) + 2.0
    db = ((r - g) * inv) + 4.0

    r_is_max = (cmax == r) & mask
    g_is_max = (cmax == g) & mask
    b_is_max = (cmax == b) & mask

    h[r_is_max] = dr[r_is_max]
    h[g_is_max] = dg[g_is_max]
    h[b_is_max] = db[b_is_max]
    h = h * (1.0 / 6.0)  # to [0,1)

    s = np.zeros_like(cmax, dtype=np.float32)
    cmask = cmax > 1e-8
    s[cmask] = delta[cmask] / (cmax[cmask] + 1e-12)

    v = cmax.astype(np.float32)
    hsv = np.stack([h, s, v], axis=-1)
    return hsv


def _edge_mag(gray):
    gx = np.zeros_like(gray, dtype=np.float32)
    gy = np.zeros_like(gray, dtype=np.float32)
    gx[:, 1:] = gray[:, 1:] - gray[:, :-1]
    gy[1:, :] = gray[1:, :] - gray[:-1, :]
    mag = np.sqrt(gx * gx + gy * gy)
    return mag


def _image_feature(arr):
    x = arr.reshape(-1, 3)
    rgb_mean = x.mean(axis=0)
    rgb_std = x.std(axis=0)

    hsv = _rgb_to_hsv_np(arr).reshape(-1, 3)
    hsv_mean = hsv.mean(axis=0)
    hsv_std = hsv.std(axis=0)

    gray = (0.2989 * arr[..., 0] + 0.5870 * arr[..., 1] + 0.1140 * arr[..., 2]).astype(
        np.float32
    )
    mag = _edge_mag(gray).reshape(-1)
    mag_mean = np.array([mag.mean()], dtype=np.float32)
    mag_std = np.array([mag.std()], dtype=np.float32)

    return np.concatenate(
        [rgb_mean, rgb_std, hsv_mean, hsv_std, mag_mean, mag_std], axis=0
    )


CACHE_DIR = "/kaggle/working/_feat_cache"
if not os.path.isdir(CACHE_DIR):
    try:
        os.makedirs(CACHE_DIR)
    except Exception:
        pass


def _feat_cache_path(split_name, image_id, size_hw):
    h, w = int(size_hw[0]), int(size_hw[1])
    base = "%s_%s_%dx%d.npy" % (split_name, str(image_id), h, w)
    return os.path.join(CACHE_DIR, base)


def _load_or_compute_feat(split_name, img_path, image_id, size_hw):
    cpath = _feat_cache_path(split_name, image_id, size_hw)
    try:
        if os.path.exists(cpath):
            feat = np.load(cpath)
            return feat
    except Exception:
        pass
    feat = _image_feature(_load_image_rgb(img_path, size=size_hw))
    try:
        np.save(cpath, feat)
    except Exception:
        pass
    return feat


num_classes = int(train_df["label"].max()) + 1
max_per_class = 2000
feat_dim = 14

protos_sum = np.zeros((num_classes, feat_dim), dtype=np.float64)
protos_sum_sq = np.zeros((num_classes, feat_dim), dtype=np.float64)
protos_count = np.zeros((num_classes,), dtype=np.int64)

train_df_sorted = train_df.sort_values("image_id", kind="mergesort").reset_index(
    drop=True
)

_join = os.path.join
_exists = os.path.exists

IMG_SIZE = (192, 192)

for row in train_df_sorted.itertuples(index=False):
    image_id = row.image_id
    y = int(row.label)
    if protos_count[y] >= max_per_class:
        continue
    img_path = _join(train_dir, str(image_id))
    if not _exists(img_path):
        continue
    try:
        feat = _load_or_compute_feat("train", img_path, image_id, IMG_SIZE).astype(
            np.float64, copy=False
        )
        if feat.shape[0] != feat_dim:
            continue
    except Exception:
        continue
    protos_sum[y] += feat
    protos_sum_sq[y] += feat * feat
    protos_count[y] += 1

class_means = np.zeros((num_classes, feat_dim), dtype=np.float64)
class_stds = np.ones((num_classes, feat_dim), dtype=np.float64)

seen = int(protos_count.sum())
if seen > 0:
    global_mu = protos_sum.sum(axis=0) / float(seen)
    global_ex2 = protos_sum_sq.sum(axis=0) / float(seen)
    global_var = np.maximum(global_ex2 - global_mu * global_mu, 1e-8)
    global_sd = np.sqrt(global_var)
else:
    global_mu = np.zeros((feat_dim,), dtype=np.float64)
    global_sd = np.ones((feat_dim,), dtype=np.float64)

for c in range(num_classes):
    if protos_count[c] > 0:
        mu = protos_sum[c] / float(protos_count[c])
        ex2 = protos_sum_sq[c] / float(protos_count[c])
        var = np.maximum(ex2 - mu * mu, 1e-8)
        sd = np.sqrt(var)
        class_means[c] = mu
        class_stds[c] = sd
    else:
        class_means[c] = global_mu
        class_stds[c] = global_sd

global_scale = class_stds.mean(axis=0)
global_scale = np.maximum(global_scale, 1e-6)

print("Prototype counts per class:", protos_count.tolist())
print("Feature dim:", feat_dim)




## === cell 6
pred_labels = np.zeros((len(test_df),), dtype=np.int64)

_join = os.path.join
_exists = os.path.exists
IMG_SIZE = (192, 192)

cm = class_means  # local
gs = global_scale  # local

test_image_ids = test_df["image_id"].values

for i in range(len(test_image_ids)):
    image_id = test_image_ids[i]
    img_path = _join(test_dir, str(image_id))
    try:
        if not _exists(img_path):
            raise IOError("missing")
        feat = _load_or_compute_feat("test", img_path, image_id, IMG_SIZE).astype(
            np.float64, copy=False
        )
        z = (cm - feat[None, :]) / gs[None, :]
        d2 = (z * z).sum(axis=1)
        pred = int(np.argmin(d2))
    except Exception:
        pred = 0
    pred_labels[i] = pred

results_new = pd.DataFrame(
    {
        "image_id": test_df["image_id"].astype(str).values,
        "label": pred_labels.astype(int),
    }
)

results_new = results_new[["image_id", "label"]]

out_path = "/kaggle/working/submission.csv"
results_new.to_csv(out_path, index=False)

print("Wrote:", out_path)
print(results_new.head())
print("Rows:", len(results_new), "Cols:", list(results_new.columns))
assert out_path.endswith(".csv")
assert results_new.shape[0] == sub_df.shape[0]
assert list(results_new.columns) == ["image_id", "label"]
