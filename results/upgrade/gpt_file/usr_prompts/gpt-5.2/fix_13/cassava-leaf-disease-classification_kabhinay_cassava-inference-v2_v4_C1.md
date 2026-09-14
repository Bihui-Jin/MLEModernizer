# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


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

# 5. Code solution

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

try:
    Image.warnings.simplefilter("ignore", Image.DecompressionBombWarning)
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
try:
    from collections import OrderedDict
except Exception:
    OrderedDict = None

try:
    import multiprocessing as mp
except Exception:
    mp = None


def _pil_to_arr01_rgb(im):
    arr = np.asarray(im, dtype=np.float32)
    arr *= 1.0 / 255.0
    return arr


def _load_pil_rgb(path):
    im = Image.open(path)
    try:
        return im.convert("RGB")
    finally:
        try:
            im.close()
        except Exception:
            pass


def _resize_pil_fast(im, size_hw):
    th, tw = int(size_hw[0]), int(size_hw[1])
    w, h = im.size
    rf_w = w // tw if tw > 0 else 1
    rf_h = h // th if th > 0 else 1
    rf = rf_w if rf_w < rf_h else rf_h
    if rf >= 2:
        try:
            im = im.reduce(int(rf))
        except Exception:
            pass
    return im.resize((tw, th), Image.BILINEAR)


def _load_image_rgb(path, size=(224, 224)):
    im = _load_pil_rgb(path)
    if size is not None:
        im = _resize_pil_fast(im, (int(size[0]), int(size[1])))
    return _pil_to_arr01_rgb(im)


def _rgb_to_hsv_np(rgb):
    r = rgb[..., 0]
    g = rgb[..., 1]
    b = rgb[..., 2]

    cmax = np.maximum(r, np.maximum(g, b))
    cmin = np.minimum(r, np.minimum(g, b))
    delta = cmax - cmin

    h = np.zeros_like(cmax, dtype=np.float32)
    s = np.zeros_like(cmax, dtype=np.float32)

    mask = delta > 1e-8
    cmask = cmax > 1e-8

    inv_delta = 1.0 / (delta + 1e-12)

    hr = ((g - b) * inv_delta) % 6.0
    hg = ((b - r) * inv_delta) + 2.0
    hb = ((r - g) * inv_delta) + 4.0

    r_is_max = (cmax == r) & mask
    g_is_max = (cmax == g) & mask
    b_is_max = (cmax == b) & mask

    h[r_is_max] = hr[r_is_max]
    h[g_is_max] = hg[g_is_max]
    h[b_is_max] = hb[b_is_max]
    h *= 1.0 / 6.0

    s[cmask] = delta[cmask] / (cmax[cmask] + 1e-12)

    v = cmax.astype(np.float32, copy=False)
    return np.stack([h, s, v], axis=-1)


def _edge_mag(gray):
    mag2 = np.zeros_like(gray, dtype=np.float32)
    dx = gray[:, 1:] - gray[:, :-1]
    dy = gray[1:, :] - gray[:-1, :]
    mag2[:, 1:] += dx * dx
    mag2[1:, :] += dy * dy
    return np.sqrt(mag2, dtype=np.float32)


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


class _LRUCache(object):
    def __init__(self, maxsize=2048):
        self.maxsize = int(maxsize)
        self._d = OrderedDict() if OrderedDict is not None else None
        self._plain = {} if self._d is None else None

    def get(self, key):
        if self._d is None:
            return self._plain.get(key, None)
        v = self._d.get(key, None)
        if v is not None:
            try:
                self._d.move_to_end(key)
            except Exception:
                self._d.pop(key, None)
                self._d[key] = v
        return v

    def set(self, key, val):
        if self._d is None:
            self._plain[key] = val
            return
        if key in self._d:
            try:
                self._d.pop(key, None)
            except Exception:
                pass
        self._d[key] = val
        if len(self._d) > self.maxsize:
            try:
                self._d.popitem(last=False)
            except Exception:
                try:
                    k0 = next(iter(self._d))
                    self._d.pop(k0, None)
                except Exception:
                    pass


_FEAT_MEM_CACHE = _LRUCache(maxsize=4096)

IMG_SIZES = [(192, 192), (224, 224)]


def _load_or_compute_feat_two_scales(split_name, img_path, image_id, sizes_hw):
    key = (split_name, str(image_id), "two_scales")
    cached = _FEAT_MEM_CACHE.get(key)
    if cached is not None:
        return cached

    im0 = _load_pil_rgb(img_path)
    sizes_hw = list(sizes_hw)

    if not (
        len(sizes_hw) == 2 and sizes_hw[0] == (192, 192) and sizes_hw[1] == (224, 224)
    ):
        raise ValueError("Expected exactly two sizes: (192,192) and (224,224)")

    im_192 = _resize_pil_fast(im0, (192, 192))
    arr_192 = _pil_to_arr01_rgb(im_192)
    f192 = _image_feature(arr_192).astype(np.float64, copy=False)

    im_224 = _resize_pil_fast(im_192, (224, 224))
    arr_224 = _pil_to_arr01_rgb(im_224)
    f224 = _image_feature(arr_224).astype(np.float64, copy=False)

    _FEAT_MEM_CACHE.set(key, (f192, f224))
    return f192, f224


def _load_or_compute_feat_multiscale(
    split_name, img_path, image_id, sizes_hw, alpha=0.5
):
    f192, f224 = _load_or_compute_feat_two_scales(
        split_name, img_path, image_id, sizes_hw
    )
    return alpha * f192 + (1.0 - alpha) * f224


def _predict_nearest_prototype(feats, class_means, inv_var_diag):
    cm_w = class_means * inv_var_diag[None, :]  # (C,D)
    cm_term = (class_means * cm_w).sum(axis=1)  # (C,)
    x_term = np.dot(inv_var_diag, (feats * feats).T)  # (N,)
    cross = np.dot(cm_w, feats.T)  # (C,N)
    d2 = cm_term[:, None] + x_term[None, :] - 2.0 * cross
    return np.argmin(d2, axis=0).astype(np.int64)


def _worker_feat2(args):
    split_name, img_path, image_id = args
    try:
        f192, f224 = _load_or_compute_feat_two_scales(
            split_name, img_path, image_id, IMG_SIZES
        )
        return (image_id, f192, f224, True)
    except Exception:
        return (image_id, None, None, False)


def _mp_map_feat2(split_name, image_ids, img_dir, processes):
    join = os.path.join
    tasks = [(split_name, join(img_dir, str(iid)), str(iid)) for iid in image_ids]
    if mp is None or processes <= 1:
        return [_worker_feat2(t) for t in tasks]
    pool = mp.Pool(processes=processes)
    try:
        chunksize = max(1, len(tasks) // (processes * 8))
        return pool.map(_worker_feat2, tasks, chunksize)
    finally:
        try:
            pool.close()
        except Exception:
            pass
        try:
            pool.join()
        except Exception:
            pass




## === cell 6
num_classes = int(train_df["label"].max()) + 1
max_per_class = 2000
feat_dim = 14

val_frac = 0.10
val_seed = 123

train_df_sorted = train_df.sort_values("image_id", kind="mergesort").reset_index(
    drop=True
)
n_all = len(train_df_sorted)
perm = np.random.RandomState(val_seed).permutation(n_all)
n_val = int(round(n_all * val_frac))
val_mask = np.zeros((n_all,), dtype=np.bool_)
val_mask[perm[:n_val]] = True

train_idx = np.where(~val_mask)[0]
val_idx = np.where(val_mask)[0]

print(
    "Split sizes: train=%d val=%d (val_frac=%.3f)"
    % (len(train_idx), len(val_idx), val_frac)
)

_join = os.path.join
_exists = os.path.exists

protos_sum_192 = np.zeros((num_classes, feat_dim), dtype=np.float64)
protos_sum_224 = np.zeros((num_classes, feat_dim), dtype=np.float64)
protos_sum_sq_192 = np.zeros((num_classes, feat_dim), dtype=np.float64)
protos_sum_sq_224 = np.zeros((num_classes, feat_dim), dtype=np.float64)
protos_count = np.zeros((num_classes,), dtype=np.int64)

train_rows = train_df_sorted.iloc[train_idx][["image_id", "label"]]
train_image_ids_all = train_rows["image_id"].astype(str).values
train_labels_all = train_rows["label"].astype(np.int64).values

keep_mask = np.zeros((len(train_image_ids_all),), dtype=np.bool_)
tmp_counts = np.zeros((num_classes,), dtype=np.int64)
for i in range(len(train_image_ids_all)):
    y = int(train_labels_all[i])
    if tmp_counts[y] < max_per_class:
        keep_mask[i] = True
        tmp_counts[y] += 1

train_image_ids = train_image_ids_all[keep_mask]
train_labels = train_labels_all[keep_mask]

cpu_cnt = 1
try:
    cpu_cnt = mp.cpu_count() if mp is not None else 1
except Exception:
    cpu_cnt = 1
n_proc = int(min(max(cpu_cnt - 1, 1), 8))

mp_results = _mp_map_feat2("train", train_image_ids, train_dir, processes=n_proc)

id_to_label = dict(
    (str(iid), int(lbl)) for iid, lbl in zip(train_image_ids, train_labels)
)
for image_id, f192, f224, ok in mp_results:
    if not ok:
        continue
    y = id_to_label.get(str(image_id), None)
    if y is None:
        continue
    if f192.shape[0] != feat_dim or f224.shape[0] != feat_dim:
        continue
    protos_sum_192[y] += f192
    protos_sum_224[y] += f224
    protos_sum_sq_192[y] += f192 * f192
    protos_sum_sq_224[y] += f224 * f224
    protos_count[y] += 1

class_mean_192 = np.zeros((num_classes, feat_dim), dtype=np.float64)
class_mean_224 = np.zeros((num_classes, feat_dim), dtype=np.float64)

seen = int(protos_count.sum())
if seen > 0:
    global_mu_192 = protos_sum_192.sum(axis=0) / float(seen)
    global_mu_224 = protos_sum_224.sum(axis=0) / float(seen)
else:
    global_mu_192 = np.zeros((feat_dim,), dtype=np.float64)
    global_mu_224 = np.zeros((feat_dim,), dtype=np.float64)

for c in range(num_classes):
    if protos_count[c] > 0:
        class_mean_192[c] = protos_sum_192[c] / float(protos_count[c])
        class_mean_224[c] = protos_sum_224[c] / float(protos_count[c])
    else:
        class_mean_192[c] = global_mu_192
        class_mean_224[c] = global_mu_224

shrink_k = 200.0
for c in range(num_classes):
    n = float(protos_count[c])
    if n > 0.0:
        a = n / (n + shrink_k)
        class_mean_192[c] = a * class_mean_192[c] + (1.0 - a) * global_mu_192
        class_mean_224[c] = a * class_mean_224[c] + (1.0 - a) * global_mu_224

if seen > 0:
    ex2_192 = protos_sum_sq_192.sum(axis=0) / float(seen)
    ex2_224 = protos_sum_sq_224.sum(axis=0) / float(seen)
    var_192 = np.maximum(ex2_192 - global_mu_192 * global_mu_192, 1e-8)
    var_224 = np.maximum(ex2_224 - global_mu_224 * global_mu_224, 1e-8)
    pooled_var = 0.5 * (var_192 + var_224)
else:
    pooled_var = np.ones((feat_dim,), dtype=np.float64)

inv_var_diag = 1.0 / (pooled_var + 1e-6)

print("Prototype counts per class:", protos_count.tolist())
print("Feature dim:", feat_dim)

alpha_grid = np.array([0.2, 0.35, 0.5, 0.65, 0.8], dtype=np.float64)

val_rows = train_df_sorted.iloc[val_idx]
val_image_ids = val_rows["image_id"].astype(str).values
val_labels = val_rows["label"].astype(np.int64).values
n_val_eff = len(val_image_ids)

val_f192 = np.zeros((n_val_eff, feat_dim), dtype=np.float64)
val_f224 = np.zeros((n_val_eff, feat_dim), dtype=np.float64)
val_ok = np.ones((n_val_eff,), dtype=np.bool_)

val_res = _mp_map_feat2("val", val_image_ids, train_dir, processes=n_proc)
val_index = dict((str(iid), i) for i, iid in enumerate(val_image_ids))
for image_id, f192, f224, ok in val_res:
    i = val_index.get(str(image_id), None)
    if i is None:
        continue
    if not ok:
        val_ok[i] = False
        val_f192[i] = 0.0
        val_f224[i] = 0.0
        continue
    val_f192[i] = f192
    val_f224[i] = f224

best_alpha = 0.5
best_acc = -1.0

for a in alpha_grid:
    feats_a = a * val_f192 + (1.0 - a) * val_f224
    pred = _predict_nearest_prototype(
        feats_a, (a * class_mean_192 + (1.0 - a) * class_mean_224), inv_var_diag
    )
    if val_ok.any():
        acc = float((pred[val_ok] == val_labels[val_ok]).mean())
    else:
        acc = 0.0
    if acc > best_acc:
        best_acc = acc
        best_alpha = float(a)

print(
    "Chosen alpha=%.3f from grid %s; val_acc=%.4f (on %d valid/%d)"
    % (best_alpha, alpha_grid.tolist(), best_acc, int(val_ok.sum()), int(len(val_ok)))
)

class_means = best_alpha * class_mean_192 + (1.0 - best_alpha) * class_mean_224



## === cell 7
pred_labels = np.zeros((len(test_df),), dtype=np.int64)

_join = os.path.join
_exists = os.path.exists

cm = class_means

test_image_ids = test_df["image_id"].astype(str).values
n_test = len(test_image_ids)
feat_dim = cm.shape[1]

test_feats = np.zeros((n_test, feat_dim), dtype=np.float64)
valid = np.ones((n_test,), dtype=np.bool_)

test_res = _mp_map_feat2("test", test_image_ids, test_dir, processes=n_proc)
test_index = dict((str(iid), i) for i, iid in enumerate(test_image_ids))

for image_id, f192, f224, ok in test_res:
    i = test_index.get(str(image_id), None)
    if i is None:
        continue
    if not ok:
        valid[i] = False
        test_feats[i] = 0.0
        continue
    feat = best_alpha * f192 + (1.0 - best_alpha) * f224
    if feat.shape[0] != feat_dim:
        valid[i] = False
        test_feats[i] = 0.0
        continue
    test_feats[i] = feat

pred_all = _predict_nearest_prototype(test_feats, cm, inv_var_diag)
pred_all[~valid] = 0
pred_labels[:] = pred_all

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
