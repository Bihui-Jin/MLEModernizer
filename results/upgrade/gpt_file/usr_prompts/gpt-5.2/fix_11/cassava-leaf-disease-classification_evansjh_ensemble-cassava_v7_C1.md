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

3.13

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
import glob
import warnings
from collections import Counter

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
np.random.seed(42)

test_image_dir = "/kaggle/input/cassava-leaf-disease-classification/test_images"
train_image_dir = "/kaggle/input/cassava-leaf-disease-classification/train_images"
sample_path = "/kaggle/input/cassava-leaf-disease-classification/sample_submission.csv"
train_csv_path = "/kaggle/input/cassava-leaf-disease-classification/train.csv"

sample_csv = pd.read_csv(sample_path)
train_df = pd.read_csv(train_csv_path)

print("sample_csv shape:", sample_csv.shape)
print("train_df shape:", train_df.shape)
print("test_image_dir exists:", os.path.isdir(test_image_dir))
print("train_image_dir exists:", os.path.isdir(train_image_dir))



## === cell 1
model_path_1 = (
    "/kaggle/input/combinedmodel3/tensorflow2/default/1/BestModel_3454_8937.h5"
)
model_path_2 = (
    "/kaggle/input/combinedmodel3/tensorflow2/default/1/best_model_0.37458707.h5"
)
model_path_3 = (
    "/kaggle/input/combinedmodel3/tensorflow2/default/1/googlenet_inceptionv3.h5"
)
model_path_4 = (
    "/kaggle/input/bestmodel_550_2/tensorflow2/default/1/BestModel_3577_8940.h5"
)
model_path_5 = (
    "/kaggle/input/bestmodel_8878/tensorflow2/default/1/BestModel_8878_0358.h5"
)
model_path_6 = "/kaggle/input/bestmodel_8875/tensorflow2/default/1/BestModel_8875.h5"

models_info = [
    (model_path_1, (550, 550)),
    (model_path_2, (512, 512)),
    (model_path_3, (448, 448)),
    (model_path_6, (512, 512)),
]



## === cell 2
class_labels = {
    0: "Cassava Bacterial Blight (CBB)",
    1: "Cassava Brown Streak Disease (CBSD)",
    2: "Cassava Green Mottle (CGM)",
    3: "Cassava Mosaic Disease (CMD)",
    4: "Healthy",
}

import tensorflow as tf
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

tf.random.set_seed(42)
try:
    tf.config.experimental.enable_op_determinism(True)
except Exception:
    pass

try:
    _CPU = max(1, (os.cpu_count() or 2))
    tf.config.threading.set_intra_op_parallelism_threads(_CPU)
    tf.config.threading.set_inter_op_parallelism_threads(2)
except Exception:
    pass

_TFDATA_OPTIONS = tf.data.Options()
_TFDATA_OPTIONS.deterministic = True
try:
    _TFDATA_OPTIONS.experimental_optimization.map_parallelization = True
    _TFDATA_OPTIONS.experimental_optimization.parallel_batch = True
    _TFDATA_OPTIONS.experimental_optimization.autotune_buffers = True
    _TFDATA_OPTIONS.experimental_optimization.apply_default_optimizations = True
except Exception:
    pass


@tf.function(reduce_retracing=True)
def _tf_decode_resize_flatten(image_bytes, height, width):
    img = tf.io.decode_jpeg(image_bytes, channels=3)
    img = tf.image.resize(
        img, [height, width], method=tf.image.ResizeMethod.BILINEAR, antialias=False
    )
    img = tf.cast(img, tf.float32) / 255.0
    return tf.reshape(img, [-1])


@tf.function(reduce_retracing=True)
def _tf_read_decode_resize_flatten(path, height, width):
    b = tf.io.read_file(path)
    return _tf_decode_resize_flatten(b, height, width)


def _features_memmap_paths(cache_prefix, size, n, dim, tag=""):
    w, h = int(size[0]), int(size[1])
    base = f"/kaggle/working/{cache_prefix}{tag}_{w}x{h}_n{n}_d{dim}"
    mm_path = base + "_X.dat"
    ok_path = base + "_ok.npy"
    return mm_path, ok_path


def _build_feature_matrix_memmap(paths, size, cache_prefix, batch_size=512, tag=""):
    """
    CHANGED (timeout fix):
      - Increase batch_size (default 512) to reduce Python loop overhead and improve CPU throughput.
      - Use a single tf.data pipeline with parallel map + prefetch + deterministic options.
      - Persist features to memmap so repeated uses are O(1) loads (no re-decode).
    Correctness preserved:
      - Feature transform is identical: decode_jpeg -> resize -> /255 -> flatten.
      - Ordering is stable because we write by explicit indices and keep deterministic=True.
    """
    h, w = int(size[1]), int(size[0])
    dim = h * w * 3
    n = len(paths)
    mm_path, ok_path = _features_memmap_paths(cache_prefix, size, n, dim, tag=tag)

    if os.path.exists(mm_path) and os.path.exists(ok_path):
        Xmm = np.memmap(mm_path, mode="r", dtype=np.float32, shape=(n, dim))
        ok = np.load(ok_path)
        if ok.shape == (n,) and ok.dtype == np.bool_:
            return Xmm, ok

    Xmm = np.memmap(mm_path, mode="w+", dtype=np.float32, shape=(n, dim))
    ok = np.zeros((n,), dtype=np.bool_)

    idx = np.arange(n, dtype=np.int32)
    paths_arr = np.asarray(paths, dtype=np.str_)

    ds = tf.data.Dataset.from_tensor_slices((idx, paths_arr)).with_options(
        _TFDATA_OPTIONS
    )

    def _map_fn(i, p):
        x = _tf_read_decode_resize_flatten(p, h, w)
        return i, x

    ds = ds.map(_map_fn, num_parallel_calls=tf.data.AUTOTUNE, deterministic=True)
    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(tf.data.AUTOTUNE)

    for ib, xb in ds:
        ib_np = ib.numpy()
        xb_np = xb.numpy()  # float32 (B, dim)
        Xmm[ib_np, :] = xb_np
        ok[ib_np] = True

    Xmm.flush()
    np.save(ok_path, ok)
    Xmm = np.memmap(mm_path, mode="r", dtype=np.float32, shape=(n, dim))
    return Xmm, ok


def train_sklearn_ensemble(
    train_df, train_image_dir, sizes, per_size_max_samples=5000, random_state=42
):
    """
    CHANGED (timeout fix):
      - Build a single set of available filenames once (O(N)) instead of repeated disk checks.
      - Cache decoded features per (size, n) via memmap.
      - Reuse the same fitted sklearn model for duplicate sizes (exactly equivalent).
    Correctness preserved:
      - Same stratified subsample (train_test_split with stratify) and same LogisticRegression settings.
      - Duplicate-size reuse is mathematically identical because the training data/features are identical.
    """
    loaded_models = []

    df = train_df.copy()
    available_train = (
        set(os.listdir(train_image_dir)) if os.path.isdir(train_image_dir) else set()
    )
    df = df.loc[df["image_id"].isin(available_train)].reset_index(drop=True)

    if len(df) == 0:
        return loaded_models

    df["path"] = (train_image_dir.rstrip("/") + "/") + df["image_id"].astype(str)

    size_to_model = {}

    for size in sizes:
        if size in size_to_model:
            loaded_models.append(
                (size_to_model[size], size, f"sklearn_logreg_{size[0]}x{size[1]}")
            )
            continue

        if per_size_max_samples is not None and len(df) > per_size_max_samples:
            df_sample, _ = train_test_split(
                df,
                train_size=per_size_max_samples,
                random_state=random_state,
                stratify=df["label"],
            )
        else:
            df_sample = df

        sample_paths = df_sample["path"].to_list()
        y_all = df_sample["label"].to_numpy(dtype=np.int64, copy=False)

        Xmm, ok = _build_feature_matrix_memmap(
            sample_paths,
            size,
            cache_prefix="train",
            batch_size=512,
            tag=f"_s{len(sample_paths)}",
        )
        if ok.sum() < 50:
            continue

        X = Xmm[ok]
        y = y_all[ok]

        clf = LogisticRegression(
            multi_class="multinomial",
            solver="lbfgs",
            max_iter=200,
            n_jobs=1,
            random_state=random_state,
        )
        clf.fit(X, y)

        size_to_model[size] = clf
        loaded_models.append((clf, size, f"sklearn_logreg_{size[0]}x{size[1]}"))
        print(
            f"Trained sklearn model for size={size} on n={len(y)} samples, dim={X.shape[1]}"
        )

    return loaded_models


def predict_one_image_ensemble(image_path, models):
    """
    Preserved for API compatibility; main path below uses batched inference for speed.
    """
    model_predictions = []
    confidence_scores = {}

    for model, input_size, _path in models:
        try:
            b = tf.io.read_file(image_path)
            flat = _tf_decode_resize_flatten(b, int(input_size[1]), int(input_size[0]))
            feat = flat.numpy()
        except Exception:
            continue

        proba = model.predict_proba(feat.reshape(1, -1))[0]
        predicted_class = int(np.argmax(proba))
        confidence_score = float(proba[predicted_class])

        model_predictions.append(predicted_class)
        confidence_scores.setdefault(predicted_class, []).append(confidence_score)

    if len(model_predictions) == 0:
        return None

    class_votes = Counter(model_predictions)
    most_common = class_votes.most_common()
    final_predicted_class = most_common[0][0]

    if len(most_common) > 1 and most_common[0][1] == most_common[1][1]:
        tied_classes = [cls for cls, count in most_common if count == most_common[0][1]]
        final_predicted_class = max(
            tied_classes,
            key=lambda cls: sum(confidence_scores.get(cls, [0.0]))
            / max(1, len(confidence_scores.get(cls, []))),
        )

    return final_predicted_class


sizes = [info[1] for info in models_info]
loaded_models = train_sklearn_ensemble(
    train_df=train_df,
    train_image_dir=train_image_dir,
    sizes=sizes,
    per_size_max_samples=5000,  # unchanged
    random_state=42,
)

print(f"Loaded {len(loaded_models)} models.")
for _, s, p in loaded_models:
    print(" model:", p, "size:", s)



## === cell 3
available_test_images = (
    set(os.listdir(test_image_dir)) if os.path.isdir(test_image_dir) else set()
)
majority_label = int(train_df["label"].value_counts().idxmax())

if len(loaded_models) == 0:
    print(
        "No models trained; writing fallback submission with majority label =",
        majority_label,
    )
    submission_df = sample_csv.copy()
    submission_df["label"] = majority_label
else:
    test_ids = sample_csv["image_id"].to_list()
    test_paths = [(test_image_dir.rstrip("/") + "/") + iid for iid in test_ids]

    present = np.fromiter(
        (iid in available_test_images for iid in test_ids),
        dtype=np.bool_,
        count=len(test_ids),
    )
    n_test = len(test_ids)

    unique_sizes = []
    seen = set()
    for _m, s, _n in loaded_models:
        if s not in seen:
            unique_sizes.append(s)
            seen.add(s)

    size_to_test_features = {}
    size_to_ok = {}

    for size in unique_sizes:
        Xmm, ok = _build_feature_matrix_memmap(
            test_paths, size, cache_prefix="test", batch_size=512, tag=""
        )
        size_to_test_features[size] = Xmm
        size_to_ok[size] = ok

    def _predict_proba_chunked(model, X, chunk=2048):
        n = X.shape[0]
        out = []
        for i in range(0, n, chunk):
            out.append(model.predict_proba(X[i : i + chunk]))
        return np.vstack(out) if len(out) > 1 else out[0]

    per_model_pred = []
    per_model_conf = []

    for model, size, _name in loaded_models:
        Xmm = size_to_test_features[size]
        ok = size_to_ok[size]
        ok2 = ok & present

        preds = np.full((n_test,), -1, dtype=np.int16)
        confs = np.zeros((n_test,), dtype=np.float32)

        if ok2.any():
            Xsel = Xmm[ok2]
            proba = _predict_proba_chunked(model, Xsel, chunk=2048)
            pcls = np.argmax(proba, axis=1).astype(np.int16, copy=False)
            pconf = proba[np.arange(proba.shape[0]), pcls].astype(
                np.float32, copy=False
            )
            idx = np.flatnonzero(ok2)
            preds[idx] = pcls
            confs[idx] = pconf

        per_model_pred.append(preds)
        per_model_conf.append(confs)

    per_model_pred = np.stack(per_model_pred, axis=0)  # (m, n)
    per_model_conf = np.stack(per_model_conf, axis=0)  # (m, n)

    final_labels = np.full((n_test,), majority_label, dtype=np.int64)

    good_cols = present & np.any(per_model_pred >= 0, axis=0)
    good_idx = np.flatnonzero(good_cols)

    if good_idx.size:
        preds_good = per_model_pred[:, good_idx]  # (m, k)

        counts = np.zeros((5, preds_good.shape[1]), dtype=np.int16)
        for c in range(5):
            counts[c] = np.sum(preds_good == c, axis=0, dtype=np.int16)

        maxc = counts.max(axis=0)
        tied_mask = counts == maxc[None, :]
        n_tied = tied_mask.sum(axis=0)

        non_tie = n_tied == 1
        if np.any(non_tie):
            cls_nt = np.argmax(counts[:, non_tie], axis=0)
            final_labels[good_idx[non_tie]] = cls_nt.astype(np.int64, copy=False)

        tie = ~non_tie
        if np.any(tie):
            avg_conf = np.zeros((5, int(np.sum(tie))), dtype=np.float32)
            conf_t = per_model_conf[:, good_idx[tie]]  # (m, t)
            preds_t = preds_good[:, tie]  # (m, t)

            for c in range(5):
                mask = preds_t == c
                denom = mask.sum(axis=0)
                num = np.sum(conf_t * mask, axis=0)
                avg = np.where(denom > 0, num / denom, 0.0)
                avg_conf[c] = avg

            avg_conf = np.where(tied_mask[:, tie], avg_conf, -1.0)
            cls_t = np.argmax(avg_conf, axis=0)
            final_labels[good_idx[tie]] = cls_t.astype(np.int64, copy=False)

    missing = int((~present).sum())
    bad_pred = int((present & ~np.any(per_model_pred >= 0, axis=0)).sum())

    if missing:
        print(
            f"Warning: {missing} images from sample_submission were not found in test_image_dir."
        )
    if bad_pred:
        print(
            f"Warning: {bad_pred} images failed feature extraction; used majority fallback for those."
        )

    submission_df = pd.DataFrame(
        {"image_id": test_ids, "label": final_labels.astype(int)}
    )

submission_df = submission_df[["image_id", "label"]]
submission_df = sample_csv[["image_id"]].merge(submission_df, on="image_id", how="left")
submission_df["label"] = submission_df["label"].fillna(majority_label).astype(int)

submission_path = "/kaggle/working/submission.csv"
submission_df.to_csv(submission_path, index=False)

print("Wrote:", submission_path)
print(submission_df.head())
print(
    "Rows:", len(submission_df), " Unique images:", submission_df["image_id"].nunique()
)



## === cell 4
submission_df
