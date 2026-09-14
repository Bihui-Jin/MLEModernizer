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

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import pandas as pd
import numpy as np

from PIL import Image

from concurrent.futures import ThreadPoolExecutor

TF_AVAILABLE = False
tf = None
load_model = None

print("TF_AVAILABLE:", TF_AVAILABLE)



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

test_image_dir = "/kaggle/input/cassava-leaf-disease-classification/test_images"
train_image_dir = "/kaggle/input/cassava-leaf-disease-classification/train_images"
sample = "/kaggle/input/cassava-leaf-disease-classification/sample_submission.csv"
train_csv_path = "/kaggle/input/cassava-leaf-disease-classification/train.csv"



## === cell 2
sample_csv = pd.read_csv(sample)
print(sample_csv.head())
print("sample_csv shape:", sample_csv.shape)

train_df = pd.read_csv(train_csv_path)
print(train_df.head())
print("train_df shape:", train_df.shape)

majority_label = int(train_df["label"].value_counts().idxmax())
label_dist = train_df["label"].value_counts().sort_index()
print("Train label distribution:\n", label_dist.to_string())
print("Majority label:", majority_label)




## === cell 3
def discover_models_under_kaggle_input(root="/kaggle/input", max_models=2):
    h5_paths = []
    savedmodel_dirs = []
    for dirpath, dirnames, filenames in os.walk(root):
        for fn in filenames:
            if fn.lower().endswith((".h5", ".hdf5")):
                h5_paths.append(os.path.join(dirpath, fn))
        if "saved_model.pb" in filenames:
            savedmodel_dirs.append(dirpath)

    h5_paths = sorted(h5_paths)
    savedmodel_dirs = sorted(savedmodel_dirs)

    candidates = h5_paths + savedmodel_dirs
    return candidates[:max_models], {
        "h5_found": len(h5_paths),
        "savedmodel_found": len(savedmodel_dirs),
    }


def try_load_model(path):
    try:
        m = load_model(path, compile=False)
        return m, None
    except Exception as e:
        return None, e


preferred_paths = [
    model_path_5,
    model_path_6,
    model_path_4,
    model_path_1,
    model_path_2,
    model_path_3,
]
existing_preferred = [p for p in preferred_paths if os.path.exists(p)]

if len(existing_preferred) < 2:
    discovered, stats = discover_models_under_kaggle_input(
        "/kaggle/input", max_models=10
    )
    print("Preferred existing:", existing_preferred)
    print("Discovered models stats:", stats)
    for p in discovered:
        if p not in existing_preferred:
            existing_preferred.append(p)

print("Model candidates (first 10):")
for p in existing_preferred[:10]:
    print(" -", p)




## === cell 4
def softmax(z):
    z = z - np.max(z, axis=1, keepdims=True)
    ez = np.exp(z)
    return ez / np.sum(ez, axis=1, keepdims=True)


def _compute_all_features(
    arr_rgb01: np.ndarray, *, size: int, block_grid: int
) -> np.ndarray:
    r = arr_rgb01[..., 0]
    g = arr_rgb01[..., 1]
    b = arr_rgb01[..., 2]

    gray = (0.2989 * r + 0.5870 * g + 0.1140 * b).astype(np.float32, copy=False)

    gx = np.abs(gray[:, 1:] - gray[:, :-1])
    gy = np.abs(gray[1:, :] - gray[:-1, :])
    edge_mean = float((gx.mean() + gy.mean()) * 0.5)
    edge_std = float((gx.std() + gy.std()) * 0.5)

    g_mean = float(gray.mean())
    g_std = float(gray.std())

    g_flat = gray.reshape(-1)
    g_q25 = float(np.quantile(g_flat, 0.25, method="linear"))
    g_q75 = float(np.quantile(g_flat, 0.75, method="linear"))

    r_mean = float(r.mean())
    gch_mean = float(g.mean())
    b_mean = float(b.mean())
    r_std = float(r.std())
    gch_std = float(g.std())
    b_std = float(b.std())

    s = r + g + b + 1e-6
    r_ch = r / s
    g_ch = g / s
    b_ch = b / s

    exg = 2.0 * g - r - b
    exr = 1.4 * r - g
    exb = 1.4 * b - g

    extra = np.array(
        [
            float(exg.mean()),
            float(exg.std()),
            float(exr.mean()),
            float(exr.std()),
            float(exb.mean()),
            float(exb.std()),
            float(r_ch.mean()),
            float(g_ch.mean()),
            float(b_ch.mean()),
            float(r_ch.std()),
            float(g_ch.std()),
            float(b_ch.std()),
        ],
        dtype=np.float32,
    )

    h, w, _ = arr_rgb01.shape
    gh = h // block_grid
    gw = w // block_grid
    if gh <= 0 or gw <= 0:
        blocks = np.zeros((3 * block_grid * block_grid,), dtype=np.float32)
    else:
        blocks = np.empty((block_grid * block_grid, 3), dtype=np.float32)
        bi = 0
        for yy in range(block_grid):
            y0 = yy * gh
            y1 = (yy + 1) * gh if yy < block_grid - 1 else h
            for xx in range(block_grid):
                x0 = xx * gw
                x1 = (xx + 1) * gw if xx < block_grid - 1 else w
                patch = arr_rgb01[y0:y1, x0:x1, :]
                blocks[bi] = patch.mean(axis=(0, 1))
                bi += 1
        blocks = blocks.reshape(-1)

    flat_rgb = arr_rgb01.reshape(-1).astype(np.float32, copy=False)

    feats = np.concatenate(
        [
            flat_rgb,
            np.array(
                [
                    g_mean,
                    g_std,
                    g_q25,
                    g_q75,
                    edge_mean,
                    edge_std,
                    r_mean,
                    gch_mean,
                    b_mean,
                    r_std,
                    gch_std,
                    b_std,
                ],
                dtype=np.float32,
            ),
            extra,
            blocks.astype(np.float32, copy=False),
        ],
        axis=0,
    ).astype(np.float32, copy=False)
    return feats


def _apply_brightness_contrast(
    arr_rgb01: np.ndarray, brightness_delta: float, contrast: float
) -> np.ndarray:
    x = arr_rgb01 + brightness_delta
    x = (x - 0.5) * contrast + 0.5
    return np.clip(x, 0.0, 1.0).astype(np.float32)


def build_dataset_quad_aug(
    df,
    img_dir,
    size=72,
    block_grid=6,
    brightness_delta=0.06,
    contrast=1.10,
    resize_resample=Image.Resampling.BICUBIC,
    num_workers=None,
):
    d = 3 * size * size + 12 + 12 + 3 * block_grid * block_grid
    n = len(df)
    X = np.zeros((n * 4, d), dtype=np.float32)
    y = None
    if "label" in df.columns:
        y0 = df["label"].to_numpy(dtype=np.int64, copy=False)
        y = np.concatenate([y0, y0, y0, y0], axis=0)

    valid = np.zeros((n,), dtype=bool)
    image_ids = df["image_id"].to_numpy()

    def _process_one(i_and_id):
        i, image_id = i_and_id
        img_path = os.path.join(img_dir, str(image_id))
        try:
            with Image.open(img_path) as im:
                img = im.convert("RGB")
                img = img.resize((size, size), resample=resize_resample)
            arr = (np.asarray(img, dtype=np.float32) / 255.0).astype(
                np.float32, copy=False
            )

            base_feats = _compute_all_features(arr, size=size, block_grid=block_grid)

            arr_hf = arr[:, ::-1, :].copy()
            hf_feats = _compute_all_features(arr_hf, size=size, block_grid=block_grid)

            arr_vf = arr[::-1, :, :].copy()
            vf_feats = _compute_all_features(arr_vf, size=size, block_grid=block_grid)

            arr_bc = _apply_brightness_contrast(
                arr, brightness_delta=brightness_delta, contrast=contrast
            )
            bc_feats = _compute_all_features(arr_bc, size=size, block_grid=block_grid)
            return i, base_feats, hf_feats, vf_feats, bc_feats, True
        except Exception:
            return i, None, None, None, None, False

    if num_workers is None:
        cpu = os.cpu_count() or 4
        num_workers = min(8, cpu)

    with ThreadPoolExecutor(max_workers=num_workers) as ex:
        for i, base_feats, hf_feats, vf_feats, bc_feats, ok in ex.map(
            _process_one, enumerate(image_ids), chunksize=64
        ):
            if not ok:
                continue
            valid[i] = True
            X[i] = base_feats
            X[i + n] = hf_feats
            X[i + 2 * n] = vf_feats
            X[i + 3 * n] = bc_feats

    return X, y, valid


def standardize_fit(X):
    mu = X.mean(axis=0, keepdims=True)
    sig = X.std(axis=0, keepdims=True)
    sig = np.where(sig < 1e-6, 1.0, sig)
    return mu.astype(np.float32), sig.astype(np.float32)


def standardize_apply(X, mu, sig):
    return ((X - mu) / sig).astype(np.float32)


def train_multinomial_logreg(
    X, y, n_classes=5, lr=0.15, reg=1e-3, epochs=60, batch_size=256, seed=123
):
    rng = np.random.default_rng(seed)
    n, d = X.shape
    W = np.zeros((d, n_classes), dtype=np.float32)
    b = np.zeros((n_classes,), dtype=np.float32)

    for ep in range(epochs):
        idx = rng.permutation(n)
        for start in range(0, n, batch_size):
            j = idx[start : start + batch_size]
            Xb = X[j]
            yb = y[j]

            logits = Xb @ W + b
            P = softmax(logits)

            bs = Xb.shape[0]
            P[np.arange(bs), yb] -= 1.0
            P /= bs

            gW = Xb.T @ P + reg * W
            gb = P.sum(axis=0)

            W -= lr * gW
            b -= lr * gb

    return W, b


def predict_logreg_proba(X, W, b):
    logits = X @ W + b
    return softmax(logits).astype(np.float32)


SEED = 123

MAX_TRAIN_SAMPLES = None

FEAT_SIZE = 72
BLOCK_GRID = 8

train_df_shuf = train_df.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
if MAX_TRAIN_SAMPLES is not None and len(train_df_shuf) > MAX_TRAIN_SAMPLES:
    train_used = train_df_shuf.iloc[:MAX_TRAIN_SAMPLES].copy()
else:
    train_used = train_df_shuf

print("Using train samples:", len(train_used), "of", len(train_df))

X_train_all, y_train_all, valid_train = build_dataset_quad_aug(
    train_used,
    train_image_dir,
    size=FEAT_SIZE,
    block_grid=BLOCK_GRID,
    brightness_delta=0.06,
    contrast=1.10,
)

n_train = len(train_used)
keep_train = np.concatenate(
    [valid_train, valid_train, valid_train, valid_train], axis=0
)
dropped = int((~valid_train).sum())
if dropped > 0:
    print(f"Dropped {dropped}/{n_train} train images due to load/feature failures.")

X_train_base_valid = X_train_all[:n_train][valid_train]
mu, sig = standardize_fit(X_train_base_valid)

X_train = X_train_all[keep_train]
y_train = y_train_all[keep_train]
X_train = standardize_apply(X_train, mu, sig)

W, b = train_multinomial_logreg(
    X_train,
    y_train,
    n_classes=5,
    lr=0.10,
    reg=5e-3,
    epochs=60,
    batch_size=256,
    seed=SEED,
)

X_test4_all, _, valid_test = build_dataset_quad_aug(
    sample_csv,
    test_image_dir,
    size=FEAT_SIZE,
    block_grid=BLOCK_GRID,
    brightness_delta=0.06,
    contrast=1.10,
)
X_test4_all = standardize_apply(X_test4_all, mu, sig)

n_test = len(sample_csv)
P_all = predict_logreg_proba(X_test4_all, W, b)  # (4*n_test, 5)

P_mean = (
    P_all[:n_test]
    + P_all[n_test : 2 * n_test]
    + P_all[2 * n_test : 3 * n_test]
    + P_all[3 * n_test :]
) / 4.0

test_pred = np.full((n_test,), majority_label, dtype=np.int64)
ok_idx = np.where(valid_test)[0]
if ok_idx.size < n_test:
    print(
        f"Test load/feature failures: {n_test - ok_idx.size}/{n_test}; using majority fallback."
    )
test_pred[ok_idx] = np.argmax(P_mean[ok_idx], axis=1).astype(np.int64)

test_pred = np.clip(test_pred, 0, 4).astype(int)



## === cell 5
image_predictions = []
test_ids = sample_csv["image_id"].tolist()

if TF_AVAILABLE:
    raise RuntimeError(
        "TF_AVAILABLE unexpectedly True in this environment; this script expects TF import to fail. "
        "If TF becomes available, re-enable the original model inference code path."
    )
else:
    image_predictions = [
        {"image_id": iid, "label": int(p)} for iid, p in zip(test_ids, test_pred)
    ]

submission_df = pd.DataFrame(image_predictions)
submission_df = submission_df[["image_id", "label"]]
assert submission_df.shape[0] == sample_csv.shape[0], (
    submission_df.shape,
    sample_csv.shape,
)

submission_path = "/kaggle/working/submission.csv"
submission_df.to_csv(submission_path, index=False)
print("Wrote:", submission_path, "shape:", submission_df.shape)



## === cell 6
print(submission_df.head())
print(submission_df["label"].value_counts().sort_index())
print("Submission columns:", submission_df.columns.tolist())
print("Submission preview saved at /kaggle/working/submission.csv")
