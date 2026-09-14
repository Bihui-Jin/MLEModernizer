# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


# 1. Kaggle task description

## Task
Create a classifier to predict the severity of diabetic retinopathy.

## Metric
Quadratic weighted kappa, which measures the agreement between two ratings. This metric typically varies from 0 (random agreement between raters) to 1 (complete agreement between raters). In the event that there is less agreement between the raters than expected by chance, this metric may go below 0. The quadratic weighted kappa is calculated between the scores assigned by the human rater and the predicted scores.

Images have five possible ratings, 0,1,2,3,4.  Each image is characterized by a tuple *(e*,*e)*, which corresponds to its scores by *Rater A* (human) and *Rater B* (predicted).  The quadratic weighted kappa is calculated as follows. First, an N x N histogram matrix *O* is constructed, such that *O* corresponds to the number of images that received a rating *i* by *A* and a rating *j* by *B*. An *N-by-N* matrix of weights, *w*, is calculated based on the difference between raters' scores:

An *N-by-N* histogram matrix of expected ratings, *E*, is calculated, assuming that there is no correlation between rating scores.  This is calculated as the outer product between each rater's histogram vector of ratings, normalized such that *E* and *O* have the same sum.

## Submission Format
```
id_code,diagnosis
0005cfc8afb6,0
003f0afdcd15,0
etc.
```

## Dataset
You are provided with a large set of retina images taken using [fundus photography](https://en.wikipedia.org/wiki/Fundus_photography) under a variety of imaging conditions.

Labels are on a scale of 0 to 4:

> 0 - No DR
> 1 - Mild
> 2 - Moderate
> 3 - Severe
> 4 - Proliferative DR

Images may contain artifacts, be out of focus, underexposed, or overexposed. The images were gathered from multiple clinics using a variety of cameras over an extended period of time, which will introduce further variation.

- **train.csv** - the training labels
- **test.csv** - the test set (you must predict the `diagnosis` value for these variables)
- **sample_submission.csv** - a sample submission file in the correct format
- **train.zip** - the training set images
- **test.zip** - the public test set images

# 2. Python version

3.10

# 3. Installed packages

No external packages required in the script and installed.

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (118 lines)
            sample_submission.csv (368 lines)
            sample_submission.csv.zip (3.2 kB)
            test.csv (368 lines)
            test.csv.zip (2.9 kB)
            test.zip (160 Bytes)
            test_images.zip (902.9 MB)
            train.csv (3296 lines)
            train.csv.zip (27.5 kB)
            train.zip (162 Bytes)
            train_images.zip (7.7 GB)
            aptos2019-blindness-detection/
                description.md (118 lines)
                sample_submission.csv (368 lines)
                ... and 9 other files
                aptos2019-blindness-detection/
                test_images/
                    218c822a3dd9.png (5.7 MB)
                    0e82bcacc475.png (5.2 MB)
                    ... and 365 other files
                    test_images/
                train_images/
                    184a185e7447.png (337.5 kB)
                    c4aef0d88d1b.png (876.6 kB)
                    ... and 3293 other files
                    train_images/
            test_images/
                218c822a3dd9.png (5.7 MB)
                0e82bcacc475.png (5.2 MB)
                ... and 365 other files
                test_images/
            train_images/
                184a185e7447.png (337.5 kB)
                c4aef0d88d1b.png (876.6 kB)
                ... and 3293 other files
                train_images/
        input/
            description.md (118 lines)
            sample_submission.csv (368 lines)
            sample_submission.csv.zip (3.2 kB)
            test.csv (368 lines)
            test.csv.zip (2.9 kB)
            test.zip (160 Bytes)
            test_images.zip (902.9 MB)
            train.csv (3296 lines)
            train.csv.zip (27.5 kB)
            train.zip (162 Bytes)
            train_images.zip (7.7 GB)
            aptos2019-blindness-detection/
                description.md (118 lines)
                sample_submission.csv (368 lines)
                ... and 9 other files
                aptos2019-blindness-detection/
                test_images/
                    218c822a3dd9.png (5.7 MB)
                    0e82bcacc475.png (5.2 MB)
                    ... and 365 other files
                    test_images/
                train_images/
                    184a185e7447.png (337.5 kB)
                    c4aef0d88d1b.png (876.6 kB)
                    ... and 3293 other files
                    train_images/
            test_images/
                218c822a3dd9.png (5.7 MB)
                0e82bcacc475.png (5.2 MB)
                ... and 365 other files
                test_images/
                    218c822a3dd9.png (5.7 MB)
                    0e82bcacc475.png (5.2 MB)
                    ... and 365 other files
                    test_images/
            train_images/
                184a185e7447.png (337.5 kB)
                c4aef0d88d1b.png (876.6 kB)
                ... and 3293 other files
                train_images/
                    184a185e7447.png (337.5 kB)
                    c4aef0d88d1b.png (876.6 kB)
                    ... and 3293 other files
                    train_images/
        working/
            aptos2019-blindness-detection/
                description.md (118 lines)
                sample_submission.csv (368 lines)
                ... and 9 other files
                aptos2019-blindness-detection/
                test_images/
                    218c822a3dd9.png (5.7 MB)
                    0e82bcacc475.png (5.2 MB)
                    ... and 365 other files
                    test_images/
                train_images/
                    184a185e7447.png (337.5 kB)
                    c4aef0d88d1b.png (876.6 kB)
                    ... and 3293 other files
                    train_images/
```

-> data/aptos2019-blindness-detection/sample_submission.csv has 367 rows and 2 columns.
The columns are: id_code, diagnosis

-> data/aptos2019-blindness-detection/test.csv has 367 rows and 1 columns.
The columns are: id_code

-> data/aptos2019-blindness-detection/train.csv has 3295 rows and 2 columns.
The columns are: id_code, diagnosis

-> data/sample_submission.csv has 367 rows and 2 columns.
The columns are: id_code, diagnosis

-> data/test.csv has 367 rows and 1 columns.
The columns are: id_code

-> data/train.csv has 3295 rows and 2 columns.
The columns are: id_code, diagnosis

-> input/aptos2019-blindness-detection/sample_submission.csv has 367 rows and 2 columns.
The columns are: id_code, diagnosis

-> (stopped after 10 files for performance)

# 5. Code solution

## === cell 0
import os
import gc
import cv2
import numpy as np
import pandas as pd

np.random.seed(42)

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"



## === cell 1
"""
    Config
"""
IMG_SIZE = 224
BATCH_SIZE = 16  # kept for compatibility; not used in this non-TF pipeline


def crop_image_from_gray(img, tol=7):
    if img.ndim == 2:
        mask = img > tol
        return img[np.ix_(mask.any(1), mask.any(0))]
    elif img.ndim == 3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        mask = gray_img > tol

        check_shape = img[:, :, 0][np.ix_(mask.any(1), mask.any(0))].shape[0]
        if check_shape == 0:
            return img
        else:
            img1 = img[:, :, 0][np.ix_(mask.any(1), mask.any(0))]
            img2 = img[:, :, 1][np.ix_(mask.any(1), mask.any(0))]
            img3 = img[:, :, 2][np.ix_(mask.any(1), mask.any(0))]
            img = np.stack([img1, img2, img3], axis=-1)
        return img


def load_ben_color(image, sigmaX=10):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = crop_image_from_gray(image).astype("uint8")
    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    image = cv2.addWeighted(image, 4, cv2.GaussianBlur(image, (0, 0), sigmaX), -4, 128)
    return image


"""
    Preprocessing for ImageDataGenerator since ImageDataGenerator reads images in rgb mode, while opencv in bgr
"""


def preprocessing(image, sigmaX=10):
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    image = crop_image_from_gray(image).astype("uint8")
    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    image = cv2.addWeighted(image, 4, cv2.GaussianBlur(image, (0, 0), sigmaX), -4, 128)
    return image.astype("float32") / 255.0




## === cell 2
def extract_features_from_ben_rgb(img_rgb_uint8):
    img = img_rgb_uint8.astype(np.float32) / 255.0

    mean_rgb = img.reshape(-1, 3).mean(axis=0)
    std_rgb = img.reshape(-1, 3).std(axis=0)

    hsv = (
        cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)
        / 255.0
    )
    mean_hsv = hsv.reshape(-1, 3).mean(axis=0)
    std_hsv = hsv.reshape(-1, 3).std(axis=0)

    gray = (
        cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(
            np.float32
        )
        / 255.0
    )
    edges = cv2.Canny((gray * 255).astype(np.uint8), 50, 150).astype(np.float32) / 255.0

    gray_hist, _ = np.histogram(gray, bins=16, range=(0.0, 1.0), density=True)
    edge_hist, _ = np.histogram(edges, bins=8, range=(0.0, 1.0), density=True)

    thumb = (
        cv2.resize(
            (img * 255).astype(np.uint8), (16, 16), interpolation=cv2.INTER_AREA
        ).astype(np.float32)
        / 255.0
    )
    thumb_flat = thumb.reshape(-1)

    feat = np.concatenate(
        [
            mean_rgb,
            std_rgb,
            mean_hsv,
            std_hsv,
            gray_hist.astype(np.float32),
            edge_hist.astype(np.float32),
            thumb_flat.astype(np.float32),
        ],
        axis=0,
    ).astype(np.float32)
    return feat




## === cell 3
DATA_ROOT_CANDIDATES = [
    "/kaggle/input/aptos2019-blindness-detection",
    "../input/aptos2019-blindness-detection",
    "/kaggle/data/aptos2019-blindness-detection",
]
DATA_ROOT = None
for p in DATA_ROOT_CANDIDATES:
    if os.path.exists(p):
        DATA_ROOT = p
        break
if DATA_ROOT is None:
    raise FileNotFoundError(
        "Could not find aptos2019-blindness-detection dataset directory in expected locations."
    )

train_csv_path = os.path.join(DATA_ROOT, "train.csv")
test_csv_path = os.path.join(DATA_ROOT, "test.csv")
sample_sub_path = os.path.join(DATA_ROOT, "sample_submission.csv")
train_img_dir = os.path.join(DATA_ROOT, "train_images")
test_img_dir = os.path.join(DATA_ROOT, "test_images")

train_df = pd.read_csv(train_csv_path)
test_df = pd.read_csv(test_csv_path)
sub_df = pd.read_csv(sample_sub_path)

sub_df = sub_df.drop(columns=["diagnosis"], errors="ignore")
sub_df = sub_df.merge(test_df[["id_code"]], on="id_code", how="right")

print("DATA_ROOT:", DATA_ROOT)
print(
    "Train rows:",
    len(train_df),
    "Test rows:",
    len(test_df),
    "Submission rows:",
    len(sub_df),
)



## === cell 4
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedShuffleSplit


def quadratic_weighted_kappa(y_true, y_pred, n_classes=5):
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    y_true = np.clip(y_true, 0, n_classes - 1)
    y_pred = np.clip(y_pred, 0, n_classes - 1)

    O = np.zeros((n_classes, n_classes), dtype=np.float64)
    for a, b in zip(y_true, y_pred):
        O[a, b] += 1.0

    act_hist = np.bincount(y_true, minlength=n_classes).astype(np.float64)
    pred_hist = np.bincount(y_pred, minlength=n_classes).astype(np.float64)

    E = np.outer(act_hist, pred_hist)
    E = E / E.sum() * O.sum()

    W = np.zeros((n_classes, n_classes), dtype=np.float64)
    for i in range(n_classes):
        for j in range(n_classes):
            W[i, j] = ((i - j) ** 2) / ((n_classes - 1) ** 2)

    num = (W * O).sum()
    den = (W * E).sum()
    return 1.0 - num / den if den > 0 else 0.0


def apply_thresholds(x, thresholds):
    t0, t1, t2, t3 = thresholds
    return np.where(
        x < t0,
        0,
        np.where(x < t1, 1, np.where(x < t2, 2, np.where(x < t3, 3, 4))),
    ).astype(int)


def fit_thresholds_qwk(x_cont, y_true, n_classes=5, n_iters=6):
    x_cont = np.asarray(x_cont, dtype=np.float64)
    y_true = np.asarray(y_true, dtype=int)

    qs = np.quantile(x_cont, [0.2, 0.4, 0.6, 0.8]).tolist()
    thr = np.array(sorted(qs), dtype=np.float64)

    xmin, xmax = float(x_cont.min()), float(x_cont.max())
    grid = np.linspace(xmin, xmax, 200, dtype=np.float64)

    best_thr = thr.copy()
    best_score = quadratic_weighted_kappa(
        y_true, apply_thresholds(x_cont, best_thr), n_classes=n_classes
    )

    for _ in range(n_iters):
        for k in range(4):
            candidate_best = best_thr.copy()
            candidate_score = best_score

            low = xmin if k == 0 else candidate_best[k - 1] + 1e-6
            high = xmax if k == 3 else candidate_best[k + 1] - 1e-6
            if low >= high:
                continue

            subgrid = grid[(grid > low) & (grid < high)]
            if subgrid.size == 0:
                continue

            for val in subgrid:
                cand = candidate_best.copy()
                cand[k] = val
                preds = apply_thresholds(x_cont, cand)
                sc = quadratic_weighted_kappa(y_true, preds, n_classes=n_classes)
                if sc > candidate_score:
                    candidate_score = sc
                    candidate_best = cand

            best_thr = candidate_best
            best_score = candidate_score

    return best_thr, best_score


MAX_TRAIN_SAMPLES = None  # use all

train_ids_all = train_df["id_code"].values
y_all = train_df["diagnosis"].astype(int).values

if MAX_TRAIN_SAMPLES is not None and len(train_ids_all) > MAX_TRAIN_SAMPLES:
    train_ids_all = train_ids_all[:MAX_TRAIN_SAMPLES]
    y_all = y_all[:MAX_TRAIN_SAMPLES]

sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, val_idx = next(sss.split(train_ids_all, y_all))
train_ids = train_ids_all[train_idx]
y_train = y_all[train_idx]
val_ids = train_ids_all[val_idx]
y_val = y_all[val_idx]


def build_features(ids, labels=None, img_dir=None, report_every=500, tag="train"):
    X_feats = []
    y_out = [] if labels is not None else None
    bad = 0
    for i, img_id in enumerate(ids):
        img_path = os.path.join(img_dir, f"{img_id}.png")
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            bad += 1
            continue
        img_rgb = load_ben_color(img_bgr)
        X_feats.append(extract_features_from_ben_rgb(img_rgb))
        if labels is not None:
            y_out.append(int(labels[i]))
        if report_every is not None and (i + 1) % report_every == 0:
            print(f"Processed {tag} images: {i+1}/{len(ids)}")
    if bad > 0:
        print(f"WARNING: {bad} {tag} images could not be read and were skipped.")
    X = np.vstack(X_feats) if len(X_feats) else np.zeros((0, 0), dtype=np.float32)
    if labels is not None:
        y_arr = np.array(y_out, dtype=int)
        return X, y_arr
    return X


print("Building train features...")
X_train, y_train = build_features(
    train_ids, labels=y_train, img_dir=train_img_dir, tag="train"
)
print("Building val features...")
X_val, y_val = build_features(
    val_ids, labels=y_val, img_dir=train_img_dir, report_every=250, tag="val"
)

print("Train feature matrix:", X_train.shape, "Train labels:", y_train.shape)
print("Val feature matrix:", X_val.shape, "Val labels:", y_val.shape)

clf = Pipeline(
    steps=[
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        (
            "lr",
            LogisticRegression(
                class_weight="balanced",
                solver="lbfgs",
                max_iter=3000,
                multi_class="multinomial",
                n_jobs=-1,
                C=2.0,
                random_state=42,
            ),
        ),
    ]
)
clf.fit(X_train, y_train)

val_proba = clf.predict_proba(X_val)
val_exp = (val_proba * np.arange(5, dtype=np.float64)[None, :]).sum(axis=1)

thr, thr_qwk = fit_thresholds_qwk(val_exp, y_val, n_classes=5, n_iters=6)
val_pred_argmax = np.argmax(val_proba, axis=1).astype(int)
val_qwk_argmax = quadratic_weighted_kappa(y_val, val_pred_argmax, n_classes=5)
val_pred_thr = apply_thresholds(val_exp, thr)
val_qwk_thr = quadratic_weighted_kappa(y_val, val_pred_thr, n_classes=5)

print("Val QWK (argmax):", float(val_qwk_argmax))
print("Val QWK (thresholded exp):", float(val_qwk_thr))
print("Learned thresholds:", thr.tolist(), "Optimizer-reported best:", float(thr_qwk))

gc.collect()



## === cell 5
print("Rebuilding full-train features and refitting model...")
full_ids = train_df["id_code"].values
full_y = train_df["diagnosis"].astype(int).values
X_full, y_full = build_features(
    full_ids, labels=full_y, img_dir=train_img_dir, report_every=500, tag="full-train"
)

clf.fit(X_full, y_full)
gc.collect()

id_code = sub_df["id_code"].values
X_test_feats = []
for i, img_id in enumerate(id_code):
    img_path = os.path.join(test_img_dir, f"{img_id}.png")
    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        raise FileNotFoundError(f"Could not read image: {img_path}")
    img_rgb = load_ben_color(img_bgr)
    X_test_feats.append(extract_features_from_ben_rgb(img_rgb))
    if (i + 1) % 100 == 0:
        print(f"Processed test images: {i+1}/{len(id_code)}")

X_test = np.vstack(X_test_feats)
proba = clf.predict_proba(X_test)

exp_test = (proba * np.arange(5, dtype=np.float64)[None, :]).sum(axis=1)
test_prediction = apply_thresholds(exp_test, thr).astype(np.int64)



## === cell 6
sub_df["diagnosis"] = test_prediction.astype("int64")
sub_path = "submission.csv"
sub_df[["id_code", "diagnosis"]].to_csv(sub_path, index=False)

unique, counts = np.unique(test_prediction, return_counts=True)
print(dict(zip(unique.tolist(), counts.tolist())))
print(f"Saved {sub_path} with shape {sub_df.shape}")
print("Done!")
