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

3.7

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

# 5. Target score

0.8723692104941825

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'I remove the Jupyter-only `%matplotlib inline` magic and fix the import stack so `Input` (and the rest of Keras symbols) are always defined. I also add a safe fallback for model weights since the referenced `../input/aptos2019-resnet50/resNet50.h5` dataset is not present in your provided input tree; the script proceed with randomly initialized weights rather than crashing, ensuring a valid `submission.csv` is always produced. Finally, I make prediction robust to missing/corrupt images and guarantee `ans` has exactly the same length as the submission template so `submit['diagnosis']=ans` cannot fail.'
- What this solution (achieved 0.0) has done: 'I fix the TensorFlow import crash (`MessageFactory.GetPrototype`) by forcing the pure-Python protobuf implementation before TensorFlow is imported, which is a common Kaggle environment incompatibility. I also move the env-var setup to the very top so it takes effect, and add a small compatibility fallback to pin protobuf parsing behavior if needed. These changes are runtime/stability fixes only (model, preprocessing, and prediction logic remain the same), and the script still write a valid `submission.csv`. With TensorFlow successfully importing, your model at least run end-to-end and produce non-empty predictions (improving from the current 0.0 which is effectively “broken run”).'
- What this solution (achieved 0.0) has done: 'I fix the TensorFlow import crash (`MessageFactory.GetPrototype`) by setting the protobuf environment variables before *any* TensorFlow-related import and by forcing the pure-Python protobuf implementation early. I also add a safe fallback that, if TensorFlow still can’t be imported in this environment, produces a valid `submission.csv` using the sample submission template (so you don’t get a broken 0.0 due to no file). These changes are runtime/stability only and keep your model definition, preprocessing, and prediction logic unchanged when TensorFlow loads successfully. The script always write a correctly formatted `.csv` submission with `id_code,diagnosis`.'

# 9. Code solution

## === cell 0
import os
import sys
import numpy as np
import pandas as pd


def _ensure_protobuf_env_and_reexec():
    desired = {
        "PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION": "python",
        "PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION": "3",
        "PYTHONHASHSEED": "42",
    }
    need_reexec = False
    for k, v in desired.items():
        if os.environ.get(k) != v:
            os.environ[k] = v
            need_reexec = True
    if need_reexec and os.environ.get("_KAGGLE_REEXEC_DONE", "0") != "1":
        os.environ["_KAGGLE_REEXEC_DONE"] = "1"
        os.execv(sys.executable, [sys.executable] + sys.argv)


_ensure_protobuf_env_and_reexec()

np.random.seed(42)

BASE_INPUT = "/kaggle/input"
BASE_DATA_FALLBACK = "/kaggle/data"

print(
    "Input dirs:",
    os.listdir(BASE_INPUT) if os.path.isdir(BASE_INPUT) else "BASE_INPUT not found",
)



## === cell 1
from PIL import Image  # noqa: F401
import cv2
from tqdm import tqdm

from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import cohen_kappa_score




## === cell 2
def crop_image_from_gray(img, tol=7):
    if img.ndim == 2:
        mask = img > tol
        return img[np.ix_(mask.any(1), mask.any(0))]
    elif img.ndim == 3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        mask = gray_img > tol
        check_shape = img[:, :, 0][np.ix_(mask.any(1), mask.any(0))].shape[0]
        if check_shape == 0:  # image is too dark so that we crop out everything
            return img  # return original image
        else:
            img1 = img[:, :, 0][np.ix_(mask.any(1), mask.any(0))]
            img2 = img[:, :, 1][np.ix_(mask.any(1), mask.any(0))]
            img3 = img[:, :, 2][np.ix_(mask.any(1), mask.any(0))]
            img = np.stack([img1, img2, img3], axis=-1)
        return img
    else:
        return img




## === cell 3
IMG_SIZE = 224
batch_size = 32
epochs = 10  # kept for compatibility; non-TF baseline doesn't use it.




## === cell 4
def preprocess_image(img_path):
    image = cv2.imread(img_path)
    if image is None:
        return None
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = crop_image_from_gray(image)
    if image is None or image.size == 0:
        return None
    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    image = cv2.addWeighted(image, 4, cv2.GaussianBlur(image, (0, 0), 30), -4, 128)
    return image




## === cell 5
def _find_first_existing(paths):
    for p in paths:
        if p and os.path.exists(p):
            return p
    return None


comp_dir = _find_first_existing(
    [
        os.path.join(BASE_INPUT, "aptos2019-blindness-detection"),
        os.path.join(
            BASE_INPUT, "aptos2019-blindness-detection", "aptos2019-blindness-detection"
        ),
        os.path.join(BASE_DATA_FALLBACK, "aptos2019-blindness-detection"),
        os.path.join(
            BASE_DATA_FALLBACK,
            "aptos2019-blindness-detection",
            "aptos2019-blindness-detection",
        ),
    ]
)
if comp_dir is None:
    raise FileNotFoundError(
        "Could not find aptos2019-blindness-detection directory in /kaggle/input or /kaggle/data"
    )

test_images_dir = _find_first_existing(
    [
        os.path.join(comp_dir, "test_images"),
        os.path.join(comp_dir, "aptos2019-blindness-detection", "test_images"),
        os.path.join(BASE_INPUT, "test_images"),
        os.path.join(BASE_DATA_FALLBACK, "test_images"),
    ]
)
train_images_dir = _find_first_existing(
    [
        os.path.join(comp_dir, "train_images"),
        os.path.join(comp_dir, "aptos2019-blindness-detection", "train_images"),
        os.path.join(BASE_INPUT, "train_images"),
        os.path.join(BASE_DATA_FALLBACK, "train_images"),
    ]
)
sample_sub_path = _find_first_existing(
    [
        os.path.join(comp_dir, "sample_submission.csv"),
        os.path.join(
            comp_dir, "aptos2019-blindness-detection", "sample_submission.csv"
        ),
        os.path.join(BASE_INPUT, "sample_submission.csv"),
        os.path.join(BASE_DATA_FALLBACK, "sample_submission.csv"),
    ]
)
train_csv_path = _find_first_existing(
    [
        os.path.join(comp_dir, "train.csv"),
        os.path.join(comp_dir, "aptos2019-blindness-detection", "train.csv"),
        os.path.join(BASE_INPUT, "train.csv"),
        os.path.join(BASE_DATA_FALLBACK, "train.csv"),
    ]
)
test_csv_path = _find_first_existing(
    [
        os.path.join(comp_dir, "test.csv"),
        os.path.join(comp_dir, "aptos2019-blindness-detection", "test.csv"),
        os.path.join(BASE_INPUT, "test.csv"),
        os.path.join(BASE_DATA_FALLBACK, "test.csv"),
    ]
)

print("Competition dir:", comp_dir)
print(
    "Train images dir:",
    train_images_dir,
    "exists:",
    os.path.isdir(train_images_dir) if train_images_dir else False,
)
print(
    "Test images dir:",
    test_images_dir,
    "exists:",
    os.path.isdir(test_images_dir) if test_images_dir else False,
)
print(
    "Sample submission:",
    sample_sub_path,
    "exists:",
    os.path.isfile(sample_sub_path) if sample_sub_path else False,
)
print(
    "Train CSV:",
    train_csv_path,
    "exists:",
    os.path.isfile(train_csv_path) if train_csv_path else False,
)
print(
    "Test CSV:",
    test_csv_path,
    "exists:",
    os.path.isfile(test_csv_path) if test_csv_path else False,
)

if sample_sub_path is None or not os.path.isfile(sample_sub_path):
    raise FileNotFoundError("sample_submission.csv not found")
if test_csv_path is None or not os.path.isfile(test_csv_path):
    raise FileNotFoundError("test.csv not found")

test_df = pd.read_csv(test_csv_path)
test_df["id_code"] = test_df["id_code"].astype(str)

sample_submit = pd.read_csv(sample_sub_path)
sample_submit["id_code"] = sample_submit["id_code"].astype(str)




## === cell 6
def _extract_features_from_image_rgb(img_rgb_224):
    if img_rgb_224 is None:
        return None

    img = img_rgb_224.astype(np.float32)

    small = cv2.resize(img, (64, 64), interpolation=cv2.INTER_AREA)  # (64,64,3)
    gray = (
        cv2.cvtColor(small.astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32)
        / 255.0
    )
    hsv = cv2.cvtColor(small.astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[:, :, 0] /= 179.0
    hsv[:, :, 1] /= 255.0
    hsv[:, :, 2] /= 255.0

    hist, _ = np.histogram(gray, bins=32, range=(0.0, 1.0), density=True)
    grid = cv2.resize(gray, (16, 16), interpolation=cv2.INTER_AREA).reshape(-1)

    hsv_mean = hsv.reshape(-1, 3).mean(axis=0)
    hsv_std = hsv.reshape(-1, 3).std(axis=0)

    feat = np.concatenate(
        [
            hist.astype(np.float32),
            grid.astype(np.float32),
            hsv_mean.astype(np.float32),
            hsv_std.astype(np.float32),
        ]
    )
    return feat


def _load_features(img_dir, id_list):
    feats = []
    bad = []
    for _id in tqdm(
        id_list,
        desc=f"Extract feats from {os.path.basename(img_dir) if img_dir else 'None'}",
        leave=False,
    ):
        p = os.path.join(img_dir, str(_id) + ".png")
        im = preprocess_image(p)
        f = _extract_features_from_image_rgb(im)
        if f is None:
            bad.append(True)
            feats.append(None)
        else:
            bad.append(False)
            feats.append(f)

    dim = None
    for f in feats:
        if f is not None:
            dim = int(f.shape[0])
            break
    if dim is None:
        return None, np.array(bad, dtype=np.bool_)

    X = np.zeros((len(id_list), dim), dtype=np.float32)
    for i, f in enumerate(feats):
        if f is not None:
            X[i] = f
    return X, np.array(bad, dtype=np.bool_)


def _qwk(y_true, y_pred):
    return cohen_kappa_score(y_true, y_pred, weights="quadratic")


def _apply_thresholds(x, thr):
    return np.digitize(x, bins=thr).astype(int)


def _fit_thresholds_by_greedy_search(x, y, initial=None, iters=4, step=0.02):
    if initial is None:
        thr = np.array([0.5, 1.5, 2.5, 3.5], dtype=np.float32)
    else:
        thr = np.array(initial, dtype=np.float32)

    best_thr = thr.copy()
    best_score = _qwk(y, _apply_thresholds(x, best_thr))

    for _ in range(iters):
        improved = False
        for i in range(4):
            for delta in (-step, step):
                cand = best_thr.copy()
                cand[i] = cand[i] + delta
                if not (0.0 <= cand[0] < cand[1] < cand[2] < cand[3] <= 4.0):
                    continue
                s = _qwk(y, _apply_thresholds(x, cand))
                if s > best_score:
                    best_score = s
                    best_thr = cand
                    improved = True
        if not improved:
            break
    return best_thr, best_score


def _train_classifier_and_calibrator():
    if train_csv_path is None or not os.path.isfile(train_csv_path):
        raise FileNotFoundError("Train CSV missing; cannot train.")
    if train_images_dir is None or not os.path.isdir(train_images_dir):
        raise FileNotFoundError("Train images dir missing; cannot train.")

    df = pd.read_csv(train_csv_path)
    df["id_code"] = df["id_code"].astype(str)
    y = df["diagnosis"].astype(int).values

    X, bad = _load_features(train_images_dir, df["id_code"].values)
    if X is None:
        raise RuntimeError("Could not read any training images to extract features.")

    mean_feat = None
    if bad.any():
        good_mask = ~bad
        if good_mask.any():
            mean_feat = X[good_mask].mean(axis=0)
            X[bad] = mean_feat

    scaler = StandardScaler(with_mean=True, with_std=True)
    Xs = scaler.fit_transform(X)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_x = np.zeros(len(y), dtype=np.float32)

    for tr_idx, val_idx in skf.split(Xs, y):
        clf_k = LogisticRegression(
            multi_class="multinomial",
            solver="lbfgs",
            max_iter=600,
            n_jobs=1,
            random_state=42,
            C=1.0,
            class_weight="balanced",
        )
        clf_k.fit(Xs[tr_idx], y[tr_idx])
        proba = clf_k.predict_proba(Xs[val_idx]).astype(np.float32)  # (n,5)
        exp_class = (proba * np.arange(5, dtype=np.float32)[None, :]).sum(axis=1)
        oof_x[val_idx] = exp_class

    thr, thr_score = _fit_thresholds_by_greedy_search(
        oof_x, y, initial=[0.5, 1.5, 2.5, 3.5], iters=10, step=0.02
    )
    print("OOF threshold calibration QWK:", float(thr_score))
    print("Learned thresholds:", thr.tolist())

    clf = LogisticRegression(
        multi_class="multinomial",
        solver="lbfgs",
        max_iter=600,
        n_jobs=1,
        random_state=42,
        C=1.0,
        class_weight="balanced",
    )
    clf.fit(Xs, y)

    return clf, scaler, thr, mean_feat


if test_images_dir is None or not os.path.isdir(test_images_dir):
    raise FileNotFoundError("Test images dir not found; cannot create submission.")

clf, scaler, thr, train_mean_feat = _train_classifier_and_calibrator()

test_ids = test_df["id_code"].astype(str).tolist()
X_test, bad_test = _load_features(test_images_dir, test_ids)

if X_test is None:
    pred = np.zeros(len(test_ids), dtype=int)
else:
    if bad_test.any():
        good_mask = ~bad_test
        if train_mean_feat is not None:
            X_test[bad_test] = train_mean_feat.astype(np.float32)
        elif good_mask.any():
            X_test[bad_test] = X_test[good_mask].mean(axis=0).astype(np.float32)
        else:
            pass

    X_test_s = scaler.transform(X_test)
    proba_test = clf.predict_proba(X_test_s).astype(np.float32)
    exp_test = (proba_test * np.arange(5, dtype=np.float32)[None, :]).sum(axis=1)
    pred = _apply_thresholds(exp_test, thr)
    pred = np.clip(pred, 0, 4).astype(int)

sub = pd.DataFrame({"id_code": test_ids, "diagnosis": pred.astype(int)})
sub = sample_submit[["id_code"]].merge(sub, on="id_code", how="left")
sub["diagnosis"] = sub["diagnosis"].fillna(0).astype(int)

sub.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", sub.shape)
print(sub.head())
print(
    "Pred label distribution:",
    pd.Series(sub["diagnosis"]).value_counts().sort_index().to_dict(),
)
