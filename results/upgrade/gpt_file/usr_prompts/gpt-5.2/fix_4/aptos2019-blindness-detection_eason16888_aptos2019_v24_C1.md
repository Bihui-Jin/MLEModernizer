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

# 5. Target score

0.8612962593545803

# 6. Current score

0.72139

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'I remove the failing `pip install tensorflow-addons` and the `tensorflow_addons` import that triggers the protobuf `MessageFactory.GetPrototype` error, because it is not used anywhere in your inference-only pipeline. I also make the script self-contained by re-importing all required libraries in the first cell so `tf`, `pd`, etc. are always defined, and I fix the broken cell numbering/order while preserving your model architecture and preprocessing logic. Since your code references a non-existent external weights file, I instead use EfficientNetB1 with ImageNet weights (same architecture) so the notebook can produce meaningful predictions and a valid `submission.csv` end-to-end. Finally, I ensure inference uses the same normalization as your `preprocessing` function (divide by 255) and read `test.csv` (not `sample_submission.csv`) for the id list, keeping the required submission columns unchanged.'
- What this solution (achieved 0.71135) has done: 'The protobuf `MessageFactory.GetPrototype` crash happens during TensorFlow import due to an incompatible protobuf runtime in this environment, so the only viable way to make this run end-to-end is to remove TensorFlow usage entirely. To keep the pipeline’s semantics (image preprocessing → model → 0–4 prediction → submission) while remaining minimal and stable, I replace the TF model inference with a lightweight OpenCV feature extractor plus a scikit-learn multiclass classifier trained on `train.csv`, which is available locally. This also resolves the current score of 0.0 (from a hard crash) by producing real predictions aligned to the evaluation labels (0–4). The script keeps your Ben Graham-style preprocessing and writes a valid `submission.csv` with the required columns.'
- What this solution (achieved 0.72139) has done: 'Your current pipeline is already producing a valid submission and is far below the target (0.71135 vs 0.8613), so we should make small, legitimate improvements that better align the predictions with quadratic weighted kappa without changing the overall approach (Ben Graham preprocessing → simple features → scikit-learn classifier). The most direct, minimal lever here is handling the strong class imbalance: logistic regression is likely biased toward the majority class, which hurts kappa. I add `class_weight="balanced"` to the same multinomial logistic regression (same model family, same training loop) to reduce that bias. I also switch the solver explicitly to `lbfgs` (standard for multinomial) to make optimization more stable without changing semantics, keeping everything else identical and still writing `submission.csv`.'

# 9. Code solution

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

MAX_TRAIN_SAMPLES = None  # use all

train_ids = train_df["id_code"].values
y_train = train_df["diagnosis"].astype(int).values

if MAX_TRAIN_SAMPLES is not None and len(train_ids) > MAX_TRAIN_SAMPLES:
    train_ids = train_ids[:MAX_TRAIN_SAMPLES]
    y_train = y_train[:MAX_TRAIN_SAMPLES]

X_feats = []
bad = 0
good_y = []
for i, (img_id, y) in enumerate(zip(train_ids, y_train)):
    img_path = os.path.join(train_img_dir, f"{img_id}.png")
    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        bad += 1
        continue
    img_rgb = load_ben_color(img_bgr)
    X_feats.append(extract_features_from_ben_rgb(img_rgb))
    good_y.append(int(y))
    if (i + 1) % 500 == 0:
        print(f"Processed train images: {i+1}/{len(train_ids)}")

if bad > 0:
    print(f"WARNING: {bad} training images could not be read and were skipped.")

X_train = np.vstack(X_feats)
y_train = np.array(good_y, dtype=int)

print("Feature matrix:", X_train.shape, "Labels:", y_train.shape)

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

gc.collect()



## === cell 5
id_code = sub_df["id_code"].values
test_prediction = np.empty(len(id_code), dtype="int64")

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
pred = np.argmax(proba, axis=1).astype(np.int64)
test_prediction[:] = pred



## === cell 6
sub_df["diagnosis"] = test_prediction.astype("int64")
sub_path = "submission.csv"
sub_df[["id_code", "diagnosis"]].to_csv(sub_path, index=False)

unique, counts = np.unique(test_prediction, return_counts=True)
print(dict(zip(unique.tolist(), counts.tolist())))
print(f"Saved {sub_path} with shape {sub_df.shape}")
print("Done!")
