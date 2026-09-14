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

0.8540525850827251

# 6. Current score

0.69178

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'The changes freeze the EfficientNetB3 backbone and double the batch size (from 16 to 32) so each epoch processes half as many batches, dramatically reducing training time while keeping the same model architecture and loss. Freezing the backbone cuts the expensive gradient calculations but still allows the classifier head to learn, preserving the core learning logic. We also enable parallel data loading (workers = 4) to keep the GPU fed efficiently. All other steps, preprocessing, validation, and submission remain unchanged.'
- What this solution (achieved 0.0) has done: 'The changes preload and preprocess all images once, replace the per‑epoch disk reads with in‑memory `ImageDataGenerator.flow` batches, and enable multiprocessing in `model.fit`. This removes the most time‑consuming I/O and repeated preprocessing while keeping the exact model, augmentation, loss, and evaluation logic, so the validation kappa and final predictions remain unchanged.'
- What this solution (achieved 0.70308) has done: 'The fix removes the TensorFlow dependency that caused import errors and replaces the deep‑learning model with a lightweight RandomForest classifier built on the same pre‑processed image data. This keeps the original data loading and preprocessing logic, preserves the train/validation split, and correctly writes a *submission.csv* file with the required columns. Using a conventional model also allows the script to run end‑to‑end in the given environment while producing a non‑zero Quadratic Weighted Kappa score, moving the result toward the target.'
- What this solution (achieved 0.69686) has done: 'The changes add parallel image loading with a thread pool, which speeds up the I/O‑ and CPU‑heavy preprocessing step without altering any model or evaluation logic. The rest of the pipeline (splitting, RandomForest training, voting, and submission) stays exactly the same, preserving correctness and results.'
- What this solution (achieved 0.69178) has done: 'I keep the overall RandomForest‑based pipeline but make two small, score‑oriented tweaks:  
1. Use the slightly richer `load_ben_color` preprocessing instead of the simpler `preprocessing` function, which preserves more image detail and often improves the kappa.  
2. Replace the hard majority‑vote on class labels with an average of the class‑probability vectors from each forest and then take the arg‑max. This smoother ensembling usually yields a higher quadratic weighted kappa.  
Both changes stay within the existing model framework and add only minor code adjustments, so the script still runs end‑to‑end and writes a correct `submission.csv`.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd
import cv2
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import cohen_kappa_score
from sklearn.model_selection import train_test_split
import concurrent.futures  # parallel image loading



## === cell 1
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 3  # kept for compatibility, not used
TRAIN_IMG_DIR = "/kaggle/input/aptos2019-blindness-detection/train_images"
TEST_IMG_DIR = "/kaggle/input/aptos2019-blindness-detection/test_images"
TRAIN_CSV = "/kaggle/input/aptos2019-blindness-detection/train.csv"
TEST_CSV = "/kaggle/input/aptos2019-blindness-detection/test.csv"


def crop_image_from_gray(img, tol=7):
    if img.ndim == 2:
        mask = img > tol
        return img[np.ix_(mask.any(1), mask.any(0))]
    elif img.ndim == 3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        mask = gray_img > tol
        if img[:, :, 0][np.ix_(mask.any(1), mask.any(0))].shape[0] == 0:
            return img
        img1 = img[:, :, 0][np.ix_(mask.any(1), mask.any(0))]
        img2 = img[:, :, 1][np.ix_(mask.any(1), mask.any(0))]
        img3 = img[:, :, 2][np.ix_(mask.any(1), mask.any(0))]
        return np.stack([img1, img2, img3], axis=-1)


def load_ben_color(image, sigmaX=10):
    """Preprocess image with the richer pipeline used in many Kaggle solutions."""
    image = crop_image_from_gray(image)
    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    image = cv2.addWeighted(image, 4, cv2.GaussianBlur(image, (0, 0), sigmaX), -4, 128)
    return image.astype("float32") / 255.0


def _process_path(path):
    """Read and preprocess a single image using the richer pipeline."""
    img = cv2.imread(path)
    img = load_ben_color(img)
    return img


def load_images(df):
    """Load and preprocess images in parallel, preserving order."""
    paths = df["filepath"].tolist()
    with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        imgs = list(executor.map(_process_path, paths))
    return np.stack(imgs)




## === cell 2
train_df = pd.read_csv(TRAIN_CSV)
train_df["filepath"] = train_df["id_code"].apply(
    lambda x: os.path.join(TRAIN_IMG_DIR, f"{x}.png")
)
train_df["diagnosis"] = train_df["diagnosis"].astype(str)

train_df, val_df = train_test_split(
    train_df,
    test_size=0.1,
    stratify=train_df["diagnosis"],
    random_state=42,
)

print("Loading and preprocessing training images...")
train_images = load_images(train_df)
print("Loading and preprocessing validation images...")
val_images = load_images(val_df)

train_labels = train_df["diagnosis"].astype(int).values
val_labels = val_df["diagnosis"].astype(int).values

X_train = train_images.reshape(len(train_images), -1)
X_val = val_images.reshape(len(val_images), -1)




## === cell 3
rf_models = []
seeds = [42, 52, 62]  # distinct seeds for diversity
for s in seeds:
    rf = RandomForestClassifier(
        n_estimators=800,  # more trees for a stronger model
        max_depth=None,
        n_jobs=-1,
        random_state=s,
        class_weight="balanced",
    )
    rf.fit(X_train, train_labels)
    rf_models.append(rf)


def ensemble_predict_proba(models, X):
    """Average class‑probability predictions from all models."""
    prob_sum = np.zeros((X.shape[0], 5), dtype=float)  # 5 classes: 0‑4
    for m in models:
        prob_sum += m.predict_proba(X)
    avg_prob = prob_sum / len(models)
    return np.argmax(avg_prob, axis=1)


val_pred = ensemble_predict_proba(rf_models, X_val)

kappa = cohen_kappa_score(val_labels, val_pred, weights="quadratic")
print(f"Validation Quadratic Weighted Kappa: {kappa:.5f}")




## === cell 4
test_df = pd.read_csv(TEST_CSV)
test_df["filepath"] = test_df["id_code"].apply(
    lambda x: os.path.join(TEST_IMG_DIR, f"{x}.png")
)


def load_test_images(df):
    """Parallel loading for test set (same logic as load_images)."""
    paths = df["filepath"].tolist()
    with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        imgs = list(executor.map(_process_path, paths))
    return np.stack(imgs)


print("Loading and preprocessing test images...")
test_images = load_test_images(test_df)
X_test = test_images.reshape(len(test_images), -1)

test_pred_labels = ensemble_predict_proba(rf_models, X_test)

submission = pd.DataFrame(
    {"id_code": test_df["id_code"], "diagnosis": test_pred_labels}
)
submission.to_csv("submission.csv", index=False)
print("Submission file saved as submission.csv")
