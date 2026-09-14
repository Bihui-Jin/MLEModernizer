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
import numpy as np
import pandas as pd
import cv2
from tqdm import tqdm
from sklearn.metrics import cohen_kappa_score
import random

tf_available = False

IMG_SIZE = 224
BATCH_SIZE = 16


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
        img1 = img[:, :, 0][np.ix_(mask.any(1), mask.any(0))]
        img2 = img[:, :, 1][np.ix_(mask.any(1), mask.any(0))]
        img3 = img[:, :, 2][np.ix_(mask.any(1), mask.any(0))]
        img = np.stack([img1, img2, img3], axis=-1)
        return img


def preprocessing(image, sigmaX=10):
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    image = crop_image_from_gray(image).astype("uint8")
    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    image = cv2.addWeighted(image, 4, cv2.GaussianBlur(image, (0, 0), sigmaX), -4, 128)
    return image.astype("float32") / 255.0


def get_base_dir():
    """Return the directory that contains train.csv / test.csv etc."""
    candidates = [
        "/kaggle/input/aptos2019-blindness-detection",
        os.path.abspath("./input/aptos2019-blindness-detection"),
        os.path.abspath("./data/aptos2019-blindness-detection"),
        os.path.abspath("./aptos2019-blindness-detection"),
    ]
    for path in candidates:
        if os.path.isdir(path):
            return path
    return os.getcwd()


base_dir = get_base_dir()


def compute_centroids_from_df(df, img_dir):
    """Compute colour and flat centroids using the rows of a dataframe."""
    colour_sums = np.zeros((5, 3), dtype=np.float64)
    flat_sums = np.zeros((5, IMG_SIZE * IMG_SIZE * 3), dtype=np.float64)
    counts = np.zeros(5, dtype=int)

    for _, row in tqdm(df.iterrows(), desc="Computing centroids"):
        code = row["id_code"]
        label = int(row["diagnosis"])
        img_path = os.path.join(img_dir, f"{code}.png")
        img = cv2.imread(img_path)
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = preprocessing(img)
        colour_means = img.mean(axis=(0, 1))
        flat_means = img.reshape(-1)

        colour_sums[label] += colour_means
        flat_sums[label] += flat_means
        counts[label] += 1

    colour_centroids = np.zeros((5, 3), dtype=np.float32)
    flat_centroids = np.zeros((5, IMG_SIZE * IMG_SIZE * 3), dtype=np.float32)

    for c in range(5):
        if counts[c] > 0:
            colour_centroids[c] = colour_sums[c] / counts[c]
            flat_centroids[c] = flat_sums[c] / counts[c]

    return {"colour": colour_centroids, "flat": flat_centroids}


def compute_overall_centroids(base_dir):
    train_csv_path = os.path.join(base_dir, "train.csv")
    train_img_dir = os.path.join(base_dir, "train_images")
    if not os.path.isfile(train_csv_path) or not os.path.isdir(train_img_dir):
        return {"colour": None, "flat": None}
    train_df = pd.read_csv(train_csv_path)
    return compute_centroids_from_df(train_df, train_img_dir)


def estimate_blend_weight(base_dir, n_val=400, seed=42):
    """Return a blend weight for colour vs flat based on a quick validation split."""
    train_csv_path = os.path.join(base_dir, "train.csv")
    train_img_dir = os.path.join(base_dir, "train_images")
    train_df = pd.read_csv(train_csv_path)

    np.random.seed(seed)
    shuffled = train_df.sample(frac=1, random_state=seed).reset_index(drop=True)
    val_df = shuffled.iloc[:n_val]
    train_subset = shuffled.iloc[n_val:]

    centroids = compute_centroids_from_df(train_subset, train_img_dir)

    def predict_subset(df, use_colour=True, use_flat=True):
        preds = []
        for _, row in df.iterrows():
            code = row["id_code"]
            img_path = os.path.join(train_img_dir, f"{code}.png")
            img = cv2.imread(img_path)
            if img is None:
                preds.append(0)
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = preprocessing(img)
            x = np.expand_dims(img, axis=0)

            colour_feats = x.mean(axis=(1, 2))
            flat_feats = x.reshape(1, -1)

            if use_colour and centroids["colour"] is not None:
                diffs_c = colour_feats[:, None, :] - centroids["colour"][None, :, :]
                dists_c = np.linalg.norm(diffs_c, axis=2)
                probs_c = np.exp(-dists_c)
                probs_c /= probs_c.sum()
            else:
                probs_c = np.full((1, 5), 1.0 / 5)

            if use_flat and centroids["flat"] is not None:
                diffs_f = flat_feats[:, None, :] - centroids["flat"][None, :, :]
                dists_f = np.sqrt(np.mean(diffs_f**2, axis=2))
                probs_f = np.exp(-dists_f)
                probs_f /= probs_f.sum()
            else:
                probs_f = np.full((1, 5), 1.0 / 5)

            probs = (probs_c + probs_f) / 2.0
            preds.append(int(np.argmax(probs, axis=1)[0]))
        return np.array(preds)

    pred_c = predict_subset(val_df, use_colour=True, use_flat=False)
    kappa_c = cohen_kappa_score(val_df["diagnosis"], pred_c, weights="quadratic")

    pred_f = predict_subset(val_df, use_colour=False, use_flat=True)
    kappa_f = cohen_kappa_score(val_df["diagnosis"], pred_f, weights="quadratic")

    if (kappa_c + kappa_f) == 0:
        return 0.5
    weight = kappa_c / (kappa_c + kappa_f)
    return float(weight)


centroids_dict = compute_overall_centroids(base_dir)
blend_weight = estimate_blend_weight(base_dir)
print(f"Estimated blend weight (colour): {blend_weight:.3f}")


class DummyModel:
    def __init__(self, num_classes=5, centroids=None, blend_weight=0.5):
        self.num_classes = num_classes
        self.colour_centroids = centroids.get("colour") if centroids else None
        self.flat_centroids = centroids.get("flat") if centroids else None
        self.blend_weight = blend_weight  # proportion for colour component

    def _prob_from_distances(self, dists):
        probs = np.exp(-dists)
        probs_sum = probs.sum(axis=1, keepdims=True)
        probs = np.where(probs_sum == 0, 1.0 / self.num_classes, probs / probs_sum)
        return probs

    def predict(self, x):
        batch_size = x.shape[0]
        colour_feats = x.mean(axis=(1, 2))

        if self.colour_centroids is not None:
            diffs_c = colour_feats[:, None, :] - self.colour_centroids[None, :, :]
            dists_c = np.linalg.norm(diffs_c, axis=2)
            probs_c = self._prob_from_distances(dists_c)
        else:
            probs_c = np.full((batch_size, self.num_classes), 1.0 / self.num_classes)

        flat_feats = x.reshape(batch_size, -1)
        if self.flat_centroids is not None:
            diffs_f = flat_feats[:, None, :] - self.flat_centroids[None, :, :]
            dists_f = np.sqrt(np.mean(diffs_f**2, axis=2))
            probs_f = self._prob_from_distances(dists_f)
        else:
            probs_f = np.full((batch_size, self.num_classes), 1.0 / self.num_classes)

        probs = self.blend_weight * probs_c + (1.0 - self.blend_weight) * probs_f
        return probs


model = DummyModel(centroids=centroids_dict, blend_weight=blend_weight)
print("Using enhanced colour‑and‑flat‑centroid dummy model for predictions.")




## === cell 1
test_csv_path = os.path.join(base_dir, "test.csv")
test_img_dir = os.path.join(base_dir, "test_images")

test_csv = pd.read_csv(test_csv_path)
id_codes = test_csv["id_code"].values
test_prediction = np.empty(len(id_codes), dtype="int64")

for i, code in enumerate(tqdm(id_codes, desc="Predicting")):
    img_path = os.path.join(test_img_dir, f"{code}.png")
    img = cv2.imread(img_path)
    if img is None:
        test_prediction[i] = 0
        continue
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = preprocessing(img)
    X = np.expand_dims(img, axis=0)
    pred = model.predict(X)
    test_prediction[i] = np.argmax(pred, axis=1)[0]




## === cell 2
submission = pd.DataFrame({"id_code": id_codes, "diagnosis": test_prediction})
submission_path = "submission.csv"
submission.to_csv(submission_path, index=False)
print(f"Submission saved to {submission_path}")
print(
    "Class distribution:",
    dict(zip(*np.unique(test_prediction, return_counts=True))),
)
