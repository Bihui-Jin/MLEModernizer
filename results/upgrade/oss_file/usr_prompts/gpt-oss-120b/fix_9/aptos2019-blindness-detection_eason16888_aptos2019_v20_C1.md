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

0.6376212321223429

# 6. Current score

0.48102

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'The fixes remove the problematic TensorFlow‑Addons installation, correctly import keras, add safe model loading with a fallback dummy model, ensure pandas (`pd`) and other libraries are available, and guard against missing images. This produces a valid `submission.csv` file without runtime errors.'
- What this solution (achieved 0.0) has done: 'I fixed the incorrect file paths, added safeguards for missing directories, and ensured that a dummy model is always available when TensorFlow cannot be loaded. The script now reads the proper `test.csv`, processes each image safely, generates predictions, and writes a correctly‑named `submission.csv` that Kaggle accept. These changes resolve the runtime errors and guarantee a valid submission file.'
- What this solution (achieved 0.03209) has done: 'Implemented fixes to eliminate TensorFlow import errors by skipping TF loading entirely and setting `tf_available=False`. Added a lightweight heuristic‑based `DummyModel` that derives a class prediction from each image’s mean intensity, producing a one‑hot probability vector instead of uniform scores. This provides varied predictions, improving the Quadratic Weighted Kappa score toward the target while preserving the overall pipeline and submission generation.'
- What this solution (achieved 0.04685) has done: 'I add a lightweight data‑driven heuristic to the dummy model: compute per‑class mean‑and‑std intensity centroids from the training images and, at prediction time, assign each test image to the nearest centroid (using Euclidean distance). This keeps the overall pipeline and dummy‑model structure intact while giving the model a bit of learned information, which should raise the Quadratic Weighted Kappa score toward the target. I also simplify the path handling so the test‑prediction cell reuses the already‑determined `base_dir`.'
- What this solution (achieved 0.0) has done: 'I enhance the lightweight heuristic by using per‑channel mean colour information instead of a single overall mean‑and‑std vector. This gives the dummy model a richer representation of each image while preserving its overall structure, so the predictions become more discriminative and the quadratic weighted kappa should move closer to the target score.'
- What this solution (achieved 0.23142) has done: 'I make the base‑directory detection robust and ensure that the same preprocessing used for computing the colour‑centroids is also applied when generating predictions. This lets the dummy model work with reliable centroids and more consistent image features, which should raise the quadratic weighted kappa score toward the target without altering the overall pipeline.'
- What this solution (achieved 0.48102) has done: 'I enhance the dummy model by adding per‑class mean image (flattened) centroids in addition to the existing colour‑channel centroids, and combine both distance‑based probability estimates. This richer representation should give more discriminative predictions and raise the quadratic weighted kappa score toward the target while preserving the overall pipeline.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd
import cv2
from tqdm import tqdm

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


def compute_class_centroids(base_dir):
    """
    Compute per‑class centroids:
      * colour_centroids: mean R,G,B values
      * flat_centroids: mean flattened pre‑processed image
    """
    train_csv_path = os.path.join(base_dir, "train.csv")
    train_img_dir = os.path.join(base_dir, "train_images")
    if not os.path.isfile(train_csv_path) or not os.path.isdir(train_img_dir):
        return {"colour": None, "flat": None}

    train_df = pd.read_csv(train_csv_path)

    colour_sums = np.zeros((5, 3), dtype=np.float64)
    flat_sums = np.zeros((5, IMG_SIZE * IMG_SIZE * 3), dtype=np.float64)
    counts = np.zeros(5, dtype=int)

    for _, row in tqdm(train_df.iterrows(), desc="Computing centroids"):
        code = row["id_code"]
        label = int(row["diagnosis"])
        img_path = os.path.join(train_img_dir, f"{code}.png")
        img = cv2.imread(img_path)
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = preprocessing(img)  # (H, W, C) float32
        colour_means = img.mean(axis=(0, 1))  # (3,)
        flat_means = img.reshape(-1)  # (H*W*C,)

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


centroids_dict = compute_class_centroids(base_dir)


class DummyModel:
    def __init__(self, num_classes=5, centroids=None):
        self.num_classes = num_classes
        self.colour_centroids = centroids.get("colour") if centroids else None
        self.flat_centroids = centroids.get("flat") if centroids else None

    def _prob_from_distances(self, dists):
        """Convert distances to probabilities."""
        probs = np.exp(-dists)  # (B, C)
        probs_sum = probs.sum(axis=1, keepdims=True)
        probs = np.where(probs_sum == 0, 1.0 / self.num_classes, probs / probs_sum)
        return probs

    def predict(self, x):
        """Return class‑probability vectors for a batch of images."""
        batch_size = x.shape[0]

        colour_feats = x.mean(axis=(1, 2))  # (B, 3)

        if self.colour_centroids is not None:
            diffs_c = (
                colour_feats[:, None, :] - self.colour_centroids[None, :, :]
            )  # (B, C, 3)
            dists_c = np.linalg.norm(diffs_c, axis=2)  # (B, C)
            probs_c = self._prob_from_distances(dists_c)
        else:
            probs_c = np.full((batch_size, self.num_classes), 1.0 / self.num_classes)

        flat_feats = x.reshape(batch_size, -1)  # (B, D)

        if self.flat_centroids is not None:
            diffs_f = (
                flat_feats[:, None, :] - self.flat_centroids[None, :, :]
            )  # (B, C, D)
            dists_f = np.sqrt(np.mean(diffs_f**2, axis=2))  # (B, C)
            probs_f = self._prob_from_distances(dists_f)
        else:
            probs_f = np.full((batch_size, self.num_classes), 1.0 / self.num_classes)

        probs = (probs_c + probs_f) / 2.0
        return probs


model = DummyModel(centroids=centroids_dict)
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
    img = preprocessing(img)  # same preprocessing as used for centroids
    X = np.expand_dims(img, axis=0)  # shape (1, H, W, C)
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
