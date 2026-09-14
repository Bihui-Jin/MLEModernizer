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

0.7657290256664656

# 6. Current score

0.1835

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved -0.05659) has done: 'I add the missing `os` and `tqdm` imports, guard the TensorFlow import so that a simple dummy model is used if TensorFlow cannot be loaded, and adjust the model‑creation cell to fall back to this dummy model. This removes the import‑time errors, ensures all variables (`os`, `tqdm`, `id_code`) are defined, and guarantees a valid `submission.csv` is written.'
- What this solution (achieved 0.12789) has done: 'I removed the failing TensorFlow import and replaced it with a lightweight, intensity‑based fallback model that computes average pixel intensity per class from the training set and predicts the class whose mean intensity is closest to the test image’s intensity. This eliminates the protobuf error, ensures a valid `submission.csv` is written, and provides a simple heuristic that should raise the quadratic weighted kappa far above the previous –0.056 score.'
- What this solution (achieved 0.23043) has done: 'I enhance the fallback intensity‑based model by using a three‑channel (R,G,B) mean vector for each class instead of a single grayscale intensity. This retains the same overall structure (still runs when TensorFlow is unavailable) but provides a richer feature for comparison, which should raise the quadratic weighted kappa toward the target without altering any core architecture or training logic.'
- What this solution (achieved 0.20758) has done: 'The changes vectorize the RGB‑mean model’s prediction (removing the per‑image Python loop) and process test images in batches instead of one‑by‑one, dramatically reducing Python overhead while keeping the exact same mathematical operations and outputs.'
- What this solution (achieved 0.18608) has done: 'I augment the fallback model by also learning class‑wise grayscale intensity statistics and combine the RGB‑mean Mahalanobis distance with a grayscale‑mean distance (simple weighted sum). This adds only a few extra calculations during the training‐phase statistics gathering and a tiny change in the `predict` method, preserving the overall architecture while providing a richer feature for discrimination, which should raise the quadratic weighted kappa toward the target score.'
- What this solution (achieved 0.21369) has done: 'The changes keep the same fallback‑model structure but make its distance metric more stable and give the grayscale feature a stronger influence (α = 1.0).  
Instead of using per‑class variance (which can be noisy with few samples), we replace it with the overall average variance for RGB channels and for gray, then broadcast this value to all classes. This small adjustment is expected to improve class discrimination and raise the quadratic weighted kappa toward the target while preserving the original logic.'
- What this solution (achieved 0.1835) has done: 'I enhance the lightweight fallback model by using per‑class variance for the RGB and grayscale features (instead of a single global variance) and reduce the gray‑distance weight (α) so the RGB mean similarity plays a larger role. These modest statistical upgrades keep the original architecture unchanged while giving the model a clearer discriminative signal, which should raise the quadratic weighted kappa toward the target score.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd
import cv2
from tqdm import tqdm

tf = None



## === cell 1
"""
    Config and preprocessing utilities
"""
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


def preprocessing(image, sigmaX=10):
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    image = crop_image_from_gray(image).astype("uint8")
    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    image = cv2.addWeighted(image, 4, cv2.GaussianBlur(image, (0, 0), sigmaX), -4, 128)
    return image.astype("float32") / 255.0




## === cell 2
if tf is not None:
    from tensorflow.keras.layers import Flatten, Dense, Dropout
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.applications import EfficientNetB2

    try:
        base_model = EfficientNetB2(
            include_top=False,
            weights="imagenet",
            input_shape=(IMG_SIZE, IMG_SIZE, 3),
        )
    except Exception:
        base_model = EfficientNetB2(
            include_top=False, weights=None, input_shape=(IMG_SIZE, IMG_SIZE, 3)
        )

    flatten_layer = Flatten()
    dense_layer_1 = Dense(4096, activation="relu")
    Dropout_1 = Dropout(0.6)
    dense_layer_2 = Dense(2048, activation="relu")
    Dropout_2 = Dropout(0.6)
    dense_layer_3 = Dense(1024, activation="relu")
    prediction_layer = Dense(5, activation="softmax")

    model = Sequential(
        [
            base_model,
            flatten_layer,
            dense_layer_1,
            Dropout_1,
            dense_layer_2,
            Dropout_2,
            dense_layer_3,
            prediction_layer,
        ]
    )

    weight_path = "../input/eff-b2-model/eff_b2_model.h5"
    if os.path.exists(weight_path):
        try:
            model.load_weights(weight_path)
            print("Loaded pretrained weights.")
        except Exception as e:
            print(f"Could not load weights from {weight_path}: {e}")
    else:
        print("Weight file not found; using randomly initialized model.")
else:
    train_csv_path = "../input/aptos2019-blindness-detection/train.csv"
    train_df = pd.read_csv(train_csv_path)

    class_sums_rgb = np.zeros((5, 3), dtype=np.float64)
    class_sq_sums_rgb = np.zeros((5, 3), dtype=np.float64)
    class_counts = np.zeros(5, dtype=np.int64)

    class_sums_gray = np.zeros(5, dtype=np.float64)
    class_sq_sums_gray = np.zeros(5, dtype=np.float64)

    for _, row in train_df.iterrows():
        img_path = (
            f"../input/aptos2019-blindness-detection/train_images/{row['id_code']}.png"
        )
        img = cv2.imread(img_path)
        if img is None:
            continue
        img = load_ben_color(img)  # RGB uint8, size IMG_SIZE×IMG_SIZE

        mean_rgb = img.mean(axis=(0, 1))  # (3,)
        sq_mean_rgb = (img.astype(np.float32) ** 2).mean(axis=(0, 1))  # (3,)

        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        mean_gray = gray.mean()
        sq_mean_gray = (gray.astype(np.float32) ** 2).mean()

        label = int(row["diagnosis"])
        class_sums_rgb[label] += mean_rgb
        class_sq_sums_rgb[label] += sq_mean_rgb
        class_sums_gray[label] += mean_gray
        class_sq_sums_gray[label] += sq_mean_gray
        class_counts[label] += 1

    class_means_rgb = np.divide(
        class_sums_rgb,
        class_counts[:, None],
        where=class_counts[:, None] != 0,
    )  # (5,3)

    class_means_gray = np.divide(
        class_sums_gray,
        class_counts,
        where=class_counts != 0,
    )  # (5,)

    class_vars_rgb = np.maximum(
        class_sq_sums_rgb
        / np.where(class_counts[:, None] != 0, class_counts[:, None], 1)
        - class_means_rgb**2,
        1e-6,
    )  # (5,3)

    class_vars_gray = np.maximum(
        class_sq_sums_gray / np.where(class_counts != 0, class_counts, 1)
        - class_means_gray**2,
        1e-6,
    )  # (5,)

    class SimpleRGBMeanModel:
        """
        Predict by nearest class mean using per‑class inverse‑variance weighting.
        Combines RGB Mahalanobis‑like distance with a grayscale intensity distance.
        """

        def __init__(
            self,
            class_means_rgb,
            class_vars_rgb,
            class_means_gray,
            class_vars_gray,
            alpha=0.2,  # lower weight for gray distance, letting RGB dominate
        ):
            self.class_means_rgb = class_means_rgb  # (5,3)
            self.class_vars_rgb = class_vars_rgb  # (5,3)
            self.class_means_gray = class_means_gray  # (5,)
            self.class_vars_gray = class_vars_gray  # (5,)
            self.alpha = alpha

        def predict(self, X, verbose=0):
            """
            X: ndarray of shape (batch, IMG_SIZE, IMG_SIZE, 3), dtype uint8
            Returns: probability array of shape (batch, 5) with a single 1.0
                     at the nearest‑class index.
            """
            batch = X.shape[0]

            rgb_means = X.mean(axis=(1, 2))  # (batch, 3)

            gray_means = (
                0.2989 * X[:, :, :, 0] + 0.5870 * X[:, :, :, 1] + 0.1140 * X[:, :, :, 2]
            ).mean(
                axis=(1, 2)
            )  # (batch,)

            diff_rgb = (
                self.class_means_rgb[None, :, :] - rgb_means[:, None, :]
            )  # (batch,5,3)
            dists_rgb = np.sqrt(
                np.sum((diff_rgb**2) / self.class_vars_rgb[None, :, :], axis=2)
            )  # (batch,5)

            diff_gray = (
                self.class_means_gray[None, :] - gray_means[:, None]
            )  # (batch,5)
            dists_gray = np.sqrt(
                (diff_gray**2) / self.class_vars_gray[None, :]
            )  # (batch,5)

            total_dist = dists_rgb + self.alpha * dists_gray

            pred_idx = np.argmin(total_dist, axis=1)  # (batch,)
            probs = np.zeros((batch, 5), dtype=np.float32)
            probs[np.arange(batch), pred_idx] = 1.0
            return probs

    model = SimpleRGBMeanModel(
        class_means_rgb,
        class_vars_rgb,
        class_means_gray,
        class_vars_gray,
        alpha=0.2,
    )
    print(
        "TensorFlow not available – using per‑class variance RGB‑mean + gray‑intensity model for predictions."
    )



## === cell 3
test_csv_path = "../input/aptos2019-blindness-detection/test.csv"
if not os.path.exists(test_csv_path):
    test_csv_path = (
        "../input/aptos2019-blindness-detection/sample_submission.csv"  # fallback
    )
test_csv = pd.read_csv(test_csv_path)
id_code = test_csv["id_code"].values
test_prediction = np.empty(len(id_code), dtype="int64")



## === cell 4
batch_imgs = []
batch_indices = []

for i, img_id in enumerate(tqdm(id_code, desc="Predicting")):
    img_path = f"../input/aptos2019-blindness-detection/test_images/{img_id}.png"
    img = cv2.imread(img_path)
    if img is None:
        img = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype="uint8")
    img = load_ben_color(img)
    batch_imgs.append(img)
    batch_indices.append(i)

    if len(batch_imgs) == BATCH_SIZE:
        X_batch = np.stack(batch_imgs)  # (BATCH_SIZE, H, W, 3)
        preds = model.predict(X_batch, verbose=0)  # vectorized
        test_prediction[batch_indices] = np.argmax(preds, axis=1).astype("int64")
        batch_imgs.clear()
        batch_indices.clear()

if batch_imgs:
    X_batch = np.stack(batch_imgs)
    preds = model.predict(X_batch, verbose=0)
    test_prediction[batch_indices] = np.argmax(preds, axis=1).astype("int64")
    batch_imgs.clear()
    batch_indices.clear()



## === cell 5
submission = pd.DataFrame({"id_code": id_code, "diagnosis": test_prediction})
submission.to_csv("submission.csv", index=False)
unique, counts = np.unique(test_prediction, return_counts=True)
print(dict(zip(unique, counts)))
print("Done!")
