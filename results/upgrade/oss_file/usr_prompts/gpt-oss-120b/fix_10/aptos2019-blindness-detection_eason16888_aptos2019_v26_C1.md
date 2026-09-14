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

0.8452750103900721

# 6. Current score

0.73458

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved -0.0033) has done: 'I remove the unnecessary tensorflow‑addons import (which caused the protobuf error), ensure TensorFlow is imported before it is used, guard the weight loading in case the file is missing, and fix the undefined‑variable errors so the script runs end‑to‑end and writes a proper submission.csv file.'
- What this solution (achieved 0.0668) has done: 'I add a small environment fix for TensorFlow import, guard the model creation/loading inside a try‑except, compute the most‑common diagnosis from the training set and fall back to that simple baseline when the TensorFlow model cannot be used. This eliminates the runtime error, guarantees a valid submission.csv, and replaces the random predictions with a reasonable constant prediction, moving the score from a negative value toward the target.'
- What this solution (achieved -0.02643) has done: 'I replace the constant fallback prediction with a simple intensity‑based heuristic that uses the training images to compute a mean‑pixel intensity centroid for each diagnosis class. When the TensorFlow model cannot be loaded, the script now assigns each test image to the class whose centroid is closest to the image’s mean intensity, which yields a more informed baseline and improves the quadratic weighted kappa score while keeping the original model architecture unchanged.'
- What this solution (achieved -0.00029) has done: 'I wrapped the TensorFlow import in a safe try/except so the script can continue when TF isn’t available, moved model construction inside a TF‑available guard, and replaced the simple intensity‑centroid fallback with a slightly richer heuristic that uses both mean and standard‑deviation of image intensities per class (still lightweight and respects the original baseline logic). These changes eliminate the protobuf import error, guarantee a valid submission.csv is produced, and give a modest but meaningful boost toward the target score.'
- What this solution (achieved -0.09969) has done: 'I keep the existing TensorFlow fallback logic but replace the simple intensity‑centroid heuristic with a lightweight scikit‑learn classifier that uses per‑channel mean and standard‑deviation features extracted from the pre‑processed images. This adds a more informative model without altering the core architecture, fixes the earlier protobuf import error by always handling the TensorFlow import safely, and produces a valid submission.csv file. The changes are confined to importing sklearn, defining a feature extractor, training the fallback classifier, and using it for predictions when the TensorFlow model isn’t available.'
- What this solution (achieved 0.73465) has done: 'I remove the TensorFlow import (which raises a protobuf error) and force the script to use the fallback classifier. I enrich the fallback feature extractor to include RGB, HSV, and Lab channel statistics plus a short intensity histogram, and I strengthen the classifier by using a larger RandomForest with balanced class weights. These changes fix the runtime failure and give the model more discriminative power, moving the quadratic weighted kappa score toward the target while keeping the overall pipeline unchanged.'
- What this solution (achieved 5e-05) has done: 'The changes parallelize image loading and feature extraction for both training and test sets using a thread pool, and batch‑predict the test features instead of looping per image. This reduces I/O‑bound waiting and removes per‑sample prediction overhead while keeping every computation (pre‑processing, feature vector, model) identical, so the final predictions and model training remain unchanged.'
- What this solution (achieved 0.73458) has done: 'I fix the AttributeError caused by treating tuples returned from `itertuples` as objects. The training loop now correctly unpack the tuple fields (`id_code` and `diagnosis`) when submitting jobs to the thread pool. This change restores proper feature extraction and model training, enabling the fallback ExtraTrees classifier to generate meaningful predictions and improve the score toward the target.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import numpy as np
import pandas as pd
import cv2
import gc
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

tf = None
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing




## === cell 1
"""
    Config
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




## === cell 2
train_csv_path = "../input/aptos2019-blindness-detection/train.csv"
train_df = pd.read_csv(train_csv_path)
fallback_class = int(train_df["diagnosis"].mode()[0])  # most common label as integer




## === cell 3
if tf is not None:
    try:
        base_model = tf.keras.applications.EfficientNetB2(
            include_top=False,
            weights="imagenet",
            input_shape=(IMG_SIZE, IMG_SIZE, 3),
        )

        flatten_layer = tf.keras.layers.Flatten()
        dense_layer_1 = tf.keras.layers.Dense(4096, activation="relu")
        dropout_1 = tf.keras.layers.Dropout(0.6)
        dense_layer_2 = tf.keras.layers.Dense(2048, activation="relu")
        dropout_2 = tf.keras.layers.Dropout(0.5)
        dense_layer_3 = tf.keras.layers.Dense(1024, activation="relu")
        dropout_3 = tf.keras.layers.Dropout(0.3)
        dense_layer_4 = tf.keras.layers.Dense(512, activation="relu")
        prediction_layer = tf.keras.layers.Dense(5, activation="softmax")

        model = tf.keras.Sequential(
            [
                base_model,
                flatten_layer,
                dense_layer_1,
                dropout_1,
                dense_layer_2,
                dropout_2,
                dense_layer_3,
                dropout_3,
                dense_layer_4,
                prediction_layer,
            ]
        )

        weight_path = "../input/eff-balance-b1-model-224/eff_balance_b1_model_224.h5"
        if os.path.exists(weight_path):
            try:
                model.load_weights(weight_path)
                print("Loaded pretrained weights.")
            except Exception as e:
                print(f"Could not load weights ({e}); proceeding with ImageNet init.")
        else:
            print("Weight file not found – using ImageNet initialization.")
    except Exception as e:
        print(
            f"Model construction failed ({e}); will use fallback constant prediction."
        )
        model = None
else:
    model = None




## === cell 4
test_csv_path = "../input/aptos2019-blindness-detection/sample_submission.csv"
test_csv = pd.read_csv(test_csv_path)
id_codes = test_csv["id_code"].values
test_prediction = np.empty(len(id_codes), dtype="int64")




## === cell 5
test_img_dir = "../input/aptos2019-blindness-detection/test_images"


def extract_features(img):
    """
    Returns a richer feature vector:
    - RGB channel mean & std (6)
    - HSV channel mean & std (6)
    - Lab channel mean & std (6)
    - 32‑bin grayscale histogram (32)
    - Laplacian variance (1)
    - Sobel edge mean (1)
    Total length = 52
    """
    img_float = img.astype("float32")

    rgb_mean = img_float.mean(axis=(0, 1))
    rgb_std = img_float.std(axis=(0, 1))

    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype("float32")
    hsv_mean = hsv.mean(axis=(0, 1))
    hsv_std = hsv.std(axis=(0, 1))

    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB).astype("float32")
    lab_mean = lab.mean(axis=(0, 1))
    lab_std = lab.std(axis=(0, 1))

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    hist = cv2.calcHist([gray], [0], None, [32], [0, 256]).flatten()
    hist = hist / hist.sum()  # normalize

    lap = cv2.Laplacian(gray, cv2.CV_64F)
    lap_var = lap.var()

    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    sobel_mag = np.sqrt(sobelx**2 + sobely**2)
    sobel_mean = sobel_mag.mean()

    return np.concatenate(
        [
            rgb_mean,
            rgb_std,
            hsv_mean,
            hsv_std,
            lab_mean,
            lab_std,
            hist,
            np.array([lap_var, sobel_mean], dtype="float32"),
        ]
    )


if model is None:
    train_img_dir = "../input/aptos2019-blindness-detection/train_images"

    def _process_train(idx, code, label):
        img_path = os.path.join(train_img_dir, f"{code}.png")
        img = cv2.imread(img_path)
        if img is None:
            return None  # skip missing
        img = load_ben_color(img)
        feats = extract_features(img)
        return (idx, feats, label)

    max_workers = multiprocessing.cpu_count()
    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(_process_train, i, code, int(label))
            for i, (code, label) in enumerate(
                train_df.itertuples(index=False, name=None)
            )
        ]
        for f in as_completed(futures):
            res = f.result()
            if res is not None:
                results.append(res)

        if not results:
            print("No training images loaded; using constant fallback.")
            test_prediction[:] = fallback_class
        else:
            results.sort(key=lambda x: x[0])
            X_train = np.stack([r[1] for r in results])
            y_train = np.array([r[2] for r in results])

            clf = make_pipeline(
                StandardScaler(),
                ExtraTreesClassifier(
                    n_estimators=1000,
                    max_depth=None,
                    random_state=42,
                    n_jobs=-1,
                    class_weight="balanced",
                ),
            )
            clf.fit(X_train, y_train)

            def _process_test(idx, code):
                img_path = os.path.join(test_img_dir, f"{code}.png")
                img = cv2.imread(img_path)
                if img is None:
                    img = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype="uint8")
                img = load_ben_color(img)
                feats = extract_features(img)
                return (idx, feats)

            test_futures = [
                executor.submit(_process_test, i, code)
                for i, code in enumerate(id_codes)
            ]
            test_results = []
            for f in as_completed(test_futures):
                test_results.append(f.result())

            test_results.sort(key=lambda x: x[0])
            X_test = np.stack([r[1] for r in test_results])
            preds = clf.predict(X_test)
            test_prediction[:] = preds
            del X_test, test_results, results
            gc.collect()
else:
    for idx, code in enumerate(id_codes):
        img_path = os.path.join(test_img_dir, f"{code}.png")
        img = cv2.imread(img_path)
        if img is None:
            img = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype="uint8")
        img = load_ben_color(img)
        X = np.expand_dims(img, axis=0)  # shape (1,224,224,3)
        pred = model.predict(X, verbose=0)
        test_prediction[idx] = np.argmax(pred, axis=1)[0]




## === cell 6
test_csv["diagnosis"] = test_prediction
submission_path = "submission.csv"
test_csv.to_csv(submission_path, index=False)
print(f"Submission written to {submission_path}")

unique, counts = np.unique(test_prediction, return_counts=True)
print(dict(zip(unique, counts)))
print("Done!")
