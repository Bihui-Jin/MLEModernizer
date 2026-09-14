# Goal

I want you to fix bugs and increase the score toward a target for a Kaggle competition solution. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (big fix and/or evaluation score improvement); avoid unrelated refactors or stylistic edits.
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

0.00724

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved -0.11833) has done: 'I removed the problematic TensorFlow‑Addons import, guarded the weight‑loading step so missing files don’t crash, and reorganised the notebook into a clean sequence of cells that correctly import TensorFlow, build the EfficientNetB1 model, run inference on the test images and write a valid `submission.csv`. The core model architecture and preprocessing remain unchanged; only the minimal fixes needed for execution and a safe fallback when pretrained weights are unavailable were added.'
- What this solution (achieved 0.0496) has done: 'I wrap the TensorFlow import in a safe try/except block, create the EfficientNet model only when TensorFlow loads correctly, and add a fallback that predicts the most frequent diagnosis from the training set when TensorFlow is unavailable. This removes the import‑time AttributeError, guarantees a valid `submission.csv` is written, and improves the score from a negative value to a reasonable positive baseline while keeping the original model code untouched for environments where TensorFlow works.'
- What this solution (achieved -0.01852) has done: 'Implemented a simple image‑based heuristic to replace the “most common class” fallback.  
When TensorFlow isn’t available, the script now loads the training images, computes the mean pixel intensity for each image, aggregates average intensities per diagnosis class, and predicts each test image by assigning the class whose average intensity is closest to the test image’s intensity. This inexpensive feature‑based approach substantially raises the expected QWK score while preserving the original structure and fallback behavior.'
- What this solution (achieved -0.01115) has done: 'I add an environment fix for the protobuf issue before importing TensorFlow and replace the simple mean‑intensity fallback with a lightweight nearest‑centroid classifier that uses each image’s mean RGB values. This keeps the original model code untouched, fixes the import error, and provides a more informative heuristic that should raise the QWK score toward the target while still producing a valid `submission.csv`.'
- What this solution (achieved 0.00724) has done: 'Implemented a more informative fallback classifier for environments without TensorFlow. The new heuristic computes per‑class average pre‑processed images from the training set and predicts each test image by finding the nearest class‑average image using L2 distance, which is far richer than the previous mean‑RGB centroid approach. This change fixes the earlier import error path and is expected to raise the QWK score toward the target while keeping the original model pipeline unchanged for TF‑enabled runs.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import numpy as np
import pandas as pd
import cv2
import gc
from tqdm import tqdm

try:
    import tensorflow as tf
    from keras.models import Sequential
    from keras.layers import Flatten, Dense, Dropout

    TF_AVAILABLE = True
except Exception as e:
    print(f"TensorFlow import failed ({e}); will use fallback baseline.")
    tf = None
    TF_AVAILABLE = False




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

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


def preprocessing(image, sigmaX=10):
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    image = crop_image_from_gray(image).astype("uint8")
    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    image = cv2.addWeighted(image, 4, cv2.GaussianBlur(image, (0, 0), sigmaX), -4, 128)
    return image.astype("float32") / 255.0




## === cell 2
if TF_AVAILABLE:
    base_model = tf.keras.applications.efficientnet.EfficientNetB1(
        include_top=False, weights="imagenet", input_shape=(IMG_SIZE, IMG_SIZE, 3)
    )
    flatten_layer = Flatten()
    dense_layer_1 = Dense(4096, activation="relu")
    dropout_1 = Dropout(0.6)
    dense_layer_2 = Dense(2048, activation="relu")
    dropout_2 = Dropout(0.5)
    dense_layer_3 = Dense(1024, activation="relu")
    dropout_3 = Dropout(0.3)
    dense_layer_4 = Dense(512, activation="relu")
    prediction_layer = Dense(5, activation="softmax")

    model = Sequential(
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
else:
    model = None  # placeholder when TF is not available




## === cell 3
if TF_AVAILABLE:
    weights_path = "../input/eff-b1-model-224/eff_b1_model_224.h5"
    if os.path.exists(weights_path):
        try:
            model.load_weights(weights_path)
            print("Loaded pretrained weights.")
        except Exception as e:
            print(
                f"Warning: could not load weights ({e}); proceeding with ImageNet backbone."
            )
    else:
        print("Pretrained weights not found; using ImageNet backbone only.")
else:
    print("Skipping weight loading – TensorFlow not available.")




## === cell 4
test_csv = pd.read_csv("../input/aptos2019-blindness-detection/sample_submission.csv")
id_codes = test_csv["id_code"].values
test_prediction = np.empty(len(id_codes), dtype="int64")

if TF_AVAILABLE and model is not None:
    for idx, code in enumerate(tqdm(id_codes, desc="Predicting")):
        img_path = f"../input/aptos2019-blindness-detection/test_images/{code}.png"
        img = cv2.imread(img_path)
        if img is None:
            img = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype="uint8")
        img = load_ben_color(img)
        X = np.expand_dims(img, axis=0)  # shape (1, IMG_SIZE, IMG_SIZE, 3)
        pred = model.predict(X, verbose=0)
        test_prediction[idx] = np.argmax(pred, axis=1)[0]
else:
    train_path = "../input/aptos2019-blindness-detection/train.csv"
    train_img_dir = "../input/aptos2019-blindness-detection/train_images"
    if os.path.exists(train_path):
        train_df = pd.read_csv(train_path)

        class_sum_img = np.zeros((5, IMG_SIZE, IMG_SIZE, 3), dtype="float64")
        class_cnt = np.zeros(5, dtype="int64")

        print("Computing per‑class average images from training data...")
        for _, row in tqdm(
            train_df.iterrows(), total=len(train_df), desc="Training stats"
        ):
            code = row["id_code"]
            label = int(row["diagnosis"])
            img_path = os.path.join(train_img_dir, f"{code}.png")
            img = cv2.imread(img_path)
            if img is None:
                continue
            img = load_ben_color(img)  # (IMG_SIZE, IMG_SIZE, 3), RGB uint8
            img_f = img.astype("float64")
            class_sum_img[label] += img_f
            class_cnt[label] += 1

        global_avg_img = class_sum_img.sum(axis=0) / max(class_cnt.sum(), 1)
        class_avg_img = np.where(
            class_cnt[:, None, None, None] > 0,
            class_sum_img / class_cnt[:, None, None, None],
            global_avg_img,
        )

        print("Predicting test set using nearest‑average‑image heuristic...")
        for idx, code in enumerate(tqdm(id_codes, desc="Predicting")):
            img_path = f"../input/aptos2019-blindness-detection/test_images/{code}.png"
            img = cv2.imread(img_path)
            if img is None:
                img = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype="uint8")
            img = load_ben_color(img).astype("float64")
            dists = np.linalg.norm(class_avg_img - img, axis=(1, 2, 3))
            test_prediction[idx] = int(np.argmin(dists))
    else:
        most_common = 0
        print("Training CSV not found – defaulting fallback prediction to class 0.")
        test_prediction.fill(most_common)




## === cell 5
test_csv["diagnosis"] = test_prediction
submission_path = "submission.csv"
test_csv.to_csv(submission_path, index=False)
print(f"Submission written to {submission_path}")

unique, counts = np.unique(test_prediction, return_counts=True)
print(dict(zip(unique, counts)))
print("Done!")
