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

0.8762123503241274

# 6. Current score

-0.17247

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'I fixed the import errors by switching to `tensorflow.keras` instead of the standalone `keras` module, which avoids the protobuf‐related crash. I also updated the Adam optimizer call to use the current‑compatible `learning_rate` argument instead of the deprecated `lr`. The rest of the pipeline is unchanged, so the model, data handling and submission creation remain the same while now running without errors and producing a valid `submission.csv` file.'
- What this solution (achieved 0.0) has done: 'Implemented parallel image loading using ThreadPoolExecutor for both training/validation and test‑time prediction, reducing I/O‑bound preprocessing time. Switched the DenseNet121 backbone to ImageNet‑pretrained weights and frozen its layers, dramatically cutting training compute while keeping the exact architecture and loss unchanged. Added deterministic seeding for reproducibility and minor clean‑ups that preserve all original semantics.'
- What this solution (achieved 0.02214) has done: 'I fixed the TensorFlow import crash by forcing the pure‑Python protobuf implementation, added a robust routine to locate the correct data directory, and guarded all TensorFlow‑dependent code so it falls back to a simple uniform‑probability dummy model when TF cannot be used. These changes let the notebook run end‑to‑end, correctly load the CSV files, train (or skip) the model, generate predictions, and write a valid `submission.csv` that matches the required format.'
- What this solution (achieved -0.04039) has done: 'The fix addresses three runtime errors that prevented the notebook from completing and writing a submission file:

1. **`image` not defined** – import `ImageDataGenerator` from `tensorflow.keras.preprocessing.image` and use it in `dataGenerator`.
2. **`EarlyStopping` / `ReduceLROnPlateau` not defined** – reference the callbacks via `tf.keras.callbacks`.
3. **TensorFlow import handling** – keep the existing safe import, and when TensorFlow is unavailable the fallback scikit‑learn model is used; the corrected imports allow the code to run end‑to‑end and produce a valid `submission.csv`.'
- What this solution (achieved 0.02135) has done: 'Implemented a robust TensorFlow‑fallback that avoids the protobuf crash and replaces the simple LogisticRegression with a balanced, feature‑engineered classifier.  
- Added a lightweight image‑feature extractor (mean & std per colour channel) used by the fallback model.  
- Trained a LogisticRegression with `class_weight='balanced'` on these features, improving predictive power and QWK score.  
- Kept the original TensorFlow path untouched; if TF loads it still be used unchanged.  
- Ensured the submission file is written correctly.'
- What this solution (achieved -0.17247) has done: 'I make the TensorFlow import safe by falling back to the sklearn model if any TensorFlow‑related error occurs, and guard the training call so it only runs when a real Keras model is created. I also enrich the fallback feature extractor (adding per‑channel min/max) to give the LogisticRegression a bit more signal, which should improve the QWK score while keeping the original pipeline untouched.'

# 9. Code solution

## === cell 0
import os
import gc
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import concurrent.futures
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score
from sklearn.exceptions import NotFittedError
from sklearn.model_selection import train_test_split

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

try:
    import tensorflow as tf
except Exception as e:
    print(f"TensorFlow import failed: {e}")
    tf = None

if tf is not None:
    try:
        from tensorflow.keras.preprocessing.image import ImageDataGenerator
    except Exception as e:
        print(f"Keras import failed: {e}")
        tf = None
        ImageDataGenerator = None
else:
    ImageDataGenerator = None

IMG_DIM = 224  # input size for DenseNet121
CHANNELS = 3
NUM_CLASSES = 5
BATCH_SIZE = 32
NORMAL_WEIGHTS = None  # No external weight file; will train from ImageNet backbone


def resolve_input_folder():
    """Find the correct path to the competition data."""
    candidates = [
        "./data/aptos2019-blindness-detection/",
        "/kaggle/input/aptos2019-blindness-detection/",
        "/kaggle/working/aptos2019-blindness-detection/",
    ]
    for p in candidates:
        if os.path.isdir(p):
            return p
    raise FileNotFoundError(
        "Unable to locate the aptos2019-blindness-detection data folder."
    )


INPUT_FOLDER = resolve_input_folder()
print(f"Using INPUT_FOLDER = {INPUT_FOLDER}")




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
def crop(gray, img, percent_smaller):
    thresh = 8
    top = 0
    left = 0
    bottom = gray.shape[0] - 1
    right = gray.shape[1] - 1

    middleCol = gray[:, int(gray.shape[1] / 2)] > thresh
    while middleCol[top] == 0:
        top += 1
    while middleCol[bottom] == 0:
        bottom -= 1

    middleRow = gray[int(gray.shape[0] / 2)] > thresh
    while middleRow[left] == 0:
        left += 1
    while middleRow[right] == 0:
        right -= 1

    height = bottom - top
    width = right - left

    bottom -= int(percent_smaller * height)
    top += int(percent_smaller * height)
    right -= int(percent_smaller * width)
    left += int(percent_smaller * width)

    if height < 100 or width < 100:
        print("Error: squareUp: bottom:", bottom, "top:", top)
        print("Error: squareUp: right:", right, "left:", left)
        return img

    return img[top:bottom, left:right]


def benYCC(bgr, weight=4, gamma=20):
    ycc = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycc)
    y = cv2.addWeighted(y, weight, cv2.GaussianBlur(y, (0, 0), gamma), -weight, 128)
    ycc_modified = cv2.merge((y, cr, cb))
    return cv2.cvtColor(ycc_modified, cv2.COLOR_YCrCb2BGR)


def benSimple(img, weight=4, gamma=20):
    return cv2.addWeighted(
        img, weight, cv2.GaussianBlur(img, (0, 0), gamma), -weight, 128
    )


def reflectAndSquareUp(img):
    height = img.shape[0]
    width = img.shape[1]
    if height > width:
        offset = int((height - width) / 2)
        return img[offset : offset + width]
    else:
        if len(img.shape) == 3:
            new_img = np.zeros((width, width, img.shape[2]), np.uint8)
        else:
            new_img = np.zeros((width, width), np.uint8)

        h1 = int((width - height) / 2)
        h2 = h1 + height
        new_img[h1:h2, :] = img
        for i in range(h1):
            new_img[h1 - i] = img[i]
        for i in range(width - h2):
            new_img[h2 + i] = img[height - i - 1]
        return new_img


def adjust_gamma(image, gamma=1.0):
    invGamma = 1.0 / gamma
    table = np.array(
        [((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]
    ).astype("uint8")
    return cv2.LUT(image, table)


def processBenNormal(bgr):
    green = bgr[:, :, 1]  # use green channel as greyscale
    if bgr.shape != (480, 640, 3):
        cropped = crop(green, bgr, 0.02)
        width = int(cropped.shape[1] * 0.9)
        height = int(width * 480 / 640)
        if height > cropped.shape[0]:
            height = cropped.shape[0] - 2
        h = int((cropped.shape[0] - height) / 2)
        w = int((cropped.shape[1] - width) / 2)
        test_crop = cropped[h : height + h, w : width + w, :]
    else:
        test_crop = bgr

    reflected = reflectAndSquareUp(test_crop)
    resized = cv2.resize(reflected, (IMG_DIM, IMG_DIM), interpolation=cv2.INTER_AREA)
    equalised = adjust_gamma(resized, 1 + np.log(90) - np.log(np.median(resized)))
    bens = benYCC(equalised, weight=3, gamma=20)
    return cv2.cvtColor(bens, cv2.COLOR_BGR2RGB)


pre_process_function = processBenNormal




## === cell 2
def dataGenerator(jitter=0.1):
    """Create an ImageDataGenerator with optional jitter for augmentation."""
    datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        horizontal_flip=(jitter > 0.01),
        vertical_flip=(jitter > 0.01),
        zoom_range=[max(0.8, 1 - 5 * jitter), 1],
        rotation_range=int(60 * jitter),  # reduced from 600*jitter
        brightness_range=[1 - jitter / 3, 1 + jitter / 3],
        fill_mode="mirror",
    )
    return datagen




## === cell 3
def _load_and_process(path):
    bgr = cv2.imread(path)
    if bgr is None:
        return np.full((IMG_DIM, IMG_DIM, CHANNELS), 128, dtype=np.uint8)
    return processBenNormal(bgr)


def load_image_block(df, images_dir):
    """Parallel load & preprocess images matching df order."""
    paths = [os.path.join(images_dir, fname) for fname in df.id_code]
    n = len(paths)
    block = np.empty((n, IMG_DIM, IMG_DIM, CHANNELS), dtype=np.uint8)

    with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        for i, img in enumerate(executor.map(_load_and_process, paths)):
            block[i] = img
    return block


print("Loading training metadata...")
train_df = pd.read_csv(os.path.join(INPUT_FOLDER, "train.csv"))
train_df.id_code = train_df.id_code.apply(lambda x: x + ".png")

train_meta, val_meta = train_test_split(
    train_df, test_size=0.1, stratify=train_df.diagnosis, random_state=42
)

print("Loading training images...")
train_images = load_image_block(train_meta, os.path.join(INPUT_FOLDER, "train_images/"))
print("Loading validation images...")
val_images = load_image_block(val_meta, os.path.join(INPUT_FOLDER, "train_images/"))

train_labels = pd.get_dummies(train_meta.diagnosis).values
val_labels = pd.get_dummies(val_meta.diagnosis).values




## === cell 4
def DummyModel(class_probs):
    class Dummy:
        def __init__(self, probs):
            self.probs = probs

        def predict(self, x, **kwargs):
            batch = x.shape[0]
            return np.tile(self.probs, (batch, 1))

        def fit(self, *args, **kwargs):
            pass

    return Dummy(class_probs)


def _extract_features(images):
    """
    Enhanced feature extractor for the sklearn fallback:
    For each image compute per‑channel mean, std, min, and max (12 values total).
    """
    means = images.mean(axis=(1, 2))
    stds = images.std(axis=(1, 2))
    mins = images.min(axis=(1, 2))
    maxs = images.max(axis=(1, 2))
    return np.concatenate([means, stds, mins, maxs], axis=1)


def create_model(weights_path=None):
    """
    Returns a model with a unified interface:
    - If TensorFlow is functional, returns a compiled Keras model.
    - Otherwise, trains a lightweight LogisticRegression on simple image statistics
      and returns a wrapper exposing a `predict` method returning class probabilities.
    """
    if tf is not None:
        try:
            from tensorflow.keras.utils import to_categorical
            from tensorflow.keras.applications import DenseNet121
            from tensorflow.keras.models import Sequential
            from tensorflow.keras.layers import GlobalAveragePooling2D, Dropout, Dense
            from tensorflow.keras.optimizers import Adam

            model = Sequential()
            backbone = DenseNet121(
                weights="imagenet",
                include_top=False,
                input_shape=(IMG_DIM, IMG_DIM, CHANNELS),
            )
            backbone.trainable = False
            model.add(backbone)
            model.add(GlobalAveragePooling2D())
            model.add(Dropout(0.5))
            model.add(Dense(NUM_CLASSES, activation="softmax"))

            if weights_path is not None and os.path.exists(weights_path):
                model.load_weights(weights_path)

            model.compile(
                optimizer=Adam(learning_rate=0.00005),
                loss="categorical_crossentropy",
                metrics=["accuracy"],
            )
            return model
        except Exception as e:
            print(f"TensorFlow model creation failed ({e}); falling back to sklearn.")

    print(
        "TensorFlow unavailable or failed – training enhanced LogisticRegression fallback model."
    )
    X = _extract_features(train_images)
    y = train_meta.diagnosis.values

    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            multi_class="multinomial",
            max_iter=3000,
            n_jobs=os.cpu_count(),
            solver="lbfgs",
            class_weight="balanced",
        ),
    )
    clf.fit(X, y)

    class SklearnWrapper:
        def __init__(self, model):
            self.model = model

        def predict(self, x, **kwargs):
            xb = _extract_features(x)
            probs = self.model.predict_proba(xb)
            return probs

        def fit(self, *args, **kwargs):
            pass  # already fitted

    return SklearnWrapper(clf)


model = create_model(NORMAL_WEIGHTS)

print("Starting model training...")
if tf is not None and hasattr(model, "fit") and not isinstance(model, type):
    try:
        model.fit(
            train_images,
            train_labels,
            validation_data=(val_images, val_labels),
            epochs=1,
            batch_size=BATCH_SIZE,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(patience=2, restore_best_weights=True),
                tf.keras.callbacks.ReduceLROnPlateau(
                    patience=1, factor=0.5, min_lr=1e-7
                ),
            ],
            verbose=1,
        )
    except Exception as e:
        print(f"Keras training failed ({e}); skipping training.")
else:
    print(
        "Skipped Keras training because TensorFlow is unavailable or fallback model is in use."
    )




## === cell 5
def make_predictions(d_set, processing_function, model, jitters=5):
    images_dir = os.path.join(INPUT_FOLDER, f"{d_set}_images/")
    df = pd.read_csv(os.path.join(INPUT_FOLDER, f"{d_set}.csv"))
    df.id_code = df.id_code.apply(lambda x: x + ".png")

    block_size = 512
    total = df.shape[0]
    predictions = np.zeros((total, NUM_CLASSES))

    print(f"Making predictions on the {d_set} dataset. Total: {total}")

    for start in range(0, total, block_size):
        end = min(start + block_size, total)
        batch_len = end - start
        batch_df = df.iloc[start:end]

        paths = [os.path.join(images_dir, fname) for fname in batch_df.id_code]
        img_block = np.empty((batch_len, IMG_DIM, IMG_DIM, CHANNELS), dtype=np.uint8)
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=os.cpu_count()
        ) as executor:
            for i, img in enumerate(executor.map(_load_and_process, paths)):
                img_block[i] = img

        if tf is None:
            predictions[start:end] = model.predict(img_block)
            print(f"{start} - {end} finished (fallback model)")
            continue

        prediction_jitters = np.zeros((batch_len, jitters, NUM_CLASSES))
        jit = 0.0
        for j in range(jitters):
            datagen = dataGenerator(jit).flow(
                img_block, batch_size=BATCH_SIZE, shuffle=False
            )
            preds = model.predict(datagen, steps=len(datagen), verbose=0)
            prediction_jitters[:, j, :] = preds
            gc.collect()
            jit += 0.02

        predictions[start:end] = np.median(prediction_jitters, axis=1)
        print(f"{start} - {end} finished")
        gc.collect()

    return predictions




## === cell 6
def prediction_convert(predictions):
    """Convert probability predictions to class labels by taking argmax."""
    return np.argmax(predictions, axis=1)




## === cell 7
preds = make_predictions("test", processBenNormal, model)

test_classes = prediction_convert(preds)
print("First 10 predictions:", test_classes[:10])

test_df = pd.read_csv(os.path.join(INPUT_FOLDER, "test.csv"))
test_df["diagnosis"] = test_classes
submission_path = "submission.csv"
test_df.to_csv(submission_path, index=False)
print(f"Submission file written to {submission_path}")
