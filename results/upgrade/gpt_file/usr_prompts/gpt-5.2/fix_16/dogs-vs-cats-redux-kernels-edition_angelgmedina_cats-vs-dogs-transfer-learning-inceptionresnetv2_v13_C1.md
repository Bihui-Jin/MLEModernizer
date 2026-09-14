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
Given a dataset of images of dogs and cats, predict if an image is a dog or a cat.

## Metric
Log loss.

## Submission Format
For each image in the test set, you must submit a probability that image is a dog. The file should have a header and be in the following format:

```
id,label
1,0.5
2,0.5
3,0.5
...
```

## Dataset
The train folder contains 25,000 images of dogs and cats. Each image in this folder has the label as part of the filename. The test folder contains 12,500 images, named according to a numeric id.

# 2. Python version

3.7

# 3. Installed packages

geopandas==0.14.4
keras==3.8.0
keras-core==0.1.7
keras-cv==0.9.0
keras-hub==0.18.1
keras-nlp==0.18.1
keras-tuner==1.4.7
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
numpy==1.26.4
opencv-python==4.12.0.88
opencv-python-headless==4.12.0.88
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
seaborn==0.12.2
sklearn-pandas==2.2.0
tf_keras==2.18.0

# 4. Data file paths

```
/
    kaggle/
        data/
            cat.1714.jpg (7.8 kB)
            cat.10025.jpg (18.4 kB)
            ... and 24998 other files
            description.md (50 lines)
            sample_submission.csv (2501 lines)
            sample_submission.csv.zip (6.0 kB)
            test.zip (56.6 MB)
            train.zip (513.0 MB)
            dogs-vs-cats-redux-kernels-edition/
                cat.1714.jpg (7.8 kB)
                cat.10025.jpg (18.4 kB)
                ... and 24998 other files
                description.md (50 lines)
                sample_submission.csv (2501 lines)
                sample_submission.csv.zip (6.0 kB)
                test.zip (56.6 MB)
                train.zip (513.0 MB)
                dogs-vs-cats-redux-kernels-edition/
                test/
                    test/
                    unknown/
                        900.jpg (42.3 kB)
                        572.jpg (30.6 kB)
                        ... and 2498 other files
                train/
                    cat/
                        cat.4838.jpg (20.2 kB)
                        cat.1314.jpg (21.7 kB)
                        ... and 11240 other files
                    dog/
                        dog.6712.jpg (35.3 kB)
                        dog.7152.jpg (36.1 kB)
                        ... and 11256 other files
                    train/
            test/
                test/
                unknown/
                    900.jpg (42.3 kB)
                    572.jpg (30.6 kB)
                    ... and 2498 other files
            train/
                cat/
                    cat.4838.jpg (20.2 kB)
                    cat.1314.jpg (21.7 kB)
                    ... and 11240 other files
                dog/
                    dog.6712.jpg (35.3 kB)
                    dog.7152.jpg (36.1 kB)
                    ... and 11256 other files
                train/
        input/
            cat.1714.jpg (7.8 kB)
            cat.10025.jpg (18.4 kB)
            ... and 24998 other files
            description.md (50 lines)
            sample_submission.csv (2501 lines)
            sample_submission.csv.zip (6.0 kB)
            test.zip (56.6 MB)
            train.zip (513.0 MB)
            dogs-vs-cats-redux-kernels-edition/
                cat.1714.jpg (7.8 kB)
                cat.10025.jpg (18.4 kB)
                ... and 24998 other files
                description.md (50 lines)
                sample_submission.csv (2501 lines)
                sample_submission.csv.zip (6.0 kB)
                test.zip (56.6 MB)
                train.zip (513.0 MB)
                dogs-vs-cats-redux-kernels-edition/
                test/
                    test/
                    unknown/
                        900.jpg (42.3 kB)
                        572.jpg (30.6 kB)
                        ... and 2498 other files
                train/
                    cat/
                        cat.4838.jpg (20.2 kB)
                        cat.1314.jpg (21.7 kB)
                        ... and 11240 other files
                    dog/
                        dog.6712.jpg (35.3 kB)
                        dog.7152.jpg (36.1 kB)
                        ... and 11256 other files
                    train/
            test/
                test/
                    test/
                    unknown/
                        900.jpg (42.3 kB)
                        572.jpg (30.6 kB)
                        ... and 2498 other files
                unknown/
                    900.jpg (42.3 kB)
                    572.jpg (30.6 kB)
                    ... and 2498 other files
            train/
                cat/
                    cat.4838.jpg (20.2 kB)
                    cat.1314.jpg (21.7 kB)
                    ... and 11240 other files
                dog/
                    dog.6712.jpg (35.3 kB)
                    dog.7152.jpg (36.1 kB)
                    ... and 11256 other files
                train/
                    cat/
                        cat.4838.jpg (20.2 kB)
                        cat.1314.jpg (21.7 kB)
                        ... and 11240 other files
                    dog/
                        dog.6712.jpg (35.3 kB)
                        dog.7152.jpg (36.1 kB)
                        ... and 11256 other files
                    train/
        working/
            dogs-vs-cats-redux-kernels-edition/
                cat.1714.jpg (7.8 kB)
                cat.10025.jpg (18.4 kB)
                ... and 24998 other files
                description.md (50 lines)
                sample_submission.csv (2501 lines)
                sample_submission.csv.zip (6.0 kB)
                test.zip (56.6 MB)
                train.zip (513.0 MB)
                dogs-vs-cats-redux-kernels-edition/
                test/
                    test/
                    unknown/
                        900.jpg (42.3 kB)
                        572.jpg (30.6 kB)
                        ... and 2498 other files
                train/
                    cat/
                        cat.4838.jpg (20.2 kB)
                        cat.1314.jpg (21.7 kB)
                        ... and 11240 other files
                    dog/
                        dog.6712.jpg (35.3 kB)
                        dog.7152.jpg (36.1 kB)
                        ... and 11256 other files
                    train/
```

-> data/dogs-vs-cats-redux-kernels-edition/sample_submission.csv has 2500 rows and 2 columns.
The columns are: id, label

-> data/sample_submission.csv has 2500 rows and 2 columns.
The columns are: id, label

-> input/dogs-vs-cats-redux-kernels-edition/sample_submission.csv has 2500 rows and 2 columns.
The columns are: id, label

-> input/sample_submission.csv has 2500 rows and 2 columns.
The columns are: id, label

-> working/dogs-vs-cats-redux-kernels-edition/sample_submission.csv has 2500 rows and 2 columns.
The columns are: id, label

# 5. Target score

0.5011

# 6. Current score

8.11063

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 18.13717) has done: 'I fix the environment-breaking Keras import issue by switching to `tf_keras` (available in your packages) while keeping the exact same InceptionResNetV2 + Dense head and training procedure. I also fix the dataset paths to point at the actual extracted folders (`../input/dogs-vs-cats-redux-kernels-edition/train/...` and `.../test/unknown/...`) so `cv2.imread()` doesn’t return `None` and crash in `cv2.resize`. I replace deprecated `fit_generator` with `fit`, restore `ImageDataGenerator` from `tf_keras`, and fix a couple of Python 3 plotting issues (integer subplot args) and removed deprecated `np.float`. Finally, I ensure the submission is written as a valid `submission.csv` with `id,label`, sorted by numeric `id`.'
- What this solution (achieved 17.90649) has done: 'I fix the import-time crash (`MessageFactory.GetPrototype`) by switching from `tf_keras` to `tensorflow.keras`, which avoids the protobuf incompatibility causing the runtime error while keeping the same InceptionResNetV2 model and training loop. Then I fix the main scoring issue: the model currently predicts “dog probability” but is trained with `dog=1`, while this competition’s `label` expects “cat probability” (1=cat, 0=dog), which explains the very high logloss; I invert predictions in the submission (`1 - p_dog`) without changing the model. Finally, I ensure test IDs are parsed correctly and the submission is sorted and written as a valid `submission.csv` with `id,label`.'
- What this solution (achieved 8.02221) has done: 'I fix the import-time crash (`MessageFactory.GetPrototype`) by avoiding TensorFlow entirely and switching to the already-installed `tf_keras` backend for Keras, keeping the same InceptionResNetV2+Dense head, training loop, and data pipeline. I also correct the label semantics to match the competition (submission `label` must be P(dog)), which is currently inverted and causes the very large logloss. Finally, I make the submission generation robust by parsing test IDs from filenames, sorting by numeric id, clipping probabilities away from 0/1 to prevent logloss blow-ups, and ensuring a valid `submission.csv` is always written.'
- What this solution (achieved 8.02221) has done: 'I fix the environment-breaking protobuf/TensorFlow import issue that causes `tf_keras` to crash by switching the imports to `tensorflow.keras`, while keeping the exact same InceptionResNetV2 + Dense head and the same training/inference flow. I also ensure the data root resolves correctly in Kaggle (falling back to `../kaggle/input/...` if `../input/...` doesn’t exist) so image reads don’t silently fail. Finally, because your current logloss is far from the target, I make a minimal score-improving adjustment that preserves the core approach: increase the training subset size (still the same model and loop) so predictions are better calibrated and logloss drops substantially toward the target band, while keeping runtime within the 600s limit.'
- What this solution (achieved 8.11063) has done: 'I fix the import-time crash by removing `tf_keras` and using `tensorflow.keras`, which is available in Kaggle and avoids the protobuf `MessageFactory.GetPrototype` error. I also fix the missing `InceptionResNetV2` / `preprocess_input` symbols by importing them from `tensorflow.keras.applications.inception_resnet_v2` (instead of `keras_applications`, which isn’t installed here). These changes keep the same core model (InceptionResNetV2 backbone + Flatten + Dense head), the same training loop, and the same preprocessing semantics, but make the notebook run end-to-end. Finally, I ensure the submission is always written as `submission.csv` with `id,label`, sorted by numeric id, with probabilities clipped to avoid logloss blow-ups.'
- What this solution (achieved 8.11063) has done: 'We fix the environment-breaking TensorFlow import crash (`MessageFactory.GetPrototype`) by switching the Keras/TensorFlow stack to the already-installed `tf_keras` package while keeping the exact same InceptionResNetV2 backbone, Dense head, preprocessing, and training loop. We keep paths and data loading logic intact, but add a small safety check to ensure we read all expected test images and IDs. Finally, we keep submission semantics as “probability of dog” (per the competition), ensure IDs are numeric-sorted, and always write a valid `submission.csv`. This should both unblock runtime and improve logloss substantially versus the current broken/incorrect run.'
- What this solution (achieved 8.11063) has done: 'I fix the import-time crash caused by the TensorFlow/protobuf incompatibility by switching the Keras stack to the installed `tf_keras` package while keeping the same InceptionResNetV2 backbone, head, preprocessing, and training loop semantics. Then I fix the calibration crash by making the logistic calibrator conditional: if the validation split accidentally contains only one class, we skip calibration and fall back to the raw model probabilities (still clipped for logloss safety). Finally, I ensure test image discovery is robust across the provided folder layouts and that the submission is always written as a valid `submission.csv` with `id,label` sorted by numeric `id`.'
- What this solution (achieved 8.11063) has done: 'You’re currently crashing at import time because `tf_keras` triggers a protobuf `MessageFactory.GetPrototype` error in this Kaggle environment, so I switch the imports to `tensorflow.keras` (which is available here) while keeping the same InceptionResNetV2 backbone, head, and training loop. Your very high logloss vs target indicates the submission probabilities are likely semantically inverted for this dataset, so I align labels to the required “probability of dog” by training with `dog=1, cat=0` and (critically) ensuring the submission uses `P(dog)` (no inversion). Finally, I make submission generation robust by matching the sample_submission IDs exactly (2500) and ordering predictions by those IDs, preventing misalignment (a common cause of huge logloss even when the model is fine).'
- What this solution (achieved 8.11063) has done: 'The runtime crash happens before any training because importing `tensorflow` triggers a protobuf incompatibility (`MessageFactory.GetPrototype`) in this environment. To make the pipeline run end-to-end, I switch the Keras stack from `tensorflow.keras` to the already-installed `tf_keras` package while keeping the exact same model (InceptionResNetV2 backbone + Flatten + Dense head), preprocessing, training loop, and calibration logic. I also keep the existing robust dataset path resolution and submission alignment to `sample_submission.csv`, but add a small fallback so the script can still produce a valid `submission.csv` even if some test IDs are not found (without changing the intended ordering when everything is present). These fixes are execution- and correctness-focused and should also reduce logloss substantially versus the current crash/unstable run, moving closer toward the target.'
- What this solution (achieved 8.11063) has done: 'I fix the immediate runtime crash by avoiding `tf_keras` (which is triggering the protobuf `MessageFactory.GetPrototype` error in this environment) and switching the imports to `tensorflow.keras` while keeping the exact same InceptionResNetV2 + Flatten + Dense head, generators, and training loop. Then I correct a key logic issue that can cause extremely high logloss: the calibrator currently outputs `P(dog)` but is being written to the submission as `label` without ensuring the calibrator is actually calibrated for the same class semantics; we explicitly calibrate and submit `P(dog)` consistently. Finally, I make submission generation stricter by ensuring we predict for all IDs in `sample_submission.csv` in the exact same order (filling missing IDs with 0.5 as you already intended) and always write `submission.csv` with `id,label`. These changes are directly tied to execution + logloss correctness and should move the score substantially toward the 0.5011 target.'
- What this solution (achieved 8.11063) has done: 'I fix the immediate runtime crash (`MessageFactory.GetPrototype`) by avoiding `tensorflow` imports entirely and switching the Keras stack to the already-installed `tf_keras`, keeping the same InceptionResNetV2 backbone, head, preprocessing, generators, and training loop semantics. Then I fix the main scoring issue causing huge logloss: this competition’s `label` is the probability of **dog**, but your pipeline is very likely submitting mis-calibrated/inverted probabilities due to label-direction mismatch between the folder labels and the calibrator; I ensure we always train and submit consistently as `P(dog)` (dog=1, cat=0) and calibrate on that same target. Finally, I keep the strict sample_submission-based ID ordering (2500 rows) and make test path resolution robust, so the produced `submission.csv` is valid and aligned, which should move logloss strongly toward the ~0.5 target.'
- What this solution (achieved 8.11063) has done: 'I remove the TensorFlow import that crashes this Kaggle environment (`MessageFactory.GetPrototype`) and switch the exact same Keras API usage to the already-installed `tf_keras` package, keeping the same InceptionResNetV2 backbone, head, preprocessing, and training loop semantics. Then I fix test image discovery so it finds the actual `unknown/` folder that contains the 2500 images matching `sample_submission.csv`, preventing the “No test images were loaded” failure. Finally, I fix the `model.predict` crash by consistently predicting from a NumPy array via `predict(..., batch_size=...)` (same inference semantics) and ensure we always write a valid `submission.csv` with `id,label` aligned to `sample_submission.csv` ordering.'

# 9. Code solution

## === cell 0
import os
import gc
import random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cv2

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

import tf_keras as keras
from tf_keras import layers, models
from tf_keras.preprocessing.image import ImageDataGenerator
from tf_keras.applications.inception_resnet_v2 import (
    InceptionResNetV2,
    preprocess_input,
)

random.seed(1)
np.random.seed(1)
try:
    keras.utils.set_random_seed(1)
except Exception:
    pass

print("Using keras module:", keras.__name__)
print("Keras version:", getattr(keras, "__version__", "unknown"))



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
print(
    "Input root listing (../input):",
    os.listdir("../input")[:20] if os.path.isdir("../input") else "MISSING",
)
print(
    "Input root listing (../kaggle/input):",
    (
        os.listdir("../kaggle/input")[:20]
        if os.path.isdir("../kaggle/input")
        else "MISSING"
    ),
)

DATA_ROOT_CANDIDATES = [
    "../input/dogs-vs-cats-redux-kernels-edition",
    "../kaggle/input/dogs-vs-cats-redux-kernels-edition",
    "/kaggle/input/dogs-vs-cats-redux-kernels-edition",
    "../kaggle/data/dogs-vs-cats-redux-kernels-edition",
    "/kaggle/data/dogs-vs-cats-redux-kernels-edition",
]
DATA_ROOT = next((p for p in DATA_ROOT_CANDIDATES if os.path.isdir(p)), None)
if DATA_ROOT is None:
    raise FileNotFoundError(f"Could not find dataset root in: {DATA_ROOT_CANDIDATES}")

train_dir = os.path.join(DATA_ROOT, "train")
test_dir = os.path.join(DATA_ROOT, "test")

print("Resolved DATA_ROOT:", DATA_ROOT)
print("Train dir exists:", os.path.isdir(train_dir), train_dir)
print("Test dir exists:", os.path.isdir(test_dir), test_dir)

SAMPLE_SUB_PATHS = [
    os.path.join(DATA_ROOT, "sample_submission.csv"),
    "../input/sample_submission.csv",
    "../kaggle/input/sample_submission.csv",
    "/kaggle/input/sample_submission.csv",
    "../kaggle/data/sample_submission.csv",
    "/kaggle/data/sample_submission.csv",
]
SAMPLE_SUB = next((p for p in SAMPLE_SUB_PATHS if os.path.isfile(p)), None)
if SAMPLE_SUB is None:
    raise FileNotFoundError(
        f"Could not find sample_submission.csv in: {SAMPLE_SUB_PATHS}"
    )
sample_sub = pd.read_csv(SAMPLE_SUB)
print("Loaded sample_submission:", SAMPLE_SUB, "shape:", sample_sub.shape)
print("Sample submission head:\n", sample_sub.head())



## === cell 2
train_cat_dir = os.path.join(train_dir, "cat")
train_dog_dir = os.path.join(train_dir, "dog")

candidate_test_dirs = [
    os.path.join(test_dir, "unknown"),  # expected: 2500 images for this kernel
    os.path.join(test_dir, "test", "unknown"),  # some nested layouts
    os.path.join(test_dir, "test"),  # sometimes 12500, but may not exist here
    os.path.join(test_dir, "test", "test"),
    os.path.join(test_dir, "test", "test", "unknown"),
]


def _count_jpgs(d):
    if not os.path.isdir(d):
        return -1
    return sum(1 for f in os.listdir(d) if f.lower().endswith(".jpg"))


best_dir = None
best_count = -1
for d in candidate_test_dirs:
    c = _count_jpgs(d)
    if c > best_count:
        best_count = c
        best_dir = d

if best_dir is None or best_count <= 0:
    raise FileNotFoundError(
        f"Could not find a non-empty test image dir in candidates: {candidate_test_dirs}"
    )
resolved_test_img_dir = best_dir

print("Resolved test image dir:", resolved_test_img_dir, "jpg_count:", best_count)

train_dogs = [
    os.path.join(train_dog_dir, f)
    for f in os.listdir(train_dog_dir)
    if f.lower().endswith(".jpg")
]
train_cats = [
    os.path.join(train_cat_dir, f)
    for f in os.listdir(train_cat_dir)
    if f.lower().endswith(".jpg")
]

test_imgs = [
    os.path.join(resolved_test_img_dir, f)
    for f in os.listdir(resolved_test_img_dir)
    if f.lower().endswith(".jpg")
]

print("Num train dogs:", len(train_dogs))
print("Num train cats:", len(train_cats))
print("Num test discovered:", len(test_imgs))



## === cell 3
size = 10000  # keep as provided (core logic preserved)
train_imgs = train_dogs[:size] + train_cats[:size]
random.shuffle(train_imgs)

img_size = 150




## === cell 4
def read_and_process_image(list_of_images):
    """
    Returns three lists:
        X: resized images (H,W,3), uint8 RGB
        y: labels (1=dog, 0=cat) for train; empty for test
        l_id: string ids (for submission; numeric filename stem for test)
    """
    X = []
    y = []
    l_id = []

    for image_path in list_of_images:
        img = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_CUBIC)
        X.append(img)

        basename = os.path.basename(image_path)
        stem = os.path.splitext(basename)[0]

        parts = stem.split(".")
        img_id = parts[-1] if parts[-1].isdigit() else parts[0]
        l_id.append(img_id)

        if "dog" in image_path:
            y.append(1)
        elif "cat" in image_path:
            y.append(0)

    return X, y, l_id




## === cell 5
X, y, l_id = read_and_process_image(train_imgs)

X = np.array(X, dtype=np.uint8)
y = np.array(y, dtype=np.int64)

print("Loaded train subset:", X.shape, y.shape)
print("Label counts:", np.bincount(y) if len(y) else "empty")



## === cell 6
plt.figure(figsize=(12, 6))
columns = 5
rows = 1
for i in range(min(columns, len(X))):
    plt.subplot(rows, columns, i + 1)
    plt.imshow(X[i])
    plt.axis("off")
plt.tight_layout()
plt.show()



## === cell 7
sns.countplot(x=y)
plt.title("Labels for Cats and Dogs (subset)")
plt.show()

print("Shape of train images is:", X.shape)
print("Shape of labels is:", y.shape)



## === cell 8
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.15, random_state=1, stratify=y
)

del X, y, train_imgs, train_dogs, train_cats
gc.collect()

print("Shape of X_train", X_train.shape)
print("Shape of X_val", X_val.shape)

ntrain = len(X_train)
nval = len(X_val)

print("Validation label bincount:", np.bincount(y_val) if len(y_val) else "empty")



## === cell 9
conv_base = InceptionResNetV2(
    weights="imagenet", include_top=False, input_shape=(150, 150, 3)
)
conv_base.trainable = False

model = models.Sequential()
model.add(conv_base)
model.add(layers.Flatten())
model.add(layers.Dense(256, activation="relu"))
model.add(layers.Dense(1, activation="sigmoid"))

model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["acc"])
model.summary()



## === cell 10
batch_size = 128

train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=30,
    horizontal_flip=True,
    fill_mode="nearest",
)
val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

train_generator = train_datagen.flow(
    X_train, y_train, batch_size=batch_size, shuffle=True
)
val_generator = val_datagen.flow(X_val, y_val, batch_size=batch_size, shuffle=False)



## === cell 11
epochs = 5
history = model.fit(
    train_generator,
    steps_per_epoch=max(1, ntrain // batch_size),
    epochs=epochs,
    validation_data=val_generator,
    validation_steps=max(1, nval // batch_size),
    verbose=1,
)



## === cell 12
hist = history.history
acc_key = "acc" if "acc" in hist else "accuracy"
val_acc_key = "val_acc" if "val_acc" in hist else "val_accuracy"

acc = hist.get(acc_key, [])
val_acc = hist.get(val_acc_key, [])
loss = hist.get("loss", [])
val_loss = hist.get("val_loss", [])

epochs_range = range(1, len(loss) + 1)

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, "b", label="Training acc")
plt.plot(epochs_range, val_acc, "r", label="Validation acc")
plt.title("Training and Validation accuracy")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, "b", label="Training loss")
plt.plot(epochs_range, val_loss, "r", label="Validation loss")
plt.title("Training and Validation loss")
plt.legend()
plt.tight_layout()
plt.show()



## === cell 13
X_test_preview, _, l_id_preview = read_and_process_image(test_imgs[:10])
x_preview = preprocess_input(np.array(X_test_preview, dtype=np.float32))

pred_preview = model.predict(x_preview, batch_size=128, verbose=0).reshape(-1)
print("Preview ids:", l_id_preview[:5])
print("Preview preds (P(dog)):", pred_preview[:5])



## === cell 14
x_val = preprocess_input(X_val.astype(np.float32))
val_pred = model.predict(x_val, batch_size=128, verbose=0).reshape(-1)
val_pred_clip = np.clip(val_pred.astype(np.float64), 1e-6, 1 - 1e-6)

unique_classes = np.unique(y_val)
calibrator = None
if unique_classes.size >= 2:
    calibrator = LogisticRegression(solver="lbfgs", max_iter=1000, random_state=1)
    calibrator.fit(val_pred_clip.reshape(-1, 1), y_val)
    print("Calibration fitted on validation predictions for P(dog).")
else:
    print(
        "Skipping calibration: validation set has only one class:",
        unique_classes.tolist(),
    )

del x_val, val_pred, val_pred_clip
gc.collect()



## === cell 15
del X_train, X_val, y_train, y_val
gc.collect()



## === cell 16
sample_ids = sample_sub["id"].astype(int).tolist()

id_to_path = {}
for p in test_imgs:
    stem = os.path.splitext(os.path.basename(p))[0]
    try:
        tid = int(stem.split(".")[-1]) if stem.split(".")[-1].isdigit() else int(stem)
        id_to_path[tid] = p
    except Exception:
        continue

ordered_test_paths = []
missing_ids = []
for i in sample_ids:
    if i in id_to_path:
        ordered_test_paths.append(id_to_path[i])
    else:
        ordered_test_paths.append(None)
        missing_ids.append(i)

if len(missing_ids) > 0:
    print(
        f"WARNING: Missing {len(missing_ids)} test images required by sample_submission. "
        f"First missing ids: {missing_ids[:10]}. Will fill with 0.5."
    )

existing_indices = [idx for idx, p in enumerate(ordered_test_paths) if p is not None]
existing_paths = [p for p in ordered_test_paths if p is not None]

X_test, _, _ = read_and_process_image(existing_paths)
if len(X_test) == 0:
    raise RuntimeError(
        "No test images were loaded; cannot create submission. "
        f"Resolved test dir was: {resolved_test_img_dir} with discovered files={len(test_imgs)}"
    )
print("Loaded existing test images:", len(X_test), "of", len(sample_ids))

x = preprocess_input(np.array(X_test, dtype=np.float32))
del X_test
gc.collect()

predictions_dog = model.predict(x, batch_size=128, verbose=1).reshape(-1)
predictions_dog = np.clip(predictions_dog.astype(np.float64), 1e-7, 1 - 1e-7)

if calibrator is not None:
    preds_in = np.clip(predictions_dog, 1e-6, 1 - 1e-6).reshape(-1, 1)
    predictions_dog_existing = calibrator.predict_proba(preds_in)[:, 1].astype(
        np.float64
    )
else:
    predictions_dog_existing = predictions_dog

predictions_dog_existing = np.clip(predictions_dog_existing, 1e-7, 1 - 1e-7)

pred_full = np.full(len(sample_ids), 0.5, dtype=np.float64)
pred_full[np.array(existing_indices, dtype=np.int64)] = predictions_dog_existing

sub = pd.DataFrame({"id": sample_ids, "label": pred_full})
print("Submission rows:", len(sub), "Unique ids:", sub["id"].nunique())

sub.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", sub.shape)
print(sub.head(10))

print(
    "Pred min/max/mean:",
    float(sub["label"].min()),
    float(sub["label"].max()),
    float(sub["label"].mean()),
)
