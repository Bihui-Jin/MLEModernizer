# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


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

3.8

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
protobuf==6.33.0
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
seaborn==0.12.2
sklearn-pandas==2.2.0
tensorflow==2.18.0
tensorflow-cloud==0.1.5
tensorflow-datasets==4.9.9
tensorflow_decision_forests==1.11.0
tensorflow-hub==0.16.1
tensorflow-io==0.37.1
tensorflow-io-gcs-filesystem==0.37.1
tensorflow-metadata==1.17.2
tensorflow-probability==0.25.0
tensorflow-text==2.18.1
tf_keras==2.18.0

# 4. Data file paths

```
/
    kaggle/
        data/
            dogs-vs-cats-redux-kernels-edition/
                description.md (50 lines)
                sample_submission.csv (2501 lines)
                ... and 2 other files
                test/
                    1332.jpg (22.2 kB)
                    617.jpg (22.9 kB)
                    ... and 2498 other files
                train/
                    dog.10425.jpg (33.5 kB)
                    cat.2462.jpg (18.8 kB)
                    ... and 22498 other files
        input/
            dogs-vs-cats-redux-kernels-edition/
                description.md (50 lines)
                sample_submission.csv (2501 lines)
                ... and 2 other files
                test/
                    1332.jpg (22.2 kB)
                    617.jpg (22.9 kB)
                    ... and 2498 other files
                train/
                    dog.10425.jpg (33.5 kB)
                    cat.2462.jpg (18.8 kB)
                    ... and 22498 other files
            test/
                test/
                    1332.jpg (22.2 kB)
                    617.jpg (22.9 kB)
                    ... and 2498 other files
            train/
                train/
                    dog.10425.jpg (33.5 kB)
                    cat.2462.jpg (18.8 kB)
                    ... and 22498 other files
        working/
            dogs-vs-cats-redux-kernels-edition/
                description.md (50 lines)
                sample_submission.csv (2501 lines)
                ... and 2 other files
                test/
                    1332.jpg (22.2 kB)
                    617.jpg (22.9 kB)
                    ... and 2498 other files
                train/
                    dog.10425.jpg (33.5 kB)
                    cat.2462.jpg (18.8 kB)
                    ... and 22498 other files
```

-> data/dogs-vs-cats-redux-kernels-edition/sample_submission.csv has 2500 rows and 2 columns.
Here is some information about the columns:
id (int64) has range: 1.00 - 2000.00, 0 nan values
label (float64) has 1 unique values: [0.5]

-> input/dogs-vs-cats-redux-kernels-edition/sample_submission.csv has 2500 rows and 2 columns.
Here is some information about the columns:
id (int64) has range: 1.00 - 2000.00, 0 nan values
label (float64) has 1 unique values: [0.5]

-> working/dogs-vs-cats-redux-kernels-edition/sample_submission.csv has 2500 rows and 2 columns.
Here is some information about the columns:
id (int64) has range: 1.00 - 2000.00, 0 nan values
label (float64) has 1 unique values: [0.5]

# 5. Code solution

## === cell 0
import os
import cv2, random, time, numpy as np, pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss, accuracy_score

random.seed(558)
np.random.seed(558)

start = time.time()




## === cell 1
BASE_PATH = "/kaggle/input/dogs-vs-cats-redux-kernels-edition"
TRAIN_DIR = os.path.join(BASE_PATH, "train")
TEST_DIR = os.path.join(BASE_PATH, "test")

all_train_files = [
    os.path.join(TRAIN_DIR, f)
    for f in os.listdir(TRAIN_DIR)
    if f.lower().endswith(".jpg")
]

train_cat = [p for p in all_train_files if os.path.basename(p).startswith("cat")]
train_dog = [p for p in all_train_files if os.path.basename(p).startswith("dog")]

train_images = train_cat + train_dog
random.shuffle(train_images)

test_images = []
for root, _, files in os.walk(TEST_DIR):
    for f in files:
        if f.lower().endswith(".jpg"):
            test_images.append(os.path.join(root, f))
test_images.sort()  # deterministic order for submission IDs




## === cell 2
IMG_WIDTH, IMG_HEIGHT = 128, 128


def load_and_resize(paths):
    arr = []
    for p in paths:
        img = cv2.imread(p)
        if img is None:
            img = np.zeros((IMG_HEIGHT, IMG_WIDTH, 3), dtype=np.uint8)
        else:
            img = cv2.resize(
                img, (IMG_WIDTH, IMG_HEIGHT), interpolation=cv2.INTER_CUBIC
            )
        arr.append(img)
    return np.array(arr, dtype="uint8")


x = load_and_resize(train_images)
y = np.array([1 if "dog" in p else 0 for p in train_images], dtype="int32")
test = load_and_resize(test_images)

print(f"Train shape: {x.shape}, Test shape: {test.shape}")
print(f"Positive ratio: {y.mean():.3f}")




## === cell 3
x_train, x_val, y_train, y_val = train_test_split(
    x, y, test_size=0.2, random_state=2020, stratify=y
)

x_train = x_train.astype("float32") / 255.0
x_val = x_val.astype("float32") / 255.0
test = test.astype("float32") / 255.0




## === cell 4
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

base_model = EfficientNetB0(
    weights="imagenet", include_top=False, input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)
)

base_model.trainable = True

model = models.Sequential(
    [base_model, layers.GlobalAveragePooling2D(), layers.Dense(1, activation="sigmoid")]
)

model.compile(
    optimizer=Adam(learning_rate=2e-4),
    loss="binary_crossentropy",
    metrics=["accuracy"],
)

model.summary()




## === cell 5
early_stop = EarlyStopping(patience=7, restore_best_weights=True, monitor="val_loss")
reduce_lr = ReduceLROnPlateau(
    monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6, verbose=1
)

history = model.fit(
    x_train,
    y_train,
    validation_data=(x_val, y_val),
    epochs=20,
    batch_size=32,
    callbacks=[early_stop, reduce_lr],
    verbose=2,
)




## === cell 6
val_preds = model.predict(x_val).ravel()
val_pred_labels = (val_preds > 0.5).astype(int)

val_acc = accuracy_score(y_val, val_pred_labels)
val_ll = log_loss(y_val, val_preds.astype("float64"), labels=[0, 1])

print(f"Validation Accuracy: {val_acc:.5f}")
print(f"Validation Log Loss: {val_ll:.5f}")




## === cell 7
test_preds = model.predict(test).ravel()

test_ids = [int(os.path.splitext(os.path.basename(p))[0]) for p in test_images]

submission = pd.DataFrame({"id": test_ids, "label": test_preds})
submission_path = "submission.csv"
submission.to_csv(submission_path, index=False)

print(f"Submission saved to {submission_path}")
print(f"Elapsed time: {time.time() - start:.2f} seconds")
