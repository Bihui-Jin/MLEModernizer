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
Create a classifier to predict whether an image contains a cactus.

## Metric
Area under the ROC curve.

## Submission Format
For each ID in the test set, you must predict a probability for the `has_cactus` variable. The file should contain a header and have the following format:

```
id,has_cactus
000940378805c44108d287872b2f04ce.jpg,0.5
0017242f54ececa4512b4d7937d1e21e.jpg,0.5
001ee6d8564003107853118ab87df407.jpg,0.5
etc.
```

## Dataset
This dataset contains a large number of 32 x 32 thumbnail images containing aerial photos of a cactus. The file name of an image corresponds to its `id`.

- **train/** - the training set images
- **test/** - the test set images (you must predict the labels of these)
- **train.csv** - the training set labels, indicates whether the image has a cactus (`has_cactus = 1`)
- **sample_submission.csv** - a sample submission file in the correct format

# 2. Python version

3.7

# 3. Installed packages

geopandas==0.14.4
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
pillow==11.3.0
protobuf==6.33.0
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
scipy==1.15.3
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
tqdm==4.67.1

# 4. Data file paths

```
/
    kaggle/
        data/
            aerial-cactus-identification/
                description.md (56 lines)
                sample_submission.csv (3326 lines)
                ... and 3 other files
        input/
            aerial-cactus-identification/
                description.md (56 lines)
                sample_submission.csv (3326 lines)
                ... and 3 other files
        working/
            aerial-cactus-identification/
                description.md (56 lines)
                sample_submission.csv (3326 lines)
                ... and 3 other files
```

-> data/aerial-cactus-identification/sample_submission.csv has 3325 rows and 2 columns.
Here is some information about the columns:
has_cactus (float64) has 1 unique values: [0.5]
id (object) has 2000 unique values. Some example values: ['09034a34de0e2015a8a28dfe18f423f6.jpg', '44b974fd955c4cd306c7dbae154646b9.jpg', '37cd593f40ba13982e7587a613a9a789.jpg', 'c892c865a5706d3072ba44beafbf2d1c.jpg']

-> data/aerial-cactus-identification/train.csv has 14175 rows and 2 columns.
Here is some information about the columns:
has_cactus (int64) has 2 unique values: [1, 0]
id (object) has 2000 unique values. Some example values: ['2de8f189f1dce439766637e75df0ee27.jpg', 'f94b18b4c6850fc44b54f5fbe3c5b0c6.jpg', 'fb0624c0ebc01643a8f4318537099078.jpg', '00b4dfbb267109b5f0d0dde365fa6161.jpg']

-> input/aerial-cactus-identification/sample_submission.csv has 3325 rows and 2 columns.
Here is some information about the columns:
has_cactus (float64) has 1 unique values: [0.5]
id (object) has 2000 unique values. Some example values: ['09034a34de0e2015a8a28dfe18f423f6.jpg', '44b974fd955c4cd306c7dbae154646b9.jpg', '37cd593f40ba13982e7587a613a9a789.jpg', 'c892c865a5706d3072ba44beafbf2d1c.jpg']

-> input/aerial-cactus-identification/train.csv has 14175 rows and 2 columns.
Here is some information about the columns:
has_cactus (int64) has 2 unique values: [1, 0]
id (object) has 2000 unique values. Some example values: ['2de8f189f1dce439766637e75df0ee27.jpg', 'f94b18b4c6850fc44b54f5fbe3c5b0c6.jpg', 'fb0624c0ebc01643a8f4318537099078.jpg', '00b4dfbb267109b5f0d0dde365fa6161.jpg']

-> working/aerial-cactus-identification/sample_submission.csv has 3325 rows and 2 columns.
Here is some information about the columns:
has_cactus (float64) has 1 unique values: [0.5]
id (object) has 2000 unique values. Some example values: ['09034a34de0e2015a8a28dfe18f423f6.jpg', '44b974fd955c4cd306c7dbae154646b9.jpg', '37cd593f40ba13982e7587a613a9a789.jpg', 'c892c865a5706d3072ba44beafbf2d1c.jpg']

-> working/aerial-cactus-identification/train.csv has 14175 rows and 2 columns.
Here is some information about the columns:
has_cactus (int64) has 2 unique values: [1, 0]
id (object) has 2000 unique values. Some example values: ['2de8f189f1dce439766637e75df0ee27.jpg', 'f94b18b4c6850fc44b54f5fbe3c5b0c6.jpg', 'fb0624c0ebc01643a8f4318537099078.jpg', '00b4dfbb267109b5f0d0dde365fa6161.jpg']

# 5. Target score

0.8526

# 6. Current score

0.9977

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plan

- What this solution (achieved 0.9977) has done: 'I fix the TensorFlow import conflict by delaying the import until it’s needed, add the missing `GlobalAveragePooling2D` layer import, and adjust the cell ordering so the script runs from cell 1 onward. These minimal changes resolve the runtime errors and allow the model to train and produce a valid `sample_submission.csv` file, moving the solution toward the target AUC score.'

# 9. Code solution

## === cell 0
import os, glob, numpy as np, pandas as pd, cv2
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
import matplotlib.pyplot as plt

print("Input directories:", os.listdir("/kaggle/input"))




## === cell 1
def load_imgs(folder):
    """Load all jpg images from *folder*, resize to 32x32 and return a dict {filename: image_array}."""
    imgs = {}
    for f in os.listdir(folder):
        path = os.path.join(folder, f)
        img = cv2.imread(path)
        if img is None:
            continue
        img = cv2.cvtColor(cv2.resize(img, (32, 32)), cv2.COLOR_BGR2RGB)
        imgs[f] = img.astype(np.float32) / 255.0
    return imgs


train_img_dir = "/kaggle/input/aerial-cactus-identification/train/"
test_img_dir = "/kaggle/input/aerial-cactus-identification/test/"

train_imgs = load_imgs(train_img_dir)
test_imgs = load_imgs(test_img_dir)




## === cell 2
train_df = pd.read_csv("/kaggle/input/aerial-cactus-identification/train.csv")
test_df = pd.read_csv(
    "/kaggle/input/aerial-cactus-identification/sample_submission.csv"
)

X = np.stack([train_imgs[row.id] for _, row in train_df.iterrows()])
y = train_df.has_cactus.values.astype(np.float32)

X_test = np.stack([test_imgs[f] for f in test_df.id])

print("Train shape:", X.shape, "Test shape:", X_test.shape)




## === cell 3
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D,
    MaxPooling2D,
    BatchNormalization,
    Activation,
    Flatten,
    Dense,
    Dropout,
    GlobalAveragePooling2D,
)
from tensorflow.keras.optimizers import RMSprop


def build_model():
    model = Sequential(
        [
            Conv2D(
                32, (3, 3), padding="same", activation="relu", input_shape=(32, 32, 3)
            ),
            MaxPooling2D(),
            BatchNormalization(),
            Conv2D(64, (3, 3), padding="same", activation="relu"),
            MaxPooling2D(),
            BatchNormalization(),
            Conv2D(128, (3, 3), padding="same", activation="relu"),
            GlobalAveragePooling2D(),
            Dropout(0.3),
            Dense(1, activation="sigmoid"),
        ]
    )
    return model


model = build_model()
model.compile(
    loss="binary_crossentropy",
    optimizer=RMSprop(learning_rate=1e-3),
    metrics=["accuracy"],
)
model.summary()




## === cell 4
X_tr, X_val, y_tr, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

history = model.fit(
    X_tr, y_tr, validation_data=(X_val, y_val), epochs=15, batch_size=64, verbose=1
)




## === cell 5
val_pred = model.predict(X_val).ravel()
auc = roc_auc_score(y_val, val_pred)
print(f"Validation AUC: {auc:.4f}")




## === cell 6
test_pred = model.predict(X_test).ravel()
submission = pd.DataFrame({"id": test_df.id, "has_cactus": test_pred})
submission_path = "sample_submission.csv"
submission.to_csv(submission_path, index=False)
print("Submission saved to", submission_path)
