# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


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
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
sklearn-pandas==2.2.0
tf_keras==2.18.0

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (56 lines)
            sample_submission.csv (3326 lines)
            sample_submission.csv.zip (67.3 kB)
            test.zip (3.5 MB)
            train.csv (14176 lines)
            train.csv.zip (285.6 kB)
            train.zip (15.0 MB)
            aerial-cactus-identification/
                description.md (56 lines)
                sample_submission.csv (3326 lines)
                ... and 5 other files
                aerial-cactus-identification/
                test/
                    76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                    f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                    ... and 3323 other files
                    test/
                train/
                    775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                    65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                    ... and 14173 other files
                    train/
            test/
                76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                ... and 3323 other files
                test/
            train/
                775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                ... and 14173 other files
                train/
        input/
            description.md (56 lines)
            sample_submission.csv (3326 lines)
            sample_submission.csv.zip (67.3 kB)
            test.zip (3.5 MB)
            train.csv (14176 lines)
            train.csv.zip (285.6 kB)
            train.zip (15.0 MB)
            aerial-cactus-identification/
                description.md (56 lines)
                sample_submission.csv (3326 lines)
                ... and 5 other files
                aerial-cactus-identification/
                test/
                    76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                    f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                    ... and 3323 other files
                    test/
                train/
                    775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                    65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                    ... and 14173 other files
                    train/
            test/
                76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                ... and 3323 other files
                test/
                    76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                    f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                    ... and 3323 other files
                    test/
            train/
                775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                ... and 14173 other files
                train/
                    775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                    65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                    ... and 14173 other files
                    train/
        working/
            aerial-cactus-identification/
                description.md (56 lines)
                sample_submission.csv (3326 lines)
                ... and 5 other files
                aerial-cactus-identification/
                test/
                    76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                    f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                    ... and 3323 other files
                    test/
                train/
                    775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                    65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                    ... and 14173 other files
                    train/
```

-> data/aerial-cactus-identification/sample_submission.csv has 3325 rows and 2 columns.
The columns are: id, has_cactus

-> data/aerial-cactus-identification/train.csv has 14175 rows and 2 columns.
The columns are: id, has_cactus

-> data/sample_submission.csv has 3325 rows and 2 columns.
The columns are: id, has_cactus

-> data/train.csv has 14175 rows and 2 columns.
The columns are: id, has_cactus

-> input/aerial-cactus-identification/sample_submission.csv has 3325 rows and 2 columns.
The columns are: id, has_cactus

-> input/aerial-cactus-identification/train.csv has 14175 rows and 2 columns.
The columns are: id, has_cactus

-> (stopped after 10 files for performance)

# 5. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import tensorflow as tf

print("Input directories:", os.listdir("../input"))




## === cell 1
train_df = pd.read_csv("../input/train.csv")
print("Train shape:", train_df.shape)




## === cell 2
train_split = train_df.iloc[:15000].reset_index(drop=True)
val_split = train_df.iloc[15000:].reset_index(drop=True)




## === cell 3
def image_generator(df, batch_size=64, shuffle=True):
    """Yield batches of (image, label) tuples."""
    img_dir = "../input/train/train/"
    n = len(df)
    while True:
        idxs = np.arange(n)
        if shuffle:
            np.random.shuffle(idxs)
        for start in range(0, n, batch_size):
            batch_idxs = idxs[start : start + batch_size]
            batch_imgs = []
            batch_labels = []
            for i in batch_idxs:
                img_path = os.path.join(img_dir, df.iloc[i]["id"])
                img = mpimg.imread(img_path)
                if img.max() > 1:
                    img = img / 255.0
                batch_imgs.append(img)
                batch_labels.append(df.iloc[i]["has_cactus"])
            yield np.array(batch_imgs, dtype=np.float32), np.array(
                batch_labels, dtype=np.float32
            ).reshape(-1, 1)




## === cell 4
model = tf.keras.models.Sequential(
    [
        tf.keras.layers.Conv2D(32, (5, 5), activation="relu", input_shape=(32, 32, 3)),
        tf.keras.layers.Conv2D(32, (5, 5), activation="relu"),
        tf.keras.layers.Conv2D(64, (5, 5), activation="relu"),
        tf.keras.layers.Conv2D(64, (5, 5), activation="relu"),
        tf.keras.layers.Conv2D(128, (3, 3), activation="relu"),
        tf.keras.layers.Conv2D(128, (3, 3), activation="relu"),
        tf.keras.layers.Conv2D(256, (3, 3), activation="relu"),
        tf.keras.layers.Conv2D(256, (3, 3), activation="relu"),
        tf.keras.layers.Conv2D(512, (3, 3), activation="relu"),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(100, activation="relu"),
        tf.keras.layers.Dense(1, activation="sigmoid"),
    ]
)
model.summary()




## === cell 5
opt = tf.keras.optimizers.Adam(learning_rate=0.0001)
model.compile(optimizer=opt, loss="binary_crossentropy", metrics=["accuracy"])




## === cell 6
batch_size = 64
steps_per_epoch = len(train_split) // batch_size
validation_steps = len(val_split) // batch_size

model.fit(
    image_generator(train_split, batch_size=batch_size, shuffle=True),
    steps_per_epoch=steps_per_epoch,
    epochs=20,
    validation_data=image_generator(val_split, batch_size=batch_size, shuffle=False),
    validation_steps=validation_steps,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(patience=2),
    ],
    verbose=2,
)




## === cell 7
val_imgs, val_labels = next(
    image_generator(val_split, batch_size=len(val_split), shuffle=False)
)
val_preds = model.predict(val_imgs).ravel()
from sklearn.metrics import roc_auc_score

val_auc = roc_auc_score(val_labels, val_preds)
print(f"Validation AUC: {val_auc:.5f}")




## === cell 8
test_dir = "../input/test/test/"
test_files = sorted([f for f in os.listdir(test_dir) if f.lower().endswith(".jpg")])
print("Number of test images:", len(test_files))




## === cell 9
batch = 40
all_preds = []

for i in range(0, len(test_files), batch):
    batch_files = test_files[i : i + batch]
    imgs = []
    for fname in batch_files:
        img_path = os.path.join(test_dir, fname)
        img = mpimg.imread(img_path)
        if img.max() > 1:
            img = img / 255.0
        imgs.append(img)
    batch_arr = np.array(imgs, dtype=np.float32)
    preds = model.predict(batch_arr).ravel()
    all_preds.extend(preds)

all_preds = np.array(all_preds)
print("Predictions shape:", all_preds.shape)




## === cell 10
submission = pd.DataFrame({"id": test_files, "has_cactus": all_preds})
submission.to_csv("sample_submission.csv", index=False)
print("Submission saved to sample_submission.csv")
