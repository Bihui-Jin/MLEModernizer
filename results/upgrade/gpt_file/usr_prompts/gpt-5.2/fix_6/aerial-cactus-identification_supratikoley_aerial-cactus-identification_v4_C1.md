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
google-api-python-client==2.177.0
ipython==7.34.0
ipython-genutils==0.2.0
ipython_pygments_lexers==1.1.1
ipython-sql==0.5.0
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
seaborn==0.12.2
sklearn-pandas==2.2.0
tf_keras==2.18.0
tqdm==4.67.1

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

# 5. Target score

0.5

# 6. Current score

0.97562

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.38432) has done: 'I fix the import crash that prevents `ImageDataGenerator` (and later `tqdm`) from being defined by switching to `tf_keras` (compatible with the Kaggle environment shown) while keeping the exact same model and training logic. I also make the input path resolution robust to both `../input/...` and `/kaggle/input/...` layouts so the code can actually find the dataset. Next, I ensure the generators are created before training, and I harden the test loading so `test_features` is always a non-empty NumPy array with the right shape (avoiding the “pop from an empty set” error). Finally, I always write a valid `submission.csv` with the required columns and IDs aligned to `sample_submission.csv`.'
- What this solution (achieved 0.98653) has done: 'I fix the crashing import by ensuring TensorFlow/Keras is imported in a way compatible with the Kaggle runtime, which resolves the protobuf `MessageFactory` error. Then I fix the validation split bug that creates an empty/invalid validation dataframe (causing “Found 0 classes”) by using a deterministic split that guarantees both classes exist in each split, while keeping the same ImageDataGenerator + model + training loop approach. Finally, I make the training/inference cells robust so later cells don’t error if something upstream fails, and I always write a correctly formatted `submission.csv` aligned to `sample_submission.csv`, which should also move the ROC-AUC up toward the 0.5 target (your current errors prevent a proper model fit).'
- What this solution (achieved 0.97562) has done: 'I fix the import crash and the missing symbols (`ImageDataGenerator`, `tqdm`) by switching to `tf_keras`, which is compatible with this Kaggle runtime while preserving the same Keras model and training flow. Then I ensure the data generators are created successfully (so downstream cells don’t fail) and make test-image loading always produce a NumPy array (so `.shape`/`.size` checks work). Finally, I always write a correctly formatted `submission.csv` aligned to `sample_submission.csv` IDs, even if something upstream goes wrong, so you get a valid file and a score that’s meaningfully above random.'

# 9. Code solution

## === cell 0
import os
import random
import numpy as np
import pandas as pd
import cv2

from tf_keras import layers, models, optimizers
from tf_keras.preprocessing.image import ImageDataGenerator

import matplotlib.pyplot as plt
from tqdm import tqdm

random.seed(42)
np.random.seed(42)



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
CANDIDATE_INPUT_ROOTS = [
    "../input/aerial-cactus-identification",
    "/kaggle/input/aerial-cactus-identification",
    "/kaggle/input/aerial-cactus-identification/aerial-cactus-identification",
    "/kaggle/data/aerial-cactus-identification",
    "/kaggle/input/aerial-cactus-identification/aerial-cactus-identification/aerial-cactus-identification",
]

INPUT_ROOT = None
for p in CANDIDATE_INPUT_ROOTS:
    if os.path.isdir(p):
        INPUT_ROOT = p
        break

if INPUT_ROOT is None:
    INPUT_ROOT = "../input/aerial-cactus-identification"

train_dir = os.path.join(INPUT_ROOT, "train")
test_dir = os.path.join(INPUT_ROOT, "test")

train_csv_path = os.path.join(INPUT_ROOT, "train.csv")
sample_sub_path = os.path.join(INPUT_ROOT, "sample_submission.csv")

print("Using INPUT_ROOT:", INPUT_ROOT)
print("Train dir exists:", os.path.isdir(train_dir), train_dir)
print("Test dir exists:", os.path.isdir(test_dir), test_dir)
print("Train CSV exists:", os.path.isfile(train_csv_path), train_csv_path)
print("Sample submission exists:", os.path.isfile(sample_sub_path), sample_sub_path)

train = pd.read_csv(train_csv_path)
df_test = pd.read_csv(sample_sub_path)



## === cell 2
train.head(5)



## === cell 3
train["has_cactus"] = train["has_cactus"].astype(str)



## === cell 4
train.shape[0], train.shape[1]



## === cell 5
train["has_cactus"].value_counts()



## === cell 6
sample_img_path = os.path.join(train_dir, train.iloc[1, 0])
print("Sample image path:", sample_img_path, "exists:", os.path.isfile(sample_img_path))



## === cell 7
datagen = ImageDataGenerator(rescale=1.0 / 255.0)
batch_size = 150



## === cell 8
train_shuf = train.sample(frac=1.0, random_state=42).reset_index(drop=True)

val_frac = 0.15
val_size = int(len(train_shuf) * val_frac)
val_df = train_shuf.iloc[:val_size].copy()
train_df = train_shuf.iloc[val_size:].copy()


def _ensure_both_classes(df_a, df_b, label_col="has_cactus"):
    classes = set(train_shuf[label_col].unique())
    for _ in range(10):
        if (
            set(df_a[label_col].unique()) == classes
            and set(df_b[label_col].unique()) == classes
        ):
            return df_a, df_b
        missing_a = classes - set(df_a[label_col].unique())
        missing_b = classes - set(df_b[label_col].unique())

        if missing_a:
            c = next(iter(missing_a))
            idx = df_b[df_b[label_col] == c].index[:1]
            if len(idx):
                df_a = pd.concat([df_a, df_b.loc[idx]], axis=0)
                df_b = df_b.drop(idx)
        if missing_b:
            c = next(iter(missing_b))
            idx = df_a[df_a[label_col] == c].index[:1]
            if len(idx):
                df_b = pd.concat([df_b, df_a.loc[idx]], axis=0)
                df_a = df_a.drop(idx)
        df_a = df_a.reset_index(drop=True)
        df_b = df_b.reset_index(drop=True)
    return df_a, df_b


val_df, train_df = _ensure_both_classes(val_df, train_df)

print("Train split size:", len(train_df), "Val split size:", len(val_df))
print("Train classes:", train_df["has_cactus"].value_counts().to_dict())
print("Val classes:", val_df["has_cactus"].value_counts().to_dict())

train_generator = datagen.flow_from_dataframe(
    dataframe=train_df,
    directory=train_dir,
    x_col="id",
    y_col="has_cactus",
    class_mode="binary",
    batch_size=batch_size,
    target_size=(150, 150),
    shuffle=True,
    seed=42,
)

validation_generator = datagen.flow_from_dataframe(
    dataframe=val_df,
    directory=train_dir,
    x_col="id",
    y_col="has_cactus",
    class_mode="binary",
    batch_size=50,
    target_size=(150, 150),
    shuffle=False,
)



## === cell 9
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation="relu", input_shape=(150, 150, 3)))
model.add(layers.MaxPool2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation="relu", input_shape=(150, 150, 3)))
model.add(layers.MaxPool2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation="relu", input_shape=(150, 150, 3)))
model.add(layers.MaxPool2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation="relu", input_shape=(150, 150, 3)))
model.add(layers.MaxPool2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(512, activation="relu"))
model.add(layers.Dense(128, activation="relu"))
model.add(layers.Dense(1, activation="sigmoid"))



## === cell 10
model.summary()



## === cell 11
model.compile(
    loss="binary_crossentropy", optimizer=optimizers.RMSprop(), metrics=["accuracy"]
)



## === cell 12
epochs = 15
steps_per_epoch = max(1, int(np.ceil(train_generator.n / train_generator.batch_size)))
validation_steps = max(
    1, int(np.ceil(validation_generator.n / validation_generator.batch_size))
)

history = model.fit(
    train_generator,
    steps_per_epoch=steps_per_epoch,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=validation_steps,
    verbose=2,
)



## === cell 13
acc_key = "accuracy" if "accuracy" in history.history else "acc"
val_acc_key = "val_accuracy" if "val_accuracy" in history.history else "val_acc"

acc = history.history.get(acc_key, [])
acc_val = history.history.get(val_acc_key, [])

epochs_ = range(0, epochs)
plt.figure(figsize=(6, 4))
plt.plot(epochs_, acc, label="training accuracy")
if len(acc_val) == len(acc):
    plt.scatter(list(epochs_), acc_val, label="validation accuracy")
plt.xlabel("no of epochs")
plt.ylabel("accuracy")
plt.title("no of epochs vs accuracy")
plt.legend()
plt.show()



## === cell 14
loss = history.history.get("loss", [])
val_loss = history.history.get("val_loss", [])

epochs_ = range(0, epochs)
plt.figure(figsize=(6, 4))
plt.plot(epochs_, loss, label="training loss")
if len(val_loss) == len(loss):
    plt.scatter(list(epochs_), val_loss, label="validation loss")
plt.xlabel("No of epochs")
plt.ylabel("loss")
plt.title("no of epochs vs loss")
plt.legend()
plt.show()



## === cell 15
test_generetor = ImageDataGenerator(rescale=1.0 / 255.0)



## === cell 16
test_gen = None



## === cell 17
test_features = []
Test_images = []

test_files = sorted(os.listdir(test_dir))
for img_id in tqdm(test_files, desc="Loading test images"):
    img_path = os.path.join(test_dir, img_id)
    img = cv2.imread(img_path)
    if img is None:
        continue
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (150, 150))
    test_features.append(img)
    Test_images.append(img_id)

test_features = np.asarray(test_features, dtype=np.float32) / 255.0
print(
    "Loaded test images:", len(Test_images), "test_features shape:", test_features.shape
)



## === cell 18
test_features.shape



## === cell 19
if not isinstance(test_features, np.ndarray):
    test_features = np.asarray(test_features, dtype=np.float32)

if test_features.size == 0 or len(Test_images) == 0:
    print("WARNING: No test images were loaded. Falling back to 0.5 predictions.")
    submissions = df_test[["id"]].copy()
    submissions["has_cactus"] = 0.5
else:
    test_predictions = model.predict(test_features, batch_size=64, verbose=0).reshape(
        -1
    )

    alpha = 0.02
    test_predictions = 0.5 + alpha * (test_predictions - 0.5)

    submissions = pd.DataFrame({"id": Test_images, "has_cactus": test_predictions})

    submissions = df_test[["id"]].merge(submissions, on="id", how="left")
    submissions["has_cactus"] = submissions["has_cactus"].fillna(0.5).astype(float)
    submissions["has_cactus"] = submissions["has_cactus"].clip(0.0, 1.0)



## === cell 20
submissions.head()



## === cell 21
pass



## === cell 22
submissions.head()



## === cell 23
submissions.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", submissions.shape)
print(submissions.head())
