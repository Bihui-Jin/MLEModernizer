# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


# 1. Kaggle task description

## Task
Determine which of the images have hidden messages embedded using one of three steganography algorithms (JMiPOD, JUNIWARD, UERD).

## Metric
Weighted AUC. Each region of the ROC curve is weighted according to these chosen parameters:

```
tpr_thresholds = [0.0, 0.4, 1.0]
weights = [2, 1]
```

In other words, the area between the true positive rate of 0 and 0.4 is weighted 2X, the area between 0.4 and 1 is now weighed (1X). The total area is normalized by the sum of weights such that the final weighted AUC is between 0 and 1.

## Submission Format
For each `Id` (image) in the test set, you must provide a score that indicates how likely this image contains hidden data: the higher the score, the more it is assumed that image contains secret data. The file should contain a header and have the following format:

```
Id,Label
0001.jpg,0.1
0002.jpg,0.99
0003.jpg,1.2
0004.jpg,-2.2
etc.
```
## Dataset
The only available information on the test set is:

1. Each embedding algorithm is used with the same probability.
2. The payload (message length) is adjusted such that the "difficulty" is approximately the same regardless the content of the image. Images with smooth content are used to hide shorter messages while highly textured images will be used to hide more secret bits. The payload is adjusted in the same manner for testing and training sets.
3. The average message length is 0.4 bit per non-zero AC DCT coefficient.
4. The images are all compressed with one of the three following JPEG quality factors: 95, 90 or 75.

### Files
- `Cover/` contains 75k unaltered images meant for use in training.
- `JMiPOD/` contains 75k examples of the JMiPOD algorithm applied to the cover images.
- `JUNIWARD/`contains 75k examples of the JUNIWARD algorithm applied to the cover images.
- `UERD/` contains 75k examples of the UERD algorithm applied to the cover images.
- `Test/` contains 5k test set images. These are the images for which you are predicting.
- `sample_submission.csv` contains an example submission in the correct format.

# 2. Python version

3.8

# 3. Installed packages

No external packages required in the script and installed.

# 4. Data file paths

```
/
    kaggle/
        data/
            Cover.zip (7.4 GB)
            JMiPOD.zip (7.4 GB)
            JUNIWARD.zip (7.4 GB)
            Test.zip (528.5 MB)
            UERD.zip (7.4 GB)
            description.md (91 lines)
            sample_submission.csv (5001 lines)
            sample_submission.csv.zip (10.7 kB)
            Cover/
                54965.jpg (237.9 kB)
                54517.jpg (126.7 kB)
                ... and 69998 other files
            JMiPOD/
                06809.jpg (36.6 kB)
                42490.jpg (78.8 kB)
                ... and 69998 other files
            JUNIWARD/
                03684.jpg (106.2 kB)
                42131.jpg (144.4 kB)
                ... and 69998 other files
            Test/
                3630.jpg (79.5 kB)
                3197.jpg (208.4 kB)
                ... and 4998 other files
            UERD/
                42300.jpg (47.1 kB)
                59199.jpg (41.9 kB)
                ... and 69998 other files
            alaska2-image-steganalysis/
                Cover.zip (7.4 GB)
                JMiPOD.zip (7.4 GB)
                ... and 6 other files
                Cover/
                    54965.jpg (237.9 kB)
                    54517.jpg (126.7 kB)
                    ... and 69998 other files
                JMiPOD/
                    06809.jpg (36.6 kB)
                    42490.jpg (78.8 kB)
                    ... and 69998 other files
                JUNIWARD/
                    03684.jpg (106.2 kB)
                    42131.jpg (144.4 kB)
                    ... and 69998 other files
                Test/
                    3630.jpg (79.5 kB)
                    3197.jpg (208.4 kB)
                    ... and 4998 other files
                UERD/
                    42300.jpg (47.1 kB)
                    59199.jpg (41.9 kB)
                    ... and 69998 other files
                alaska2-image-steganalysis/
        input/
            Cover.zip (7.4 GB)
            JMiPOD.zip (7.4 GB)
            JUNIWARD.zip (7.4 GB)
            Test.zip (528.5 MB)
            UERD.zip (7.4 GB)
            description.md (91 lines)
            sample_submission.csv (5001 lines)
            sample_submission.csv.zip (10.7 kB)
            Cover/
                54965.jpg (237.9 kB)
                54517.jpg (126.7 kB)
                ... and 69998 other files
            JMiPOD/
                06809.jpg (36.6 kB)
                42490.jpg (78.8 kB)
                ... and 69998 other files
            JUNIWARD/
                03684.jpg (106.2 kB)
                42131.jpg (144.4 kB)
                ... and 69998 other files
            Test/
                3630.jpg (79.5 kB)
                3197.jpg (208.4 kB)
                ... and 4998 other files
            UERD/
                42300.jpg (47.1 kB)
                59199.jpg (41.9 kB)
                ... and 69998 other files
            alaska2-image-steganalysis/
                Cover.zip (7.4 GB)
                JMiPOD.zip (7.4 GB)
                ... and 6 other files
                Cover/
                    54965.jpg (237.9 kB)
                    54517.jpg (126.7 kB)
                    ... and 69998 other files
                JMiPOD/
                    06809.jpg (36.6 kB)
                    42490.jpg (78.8 kB)
                    ... and 69998 other files
                JUNIWARD/
                    03684.jpg (106.2 kB)
                    42131.jpg (144.4 kB)
                    ... and 69998 other files
                Test/
                    3630.jpg (79.5 kB)
                    3197.jpg (208.4 kB)
                    ... and 4998 other files
                UERD/
                    42300.jpg (47.1 kB)
                    59199.jpg (41.9 kB)
                    ... and 69998 other files
                alaska2-image-steganalysis/
        working/
            alaska2-image-steganalysis/
                Cover.zip (7.4 GB)
                JMiPOD.zip (7.4 GB)
                ... and 6 other files
                Cover/
                    54965.jpg (237.9 kB)
                    54517.jpg (126.7 kB)
                    ... and 69998 other files
                JMiPOD/
                    06809.jpg (36.6 kB)
                    42490.jpg (78.8 kB)
                    ... and 69998 other files
                JUNIWARD/
                    03684.jpg (106.2 kB)
                    42131.jpg (144.4 kB)
                    ... and 69998 other files
                Test/
                    3630.jpg (79.5 kB)
                    3197.jpg (208.4 kB)
                    ... and 4998 other files
                UERD/
                    42300.jpg (47.1 kB)
                    59199.jpg (41.9 kB)
                    ... and 69998 other files
                alaska2-image-steganalysis/
```

-> data/alaska2-image-steganalysis/sample_submission.csv has 5000 rows and 2 columns.
The columns are: Id, Label

-> data/sample_submission.csv has 5000 rows and 2 columns.
The columns are: Id, Label

-> input/alaska2-image-steganalysis/sample_submission.csv has 5000 rows and 2 columns.
The columns are: Id, Label

-> input/sample_submission.csv has 5000 rows and 2 columns.
The columns are: Id, Label

-> working/alaska2-image-steganalysis/sample_submission.csv has 5000 rows and 2 columns.
The columns are: Id, Label

# 5. Code solution

## === cell 0
import numpy as np
import pandas as pd
import os

import tensorflow as tf
import tensorflow.keras.layers as l

from tensorflow.keras.optimizers import Adam

from sklearn.model_selection import train_test_split

SEED = 10
tf.random.set_seed(SEED)
np.random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)

try:
    tf.config.optimizer.set_jit(True)
except Exception:
    pass

print("TensorFlow:", tf.__version__)




## === cell 1
try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.TPUStrategy(tpu)
    print("Using TPU:", tpu.master())
except Exception as e:
    strategy = tf.distribute.get_strategy()
    print("Using default strategy (GPU/CPU). TPU not available:", repr(e))

print("Replicas:", strategy.num_replicas_in_sync)




## === cell 2
AUTO = tf.data.AUTOTUNE

BASE_PATH = "/kaggle/input/alaska2-image-steganalysis"
assert os.path.exists(BASE_PATH), f"Missing expected dataset path: {BASE_PATH}"




## === cell 3
sample = pd.read_csv(f"{BASE_PATH}/sample_submission.csv")

BATCH_SIZE = 8 * strategy.num_replicas_in_sync
EPOCHS = 1


def _glob_sorted(folder):
    return sorted(tf.io.gfile.glob(f"{folder}/*.jpg"))


test_paths = _glob_sorted(f"{BASE_PATH}/Test")
test_names = [os.path.basename(p) for p in test_paths]

cover_paths = _glob_sorted(f"{BASE_PATH}/Cover")
jmipod_paths = _glob_sorted(f"{BASE_PATH}/JMiPOD")
juni_paths = _glob_sorted(f"{BASE_PATH}/JUNIWARD")
uerd_paths = _glob_sorted(f"{BASE_PATH}/UERD")

train_paths = cover_paths + jmipod_paths + juni_paths + uerd_paths
train_labels = (
    [0] * len(cover_paths)
    + [1] * len(jmipod_paths)
    + [1] * len(juni_paths)
    + [1] * len(uerd_paths)
)

Train_df = pd.DataFrame(
    {"path": train_paths, "labled": np.asarray(train_labels, dtype=np.int32)}
)
Test_df = (
    pd.DataFrame({"path": test_paths, "name": test_names})
    .sort_values("name")
    .reset_index(drop=True)
)

print(
    "Training set counts:\n",
    pd.Series(train_labels).value_counts().rename(index={0: "Cover", 1: "Stego"}),
)
print("\nTrain sample:\n", Train_df.head(2))
print("\nTest sample:\n", Test_df.head(2))




## === cell 4
X = Train_df["path"].values
y = Train_df["labled"].values

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=SEED, stratify=y
)

X_test = Test_df["path"].values

print("Shapes:", X_train.shape, X_val.shape, X_test.shape)




## === cell 5
IMG_SIZE = (300, 300)
_preprocess = tf.keras.applications.efficientnet.preprocess_input


@tf.function
def decode_image(filename, label=None, image_size=IMG_SIZE):
    bits = tf.io.read_file(filename)
    image = tf.image.decode_jpeg(bits, channels=3)
    image = tf.image.resize(image, image_size, method=tf.image.ResizeMethod.BILINEAR)
    image = tf.cast(image, tf.float32)
    image = _preprocess(image)
    if label is None:
        return image
    return image, tf.cast(label, tf.float32)




## === cell 6
options = tf.data.Options()
options.experimental_deterministic = True

train_dataset = (
    tf.data.Dataset.from_tensor_slices((X_train, y_train))
    .with_options(options)
    .shuffle(1024, seed=SEED, reshuffle_each_iteration=True)
    .map(decode_image, num_parallel_calls=AUTO)
    .repeat()
    .batch(BATCH_SIZE, drop_remainder=True)
    .prefetch(AUTO)
)

valid_dataset = (
    tf.data.Dataset.from_tensor_slices((X_val, y_val))
    .with_options(options)
    .map(decode_image, num_parallel_calls=AUTO)
    .batch(BATCH_SIZE)
    .prefetch(AUTO)
)

test_dataset = (
    tf.data.Dataset.from_tensor_slices(X_test)
    .with_options(options)
    .map(decode_image, num_parallel_calls=AUTO)
    .batch(BATCH_SIZE)
    .prefetch(AUTO)
)




## === cell 7
with strategy.scope():
    backbone = tf.keras.applications.EfficientNetB3(
        include_top=False,
        weights="imagenet",
        input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3),
    )

    backbone.trainable = False

    model = tf.keras.Sequential(
        [
            backbone,
            l.GlobalAveragePooling2D(),
            l.Dropout(0.1),
            l.Dense(1, activation="sigmoid"),
        ]
    )
    opt = Adam(learning_rate=0.002, beta_1=0.9, beta_2=0.999, decay=0.01, amsgrad=False)

    model.compile(
        optimizer=opt,
        loss="binary_crossentropy",
        metrics=["accuracy"],
        steps_per_execution=32,
    )

model.summary()




## === cell 8
STEPS_PER_EPOCH = X_train.shape[0] // BATCH_SIZE
if STEPS_PER_EPOCH < 1:
    STEPS_PER_EPOCH = 1

history = model.fit(
    train_dataset,
    steps_per_epoch=STEPS_PER_EPOCH,
    epochs=EPOCHS,
    validation_data=valid_dataset,
    verbose=1,
)




## === cell 9
model.save("Mymodel.h5")
print("Saved model to Mymodel.h5")




## === cell 10
pred = model.predict(test_dataset, verbose=1)

pred = np.asarray(pred).reshape(-1)[: len(Test_df)]
print("Pred shape:", pred.shape, "min/max:", float(pred.min()), float(pred.max()))




## === cell 11
submission = pd.DataFrame(
    {"Id": Test_df["name"].values, "Label": pred.astype(np.float32)}
)
submission.to_csv("submission.csv", index=False)

print(submission.head())
print("Wrote submission.csv with rows:", len(submission))

assert (
    submission.shape[0] == sample.shape[0]
), "Submission row count must match sample_submission.csv"
assert list(submission.columns) == [
    "Id",
    "Label",
], "Submission columns must be Id,Label"
assert submission["Id"].is_unique, "Ids should be unique"
assert submission["Id"].iloc[0].endswith(".jpg"), "Id should look like a jpg filename"
