# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


# 1. Kaggle task description

## Task
Detect apple diseases from images.

## Metric
Mean column-wise ROC AUC.

## Submission Format
For each image_id in the test set, you must predict a probability for each target variable. The file should contain a header and have the following format:

```
image_id,
test_0,0.25,0.25,0.25,0.25
test_1,0.25,0.25,0.25,0.25
test_2,0.25,0.25,0.25,0.25
etc.
```

## Dataset
Given a photo of an apple leaf, can you accurately assess its health? This competition will challenge you to distinguish between leaves which are healthy, those which are infected with apple rust, those that have apple scab, and those with more than one disease.

**train.csv**

- `image_id`: the foreign key
- combinations: one of the target labels
- healthy: one of the target labels
- rust: one of the target labels
- scab: one of the target labels

**images**

A folder containing the train and test images, in jpg format.

**test.csv**

- `image_id`: the foreign key

**sample_submission.csv**

- `image_id`: the foreign key
- combinations: one of the target labels
- healthy: one of the target labels
- rust: one of the target labels
- scab: one of the target labels

# 2. Python version

3.9

# 3. Installed packages

No external packages required in the script and installed.

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (94 lines)
            images.zip (397.8 MB)
            sample_submission.csv (184 lines)
            sample_submission.csv.zip (682 Bytes)
            test.csv (184 lines)
            test.csv.zip (542 Bytes)
            train.csv (1639 lines)
            train.csv.zip (4.6 kB)
            images/
                Train_370.jpg (133.2 kB)
                Test_59.jpg (220.5 kB)
                ... and 1819 other files
            plant-pathology-2020-fgvc7/
                description.md (94 lines)
                images.zip (397.8 MB)
                ... and 6 other files
                images/
                    Train_370.jpg (133.2 kB)
                    Test_59.jpg (220.5 kB)
                    ... and 1819 other files
                plant-pathology-2020-fgvc7/
        input/
            description.md (94 lines)
            images.zip (397.8 MB)
            sample_submission.csv (184 lines)
            sample_submission.csv.zip (682 Bytes)
            test.csv (184 lines)
            test.csv.zip (542 Bytes)
            train.csv (1639 lines)
            train.csv.zip (4.6 kB)
            images/
                Train_370.jpg (133.2 kB)
                Test_59.jpg (220.5 kB)
                ... and 1819 other files
            plant-pathology-2020-fgvc7/
                description.md (94 lines)
                images.zip (397.8 MB)
                ... and 6 other files
                images/
                    Train_370.jpg (133.2 kB)
                    Test_59.jpg (220.5 kB)
                    ... and 1819 other files
                plant-pathology-2020-fgvc7/
        working/
            plant-pathology-2020-fgvc7/
                description.md (94 lines)
                images.zip (397.8 MB)
                ... and 6 other files
                images/
                    Train_370.jpg (133.2 kB)
                    Test_59.jpg (220.5 kB)
                    ... and 1819 other files
                plant-pathology-2020-fgvc7/
```

-> data/plant-pathology-2020-fgvc7/sample_submission.csv has 183 rows and 5 columns.
The columns are: image_id, healthy, multiple_diseases, rust, scab

-> data/plant-pathology-2020-fgvc7/test.csv has 183 rows and 1 columns.
The columns are: image_id

-> data/plant-pathology-2020-fgvc7/train.csv has 1638 rows and 5 columns.
The columns are: image_id, healthy, multiple_diseases, rust, scab

-> data/sample_submission.csv has 183 rows and 5 columns.
The columns are: image_id, healthy, multiple_diseases, rust, scab

-> data/test.csv has 183 rows and 1 columns.
The columns are: image_id

-> data/train.csv has 1638 rows and 5 columns.
The columns are: image_id, healthy, multiple_diseases, rust, scab

-> (stopped after 10 files for performance)

# 5. Code solution

## === cell 0
import os, numpy as np, pandas as pd, tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split
from tensorflow.keras import mixed_precision

tf.config.optimizer.set_jit(True)
mixed_precision.set_global_policy("mixed_float16")

tf.random.set_seed(2023)
np.random.seed(2023)

print("TensorFlow version:", tf.__version__)



## === cell 1
IMG_SIZE = 224  # input resolution
BATCH_SIZE = 128  # larger batch reduces steps/epoch
AUTO = tf.data.AUTOTUNE
BASE_IMG_DIR = "/kaggle/input/plant-pathology-2020-fgvc7/images"



## === cell 2
train_df = pd.read_csv("/kaggle/input/plant-pathology-2020-fgvc7/train.csv")
test_df = pd.read_csv("/kaggle/input/plant-pathology-2020-fgvc7/test.csv")
sample_submission = pd.read_csv(
    "/kaggle/input/plant-pathology-2020-fgvc7/sample_submission.csv"
)


def format_path(img_id):
    return os.path.join(BASE_IMG_DIR, f"{img_id}.jpg")


train_paths = train_df["image_id"].apply(format_path).values
test_paths = test_df["image_id"].apply(format_path).values

label_cols = ["healthy", "multiple_diseases", "rust", "scab"]
train_labels = train_df[label_cols].values.astype(np.float32)

train_paths, valid_paths, train_labels, valid_labels = train_test_split(
    train_paths,
    train_labels,
    test_size=0.15,
    random_state=2020,
    stratify=train_labels,
)




## === cell 3
def _decode_and_resize(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)  # 0‑1
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
    return img


train_ds = (
    tf.data.Dataset.from_tensor_slices((train_paths, train_labels))
    .map(lambda p, y: (_decode_and_resize(p), y), num_parallel_calls=AUTO)
    .cache()
    .shuffle(buffer_size=len(train_paths), seed=2023)
    .batch(BATCH_SIZE)
    .prefetch(AUTO)
)

valid_ds = (
    tf.data.Dataset.from_tensor_slices((valid_paths, valid_labels))
    .map(lambda p, y: (_decode_and_resize(p), y), num_parallel_calls=AUTO)
    .batch(BATCH_SIZE)
    .prefetch(AUTO)
)

test_ds = (
    tf.data.Dataset.from_tensor_slices(test_paths)
    .map(_decode_and_resize, num_parallel_calls=AUTO)
    .batch(BATCH_SIZE)
    .prefetch(AUTO)
)




## === cell 4
def build_model():
    base = EfficientNetB0(
        weights="imagenet",
        include_top=False,
        pooling="avg",
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
    )
    outputs = Dense(len(label_cols), activation="softmax", dtype="float32")(base.output)
    model = Model(inputs=base.input, outputs=outputs)
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["categorical_accuracy"],
    )
    return model


model = build_model()
model.summary()



## === cell 5
EPOCHS = 5
model.fit(
    train_ds,
    validation_data=valid_ds,
    epochs=EPOCHS,
    verbose=2,
)



## === cell 6
probabilities = model.predict(test_ds, verbose=1)  # shape (num_test, 4)

sub = sample_submission.copy()
sub.loc[:, label_cols] = probabilities
sub.to_csv("submission.csv", index=False)
print("Submission saved to submission.csv")
print(sub.head())
