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
Detect apple diseases from images.

## Metric
Mean F1-Score

## Submission Format
labels should be a space-delimited list.

The file should contain a header and have the following format:

```
image, labels
85f8cb619c66b863.jpg,healthy
ad8770db05586b59.jpg,healthy
c7b03e718489f3ca.jpg,healthy
```

## Dataset
**train.csv** - the training set metadata.

- `image` - the image ID.
- `labels` - the target classes, a space delimited list of all diseases found in the image. Unhealthy leaves with too many diseases to classify visually will have the `complex` class, and may also have a subset of the diseases identified.

**sample_submission.csv** - A sample submission file in the correct format.

- `image`
- `labels`

**train_images** - The training set images.

**test_images** - The test set images. This competition has a hidden test set: only three images are provided here as samples while the remaining 5,000 images will be available to your notebook once it is submitted.

# 2. Python version

3.9

# 3. Installed packages

geopandas==0.14.4
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
protobuf==6.33.0
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

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (101 lines)
            sample_submission.csv (3728 lines)
            sample_submission.csv.zip (39.6 kB)
            test.zip (160 Bytes)
            test_images.zip (3.2 GB)
            train.csv (14906 lines)
            train.csv.zip (171.3 kB)
            train.zip (162 Bytes)
            train_images.zip (12.7 GB)
            plant-pathology-2021-fgvc8/
                description.md (101 lines)
                sample_submission.csv (3728 lines)
                ... and 7 other files
                plant-pathology-2021-fgvc8/
                test_images/
                    df98c83c4d383c2d.jpg (802.8 kB)
                    817e97dad0c33ae0.jpg (667.3 kB)
                    ... and 3725 other files
                    test_images/
                train_images/
                    c19a7aca95e54c35.jpg (1.1 MB)
                    8476bd24bd4b89a5.jpg (985.0 kB)
                    ... and 14903 other files
                    train_images/
            test_images/
                df98c83c4d383c2d.jpg (802.8 kB)
                817e97dad0c33ae0.jpg (667.3 kB)
                ... and 3725 other files
                test_images/
            train_images/
                c19a7aca95e54c35.jpg (1.1 MB)
                8476bd24bd4b89a5.jpg (985.0 kB)
                ... and 14903 other files
                train_images/
        input/
            description.md (101 lines)
            sample_submission.csv (3728 lines)
            sample_submission.csv.zip (39.6 kB)
            test.zip (160 Bytes)
            test_images.zip (3.2 GB)
            train.csv (14906 lines)
            train.csv.zip (171.3 kB)
            train.zip (162 Bytes)
            train_images.zip (12.7 GB)
            plant-pathology-2021-fgvc8/
                description.md (101 lines)
                sample_submission.csv (3728 lines)
                ... and 7 other files
                plant-pathology-2021-fgvc8/
                test_images/
                    df98c83c4d383c2d.jpg (802.8 kB)
                    817e97dad0c33ae0.jpg (667.3 kB)
                    ... and 3725 other files
                    test_images/
                train_images/
                    c19a7aca95e54c35.jpg (1.1 MB)
                    8476bd24bd4b89a5.jpg (985.0 kB)
                    ... and 14903 other files
                    train_images/
            test_images/
                df98c83c4d383c2d.jpg (802.8 kB)
                817e97dad0c33ae0.jpg (667.3 kB)
                ... and 3725 other files
                test_images/
                    df98c83c4d383c2d.jpg (802.8 kB)
                    817e97dad0c33ae0.jpg (667.3 kB)
                    ... and 3725 other files
                    test_images/
            train_images/
                c19a7aca95e54c35.jpg (1.1 MB)
                8476bd24bd4b89a5.jpg (985.0 kB)
                ... and 14903 other files
                train_images/
                    c19a7aca95e54c35.jpg (1.1 MB)
                    8476bd24bd4b89a5.jpg (985.0 kB)
                    ... and 14903 other files
                    train_images/
        working/
            plant-pathology-2021-fgvc8/
                description.md (101 lines)
                sample_submission.csv (3728 lines)
                ... and 7 other files
                plant-pathology-2021-fgvc8/
                test_images/
                    df98c83c4d383c2d.jpg (802.8 kB)
                    817e97dad0c33ae0.jpg (667.3 kB)
                    ... and 3725 other files
                    test_images/
                train_images/
                    c19a7aca95e54c35.jpg (1.1 MB)
                    8476bd24bd4b89a5.jpg (985.0 kB)
                    ... and 14903 other files
                    train_images/
```

-> data/plant-pathology-2021-fgvc8/sample_submission.csv has 3727 rows and 2 columns.
The columns are: image, labels

-> data/plant-pathology-2021-fgvc8/train.csv has 14905 rows and 2 columns.
The columns are: image, labels

-> data/sample_submission.csv has 3727 rows and 2 columns.
The columns are: image, labels

-> data/train.csv has 14905 rows and 2 columns.
The columns are: image, labels

-> input/plant-pathology-2021-fgvc8/sample_submission.csv has 3727 rows and 2 columns.
The columns are: image, labels

-> input/plant-pathology-2021-fgvc8/train.csv has 14905 rows and 2 columns.
The columns are: image, labels

-> (stopped after 10 files for performance)

# 5. Target score

0.5919852262234518

# 6. Current score

0.25629

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.32779) has done: 'I fixed the import‑ and path‑related crashes, skipped directories when iterating over the test folder, and added a simple frequency‑based baseline that predicts the most common disease labels for every test image when the deep‑learning model cannot be built. This guarantees a valid `submission.csv` is written (to the working directory) and moves the solution past the runtime errors toward the target score.'
- What this solution (achieved 0.15848) has done: 'I replace the frequency‑based multi‑label baseline with a single‑label baseline that always predicts the most common disease (typically “healthy”). This avoids the protobuf error that prevents the deep model from loading, ensures a valid `submission.csv` is created, and should raise the mean F1 score toward the target by improving precision while keeping reasonable recall.'
- What this solution (achieved 0.26923) has done: 'I adjust the baseline to predict the ten most frequent disease labels (instead of only the single most common one) so that the multi‑label submission contains several likely classes, which improves the mean F1‑Score while still avoiding the TensorFlow protobuf error. I also renumbered the cells to start from 1 and kept the existing logic unchanged otherwise, ensuring a valid `submission.csv` is written to `/kaggle/working/`.'
- What this solution (achieved 0.20406) has done: 'I adjust the baseline to predict only those disease labels that appear in at least 5 % of the training images instead of always outputting the ten most frequent labels. This reduces noise, improves precision, and should raise the mean F1‑Score toward the target while keeping the existing fallback logic unchanged. The change is confined to the baseline‑creation cell and does not affect any other part of the pipeline.'
- What this solution (achieved 0.25629) has done: 'I replace the frequency‑based label list with a simpler yet stronger baseline: use the most common full label combination from the training set as the prediction for every test image. This keeps the existing fallback logic intact, avoids the TensorFlow loading error, and is likely to raise the mean F1‑Score toward the target while requiring only a small change in the baseline‑creation cell.'

# 9. Code solution

## === cell 0
import pandas as pd
import numpy as np
import os

train_csv_path = "/kaggle/input/plant-pathology-2021-fgvc8/train.csv"
train_images_path = "/kaggle/input/plant-pathology-2021-fgvc8/train_images/"
test_images_path = "/kaggle/input/plant-pathology-2021-fgvc8/test_images/"

print("number of training images:", len(os.listdir(train_images_path)))
print("number of test images:", len(os.listdir(test_images_path)))

training_csv = pd.read_csv(train_csv_path)

training_class = np.array([])
for labels in pd.unique(training_csv["labels"]):
    training_class = np.append(training_class, labels.split())
training_class = np.unique(training_class)
print("\nnumber of classes:", len(training_class))
print(training_class)




## === cell 1
def predict2strings(pred, threshold):
    """
    Convert sigmoid predictions to space‑delimited label string.
    `threshold` can be a scalar or an array of the same length as `training_class`.
    """
    t = np.array(threshold)
    if t.size == 1:
        t = np.full_like(pred, t.item())
    return " ".join(training_class[np.where(pred >= t)])


def predict2n_hot(pred, threshold):
    t = np.array(threshold)
    if t.size == 1:
        t = np.full_like(pred, t.item())
    return np.where(pred >= t, np.ones(pred.shape), np.zeros(pred.shape))


def n_hot2string(n_hot):
    return " ".join(training_class[np.where(n_hot >= 1)])




## === cell 2
label_counts = {lbl: 0 for lbl in training_class}
total_images = len(training_csv)
for lbls in training_csv["labels"]:
    for l in lbls.split():
        label_counts[l] += 1

most_common_combo = training_csv["labels"].value_counts().idxmax()
baseline_pred_str = most_common_combo
print("Baseline prediction (most common label combo):", baseline_pred_str)

threshold = [0.5] * len(training_class)


def write_csv_kaggle_tags():
    import csv

    rows = [["image", "labels"]]
    for filename in os.listdir(test_images_path):
        if os.path.isdir(os.path.join(test_images_path, filename)):
            continue
        img_path = os.path.join(test_images_path, filename)

        if model_available:
            image = tf.keras.preprocessing.image.load_img(
                img_path, target_size=(299, 299)
            )
            image = tf.keras.preprocessing.image.img_to_array(image)
            image = tf.image.per_image_standardization(image)
            image = tf.expand_dims(image, axis=0)
            pred_score = model(image)[0].numpy()
            label_str = predict2strings(pred_score, threshold=threshold)
        else:
            label_str = baseline_pred_str

        rows.append([filename, label_str])

    submission_path = "/kaggle/working/submission.csv"
    with open(submission_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(rows)
    print(f"submission.csv written to {submission_path} with {len(rows) - 1} records.")




## === cell 3
try:
    import tensorflow as tf

    model_input = tf.keras.layers.Input(shape=(299, 299, 3))
    xception_layer = tf.keras.applications.Xception(
        include_top=False,
        weights="imagenet",
        input_shape=(299, 299, 3),
        input_tensor=model_input,
    ).output
    at = tf.keras.layers.Conv2D(2048, 3, padding="same", activation="relu")(
        xception_layer
    )
    at = tf.keras.layers.BatchNormalization()(at)
    at = tf.keras.layers.Conv2D(2048, 1, padding="same", activation="relu")(at)
    at = tf.keras.layers.BatchNormalization()(at)
    at = tf.keras.layers.Conv2D(1, 1, padding="same", activation="relu")(at)
    attention_heatmap = tf.keras.layers.Softmax(name="attention_layer")(at)
    attention = tf.keras.layers.Multiply()([attention_heatmap, xception_layer])
    fusion_attention_spectrum = tf.keras.layers.Add()([attention, xception_layer])
    fusion_attention_spectrum = tf.keras.layers.BatchNormalization()(
        fusion_attention_spectrum
    )
    fc_1 = tf.keras.layers.Flatten()(fusion_attention_spectrum)
    fc_1 = tf.keras.layers.Dense(2048, activation="relu")(fc_1)
    fc_1 = tf.keras.layers.BatchNormalization()(fc_1)
    fc_2 = tf.keras.layers.Dense(4096 + 2048, activation="relu")(fc_1)
    fc_2 = tf.keras.layers.BatchNormalization()(fc_2)
    average_pooling = tf.keras.layers.GlobalAveragePooling2D()(
        fusion_attention_spectrum
    )
    fc_2 = tf.keras.layers.concatenate([fc_2, average_pooling])
    model_output = tf.keras.layers.Dense(len(training_class), activation="sigmoid")(
        fc_2
    )
    model = tf.keras.Model(inputs=model_input, outputs=model_output)

    try:
        model.load_weights(
            "../input/fork-of-cs5489-project-train-cont-12/my_model/attention_1"
        )
        print("Custom weights loaded.")
    except Exception as e:
        print("Custom weights not found; using ImageNet pretrained backbone.", e)

    model_available = True
except Exception as e:
    print(
        "Model construction failed (likely protobuf/TensorFlow issue). Falling back to baseline.",
        e,
    )
    model = None
    model_available = False




## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 4
write_csv_kaggle_tags()
