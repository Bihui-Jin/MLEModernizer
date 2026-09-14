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

0.7456140350877198

# 6. Current score

0.07232

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.21436) has done: 'I fix the crashes by (1) replacing the Xception backbone with a tiny pure‑CNN that works with the installed TensorFlow version, (2) removing the unsupported weight‑loading step, and (3) guarding the test‑image loop against sub‑folders so the image loader receives only actual files. These minimal changes unblock the notebook, let it write a correctly formatted `submission.csv`, and keep the original multi‑label handling logic intact.'
- What this solution (achieved 0.07232) has done: 'I remove the early TensorFlow import that caused a protobuf error, add the import later where needed, and introduce a simple training loop that loads the images, creates multi‑hot label vectors, and fits the tiny CNN for a few epochs. This enables the model to learn from the data, which should raise the mean F1‑Score toward the target while keeping the original architecture and inference code intact. The script now also reliably writes a correctly‑formatted `submission.csv`.'

# 9. Code solution

## === cell 0
import pandas as pd
import numpy as np
import os

print("number of training images")
train_img_dir = "/kaggle/input/plant-pathology-2021-fgvc8/train_images/"
test_img_dir = "/kaggle/input/plant-pathology-2021-fgvc8/test_images/"
print(
    len(
        [
            name
            for name in os.listdir(train_img_dir)
            if os.path.isfile(os.path.join(train_img_dir, name))
        ]
    )
)
print(
    len(
        [
            name
            for name in os.listdir(test_img_dir)
            if os.path.isfile(os.path.join(test_img_dir, name))
        ]
    )
)

training_csv = pd.read_csv("/kaggle/input/plant-pathology-2021-fgvc8/train.csv")
training_class = np.array([])
for labels in pd.unique(training_csv["labels"]):
    training_class = np.append(training_class, labels.split())
training_class = np.unique(training_class)
print("\nnumber of class")
print(training_class)
print(len(training_class))




## === cell 1
def predict2strings(pred, threshold):
    """
    Convert sigmoid outputs to a space‑delimited string of class names.
    `threshold` is a scalar applied to every class.
    """
    return " ".join(training_class[pred >= threshold])


def predict2n_hot(pred, threshold):
    """
    Return a binary (n‑hot) array using the same scalar threshold.
    """
    return np.where(pred >= threshold, 1, 0)


def n_hot2string(n_hot):
    """Convert an n‑hot vector back to the space‑delimited label string."""
    return " ".join(training_class[n_hot == 1])




## === cell 2
import tensorflow as tf

model_input = tf.keras.layers.Input(shape=(299, 299, 3))

x = tf.keras.layers.Rescaling(1.0 / 255.0)(model_input)

x = tf.keras.layers.Conv2D(32, (3, 3), activation="relu", padding="same")(x)
x = tf.keras.layers.MaxPooling2D()(x)
x = tf.keras.layers.Conv2D(64, (3, 3), activation="relu", padding="same")(x)
x = tf.keras.layers.MaxPooling2D()(x)
x = tf.keras.layers.Conv2D(128, (3, 3), activation="relu", padding="same")(x)
x = tf.keras.layers.GlobalAveragePooling2D()(x)

x = tf.keras.layers.Dense(256, activation="relu")(x)
model_output = tf.keras.layers.Dense(len(training_class), activation="sigmoid")(x)

model = tf.keras.Model(inputs=model_input, outputs=model_output)

model.compile(optimizer="adam", loss="binary_crossentropy")



## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 3
train_dir = "/kaggle/input/plant-pathology-2021-fgvc8/train_images/"
train_df = training_csv.copy()
train_df["filepath"] = train_dir + train_df["image"]

label_to_idx = {c: i for i, c in enumerate(training_class)}


def labels_to_multi_hot(label_str):
    vec = np.zeros(len(training_class), dtype=np.float32)
    for lbl in label_str.split():
        vec[label_to_idx[lbl]] = 1.0
    return vec


train_df["multi_hot"] = train_df["labels"].apply(labels_to_multi_hot)

paths = train_df["filepath"].values
labels = np.stack(train_df["multi_hot"].values)

AUTOTUNE = tf.data.AUTOTUNE
dataset = tf.data.Dataset.from_tensor_slices((paths, labels))


def _process(path, label):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [299, 299])
    img = tf.image.per_image_standardization(img)
    return img, label


dataset = dataset.map(_process, num_parallel_calls=AUTOTUNE)
dataset = dataset.shuffle(buffer_size=1000).batch(32).prefetch(AUTOTUNE)

model.fit(dataset, epochs=3, verbose=2)



## === cell 4
threshold = 0.5
test_img_path = "/kaggle/input/plant-pathology-2021-fgvc8/test_images/"


def write_csv_kaggle_tags():
    import csv

    rows = [["image", "labels"]]
    for filename in sorted(os.listdir(test_img_path)):
        full_path = os.path.join(test_img_path, filename)
        if not os.path.isfile(full_path):
            continue
        img = tf.keras.preprocessing.image.load_img(full_path, target_size=(299, 299))
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = tf.image.per_image_standardization(img_array)
        img_array = tf.expand_dims(img_array, axis=0)  # shape (1, 299, 299, 3)

        pred_scores = model(img_array, training=False)[0].numpy()
        label_str = predict2strings(pred_scores, threshold)
        rows.append([filename, label_str])

    output_path = "/kaggle/working/submission.csv"
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(rows)
    print(f"Submission written to {output_path}")




## === cell 5
write_csv_kaggle_tags()
