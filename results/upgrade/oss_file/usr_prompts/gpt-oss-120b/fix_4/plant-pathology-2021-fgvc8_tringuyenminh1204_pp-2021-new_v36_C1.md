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

No external packages required in the script and installed.

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

0.7931117266851357

# 6. Current score

0.21629

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.28656) has done: 'I remove the failing imports and the nonexistent model load, then replace the model‑based prediction with a simple baseline that assigns the most frequent label from the training data to every test image. This fixes the runtime errors, ensures the prediction list matches the number of test files, and writes a correctly‑formatted `submission.csv` so the notebook runs end‑to‑end. The changes are minimal and keep the original workflow structure.'
- What this solution (achieved 0.38173) has done: 'I wrap the TensorFlow import in a try‑except to avoid the protobuf error and keep the script runnable. Then I expand the baseline prediction from a single most‑common label to the three most frequent labels (space‑delimited) so the submission contains a richer multi‑label prediction, which should raise the F1 score toward the target while preserving the original workflow.'
- What this solution (achieved 0.21629) has done: 'I fix the TensorFlow import by setting the protobuf implementation environment variable, then add a lightweight image‑classification pipeline using a small MobileNetV2 backbone trained for a few epochs on the training images. This replace the naïve “most‑common labels” baseline with a learned multi‑label model, improving the mean F1‑score while keeping the overall workflow and file paths unchanged. The script now creates a proper training dataset, trains the model, generates predictions on the test set, thresholds them, and writes a correctly formatted `submission.csv`.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

try:
    import tensorflow as tf

    print("TensorFlow version:", tf.__version__)
except Exception as e:
    print("TensorFlow import failed:", e)
    raise



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
import pandas as pd, re
from collections import Counter

train_dir = "../input/plant-pathology-2021-fgvc8/train_images"
test_dir = "../input/plant-pathology-2021-fgvc8/test_images"
train_csv_path = "../input/plant-pathology-2021-fgvc8/train.csv"



## === cell 2
train_df = pd.read_csv(train_csv_path)

all_labels = set()
for lbls in train_df["labels"]:
    all_labels.update(lbls.split())
all_labels = sorted(all_labels)
label_to_idx = {lbl: i for i, lbl in enumerate(all_labels)}
idx_to_label = {i: lbl for lbl, i in label_to_idx.items()}
num_classes = len(all_labels)
print("Number of classes:", num_classes)


def encode_labels(label_str):
    vec = [0] * num_classes
    for lbl in label_str.split():
        vec[label_to_idx[lbl]] = 1
    return vec


train_df["label_vec"] = train_df["labels"].apply(encode_labels)



## === cell 3
image_size = (128, 128)
batch_size = 32


def preprocess_path(path, label):
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, image_size)
    image = image / 255.0
    return image, label


train_paths = [os.path.join(train_dir, fname) for fname in train_df["image"]]
train_labels = list(train_df["label_vec"])

train_ds = tf.data.Dataset.from_tensor_slices((train_paths, train_labels))
train_ds = train_ds.map(preprocess_path, num_parallel_calls=tf.data.AUTOTUNE)
train_ds = train_ds.shuffle(buffer=1024).batch(batch_size).prefetch(tf.data.AUTOTUNE)



## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2159915644.py in <cell line: 0>()
     18 train_ds = tf.data.Dataset.from_tensor_slices((train_paths, train_labels))
     19 train_ds = train_ds.map(preprocess_path, num_parallel_calls=tf.data.AUTOTUNE)
---> 20 train_ds = train_ds.shuffle(buffer=1024).batch(batch_size).prefetch(tf.data.AUTOTUNE)
     21 

TypeError: DatasetV2.shuffle() got an unexpected keyword argument 'buffer'

## === cell 4
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(*image_size, 3), include_top=False, weights="imagenet", pooling="avg"
)
base_model.trainable = False  # freeze pretrained weights

inputs = tf.keras.Input(shape=(*image_size, 3))
x = base_model(inputs, training=False)
outputs = tf.keras.layers.Dense(num_classes, activation="sigmoid")(x)

model = tf.keras.Model(inputs, outputs)
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), loss="binary_crossentropy"
)

model.summary()



## === cell 5
epochs = 3
model.fit(train_ds, epochs=epochs, verbose=1)



## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/2224162337.py in <cell line: 0>()
      1 # Train for a small number of epochs (kept short for speed)
      2 epochs = 3
----> 3 model.fit(train_ds, epochs=epochs, verbose=1)
      4 

/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py in error_handler(*args, **kwargs)
    120             # To get the full stack trace, call:
    121             # `keras.config.disable_traceback_filtering()`
--> 122             raise e.with_traceback(filtered_tb) from None
    123         finally:
    124             del filtered_tb

/usr/local/lib/python3.11/dist-packages/keras/src/layers/input_spec.py in assert_input_compatibility(input_spec, inputs, layer_name)
    243                 if spec_dim is not None and dim is not None:
    244                     if spec_dim != dim:
--> 245                         raise ValueError(
    246                             f'Input {input_index} of layer "{layer_name}" is '
    247                             "incompatible with the layer: "

ValueError: Input 0 of layer "functional" is incompatible with the layer: expected shape=(None, 128, 128, 3), found shape=(128, 128, 3)

## === cell 6
test_files = [
    f
    for f in os.listdir(test_dir)
    if re.search(r"([a-zA-Z0-9\s_\\.\-\(\):])+(\.jpg|\.jpeg|\.png)$", f, re.IGNORECASE)
]
test_paths = [os.path.join(test_dir, f) for f in test_files]

test_ds = tf.data.Dataset.from_tensor_slices(test_paths)
test_ds = test_ds.map(
    lambda p: tf.image.resize(
        tf.image.decode_jpeg(tf.io.read_file(p), channels=3), image_size
    )
    / 255.0,
    num_parallel_calls=tf.data.AUTOTUNE,
)
test_ds = test_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

pred_probs = model.predict(test_ds, verbose=0)

threshold = 0.5
pred_labels = []
for probs in pred_probs:
    lbl_idxs = [i for i, p in enumerate(probs) if p >= threshold]
    if not lbl_idxs:  # ensure at least one label per image
        lbl_idxs = [int(tf.argmax(probs).numpy())]
    lbls = " ".join([idx_to_label[i] for i in lbl_idxs])
    pred_labels.append(lbls)



## === cell 7
submission_df = pd.DataFrame({"image": test_files, "labels": pred_labels})
submission_path = "submission.csv"
submission_df.to_csv(submission_path, index=False)
print(f"Submission file written to {submission_path} with {len(submission_df)} rows.")
display(submission_df.head())
