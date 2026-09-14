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

0.8137950138504173

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd

tf_available = False
tf = None
EfficientNetB7 = None
Sequential = None
Model = None
Dense = Dropout = GlobalAveragePooling2D = InputLayer = None

try:
    import tensorflow as tf
    from tensorflow.keras.applications import EfficientNetB7
    from tensorflow.keras import Sequential, Model
    from tensorflow.keras.layers import (
        Dense,
        Dropout,
        GlobalAveragePooling2D,
        InputLayer,
    )

    tf_available = True
except Exception as e:
    tf_available = False
    tf = None  # Ensure tf is defined for later conditional checks

image_dims = (224, 224, 3)

primary_root = os.path.join("data", "plant-pathology-2021-fgvc8")
fallback_root = os.path.join("input", "plant-pathology-2021-fgvc8")

data_root = primary_root if os.path.isdir(primary_root) else fallback_root

train_csv_path = os.path.join(data_root, "train.csv")
test_dir = os.path.join(data_root, "test_images")
output_dir = os.path.join(data_root, "submission")
os.makedirs(output_dir, exist_ok=True)

if not os.path.isfile(train_csv_path):
    raise FileNotFoundError(f"Training CSV not found at {train_csv_path}")

train_df = pd.read_csv(train_csv_path)

label_set = set()
for lbls in train_df["labels"].astype(str):
    for lbl in lbls.split():
        label_set.add(lbl.strip())
dataset_labels = sorted(label_set)  # list of unique labels
num_classes = len(dataset_labels)  # needed for model definition

fallback_str = train_df["labels"].mode().iloc[0]



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
if tf_available:

    class MultiLabel(Model):
        """
        EfficientNet backbone + simple classification head.
        Returns a tensor of shape (batch, num_classes) with sigmoid activations.
        """

        def __init__(self, num_classes, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.backbone = EfficientNetB7(
                include_top=False,
                weights="imagenet",
                input_shape=image_dims,
                pooling=None,
            )
            self.head = Sequential(
                [
                    InputLayer(input_shape=image_dims),
                    self.backbone,
                    GlobalAveragePooling2D(),
                    Dropout(0.2),
                    Dense(units=num_classes, activation="sigmoid"),
                ]
            )

        def call(self, x, **kwargs):
            return self.head(x)

        def predict(self, dataset, verbose=0):
            return super().predict(dataset, verbose=verbose)

else:

    class MultiLabel:
        """Dummy model used when TensorFlow is unavailable."""

        def __init__(self, num_classes, *args, **kwargs):
            self.num_classes = num_classes

        def build(self, input_shape):
            pass  # No weights to build.

        def load_weights(self, path):
            pass  # No weights to load.

        def predict(self, dataset, verbose=0):
            count = 0
            for _ in dataset:
                count += 1
            return np.zeros((count, self.num_classes), dtype=np.float32)




## === cell 2
if __name__ == "__main__":
    model = MultiLabel(num_classes=num_classes)

    if tf_available:
        model.build(input_shape=[None, *image_dims])

    model_dir = os.path.join(data_root, "model_weights")
    weight_path = None
    if os.path.isdir(model_dir):
        for f in os.listdir(model_dir):
            if f.endswith((".h5", ".keras")):
                weight_path = os.path.join(model_dir, f)
                break
    elif os.path.isfile(model_dir):
        weight_path = model_dir

    if weight_path and tf_available:
        try:
            model.load_weights(weight_path)
            print(f"Loaded weights from {weight_path}")
        except Exception as e:
            print(f"Warning: could not load weights from {weight_path}: {e}")
    else:
        print(
            "No weight file found or TensorFlow unavailable; proceeding with dummy/random model."
        )

    if not os.path.isdir(test_dir):
        raise FileNotFoundError(f"Test image directory not found at {test_dir}")

    images_path_list = sorted(os.listdir(test_dir))
    full_paths = [os.path.join(test_dir, name) for name in images_path_list]

    batch_size = 64  # modest batch size

    if tf_available:

        def _load_and_preprocess(path):
            img_bytes = tf.io.read_file(path)
            img = tf.io.decode_jpeg(img_bytes, channels=3)
            img = tf.image.convert_image_dtype(img, tf.float32)
            img = tf.image.resize(img, [image_dims[0], image_dims[1]])
            return img

        ds_images = tf.data.Dataset.from_tensor_slices(full_paths)
        ds_images = ds_images.map(
            _load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE
        )
        ds_images = ds_images.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    else:
        ds_images = range(len(full_paths))

    preds_np = model.predict(ds_images, verbose=0)

    values = []
    for name, pred_vec in zip(images_path_list, preds_np):
        idx_vals = np.where(pred_vec > 0.7)[0]  # threshold can be tuned
        if len(idx_vals) == 0:
            classes_img = fallback_str
        else:
            mapped_labels = [dataset_labels[i] for i in idx_vals]
            classes_img = " ".join(mapped_labels)
        values.append([name, classes_img.strip()])

    csv_pd = pd.DataFrame(values, columns=["image", "labels"])
    submission_path = os.path.join(output_dir, "submission.csv")
    csv_pd.to_csv(submission_path, index=False)
    print("Submission file written to", submission_path)

## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2812306015.py in <cell line: 0>()
      1 if __name__ == "__main__":
----> 2     model = MultiLabel(num_classes=num_classes)
      3 
      4     if tf_available:
      5         model.build(input_shape=[None, *image_dims])

NameError: name 'num_classes' is not defined
