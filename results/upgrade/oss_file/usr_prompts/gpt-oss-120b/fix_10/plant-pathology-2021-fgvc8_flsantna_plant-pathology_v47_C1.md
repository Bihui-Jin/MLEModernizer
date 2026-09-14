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

0.7607940904893831

# 6. Current score

0.28656

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved 0.28656) has done: 'I make three focused fixes: (1) force TensorFlow to be treated as unavailable so the fallback “always‑predict‑top‑k” branch runs safely, (2) locate the training CSV and test image folder using the standard Kaggle `/kaggle/input/...` paths (with a fallback search), and (3) ensure `num_classes` and related variables are defined before they are used in the main block. These changes resolve the import error, the missing‑file error, and the NameError, letting the notebook run end‑to‑end and output a valid `submission.csv`.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd

try:
    import tensorflow as tf

    _TF_AVAILABLE = True
except Exception:
    tf = None
    _TF_AVAILABLE = False

_TF_AVAILABLE = False




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
def find_file(relative_parts):
    """
    Search for a file by joining the given path parts to the root
    '/kaggle/input' and also checking the current working directory.
    Returns the first existing path or None.
    """
    candidates = [
        os.path.join("/kaggle", "input", *relative_parts),
        os.path.join(*relative_parts),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return None


train_csv_path = find_file(["plant-pathology-2021-fgvc8", "train.csv"])
if train_csv_path is None:
    raise FileNotFoundError(
        "Training CSV not found in expected Kaggle input locations."
    )
data_set = pd.read_csv(train_csv_path)

df_labels = data_set["labels"]
one_hot = df_labels.str.get_dummies(sep=" ")
dataset_labels = one_hot.columns.tolist()
num_classes = len(dataset_labels)

class_counts = one_hot.sum(axis=0).values
class_probs = class_counts / len(df_labels)  # probability for each class

avg_labels_per_image = (
    df_labels.str.split().apply(len).mean()
)  # e.g., 1.23 → round to nearest int later

test_dir = find_file(["plant-pathology-2021-fgvc8", "test_images"])
if test_dir is None or not os.path.isdir(test_dir):
    raise FileNotFoundError(f"Test directory not found: {test_dir}")

image_dims = (300, 300, 3)
output_dir = "./"
model_dir = "/kaggle/input/plant-pathology-2021-fgvc8/effb7-e8/epoch-8"  # optional checkpoint directory (may not exist)




## === cell 2
if _TF_AVAILABLE:
    from tensorflow.keras.applications import EfficientNetB7
    from tensorflow.keras import Model
    from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout

    class MultiLabel(Model):
        def __init__(self, num_classes, **kwargs):
            super().__init__(**kwargs)
            self.backbone = EfficientNetB7(
                include_top=False,
                weights="imagenet",
                input_shape=image_dims,
            )
            self.pool = GlobalAveragePooling2D()
            self.dropout = Dropout(0.2)
            self.classifier = Dense(num_classes, activation="sigmoid")

        def call(self, inputs, training=False):
            x = self.backbone(inputs, training=training)
            x = self.pool(x)
            x = self.dropout(x, training=training)
            return self.classifier(x)

else:

    class MultiLabel:
        def __init__(self, num_classes, **kwargs):
            self.num_classes = num_classes

        def build(self, input_shape):
            pass

        def __call__(self, inputs, training=False):
            return np.zeros((inputs.shape[0], self.num_classes), dtype=np.float32)




## === cell 3
if __name__ == "__main__":
    model = MultiLabel(num_classes=num_classes)
    if _TF_AVAILABLE:
        model.build(input_shape=[None, *image_dims])

        if os.path.isdir(model_dir) or os.path.isfile(model_dir):
            try:
                model.load_weights(model_dir)
            except Exception as e:
                print(f"Warning: could not load weights from {model_dir}: {e}")

    images_path_list = sorted(
        [f for f in os.listdir(test_dir) if f.lower().endswith(".jpg")]
    )

    results = []

    if _TF_AVAILABLE:
        batch_size = 32
        test_paths = [os.path.join(test_dir, fname) for fname in images_path_list]

        def _load_path(path):
            """Load, decode, resize, and scale a single image; also return its filename."""
            name = tf.strings.split(path, os.sep)[-1]
            img = tf.io.read_file(path)
            img = tf.io.decode_image(img, channels=3, dtype=tf.dtypes.float32)
            img = tf.image.resize(img, [image_dims[0], image_dims[1]])
            img = img * 255.0
            return name, img

        ds = tf.data.Dataset.from_tensor_slices(test_paths)
        ds = ds.map(_load_path, num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.batch(batch_size)

        for batch_names, batch_imgs in ds:
            preds_raw = model(batch_imgs, training=False)
            preds = preds_raw.numpy()

            for name_tensor, pred_vec in zip(batch_names, preds):
                name = name_tensor.numpy().decode()
                selected = [
                    dataset_labels[idx] for idx, p in enumerate(pred_vec) if p > 0.5
                ]
                label_str = " ".join(selected)
                results.append([name, label_str])
    else:
        k = max(1, int(round(avg_labels_per_image)))
        top_k_indices = np.argsort(-class_probs)[:k]
        top_k_labels = [dataset_labels[i] for i in top_k_indices]

        for name in images_path_list:
            label_str = " ".join(top_k_labels)
            results.append([name, label_str])

    submission_df = pd.DataFrame(results, columns=["image", "labels"])
    submission_path = os.path.join(output_dir, "submission.csv")
    submission_df.to_csv(submission_path, index=False)
    print(f"Submission saved to {submission_path}")
