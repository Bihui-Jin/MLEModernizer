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

3.10

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

# 5. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

os.environ.setdefault("TF_XLA_FLAGS", "--tf_xla_auto_jit=2")
os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "2")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "2")

import tensorflow as tf

tf.random.set_seed(42)
np.random.seed(42)

try:
    tf.config.threading.set_intra_op_parallelism_threads(
        int(os.environ["TF_NUM_INTRAOP_THREADS"])
    )
    tf.config.threading.set_inter_op_parallelism_threads(
        int(os.environ["TF_NUM_INTEROP_THREADS"])
    )
except Exception:
    pass

output_dir = "./"

test_dir_candidates = [
    "../input/plant-pathology-2021-fgvc8/test_images/",
    "/kaggle/input/plant-pathology-2021-fgvc8/test_images/",
    "../kaggle/input/plant-pathology-2021-fgvcvc8/test_images/",
    "../kaggle/input/plant-pathology-2021-fgvc8/test_images/",
]
test_dir = None
for c in test_dir_candidates:
    if os.path.isdir(c):
        test_dir = c
        break
if test_dir is None:
    raise FileNotFoundError(
        f"Could not find test_images/ in any of: {test_dir_candidates}"
    )

model_dir = "../input/conve01/eff7-e17/epoch-17"
resnet50_weights = "../input/resnet50weights/last_epoch-20"
efficientB7 = "../input/efficientb7/effb7"
resnet50 = "../input/resnet50/Model-Resnet"

image_dims = (300, 300, 3)

train_csv_candidates = [
    "../input/plant-pathology-2021-fgvc8/train.csv",
    "/kaggle/input/plant-pathology-2021-fgvc8/train.csv",
    "../kaggle/input/plant-pathology-2021-fgvc8/train.csv",
]
train_csv_path = None
for c in train_csv_candidates:
    if os.path.isfile(c):
        train_csv_path = c
        break
if train_csv_path is None:
    raise FileNotFoundError(
        f"Could not find train.csv in any of: {train_csv_candidates}"
    )

data_set = pd.read_csv(train_csv_path)
df_labels = data_set["labels"]
one_hot = df_labels.str.get_dummies(sep=" ")
dataset_labels = one_hot.columns.to_list()

print("TF version:", tf.__version__)
print("Test dir:", test_dir)
print("Train csv:", train_csv_path)
print("Num label columns from train.csv:", len(dataset_labels), dataset_labels)



## === cell 1
from tensorflow.keras import Sequential, Model
from tensorflow.keras.layers import (
    Dense,
    BatchNormalization,
    Dropout,
    GlobalMaxPool2D,
    Conv2D,
    InputLayer,
    Lambda,
)


def _build_backbone_layer(backbone_path: str):
    if os.path.isdir(backbone_path):
        try:
            loaded = tf.keras.models.load_model(backbone_path, compile=False)
            loaded.trainable = False
            return loaded
        except Exception as e:
            print("WARNING: load_model failed for backbone dir:", backbone_path)
            print("Reason:", repr(e))
            if os.path.isfile(
                os.path.join(backbone_path, "saved_model.pb")
            ) or os.path.isfile(os.path.join(backbone_path, "saved_model.pbtxt")):
                layer = tf.keras.layers.TFSMLayer(
                    backbone_path, call_endpoint="serving_default"
                )
                return layer

    if os.path.isfile(backbone_path) and backbone_path.lower().endswith(
        (".h5", ".keras")
    ):
        loaded = tf.keras.models.load_model(backbone_path, compile=False)
        loaded.trainable = False
        return loaded

    print(
        f"WARNING: Backbone not found/recognized at: {backbone_path}. Using identity backbone."
    )
    return Lambda(lambda x: x, name="identity_backbone")


class MultiLabel(Model):
    def __init__(self):
        super().__init__()

        self.model_backbone = _build_backbone_layer(resnet50)

        self.model = Sequential()
        self.model.add(InputLayer(input_shape=image_dims))
        self.model.add(self.model_backbone)
        self.model.add(Conv2D(filters=1024, kernel_size=(1, 1), padding="same"))
        self.model.add(BatchNormalization(momentum=0.7))
        self.model.add(Dropout(0.2))
        self.model.add(Conv2D(filters=256, kernel_size=(1, 1), padding="same"))
        self.model.add(BatchNormalization(momentum=0.7))
        self.model.add(Dropout(0.1))
        self.model.add(Conv2D(filters=128, kernel_size=(1, 1), padding="same"))
        self.model.add(GlobalMaxPool2D())
        self.model.add(Dense(units=6, activation="sigmoid"))

    def call(self, predict_input):
        return self.model(predict_input)

    def create_model(self):
        return self.model




## === cell 2
if __name__ == "__main__":
    model = MultiLabel()
    model.build(input_shape=[None, image_dims[0], image_dims[1], image_dims[2]])

    try:
        model.load_weights(resnet50_weights)
        print("Loaded weights:", resnet50_weights)
    except Exception as e:
        print("WARNING: Could not load weights from:", resnet50_weights)
        print("Reason:", repr(e))
        print(
            "Continuing without loading weights (submission will be produced but score may be low)."
        )

    filepaths_py = sorted(tf.io.gfile.glob(os.path.join(test_dir, "*.jpg")))
    if len(filepaths_py) == 0:
        filepaths_py = sorted(tf.io.gfile.glob(os.path.join(test_dir, "*")))
    if len(filepaths_py) == 0:
        raise RuntimeError(f"No test images found in: {test_dir}")
    images_path_list = [os.path.basename(p) for p in filepaths_py]
    n_test = len(filepaths_py)

    AUTOTUNE = tf.data.AUTOTUNE
    BATCH_SIZE = 256

    @tf.function(reduce_retracing=True, jit_compile=False)
    def _load_and_preprocess(fp):
        img_bytes = tf.io.read_file(fp)
        img = tf.io.decode_jpeg(img_bytes, channels=3, fancy_upscaling=False)
        img = tf.image.resize(
            img, [image_dims[0], image_dims[1]], method="bilinear", antialias=False
        )
        img = tf.image.convert_image_dtype(img, tf.float32)
        img.set_shape([image_dims[0], image_dims[1], image_dims[2]])
        return img

    options = tf.data.Options()
    options.experimental_deterministic = False
    options.experimental_optimization.apply_default_optimizations = True
    options.experimental_optimization.map_parallelization = True

    ds = tf.data.Dataset.from_tensor_slices(tf.constant(filepaths_py)).with_options(
        options
    )
    ds = ds.map(_load_and_preprocess, num_parallel_calls=AUTOTUNE, deterministic=False)
    ds = ds.apply(tf.data.experimental.ignore_errors())
    ds = ds.batch(BATCH_SIZE, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)

    preds = model.predict(ds, verbose=0)
    preds = np.asarray(preds, dtype=np.float32)

    if preds.shape[0] != n_test:
        n_test = int(preds.shape[0])
        images_path_list = images_path_list[:n_test]

    n_out = int(preds.shape[1])
    if len(dataset_labels) != n_out:
        print(
            f"WARNING: train.csv has {len(dataset_labels)} label columns but model outputs {n_out}. "
            "Using a safe fallback class list to prevent misalignment."
        )
        classes = np.asarray([f"class_{i}" for i in range(n_out)], dtype=object)
    else:
        classes = np.asarray(dataset_labels, dtype=object)

    above = preds > 0.5

    rows, cols = np.where(above)
    labels_out = np.full((above.shape[0],), "healthy", dtype=object)
    if rows.size:
        order = np.argsort(rows, kind="stable")
        rows_s = rows[order]
        cols_s = cols[order]

        split_idx = np.flatnonzero(np.diff(rows_s)) + 1
        row_starts = np.r_[0, split_idx]
        row_ends = np.r_[split_idx, rows_s.size]

        for s, e in zip(row_starts, row_ends):
            r = int(rows_s[s])
            labels_out[r] = " ".join(classes[cols_s[s:e]].tolist()).strip()

    csv_pd = pd.DataFrame({"image": images_path_list, "labels": labels_out.tolist()})
    csv_path = os.path.join(output_dir, "submission.csv")
    csv_pd.to_csv(csv_path, index=False)
    print("Wrote:", csv_path, "rows:", len(csv_pd))
    print(csv_pd.head())
