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

# 5. Code solution

## === cell 0
import os
import glob
import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import Sequential, Model
from tensorflow.keras.layers import (
    Dense,
    BatchNormalization,
    Dropout,
    GlobalMaxPool2D,
    Conv2D,
    InputLayer,
)



## === cell 1
output_dir = "./"
test_dir = "../input/plant-pathology-2021-fgvc8/test_images/"
model_dir = "../input/conve01/eff7-e9/epoch-9"
efficientB7 = "../input/efficientb7/effb7"

image_dims = (300, 300, 3)

data_set = pd.read_csv("../input/plant-pathology-2021-fgvc8/train.csv")
df_labels = data_set["labels"]
one_hot = df_labels.str.get_dummies(sep=" ")
dataset_labels = one_hot.columns.to_list()
NUM_CLASSES = len(dataset_labels)

print("NUM_CLASSES:", NUM_CLASSES)
print("Labels:", dataset_labels)




## === cell 2
def _find_saved_model_dir(path: str):
    """Return a directory containing a valid TF SavedModel (has saved_model.pb or saved_model.pbtxt)."""
    if not path or not os.path.exists(path):
        return None

    def is_savedmodel_dir(d):
        return os.path.isdir(d) and (
            os.path.exists(os.path.join(d, "saved_model.pb"))
            or os.path.exists(os.path.join(d, "saved_model.pbtxt"))
        )

    if is_savedmodel_dir(path):
        return path

    candidates = []
    for root, dirs, files in os.walk(path):
        if "saved_model.pb" in files or "saved_model.pbtxt" in files:
            candidates.append(root)
        rel_depth = os.path.relpath(root, path).count(os.sep)
        if rel_depth >= 2:
            dirs[:] = []

    if candidates:
        candidates = sorted(candidates, key=lambda p: (len(p), -os.path.getmtime(p)))
        return candidates[0]
    return None


class MultiLabel(Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.using_savedmodel = False
        savedmodel_dir = _find_saved_model_dir(efficientB7)

        self.model_backbone = None
        if savedmodel_dir is not None:
            try:
                self.model_backbone = keras.layers.TFSMLayer(
                    savedmodel_dir, call_endpoint="serving_default"
                )
                self.using_savedmodel = True
            except Exception:
                try:
                    self.model_backbone = keras.layers.TFSMLayer(
                        savedmodel_dir, call_endpoint="serve"
                    )
                    self.using_savedmodel = True
                except Exception:
                    self.model_backbone = None
                    self.using_savedmodel = False

        if self.model_backbone is None:
            backbone = keras.applications.EfficientNetB7(
                include_top=False,
                weights="imagenet",
                input_shape=image_dims,
                pooling=None,
            )
            backbone.trainable = False
            self.model_backbone = backbone
            self.using_savedmodel = False

        self.model = Sequential()
        self.model.add(InputLayer(input_shape=image_dims))
        self.model.add(self.model_backbone)
        self.model.add(Conv2D(filters=1024, kernel_size=(1, 1), padding="same"))
        self.model.add(BatchNormalization(momentum=0.7))
        self.model.add(Dropout(0.2))
        self.model.add(Conv2D(filters=1024, kernel_size=(1, 1), padding="same"))
        self.model.add(
            Conv2D(filters=400 * NUM_CLASSES, kernel_size=(1, 1), padding="same")
        )

        self.pred_conv1 = Conv2D(filters=1024, kernel_size=(1, 1), padding="same")
        self.pred_bn = BatchNormalization()
        self.pred_conv2 = Conv2D(filters=512, kernel_size=(1, 1), padding="same")
        self.pred_pool = GlobalMaxPool2D()
        self.pred_dense = Dense(units=1, activation="sigmoid")

    def _apply_pred_head(self, x):
        x = self.pred_conv1(x)
        x = self.pred_bn(x)
        x = self.pred_conv2(x)
        x = self.pred_pool(x)
        x = self.pred_dense(x)
        return x

    def call(self, x, training=False, **kwargs):
        y = self.model(x, training=training)

        if isinstance(y, dict):
            if "outputs" in y:
                y = y["outputs"]
            else:
                y = next(iter(y.values()))

        shape = tf.shape(y)
        b, h, w = shape[0], shape[1], shape[2]
        y = tf.reshape(y, [b, h, w, NUM_CLASSES, 400])
        y = tf.transpose(y, [0, 3, 1, 2, 4])
        y = tf.reshape(y, [b * NUM_CLASSES, h, w, 400])

        p = self._apply_pred_head(y)  # [B*NUM_CLASSES, 1]
        p = tf.reshape(p, [b, NUM_CLASSES])
        return p

    def create_model(self):
        return self.model




## === cell 3
if __name__ == "__main__":
    tf.random.set_seed(42)
    np.random.seed(42)

    try:
        tf.config.experimental.enable_op_determinism(True)
    except Exception:
        pass

    try:
        tf.config.optimizer.set_jit(True)
    except Exception:
        pass

    try:
        tf.config.threading.set_intra_op_parallelism_threads(
            max(1, (os.cpu_count() or 4) // 2)
        )
        tf.config.threading.set_inter_op_parallelism_threads(2)
    except Exception:
        pass

    model = MultiLabel()
    model.build(input_shape=[None, image_dims[0], image_dims[1], image_dims[2]])

    weights_path = model_dir
    if os.path.isdir(weights_path):
        patterns = ("*.weights.h5", "*.h5", "ckpt*.index", "ckpt*")
        candidates = []
        for pat in patterns:
            candidates.extend(glob.glob(os.path.join(weights_path, pat)))
        candidates = [c for c in candidates if os.path.isfile(c)]
        if candidates:
            weights_path = max(candidates, key=os.path.getmtime)
        else:
            weights_path = None

    if weights_path is not None and os.path.exists(weights_path):
        try:
            model.load_weights(weights_path)
            print("Loaded weights:", weights_path)
        except Exception:
            ckpt_prefix = weights_path
            if ckpt_prefix.endswith(".index"):
                ckpt_prefix = ckpt_prefix[: -len(".index")]
            model.load_weights(ckpt_prefix)
            print("Loaded checkpoint weights:", ckpt_prefix)
    else:
        print("No external weights found; using backbone default weights (if any).")

    image_files = sorted(
        [
            os.path.join(test_dir, p)
            for p in os.listdir(test_dir)
            if p.lower().endswith(".jpg")
        ]
    )

    AUTOTUNE = tf.data.AUTOTUNE

    BATCH_SIZE = 64

    @tf.function(
        reduce_retracing=True,
        input_signature=[tf.TensorSpec(shape=(), dtype=tf.string)],
    )
    def _load_and_preprocess(path):
        input_img = tf.io.read_file(path)
        image = tf.io.decode_jpeg(input_img, channels=3, dct_method="INTEGER_FAST")
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)  # [0,1]
        image = tf.image.resize(image, [image_dims[0], image_dims[1]])
        image = image * 255.0
        image = tf.ensure_shape(image, image_dims)
        name = tf.strings.split(path, os.sep)[-1]
        return name, image

    options = tf.data.Options()
    options.experimental_deterministic = True
    try:
        options.experimental_optimization.apply_default_optimizations = True
        options.experimental_optimization.autotune_buffers = True
        options.experimental_optimization.autotune_cpu_budget = 0
        options.experimental_optimization.map_parallelization = True
    except Exception:
        pass

    files_tf = tf.constant(image_files)
    ds = tf.data.Dataset.from_tensor_slices(files_tf).with_options(options)
    ds = ds.map(_load_and_preprocess, num_parallel_calls=AUTOTUNE)
    ds = ds.apply(tf.data.experimental.ignore_errors())
    ds = ds.batch(BATCH_SIZE, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)

    classes = np.asarray(dataset_labels, dtype=object)
    thr = 0.7

    @tf.function(jit_compile=True, reduce_retracing=True)
    def _predict_mask(images):
        p = model(images, training=False)
        return p > thr

    values = []
    for names_b, images_b in ds:
        mask = _predict_mask(images_b).numpy()  # [B, C] bool
        names_np = names_b.numpy().astype("U")  # bytes -> str

        cls = classes
        append = values.append
        for name, row in zip(names_np, mask):
            idx = np.flatnonzero(row)
            append([name, " ".join(cls[idx]) if idx.size else "healthy"])

    csv_pd = pd.DataFrame(values, columns=["image", "labels"])
    out_path = os.path.join(output_dir, "submission.csv")
    csv_pd.to_csv(out_path, index=False)
    print("Wrote submission.csv with shape:", csv_pd.shape, "to", out_path)
    print(csv_pd.head())
