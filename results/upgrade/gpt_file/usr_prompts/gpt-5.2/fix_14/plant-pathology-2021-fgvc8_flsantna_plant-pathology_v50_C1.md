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
import warnings
import pandas as pd
import tensorflow as tf

warnings.filterwarnings("ignore")
tf.get_logger().setLevel("ERROR")

tf.random.set_seed(42)
try:
    tf.config.experimental.enable_op_determinism(True)
except Exception:
    pass

try:
    tf.config.threading.set_intra_op_parallelism_threads(0)
    tf.config.threading.set_inter_op_parallelism_threads(0)
except Exception:
    pass

try:
    gpus = tf.config.list_physical_devices("GPU")
    for g in gpus:
        tf.config.experimental.set_memory_growth(g, True)
except Exception:
    pass




## === cell 1
output_dir = "./"
test_dir = "../input/plant-pathology-2021-fgvc8/test_images/"
model_dir = "../input/conve01/eff7-e14/epoch-14"
efficientB7 = "../input/efficientb7/effb7"

image_dims = (300, 300, 3)

data_set = pd.read_csv("../input/plant-pathology-2021-fgvc8/train.csv")
df_labels = data_set["labels"]
one_hot = df_labels.str.get_dummies(sep=" ")
dataset_labels = one_hot.columns.to_list()

if len(dataset_labels) == 0:
    raise ValueError("No labels found in train.csv; cannot build label list.")




## === cell 2
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
from tensorflow import concat

try:
    from keras.layers import TFSMLayer  # Keras 3
except Exception:
    from tensorflow.keras.layers import TFSMLayer


def _is_savedmodel_dir(p: str) -> bool:
    return os.path.isdir(p) and (
        os.path.exists(os.path.join(p, "saved_model.pb"))
        or os.path.exists(os.path.join(p, "saved_model.pbtxt"))
    )


def _resolve_saved_model_dir(p: str) -> str:
    p = os.path.expanduser(p)
    if os.path.isfile(p):
        raise ValueError(f"Expected a directory for SavedModel, got file: {p}")
    if _is_savedmodel_dir(p):
        return p
    return p


def _find_existing_dir(candidates):
    for c in candidates:
        if c and os.path.isdir(c):
            return c
    return None


_EFFNET_SAVEDMODEL_DIR_CACHE = None


def _guess_efficientnet_savedmodel_dir() -> str:
    global _EFFNET_SAVEDMODEL_DIR_CACHE
    if _EFFNET_SAVEDMODEL_DIR_CACHE is not None:
        return _EFFNET_SAVEDMODEL_DIR_CACHE

    candidate_roots = [
        efficientB7,
        "../input/efficientb7/effb7",
        "../input/efficientb7",
        "../input/efficientnetb7",
        "../input",
        "/kaggle/input",
    ]
    root = _find_existing_dir(candidate_roots) or efficientB7

    to_check = [root]
    if os.path.isdir(root):
        try:
            lvl1 = [os.path.join(root, d) for d in os.listdir(root)]
        except Exception:
            lvl1 = []
        to_check.extend([p for p in lvl1 if os.path.isdir(p)])
        lvl2 = []
        for p in to_check[1:]:
            try:
                lvl2.extend([os.path.join(p, d) for d in os.listdir(p)])
            except Exception:
                continue
        to_check.extend([p for p in lvl2 if os.path.isdir(p)])

    resolved = None
    for p in to_check:
        if _is_savedmodel_dir(p):
            resolved = p
            break

    if resolved is None:
        resolved = _resolve_saved_model_dir(root)

    _EFFNET_SAVEDMODEL_DIR_CACHE = resolved
    return resolved


class MultiLabel(Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        backbone_dir = _guess_efficientnet_savedmodel_dir()
        if _is_savedmodel_dir(backbone_dir):
            self.model_backbone = TFSMLayer(
                backbone_dir, call_endpoint="serving_default"
            )
            backbone_layer = self.model_backbone
        else:
            backbone_layer = Lambda(lambda x: x, name="identity_backbone")

        self.model = Sequential()
        self.model.add(InputLayer(input_shape=image_dims))
        self.model.add(backbone_layer)
        self.model.add(Conv2D(filters=1024, kernel_size=(1, 1), padding="same"))
        self.model.add(BatchNormalization(momentum=0.7))
        self.model.add(Dropout(0.2))
        self.model.add(Conv2D(filters=1024, kernel_size=(1, 1), padding="same"))
        self.model.add(Conv2D(filters=2400, kernel_size=(1, 1), padding="same"))

        self.model_pred1_2 = Conv2D(filters=1024, kernel_size=(1, 1), padding="same")
        self.model_pred1_3 = BatchNormalization()
        self.model_pred1_4 = Conv2D(filters=512, kernel_size=(1, 1), padding="same")
        self.model_pred1_5 = GlobalMaxPool2D()
        self.model_pred1_6 = Dense(units=1, activation="sigmoid")

        self.model_pred2_2 = Conv2D(filters=1024, kernel_size=(1, 1), padding="same")
        self.model_pred2_3 = BatchNormalization()
        self.model_pred2_4 = Conv2D(filters=512, kernel_size=(1, 1), padding="same")
        self.model_pred2_5 = GlobalMaxPool2D()
        self.model_pred2_6 = Dense(units=1, activation="sigmoid")

        self.model_pred3_2 = Conv2D(filters=1024, kernel_size=(1, 1), padding="same")
        self.model_pred3_3 = BatchNormalization()
        self.model_pred3_4 = Conv2D(filters=512, kernel_size=(1, 1), padding="same")
        self.model_pred3_5 = GlobalMaxPool2D()
        self.model_pred3_6 = Dense(units=1, activation="sigmoid")

        self.model_pred4_2 = Conv2D(filters=1024, kernel_size=(1, 1), padding="same")
        self.model_pred4_3 = BatchNormalization()
        self.model_pred4_4 = Conv2D(filters=512, kernel_size=(1, 1), padding="same")
        self.model_pred4_5 = GlobalMaxPool2D()
        self.model_pred4_6 = Dense(units=1, activation="sigmoid")

        self.model_pred5_2 = Conv2D(filters=1024, kernel_size=(1, 1), padding="same")
        self.model_pred5_3 = BatchNormalization()
        self.model_pred5_4 = Conv2D(filters=512, kernel_size=(1, 1), padding="same")
        self.model_pred5_5 = GlobalMaxPool2D()
        self.model_pred5_6 = Dense(units=1, activation="sigmoid")

        self.model_pred6_2 = Conv2D(filters=1024, kernel_size=(1, 1), padding="same")
        self.model_pred6_3 = BatchNormalization()
        self.model_pred6_4 = Conv2D(filters=512, kernel_size=(1, 1), padding="same")
        self.model_pred6_5 = GlobalMaxPool2D()
        self.model_pred6_6 = Dense(units=1, activation="sigmoid")

    def call(self, x, **kwargs):
        y = self.model(x)

        pred1 = y[:, :, :, :400]
        pred2 = y[:, :, :, 400:800]
        pred3 = y[:, :, :, 800:1200]
        pred4 = y[:, :, :, 1200:1600]
        pred5 = y[:, :, :, 1600:2000]
        pred6 = y[:, :, :, 2000:2400]

        pred1 = self.model_pred1_2(pred1)
        pred1 = self.model_pred1_3(pred1)
        pred1 = self.model_pred1_4(pred1)
        pred1 = self.model_pred1_5(pred1)
        pred1 = self.model_pred1_6(pred1)

        pred2 = self.model_pred2_2(pred2)
        pred2 = self.model_pred2_3(pred2)
        pred2 = self.model_pred2_4(pred2)
        pred2 = self.model_pred2_5(pred2)
        pred2 = self.model_pred2_6(pred2)

        pred3 = self.model_pred3_2(pred3)
        pred3 = self.model_pred3_3(pred3)
        pred3 = self.model_pred3_4(pred3)
        pred3 = self.model_pred3_5(pred3)
        pred3 = self.model_pred3_6(pred3)

        pred4 = self.model_pred4_2(pred4)
        pred4 = self.model_pred4_3(pred4)
        pred4 = self.model_pred4_4(pred4)
        pred4 = self.model_pred4_5(pred4)
        pred4 = self.model_pred4_6(pred4)

        pred5 = self.model_pred5_2(pred5)
        pred5 = self.model_pred5_3(pred5)
        pred5 = self.model_pred5_4(pred5)
        pred5 = self.model_pred5_5(pred5)
        pred5 = self.model_pred5_6(pred5)

        pred6 = self.model_pred6_2(pred6)
        pred6 = self.model_pred6_3(pred6)
        pred6 = self.model_pred6_4(pred6)
        pred6 = self.model_pred6_5(pred6)
        pred6 = self.model_pred6_6(pred6)

        return concat([pred1, pred2, pred3, pred4, pred5, pred6], axis=1)

    def create_model(self):
        return self.model


def _resolve_weights_path(p: str) -> str:
    if os.path.isfile(p):
        return p

    if os.path.isdir(p):
        candidates = [
            os.path.join(p, "weights.h5"),
            os.path.join(p, "model.weights.h5"),
            os.path.join(p, "weights.weights.h5"),
            os.path.join(p, "checkpoint"),
        ]
        for c in candidates:
            if os.path.exists(c):
                return c

        try:
            fns = sorted(os.listdir(p))
        except Exception:
            fns = []

        for fn in fns:
            if fn.endswith(".index") and (
                fn.startswith("ckpt") or fn.startswith("checkpoint")
            ):
                return os.path.join(p, fn[: -len(".index")])

        for fn in fns:
            if fn.endswith(".h5") or fn.endswith(".keras"):
                return os.path.join(p, fn)

    return p




## === cell 3
if __name__ == "__main__":
    model = MultiLabel()
    model.build(input_shape=[None, image_dims[0], image_dims[1], image_dims[2]])

    weights_path = _resolve_weights_path(model_dir)
    try:
        model.load_weights(weights_path)
    except Exception as e:
        print(
            f"[WARN] Could not load weights from: {weights_path}\n{type(e).__name__}: {e}"
        )

    images_path_list = tf.io.gfile.glob(os.path.join(test_dir, "*.jpg"))
    images_path_list = sorted(images_path_list)

    AUTOTUNE = tf.data.AUTOTUNE

    def _load_and_preprocess(path):
        img_bytes = tf.io.read_file(path)
        img = tf.io.decode_jpeg(img_bytes, channels=3, dct_method="INTEGER_FAST")
        img = tf.image.convert_image_dtype(img, tf.float32)
        img = tf.image.resize(
            img, [image_dims[0], image_dims[1]], method="bilinear", antialias=False
        )
        img = img * 255.0  # keep original inference scaling
        img.set_shape(image_dims)
        name = tf.strings.split(path, os.sep)[-1]
        return name, img

    ds_paths = tf.data.Dataset.from_tensor_slices(images_path_list)

    options = tf.data.Options()
    options.experimental_deterministic = True
    options.experimental_optimization.apply_default_optimizations = True
    options.experimental_optimization.map_parallelization = True
    options.experimental_optimization.parallel_batch = True
    options.threading.private_threadpool_size = 0
    options.threading.max_intra_op_parallelism = 0

    BATCH_SIZE = 128

    ds = (
        ds_paths.with_options(options)
        .map(_load_and_preprocess, num_parallel_calls=AUTOTUNE, deterministic=True)
        .cache()
        .batch(BATCH_SIZE, drop_remainder=False)
        .prefetch(AUTOTUNE)
    )

    classes = dataset_labels
    classes_rev = list(reversed(classes))
    classes_rev_tf = tf.constant(classes_rev, dtype=tf.string)

    @tf.function(reduce_retracing=True)
    def _infer_and_format(batch_imgs):
        preds = model(batch_imgs, training=False)  # (B, 6)
        mask = preds > 0.7
        any_pos = tf.reduce_any(mask, axis=1)  # (B,)
        argmax_idx = tf.argmax(preds, axis=1, output_type=tf.int32)  # (B,)

        b = tf.shape(preds)[0]
        row_idx = tf.range(b, dtype=tf.int32)
        scatter_idx = tf.stack([row_idx, argmax_idx], axis=1)
        fallback_mask = tf.scatter_nd(
            scatter_idx, tf.ones([b], tf.bool), tf.shape(mask)
        )
        mask = tf.where(any_pos[:, None], mask, fallback_mask)

        mask_rev = tf.reverse(mask, axis=[1])
        tokens = tf.where(mask_rev, classes_rev_tf[None, :], tf.constant("", tf.string))
        joined = tf.strings.reduce_join(tokens, axis=1, separator=" ")
        joined = tf.strings.regex_replace(joined, r"\s+", " ")
        joined = tf.strings.strip(joined)
        return joined

    def _infer_pair(batch_names, batch_imgs):
        return batch_names, _infer_and_format(batch_imgs)

    ds_out = ds.map(
        _infer_pair, num_parallel_calls=AUTOTUNE, deterministic=True
    ).unbatch()

    out_images_t = []
    out_labels_t = []
    for name_t, label_t in ds_out:
        out_images_t.append(name_t)
        out_labels_t.append(label_t)

    out_images = tf.stack(out_images_t).numpy().astype("U").tolist()
    out_labels = tf.stack(out_labels_t).numpy().astype("U").tolist()

    csv_pd = pd.DataFrame({"image": out_images, "labels": out_labels})
    csv_pd.to_csv(os.path.join(output_dir, "submission.csv"), index=False)
    print("Wrote submission.csv with shape:", csv_pd.shape)
