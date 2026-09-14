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

os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", None)
os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")

import pandas as pd
import tensorflow as tf

tf.random.set_seed(123)



## === cell 1
output_dir = "./"
test_dir = "../input/plant-pathology-2021-fgvc8/test_images/"
model_dir = "../input/conve01/eff7-e5/epoch-5"
efficientB7 = "../input/efficientb7/effb7"

image_dims = (300, 300, 3)
data_set = pd.read_csv("../input/plant-pathology-2021-fgvc8/train.csv")
df_labels = data_set["labels"]
one_hot = df_labels.str.get_dummies(sep=" ")
dataset_labels = one_hot.columns.to_list()



## === cell 2
from tensorflow.keras import Sequential, Model
from tensorflow.keras.layers import (
    Dense,
    BatchNormalization,
    Dropout,
    GlobalMaxPool2D,
    Conv2D,
    InputLayer,
)
from tensorflow import concat

try:
    from keras.layers import TFSMLayer  # Keras 3 (may or may not be available)
except Exception:
    TFSMLayer = None


def _find_savedmodel_dir(path: str) -> str:
    """
    Search recursively for a directory containing saved_model.pb / saved_model.pbtxt.
    """

    def _search_under(root: str) -> str:
        if not tf.io.gfile.exists(root):
            return ""
        if tf.io.gfile.isdir(root):
            pb = os.path.join(root, "saved_model.pb")
            pbtxt = os.path.join(root, "saved_model.pbtxt")
            if tf.io.gfile.exists(pb) or tf.io.gfile.exists(pbtxt):
                return root
            for r, _, files in tf.io.gfile.walk(root):
                if "saved_model.pb" in files or "saved_model.pbtxt" in files:
                    return r
        return ""

    found = _search_under(path)
    if found:
        return found

    parent = os.path.dirname(path.rstrip("/"))
    found = _search_under(parent)
    if found:
        return found

    return path


def _infer_savedmodel_endpoint(savedmodel_dir: str) -> str:
    """
    Try to pick a valid SavedModel endpoint name.
    """
    try:
        loaded = tf.saved_model.load(savedmodel_dir)
        sigs = list(getattr(loaded, "signatures", {}).keys())
        if "serving_default" in sigs:
            return "serving_default"
        if len(sigs) > 0:
            return sigs[0]
    except Exception:
        pass
    return "serving_default"


class MultiLabel(Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._using_savedmodel_backbone = False
        backbone_layer = None

        resolved_backbone_dir = _find_savedmodel_dir(efficientB7)
        pb = os.path.join(resolved_backbone_dir, "saved_model.pb")
        pbtxt = os.path.join(resolved_backbone_dir, "saved_model.pbtxt")

        if TFSMLayer is not None and (
            tf.io.gfile.exists(pb) or tf.io.gfile.exists(pbtxt)
        ):
            endpoint = _infer_savedmodel_endpoint(resolved_backbone_dir)
            backbone_layer = TFSMLayer(resolved_backbone_dir, call_endpoint=endpoint)
            self._using_savedmodel_backbone = True
        else:
            eff = tf.keras.applications.EfficientNetB7(
                include_top=False,
                weights="imagenet",
                input_shape=image_dims,
            )

            class _EffBackbone(tf.keras.layers.Layer):
                def __init__(self, eff_model):
                    super().__init__()
                    self.eff_model = eff_model

                def call(self, x, training=False):
                    x = tf.keras.applications.efficientnet.preprocess_input(x)
                    return self.eff_model(x, training=training)

            backbone_layer = _EffBackbone(eff)

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

    def call(self, x, training=False, **kwargs):
        y = self.model(x, training=training)

        pred1 = y[:, :, :, :400]
        pred2 = y[:, :, :, 400:800]
        pred3 = y[:, :, :, 800:1200]
        pred4 = y[:, :, :, 1200:1600]
        pred5 = y[:, :, :, 1600:2000]
        pred6 = y[:, :, :, 2000:2400]

        pred1 = self.model_pred1_2(pred1)
        pred1 = self.model_pred1_3(pred1, training=training)
        pred1 = self.model_pred1_4(pred1)
        pred1 = self.model_pred1_5(pred1)
        pred1 = self.model_pred1_6(pred1)

        pred2 = self.model_pred2_2(pred2)
        pred2 = self.model_pred2_3(pred2, training=training)
        pred2 = self.model_pred2_4(pred2)
        pred2 = self.model_pred2_5(pred2)
        pred2 = self.model_pred2_6(pred2)

        pred3 = self.model_pred3_2(pred3)
        pred3 = self.model_pred3_3(pred3, training=training)
        pred3 = self.model_pred3_4(pred3)
        pred3 = self.model_pred3_5(pred3)
        pred3 = self.model_pred3_6(pred3)

        pred4 = self.model_pred4_2(pred4)
        pred4 = self.model_pred4_3(pred4, training=training)
        pred4 = self.model_pred4_4(pred4)
        pred4 = self.model_pred4_5(pred4)
        pred4 = self.model_pred4_6(pred4)

        pred5 = self.model_pred5_2(pred5)
        pred5 = self.model_pred5_3(pred5, training=training)
        pred5 = self.model_pred5_4(pred5)
        pred5 = self.model_pred5_5(pred5)
        pred5 = self.model_pred5_6(pred5)

        pred6 = self.model_pred6_2(pred6)
        pred6 = self.model_pred6_3(pred6, training=training)
        pred6 = self.model_pred6_4(pred6)
        pred6 = self.model_pred6_5(pred6)
        pred6 = self.model_pred6_6(pred6)

        return concat([pred1, pred2, pred3, pred4, pred5, pred6], axis=1)

    def create_model(self):
        return self.model


def _resolve_weights_path(weights_dir_or_file: str) -> str:
    """
    model.load_weights() expects a checkpoint prefix/file path, not a directory.
    Resolve the latest/best candidate in the given directory.
    """
    if tf.io.gfile.exists(weights_dir_or_file) and not tf.io.gfile.isdir(
        weights_dir_or_file
    ):
        return weights_dir_or_file

    if not tf.io.gfile.isdir(weights_dir_or_file):
        return weights_dir_or_file

    ckpt_state = tf.train.get_checkpoint_state(weights_dir_or_file)
    if ckpt_state and ckpt_state.model_checkpoint_path:
        return ckpt_state.model_checkpoint_path

    candidates = []
    for ext in (".weights.h5", ".h5", ".ckpt", ".keras"):
        candidates.extend(
            tf.io.gfile.glob(os.path.join(weights_dir_or_file, f"*{ext}"))
        )
    index_files = tf.io.gfile.glob(os.path.join(weights_dir_or_file, "*.index"))
    for f in index_files:
        candidates.append(f[:-6])  # strip ".index" to get checkpoint prefix

    if not candidates:
        all_files = [p for p in tf.io.gfile.listdir(weights_dir_or_file)]
        all_files = [
            os.path.join(weights_dir_or_file, p)
            for p in all_files
            if not p.endswith(".data-00000-of-00001")
        ]
        if all_files:
            all_files = sorted(all_files)
            return all_files[-1]
        return weights_dir_or_file

    candidates = sorted(set(candidates))
    return candidates[-1]




## === cell 3
if __name__ == "__main__":
    try:
        tf.config.threading.set_intra_op_parallelism_threads(0)
        tf.config.threading.set_inter_op_parallelism_threads(0)
    except Exception:
        pass

    model = MultiLabel()
    model.build(input_shape=[None, image_dims[0], image_dims[1], image_dims[2]])

    weights_path = _resolve_weights_path(model_dir)
    try:
        if tf.io.gfile.exists(weights_path) or tf.io.gfile.exists(
            weights_path + ".index"
        ):
            model.load_weights(weights_path)
        else:
            print(
                f"Warning: weights not found at '{weights_path}'. Proceeding with current initialization."
            )
    except Exception as e:
        print(
            f"Warning: load_weights failed for '{weights_path}': {e}. Proceeding without loaded weights."
        )

    images_path_list = sorted(os.listdir(test_dir))
    file_paths = [os.path.join(test_dir, fn) for fn in images_path_list]

    threshold = 0.7
    AUTOTUNE = tf.data.AUTOTUNE

    def _load_and_preprocess(path):
        img_bytes = tf.io.read_file(path)
        img = tf.io.decode_jpeg(img_bytes, channels=3)
        img = tf.image.convert_image_dtype(img, tf.float32)  # [0,1] float32
        img = tf.image.resize(img, [image_dims[0], image_dims[1]], method="bilinear")
        img = img * 255.0
        return img

    ds = tf.data.Dataset.from_tensor_slices(file_paths)
    options = tf.data.Options()
    options.experimental_deterministic = True
    ds = ds.with_options(options)

    ds = ds.map(_load_and_preprocess, num_parallel_calls=AUTOTUNE, deterministic=True)

    BATCH_SIZE = 64  # CHANGE: safer throughput/latency tradeoff for EfficientNetB7; semantics unchanged.
    ds = ds.batch(BATCH_SIZE, drop_remainder=False).prefetch(AUTOTUNE)

    @tf.function(
        reduce_retracing=True,
        input_signature=[
            tf.TensorSpec(
                shape=[None, image_dims[0], image_dims[1], image_dims[2]],
                dtype=tf.float32,
            )
        ],
    )
    def _predict_batch(x):
        return model(x, training=False)

    import numpy as np

    label_arr = np.asarray(dataset_labels, dtype=object)
    n_files = len(images_path_list)

    values = []
    values_append = values.append

    for batch_idx, batch in enumerate(ds):
        preds = _predict_batch(batch).numpy()  # [B, 6]
        over = preds > threshold
        argm = preds.argmax(axis=1)

        base = batch_idx * BATCH_SIZE
        bsz = preds.shape[0]
        for i in range(bsz):
            row_over = over[i]
            if row_over.any():
                idxs = np.flatnonzero(row_over)
                if idxs.size == 1:
                    classes_img = label_arr[int(idxs[0])]
                else:
                    classes_img = " ".join(label_arr[int(j)] for j in idxs[::-1])
            else:
                classes_img = label_arr[int(argm[i])]

            j = base + i
            if j < n_files:
                values_append((images_path_list[j], classes_img))

    csv_pd = pd.DataFrame(values, columns=["image", "labels"])
    csv_pd.to_csv(os.path.join(output_dir, "submission.csv"), index=False)
    print(
        f"Wrote submission to: {os.path.join(output_dir, 'submission.csv')}  rows={len(csv_pd)}"
    )
