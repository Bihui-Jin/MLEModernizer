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

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", None)

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import pandas as pd
import tensorflow as tf

tf.keras.utils.set_random_seed(42)
try:
    tf.config.experimental.enable_op_determinism(True)
except Exception:
    pass




## === cell 1
output_dir = "./"
test_dir = "../input/plant-pathology-2021-fgvc8/test_images/"
model_dir = "../input/conve01/effb7-e13/epoch-13"
efficientB7 = "../input/efficientb7/effb7"

image_dims = (300, 300, 3)


def _resolve_existing_dir(candidates):
    for p in candidates:
        if p and os.path.isdir(p):
            return p
    return None


train_csv_path = None
for p in [
    "../input/plant-pathology-2021-fgvc8/train.csv",
    "../input/train.csv",
    "../kaggle/input/plant-pathology-2021-fgvc8/train.csv",
    "../kaggle/input/train.csv",
]:
    if os.path.exists(p):
        train_csv_path = p
        break
if train_csv_path is None:
    raise FileNotFoundError(
        "Could not locate train.csv under ../input or ../kaggle/input"
    )

test_dir_resolved = _resolve_existing_dir(
    [
        test_dir,
        "../input/test_images",
        "../input/plant-pathology-2021-fgvc8/test_images",
        "../kaggle/input/test_images",
        "../kaggle/input/plant-pathology-2021-fgvc8/test_images",
    ]
)
if test_dir_resolved is None:
    raise FileNotFoundError(
        "Could not locate test_images directory under ../input or ../kaggle/input"
    )
test_dir = (
    test_dir_resolved
    if test_dir_resolved.endswith(os.sep)
    else (test_dir_resolved + os.sep)
)

data_set = pd.read_csv(train_csv_path)
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
    Lambda,
)
from tensorflow import concat

try:
    from keras.layers import TFSMLayer  # keras>=3
except Exception:
    try:
        from tensorflow.keras.layers import TFSMLayer  # some TF builds expose it here
    except Exception:
        TFSMLayer = None


def _find_savedmodel_dir(path_hint: str) -> str:
    if not path_hint:
        return None
    if os.path.isdir(path_hint) and (
        os.path.exists(os.path.join(path_hint, "saved_model.pb"))
        or os.path.exists(os.path.join(path_hint, "saved_model.pbtxt"))
    ):
        return path_hint
    return None


class MultiLabel(Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.model = Sequential()
        self.model.add(InputLayer(input_shape=image_dims))

        resolved_effb7 = _find_savedmodel_dir(efficientB7)
        if resolved_effb7 is not None and TFSMLayer is not None:
            backbone = TFSMLayer(resolved_effb7, call_endpoint="serving_default")

            def _unwrap_savedmodel_output(x):
                y = backbone(x)
                if isinstance(y, dict):
                    for k in ("outputs", "output_0", "predictions", "logits"):
                        if k in y:
                            return y[k]
                    return list(y.values())[0]
                return y

            self.model.add(Lambda(_unwrap_savedmodel_output))
        else:
            eff = tf.keras.applications.EfficientNetB7(
                include_top=False,
                weights="imagenet",
                input_shape=image_dims,
            )
            eff.trainable = False
            self.model.add(eff)

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

        pred1 = self.model_pred1_2(pred1, training=training)
        pred1 = self.model_pred1_3(pred1, training=training)
        pred1 = self.model_pred1_4(pred1, training=training)
        pred1 = self.model_pred1_5(pred1)
        pred1 = self.model_pred1_6(pred1)

        pred2 = self.model_pred2_2(pred2, training=training)
        pred2 = self.model_pred2_3(pred2, training=training)
        pred2 = self.model_pred2_4(pred2, training=training)
        pred2 = self.model_pred2_5(pred2)
        pred2 = self.model_pred2_6(pred2)

        pred3 = self.model_pred3_2(pred3, training=training)
        pred3 = self.model_pred3_3(pred3, training=training)
        pred3 = self.model_pred3_4(pred3, training=training)
        pred3 = self.model_pred3_5(pred3)
        pred3 = self.model_pred3_6(pred3)

        pred4 = self.model_pred4_2(pred4, training=training)
        pred4 = self.model_pred4_3(pred4, training=training)
        pred4 = self.model_pred4_4(pred4, training=training)
        pred4 = self.model_pred4_5(pred4)
        pred4 = self.model_pred4_6(pred4)

        pred5 = self.model_pred5_2(pred5, training=training)
        pred5 = self.model_pred5_3(pred5, training=training)
        pred5 = self.model_pred5_4(pred5, training=training)
        pred5 = self.model_pred5_5(pred5)
        pred5 = self.model_pred5_6(pred5)

        pred6 = self.model_pred6_2(pred6, training=training)
        pred6 = self.model_pred6_3(pred6, training=training)
        pred6 = self.model_pred6_4(pred6, training=training)
        pred6 = self.model_pred6_5(pred6)
        pred6 = self.model_pred6_6(pred6)

        return concat([pred1, pred2, pred3, pred4, pred5, pred6], axis=1)

    def create_model(self):
        return self.model




## === cell 3
if __name__ == "__main__":
    try:
        tf.config.optimizer.set_jit(True)
    except Exception:
        pass

    model = MultiLabel()
    model.build(input_shape=[None, image_dims[0], image_dims[1], image_dims[2]])

    loaded = False
    if model_dir and (
        os.path.exists(model_dir)
        or os.path.isdir(model_dir)
        or os.path.exists(model_dir + ".index")
    ):
        try_paths = [model_dir, model_dir.rstrip("/"), model_dir + ".ckpt"]
        for p in try_paths:
            try:
                if os.path.exists(p) or os.path.exists(p + ".index"):
                    model.load_weights(p)
                    loaded = True
                    break
            except Exception:
                pass

        if not loaded:
            try:
                ckpt = tf.train.latest_checkpoint(
                    model_dir
                    if os.path.isdir(model_dir)
                    else os.path.dirname(model_dir) or "."
                )
                if ckpt is not None:
                    model.load_weights(ckpt)
                    loaded = True
            except Exception:
                loaded = False

    import numpy as np

    AUTOTUNE = tf.data.AUTOTUNE

    @tf.function
    def _load_and_preprocess_from_path(img_path):
        b = tf.io.read_file(img_path)
        img = tf.io.decode_jpeg(b, channels=3, dct_method="INTEGER_FAST")
        img = tf.image.convert_image_dtype(img, tf.float32)  # [0,1]
        img = tf.image.resize(img, [image_dims[0], image_dims[1]])
        img = img * 255.0
        return img_path, img

    @tf.function
    def _basename(path):
        return tf.strings.regex_replace(path, r"^.*[\\/]", "")

    opts = tf.data.Options()
    opts.experimental_deterministic = False
    opts.experimental_optimization.map_parallelization = True
    opts.experimental_optimization.parallel_batch = True
    try:
        cpu = os.cpu_count() or 8
        opts.threading.private_threadpool_size = max(8, cpu)
        opts.threading.max_intra_op_parallelism = max(1, cpu // 2)
    except Exception:
        pass

    bs = 16  # preserve original choice

    file_pattern = os.path.join(test_dir, "*.jpg")
    files_ds = tf.data.Dataset.list_files(file_pattern, shuffle=False)
    files_ds = files_ds.with_options(opts)

    try:
        n_files = int(tf.data.experimental.cardinality(files_ds).numpy())
        if n_files < 0:
            raise ValueError
    except Exception:
        n_files = len(
            [
                f
                for f in os.listdir(test_dir)
                if f.lower().endswith((".jpg", ".jpeg", ".png"))
            ]
        )

    ds = files_ds.map(
        _load_and_preprocess_from_path, num_parallel_calls=AUTOTUNE, deterministic=False
    )
    ds = ds.batch(bs, drop_remainder=False).prefetch(AUTOTUNE)

    n_labels = len(dataset_labels)
    preds_all = np.empty((n_files, n_labels), dtype=np.float32)
    names_all = np.empty((n_files,), dtype=object)

    infer = tf.function(lambda x: model(x, training=False), jit_compile=False)

    _ = infer(
        tf.zeros([1, image_dims[0], image_dims[1], image_dims[2]], dtype=tf.float32)
    )

    write_i = 0
    for batch_paths, batch_imgs in ds:
        batch_pred = infer(batch_imgs).numpy()
        bsz = batch_pred.shape[0]
        preds_all[write_i : write_i + bsz, :] = batch_pred

        batch_names = _basename(batch_paths).numpy()
        names_all[write_i : write_i + bsz] = [
            n.decode("utf-8") if isinstance(n, (bytes, bytearray)) else str(n)
            for n in batch_names
        ]

        write_i += bsz

    if write_i != n_files:
        preds_all = preds_all[:write_i]
        names_all = names_all[:write_i]
        n_files = write_i

    keep = preds_all > 0.7

    labels_arr = dataset_labels
    out_labels = []
    for row in keep:
        idx = row.nonzero()[0]
        if idx.size == 0:
            out_labels.append("healthy")
        else:
            out_labels.append(
                " ".join(labels_arr[int(j)] for j in idx if int(j) < n_labels)
            )

    csv_pd = pd.DataFrame({"image": names_all.tolist(), "labels": out_labels})
    csv_pd = csv_pd.sort_values("image").reset_index(drop=True)

    out_path = os.path.join(output_dir, "submission.csv")
    csv_pd.to_csv(out_path, index=False)
    print(
        f"Wrote submission to: {out_path}  rows={len(csv_pd)}  weights_loaded={loaded}"
    )
