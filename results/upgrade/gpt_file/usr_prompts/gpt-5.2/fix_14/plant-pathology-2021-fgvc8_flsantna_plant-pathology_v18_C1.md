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

0.6584487534626026

# 6. Current score

0.35717

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.24507) has done: 'I fix the protobuf/TensorFlow import crash by pinning protobuf to the compatible pure-Python implementation at runtime (via environment variables) and by avoiding printing `tf.keras.__version__`, which can trigger the failing protobuf path. Then I remove the hard dependency on a missing `../input/model-25/epoch-25` artifact by enabling a minimal fallback training path using the exact same model architecture/loss/optimizer already defined, so the notebook always produces predictions. Finally, I ensure the submission rows align exactly to `sample_submission.csv` image order and that label strings are space-delimited, producing a valid `submission.csv`.'
- What this solution (achieved 0.25203) has done: 'I fix the TensorFlow/protobuf crash by forcing the pure-Python protobuf implementation *before* TensorFlow is imported and by using `tf.keras.layers.TFSMLayer` only when the SavedModel can be safely loaded. Then I fix a scoring bug in post-processing: the code currently reverses class indices (`idxs[::-1]`), which scrambles label names and tanks mean F1; I output labels in the correct class order. Finally, I keep the core model/training logic intact, but adjust the decision threshold from an overly strict 0.7 to 0.5 (a standard calibration for sigmoid multilabel) to move the score upward toward the target.'
- What this solution (achieved 0.26182) has done: 'I fix the crash happening before TensorFlow fully imports by removing the protobuf override that forces an incompatible pure-Python protobuf path in this environment, and instead keep TensorFlow logs quiet and determinism settings as before. Then I add a small safety fallback around the protobuf version print so it never trips import-time protobuf internals. The rest of the pipeline (data loading, label binarization, model architecture, fallback training, thresholding, and submission alignment/format) remain unchanged to preserve core logic while allowing the run to complete and produce `submission.csv`. This should also restore expected TensorFlow behavior and improve score versus the currently-crashing run, without altering the intended modeling semantics.'
- What this solution (achieved 0.32444) has done: 'The crash happens before your notebook can run because TensorFlow 2.18 + protobuf 6 can hit a known runtime incompatibility (`MessageFactory.GetPrototype` missing). I fix this by forcing the pure-Python protobuf implementation *before importing TensorFlow*, which avoids the failing C++ path in this environment and lets the rest of your pipeline run. Then I make a minimal, metric-aligned post-processing improvement by ensuring each prediction always outputs at least one label by falling back to the top-1 class (instead of hardcoding `healthy` when nothing passes the threshold), which should move mean F1 upward toward your target without changing the model/training core. All paths, model architecture, training loop, and submission format remain the same, and the script write `submission.csv`.'
- What this solution (achieved 0.35717) has done: 'I fix the TensorFlow import crash by forcing the pure-Python protobuf implementation before TensorFlow is imported (your current `"cpp"` setting triggers the missing `google.protobuf.pyext._message` error), so `tf` is defined for later cells. Then I keep your existing model/training/inference logic unchanged, only making the import path robust and ensuring the pipeline always reaches submission writing. Finally, I keep the submission alignment to `sample_submission.csv` order and the space-delimited label formatting exactly as required, so a valid `submission.csv` is produced end-to-end.'

# 9. Code solution

## === cell 0
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "2")

import sys
import numpy as np
import pandas as pd

import tensorflow as tf

keras = tf.keras

print("Python:", sys.version.split()[0])
print("TF:", tf.__version__)
try:
    from google.protobuf import __version__ as pb_ver
except Exception:
    pb_ver = None
print("protobuf:", pb_ver)

tf.random.set_seed(42)
np.random.seed(42)

try:
    tf.config.threading.set_intra_op_parallelism_threads(0)
    tf.config.threading.set_inter_op_parallelism_threads(0)
except Exception:
    pass

try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
output_dir = "./"
test_dir = "../input/plant-pathology-2021-fgvc8/test_images/"
model_dir = "../input/model-25/epoch-25"

image_dims = (300, 300, 3)

train_csv_path = "../input/plant-pathology-2021-fgvc8/train.csv"
train_img_dir = "../input/plant-pathology-2021-fgvc8/train_images/"
sample_sub_path = (
    "../input/plant-pathology-2021-fgvcvc8/sample_submission.csv"
    if False
    else "../input/plant-pathology-2021-fgvc8/sample_submission.csv"
)

data_set = pd.read_csv(train_csv_path)
df_labels = data_set["labels"]
one_hot = df_labels.str.get_dummies(sep=" ")
dataset_labels = one_hot.columns.to_list()

print("Num classes:", len(dataset_labels))
print("First classes:", dataset_labels[:10])

Y_ALL = one_hot.reindex(columns=dataset_labels, fill_value=0).to_numpy(
    dtype=np.float32, copy=False
)
PATHS_ALL = (train_img_dir + data_set["image"]).to_numpy()




## === cell 2
def _is_savedmodel_dir(path: str) -> bool:
    return os.path.isdir(path) and (
        os.path.exists(os.path.join(path, "saved_model.pb"))
        or os.path.exists(os.path.join(path, "saved_model.pbtxt"))
    )


def _find_model_path(base_path: str) -> str:
    if base_path is None:
        return None
    if os.path.isfile(base_path) and base_path.endswith((".keras", ".h5")):
        return base_path
    if _is_savedmodel_dir(base_path):
        return base_path
    if not os.path.isdir(base_path):
        return None

    try:
        entries = [os.path.join(base_path, n) for n in os.listdir(base_path)]
    except Exception:
        entries = []

    for p in entries:
        if os.path.isfile(p) and p.endswith((".keras", ".h5")):
            return p
        if _is_savedmodel_dir(p):
            return p

    base_depth = base_path.rstrip(os.sep).count(os.sep)
    best_savedmodel = None
    best_file = None

    for root, dirs, files in os.walk(base_path):
        depth = root.rstrip(os.sep).count(os.sep) - base_depth
        if depth > 3:
            dirs[:] = []
            continue

        if _is_savedmodel_dir(root):
            best_savedmodel = root
            break

        for fn in files:
            if fn.endswith(".keras") or fn.endswith(".h5"):
                best_file = os.path.join(root, fn)
                break

    return best_savedmodel or best_file


def _load_inference_model(path: str):
    if path is None:
        return None, "no_model_found"

    if os.path.isfile(path) and path.endswith((".keras", ".h5")):
        model = tf.keras.models.load_model(path, compile=False)

        @tf.function(input_signature=[tf.TensorSpec([None, *image_dims], tf.float32)])
        def _call(x):
            return model(x, training=False)

        return _call, f"keras_model_file:{path}"

    if _is_savedmodel_dir(path):
        try:
            layer = tf.keras.layers.TFSMLayer(path, call_endpoint="serving_default")

            @tf.function(
                input_signature=[tf.TensorSpec([None, *image_dims], tf.float32)]
            )
            def _call(x):
                return layer(x)

            return _call, f"savedmodel_dir:{path}"
        except Exception as e:
            print("Failed to load SavedModel via TFSMLayer:", repr(e))
            return None, f"savedmodel_load_failed:{path}"

    return None, f"unusable_path:{path}"


resolved_model_path = _find_model_path(model_dir)
print("Resolved model path:", resolved_model_path)

inference_fn, model_kind = _load_inference_model(resolved_model_path)
print("Model kind:", model_kind)



## === cell 3
num_classes = len(dataset_labels)


def _build_model():
    inputs = tf.keras.Input(shape=image_dims, name="image")
    x = tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu")(inputs)
    x = tf.keras.layers.MaxPooling2D()(x)
    x = tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.MaxPooling2D()(x)
    x = tf.keras.layers.Conv2D(128, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(256, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="sigmoid", name="pred")(x)
    model = tf.keras.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
    )
    return model


@tf.function
def _decode_and_resize_from_bytes(img_bytes):
    img = tf.io.decode_jpeg(img_bytes, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.resize(img, [image_dims[0], image_dims[1]], antialias=False)
    return img


@tf.function
def _read_bytes(path):
    return tf.io.read_file(path)


def _make_train_dataset_from_arrays(
    paths, y, batch_size=16, shuffle=True, cache_in_memory=True
):
    ds = tf.data.Dataset.from_tensor_slices((paths, y))
    if shuffle:
        ds = ds.shuffle(
            buffer_size=min(int(len(paths)), 2048),
            seed=42,
            reshuffle_each_iteration=True,
        )

    ds = ds.map(
        lambda p, label: (_read_bytes(p), label), num_parallel_calls=tf.data.AUTOTUNE
    )

    if cache_in_memory:
        ds = ds.cache()

    ds = ds.map(
        lambda img_bytes, label: (_decode_and_resize_from_bytes(img_bytes), label),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    opts = tf.data.Options()
    opts.experimental_deterministic = True
    ds = ds.with_options(opts)
    return ds


if inference_fn is None:
    print(
        "Pretrained model not found/failed to load under model_dir; running fallback training "
        "with the same model architecture to generate a valid submission."
    )
    model = _build_model()
    train_ds = _make_train_dataset_from_arrays(
        PATHS_ALL, Y_ALL, batch_size=16, shuffle=True, cache_in_memory=True
    )

    model.fit(train_ds, epochs=1, verbose=2)

    @tf.function(input_signature=[tf.TensorSpec([None, *image_dims], tf.float32)])
    def inference_fn(x):
        return model(x, training=False)

    model_kind = "fallback_trained_model"
else:
    print("Using provided pretrained model.")



## === cell 4
sub_df = pd.read_csv(sample_sub_path)
test_paths = (test_dir + sub_df["image"]).to_numpy()

test_paths_ds = tf.data.Dataset.from_tensor_slices(test_paths)

print("Num test images (from sample_submission):", len(test_paths))
print("First test path:", test_paths[0])


def _make_test_dataset(paths_ds, batch_size=32):
    ds = paths_ds.map(
        lambda p: (p, _read_bytes(p)), num_parallel_calls=tf.data.AUTOTUNE
    )
    ds = ds.map(
        lambda p, b: (p, _decode_and_resize_from_bytes(b)),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    ds = ds.batch(batch_size, drop_remainder=False).prefetch(tf.data.AUTOTUNE)
    opts = tf.data.Options()
    opts.experimental_deterministic = True
    ds = ds.with_options(opts)
    return ds


def _coerce_model_output_to_probs(out, num_classes: int):
    if isinstance(out, dict):
        for k in ("pred", "outputs", "output_0", "logits", "probabilities"):
            if k in out:
                out = out[k]
                break
        else:
            out = out[next(iter(out.keys()))]

    out = tf.convert_to_tensor(out)

    out = tf.squeeze(out)
    if out.shape.rank == 1:
        out = tf.expand_dims(out, axis=0)

    if (
        out.shape.rank == 2
        and (out.shape[-1] != num_classes)
        and (out.shape[1] != num_classes)
    ):
        out = tf.reshape(out, [-1, num_classes])

    return out


test_ds = _make_test_dataset(test_paths_ds, batch_size=32)

values = []
classes = dataset_labels
classes_arr = np.asarray(classes, dtype=object)

threshold = 0.5

for batch_paths, batch_imgs in test_ds:
    out = inference_fn(batch_imgs)
    out = _coerce_model_output_to_probs(out, num_classes=len(classes))
    out_np = out.numpy()

    mask = out_np > threshold

    names = batch_paths.numpy()
    names = [os.path.basename(n.decode("utf-8")) for n in names]

    for name, row_mask, row_prob in zip(names, mask, out_np):
        idxs = np.flatnonzero(row_mask)

        if idxs.size == 0:
            top1 = int(np.argmax(row_prob))
            classes_img = str(classes_arr[top1])
        else:
            classes_img = " ".join(classes_arr[idxs])

        values.append([name, classes_img])

pred_df = pd.DataFrame(values, columns=["image", "labels"])

pred_map = dict(zip(pred_df["image"].values, pred_df["labels"].values))
sub_df["labels"] = sub_df["image"].map(pred_map).fillna("healthy")

out_path = os.path.join(output_dir, "submission.csv")
sub_df.to_csv(out_path, index=False)

print("Model kind used:", model_kind)
print("Wrote:", out_path)
print(sub_df.head())
print("Submission shape:", sub_df.shape)
