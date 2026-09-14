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

0.6601477377654648

# 6. Current score

0.22303

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.09848) has done: 'The timeout is dominated by per-image Python overhead and calling the model one image at a time, which prevents TensorFlow from efficiently pipelining CPU decode/resize and batching inference. I keep the exact same model loading/inference semantics and thresholding logic, but switch to a `tf.data` input pipeline that decodes/resizes in parallel, batches images, and prefetches to overlap I/O and compute. I also wrap inference in a `tf.function` (same computations, less eager overhead) and avoid repeated conversions by moving to batched NumPy post-processing. These changes are equivalent in results (same preprocessing ops, same thresholding/argmax fallback), but drastically reduce runtime.'
- What this solution (achieved 0.28474) has done: 'You’re hitting a TensorFlow/Protobuf incompatibility caused by forcing `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python`, which triggers the `MessageFactory.GetPrototype` error before the model can even load; removing those env overrides fixes runtime. Next, the current low score is consistent with label-index mismatch: you build `dataset_labels` from `train.csv`, but the external SavedModel likely uses a fixed class order (commonly the competition’s standard order), so predictions are being mapped to the wrong label names; I pin the label order to the known Plant Pathology 2021 class list to align outputs correctly. Finally, I keep your exact inference/thresholding semantics, but also ensure the submission rows are aligned to `sample_submission.csv` ordering (safer for Kaggle ingestion) and keep output as a valid `submission.csv`.'
- What this solution (achieved 0.23993) has done: 'We fix the immediate crash by ensuring no protobuf-breaking environment variables are set *before* importing TensorFlow, and by forcing the safe pure-Python protobuf implementation consistently. Then we keep your exact model/inference/threshold logic, but make the external SavedModel output selection more robust (prefer the common “predictions/probabilities” keys) to avoid silently grabbing the wrong tensor. Finally, we keep the submission aligned to `sample_submission.csv` order and always write `submission.csv` with the required columns.'
- What this solution (achieved 0.22303) has done: 'I fix the immediate TensorFlow/Protobuf crash by removing the environment override that forces the pure-Python protobuf implementation (it’s incompatible with the protobuf version in this Kaggle image and triggers `MessageFactory.GetPrototype` errors). Then I keep your exact inference + threshold/argmax fallback logic, but make the SavedModel output tensor selection slightly safer by preferring 2D `(batch, classes)` outputs when multiple tensors exist (this is score-positive without changing the model). Finally, I ensure test image paths are correct, predictions align to `sample_submission.csv` order, and `submission.csv` is always written with the required columns.'

# 9. Code solution

## === cell 0
import os

os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", None)
os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", None)

import pandas as pd
import tensorflow as tf



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
output_dir = "./"
test_dir = "../input/plant-pathology-2021-fgvc8/test_images/"

model_dir = "../input/model-aug-epoch20/model_complete_with_augEpoch:20"

image_dims = (300, 300, 3)

dataset_labels = [
    "complex",
    "frog_eye_leaf_spot",
    "healthy",
    "powdery_mildew",
    "rust",
    "scab",
]


def _find_savedmodel_dir(preferred_path: str, fallback_root: str) -> str:
    """Return a directory that contains a TensorFlow SavedModel (saved_model.pb)."""
    preferred_path = os.path.abspath(preferred_path)
    fallback_root = os.path.abspath(fallback_root)

    def is_savedmodel_dir(p: str) -> bool:
        return os.path.isdir(p) and os.path.exists(os.path.join(p, "saved_model.pb"))

    if is_savedmodel_dir(preferred_path):
        return preferred_path

    candidates = []
    candidates.append(preferred_path.replace(":", "_"))
    candidates.append(preferred_path.replace(":", ""))
    candidates.append(preferred_path.split(":")[0])
    for c in candidates:
        if is_savedmodel_dir(c):
            return c

    if os.path.isdir(fallback_root):
        for name in sorted(os.listdir(fallback_root)):
            p = os.path.join(fallback_root, name)
            if is_savedmodel_dir(p):
                return p

        for root, dirs, files in os.walk(fallback_root):
            if "saved_model.pb" in files:
                return root

    raise FileNotFoundError(
        "Could not locate a SavedModel directory. Tried preferred path and searched under: "
        f"{fallback_root}\nPreferred: {preferred_path}"
    )


try:
    model_dir = _find_savedmodel_dir(model_dir, "../input/model-aug-epoch20")
    has_external_model = True
except FileNotFoundError:
    has_external_model = False
    model_dir = None



## === cell 2
if __name__ == "__main__":
    if not os.path.isdir(test_dir):
        alt_test_dir = "/kaggle/input/plant-pathology-2021-fgvc8/test_images"
        if os.path.isdir(alt_test_dir):
            test_dir = alt_test_dir
        else:
            raise FileNotFoundError(f"test_dir not found: {test_dir}")

    sample_sub_path = "../input/plant-pathology-2021-fgvc8/sample_submission.csv"
    if not os.path.exists(sample_sub_path):
        sample_sub_path = (
            "/kaggle/input/plant-pathology-2021-fgvc8/sample_submission.csv"
        )
    sample_df = pd.read_csv(sample_sub_path)
    sample_images = sample_df["image"].tolist()

    if has_external_model:
        tfsml = tf.keras.layers.TFSMLayer(model_dir, call_endpoint="serving_default")

        inp = tf.keras.Input(shape=image_dims, dtype=tf.float32, name="input_image")
        out = tfsml(inp)

        if isinstance(out, dict):
            preferred_keys = [
                "predictions",
                "prediction",
                "probabilities",
                "probs",
                "outputs",
                "output",
                "dense",
                "sigmoid",
            ]
            picked_key = None
            for k in preferred_keys:
                if k in out:
                    picked_key = k
                    break

            if picked_key is not None:
                out_tensor = out[picked_key]
            else:
                rank2_keys = []
                for k, v in out.items():
                    try:
                        if (
                            hasattr(v, "shape")
                            and v.shape is not None
                            and len(v.shape) == 2
                        ):
                            rank2_keys.append(k)
                    except Exception:
                        pass
                if len(rank2_keys) > 0:
                    picked_key = sorted(rank2_keys)[0]
                else:
                    picked_key = sorted(out.keys())[0]
                out_tensor = out[picked_key]
            out = out_tensor
        elif isinstance(out, (list, tuple)):
            out = out[0]

        infer_model = tf.keras.Model(inputs=inp, outputs=out)
    else:
        base = tf.keras.applications.EfficientNetB0(
            include_top=False,
            weights="imagenet",
            input_shape=image_dims,
            pooling="avg",
        )
        inp = tf.keras.Input(shape=image_dims, dtype=tf.float32, name="input_image")
        x = tf.keras.applications.efficientnet.preprocess_input(inp * 255.0)
        x = base(x, training=False)
        out = tf.keras.layers.Dense(
            len(dataset_labels), activation="sigmoid", name="pred"
        )(x)
        infer_model = tf.keras.Model(inputs=inp, outputs=out)

    images_path_list = sample_images

    def _load_and_preprocess(path):
        img_bytes = tf.io.read_file(path)
        img = tf.io.decode_jpeg(img_bytes, channels=3)
        img = tf.image.convert_image_dtype(img, dtype=tf.float32)  # -> [0,1]
        img.set_shape([None, None, 3])
        img = tf.image.resize(
            img, [image_dims[0], image_dims[1]], method="bilinear", antialias=True
        )
        return img

    @tf.function(reduce_retracing=True)
    def _infer_batch(batch_images):
        return infer_model(batch_images, training=False)

    threshold = 0.6
    batch_size = 64

    file_paths = [os.path.join(test_dir, n) for n in images_path_list]
    ds = tf.data.Dataset.from_tensor_slices(file_paths)
    ds = ds.map(_load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(tf.data.AUTOTUNE)

    values = []
    offset = 0

    for batch_imgs in ds:
        batch_probs = _infer_batch(batch_imgs)
        batch_probs = tf.convert_to_tensor(batch_probs)

        if batch_probs.shape.rank is not None and batch_probs.shape.rank > 2:
            batch_probs = tf.reshape(batch_probs, [tf.shape(batch_probs)[0], -1])

        batch_probs_np = batch_probs.numpy().astype("float32")  # (B, C)

        bsz = batch_probs_np.shape[0]
        for i in range(bsz):
            probs_np = batch_probs_np[i]
            picked = [j for j, v in enumerate(probs_np) if v > threshold]
            if len(picked) == 0:
                picked = [int(probs_np.argmax())]
            classes_img = " ".join([dataset_labels[j] for j in picked]).strip()
            values.append([images_path_list[offset + i], classes_img])
        offset += bsz

    pred_df = pd.DataFrame(values, columns=["image", "labels"])

    sub_df = sample_df[["image"]].merge(pred_df, on="image", how="left")
    sub_df["labels"] = sub_df["labels"].fillna("healthy")

    out_path = os.path.join(output_dir, "submission.csv")
    sub_df.to_csv(out_path, index=False)
    print(f"Wrote submission to: {out_path} with shape={sub_df.shape}")
    print(sub_df.head())
