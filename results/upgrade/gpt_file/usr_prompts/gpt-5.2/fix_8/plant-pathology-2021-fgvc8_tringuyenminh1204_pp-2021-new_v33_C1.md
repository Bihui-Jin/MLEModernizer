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

0.7933702677747018

# 6. Current score

0.272

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.272) has done: 'I remove the failing `kaggle_datasets` import that’s triggering the protobuf `GetPrototype` crash, since it’s not used anywhere in your pipeline. Then I fix the missing model file issue by loading the pretrained `.h5` if it exists, otherwise falling back to a small TF/Keras model so the notebook always runs end-to-end and produces predictions (this preserves your overall “load model → predict → threshold → write submission” logic). Finally, I fix the submission-length mismatch by keeping a single, consistently ordered list of test filenames and using it both for the dataset and for the `image` column, ensuring `submission.csv` is valid and aligned.'
- What this solution (achieved 0.272) has done: 'I fix the immediate runtime crash that happens on `import tensorflow` by forcing TensorFlow to use the pure-Python protobuf implementation (this avoids the `MessageFactory.GetPrototype` incompatibility in many Kaggle images). Then I correct a logic bug in your label mapping (`healthy` was keyed as `6` instead of `5`) which was preventing the default class from ever matching the model’s 6 outputs and hurting score. Finally, I keep your existing “load model → predict → threshold → write submission.csv” flow intact, but make the label loop robust to exactly 6 outputs and ensure the submission uses the sample-submission image order for perfect alignment.'
- What this solution (achieved 0.272) has done: 'I fix the TensorFlow import crash (`MessageFactory` missing `GetPrototype`) by forcing TensorFlow/Keras to use the legacy pure-Python protobuf *before any TensorFlow-related import* and by removing the direct `tensorflow.keras.backend` import that triggers the failing protobuf path. Then I add a safe fallback that uses `sample_submission.csv` directly if images aren’t present in the visible filesystem (common with hidden test sets), so the notebook still writes a correctly shaped `submission.csv`. Finally, I keep your existing “load model → predict → thresholds → write submission” logic intact, but make the fallback model compile-less and deterministic so it runs end-to-end without changing the intended semantics beyond what’s necessary to produce a valid submission and improve from the current broken state.'
- What this solution (achieved 0.272) has done: 'I fix the immediate TensorFlow/protobuf crash by avoiding the brittle “force pure-Python protobuf” env vars and by removing the `tensorflow.keras.backend as K` dependency that triggers the failing protobuf path in this environment. To preserve your core flow (“build/load model → predict → threshold → write submission”), I re-implement `FixedDropout` using only TensorFlow ops so it no longer needs `K.shape`. Finally, I make the code robust to Kaggle path variants by falling back to `../input/plant-pathology-2021-fgvc8/...` if the shorter path isn’t present, ensuring the notebook runs end-to-end and always produces a valid `submission.csv`.'
- What this solution (achieved 0.272) has done: 'We fix the runtime crash that happens at `import tensorflow` by setting `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` (and version) before importing TensorFlow, which avoids the `MessageFactory.GetPrototype` protobuf incompatibility in this Kaggle image. Then we keep your exact prediction + thresholding + submission-writing logic intact, only ensuring paths are robust and `have_images` is computed correctly so inference actually runs when images exist. This should both unblock execution and substantially improve score versus the current fallback “all healthy” behavior (0.272) by enabling your pretrained model (when present) to generate real predictions. Finally, we ensure a valid `submission.csv` with correct columns and row alignment is always produced.'
- What this solution (achieved 0.272) has done: 'I fix the root cause of the crash by forcing protobuf to use the pure-Python implementation *before* importing TensorFlow (your current `"cpp"` setting triggers the missing `_message` import). Then I make the `have_images` logic robust so it doesn’t wrongly disable inference when a few files are missing, and I ensure we always produce exactly one prediction per row in `sample_submission.csv` (padding/truncating if needed) to avoid the “arrays must be same length” submission error. These changes keep your core flow intact (load model → predict on tf.data → threshold to space-delimited labels → write `submission.csv`) while enabling real predictions instead of crashing or producing misaligned outputs. Finally, the script always write a valid `submission.csv` with the required columns and row order.'

# 9. Code solution

## === cell 0
import os, re, math, random, pathlib

os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", None)
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow import keras

print("tf:", tf.__version__)
print("tf.keras:", tf.keras.__version__)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
def decode_image(filename, label=None, image_size=(512, 512)):
    bits = tf.io.read_file(filename)
    image = tf.image.decode_jpeg(bits, channels=3)
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.image.resize(image, image_size)
    if label is None:
        return image
    else:
        return image, label




## === cell 2
BATCH_SIZE = 32



## === cell 3
candidate_roots = [
    "../input/plant-pathology-2021-fgvc8",
    "/kaggle/input/plant-pathology-2021-fgvc8",
    "/kaggle/input/plant-pathology-2021-fgvc8/plant-pathology-2021-fgvc8",
]
data_root = None
for r in candidate_roots:
    if os.path.exists(r):
        data_root = r
        break
if data_root is None:
    data_root = "../input/plant-pathology-2021-fgvc8"

source = os.path.join(data_root, "test_images")
sample_path = os.path.join(data_root, "sample_submission.csv")

sample_sub = pd.read_csv(sample_path)
image_files = sample_sub["image"].astype(str).tolist()
IMAGE_PATHS = [os.path.join(source, f) for f in image_files]

missing = [f for f, p in zip(image_files, IMAGE_PATHS) if not os.path.exists(p)]
if len(missing) > 0:
    print(
        f"WARNING: Missing {len(missing)} test images in filesystem. Example: {missing[:3]}"
    )
else:
    print("Num test images found:", len(IMAGE_PATHS))
print("First 3:", image_files[:3])



## === cell 4
IMAGE_PATHS[:5]



## === cell 5
existing_paths = [p for p in IMAGE_PATHS if os.path.exists(p)]
if len(existing_paths) == 0:
    print(
        f"WARNING: No readable images found in {source}. Will generate a valid submission via fallback."
    )
else:
    print(f"Readable test images found: {len(existing_paths)}/{len(IMAGE_PATHS)}")



## === cell 6
AUTO = tf.data.experimental.AUTOTUNE



## === cell 7
existing_mask = np.array([os.path.exists(p) for p in IMAGE_PATHS], dtype=bool)
existing_indices = np.where(existing_mask)[0].tolist()
have_images = len(existing_indices) > 0

if have_images:
    idx_ds = tf.data.Dataset.from_tensor_slices(existing_indices)
    path_ds = tf.data.Dataset.from_tensor_slices(
        [IMAGE_PATHS[i] for i in existing_indices]
    )
    test_dataset = tf.data.Dataset.zip((idx_ds, path_ds))

    def _map_idx_path(i, p):
        return i, decode_image(p)

    test_dataset = (
        test_dataset.map(_map_idx_path, num_parallel_calls=AUTO)
        .batch(BATCH_SIZE)
        .prefetch(AUTO)
    )
else:
    test_dataset = None



## === cell 9
class FixedDropout(tf.keras.layers.Dropout):
    def _get_noise_shape(self, inputs):
        if self.noise_shape is None:
            return None
        symbolic_shape = tf.shape(inputs)
        noise_shape = [
            symbolic_shape[axis] if shape is None else shape
            for axis, shape in enumerate(self.noise_shape)
        ]
        return tuple(noise_shape)




## === cell 10
candidate_model_paths = [
    "../input/smresnet50/SMResNet50.h5",
    "/kaggle/input/smresnet50/SMResNet50.h5",
]
model_path = None
for mp in candidate_model_paths:
    if os.path.exists(mp):
        model_path = mp
        break

if model_path is not None:
    model = tf.keras.models.load_model(
        model_path, compile=False, custom_objects={"FixedDropout": FixedDropout}
    )
    print("Loaded model:", model_path)
else:
    print(f"WARNING: Model file not found at any of: {candidate_model_paths}")
    print("Falling back to a small baseline model to produce a valid submission.")
    inp = keras.Input(shape=(512, 512, 3))
    x = keras.layers.Conv2D(16, 3, strides=2, padding="same", activation="relu")(inp)
    x = keras.layers.Conv2D(32, 3, strides=2, padding="same", activation="relu")(x)
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dense(64, activation="relu")(x)
    out = keras.layers.Dense(6, activation="sigmoid")(x)
    model = keras.Model(inp, out)



## === cell 11
if have_images and test_dataset is not None:
    all_idx = []
    all_probs = []
    for batch in test_dataset:
        idx_batch, img_batch = batch
        probs_batch = model(img_batch, training=False).numpy()
        all_idx.extend(idx_batch.numpy().tolist())
        all_probs.append(probs_batch)
    probs_existing = (
        np.concatenate(all_probs, axis=0)
        if len(all_probs)
        else np.zeros((0, 6), dtype=np.float32)
    )
    print("probs_existing shape:", probs_existing.shape, "num indices:", len(all_idx))
else:
    probs_existing = None
    all_idx = []
    print(
        "Skipping model prediction because test images are not accessible in this environment."
    )



## === cell 12
if have_images and probs_existing is not None and len(probs_existing) > 0:
    probs_existing[:2]
else:
    print("No probability preview available (no predictions computed).")



## === cell 13
name = {
    0: "scab",
    1: "frog_eye_leaf_spot",
    2: "complex",
    3: "rust",
    4: "powdery_mildew",
    5: "healthy",
}

threshold = {
    0: 0.5,
    1: 0.4,
    2: 0.3,
    3: 0.4,
    4: 0.4,
}

temp_probs = None
if probs_existing is not None:
    temp_probs = np.full((len(image_files), 6), np.nan, dtype=np.float32)
    for k, idx in enumerate(all_idx):
        if 0 <= idx < len(image_files):
            temp_probs[idx, : probs_existing.shape[1]] = probs_existing[k]

pred_string = []
if temp_probs is not None:
    for line in temp_probs:
        s = ""
        count = 0

        max_i = min(6, int(line.shape[0]))

        for i in range(max_i):
            if i in threshold and np.isfinite(line[i]) and line[i] > threshold[i]:
                s = s + name[i] + " "
                count += 1

        if count >= 2:
            notComplex = True
            for i in range(max_i):
                if (
                    i in threshold
                    and np.isfinite(line[i])
                    and line[i] > threshold[i]
                    and name.get(i, "") == "complex"
                ):
                    notComplex = False
                    break
            if notComplex is True:
                s = s + "complex" + " "

        if s == "":
            s = name[5]

        pred_string.append(s.strip())

    print("Num predictions:", len(pred_string))
    print("First 5 predictions:", pred_string[:5])
else:
    pred_string = ["healthy"] * len(image_files)
    print("Generated fallback predictions:", len(pred_string))

if len(pred_string) < len(image_files):
    pred_string = pred_string + ["healthy"] * (len(image_files) - len(pred_string))
elif len(pred_string) > len(image_files):
    pred_string = pred_string[: len(image_files)]



## === cell 14
pred_string[:10]



## === cell 15
df = pd.DataFrame({"image": image_files, "labels": pred_string})
assert (
    len(df) == len(image_files) == len(pred_string)
), "Submission lengths do not match."

df.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", df.shape)
print(df.head())
