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

0.8240627885503237

# 6. Current score

0.24507

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.272) has done: 'I fix the immediate runtime error caused by importing `kaggle_datasets` (protobuf incompatibility) by removing that unused import. Then I make the model-loading robust: if the referenced external `.h5` file isn’t present in this environment, the script fall back to a standard Keras image model (kept strictly inference-only) so it can still generate predictions and a valid submission. Finally, I fix the submission alignment bug by ensuring `image` filenames are taken from the same sorted list used to build `IMAGE_PATHS`, so the number/order of filenames matches `pred_string` exactly and `submission.csv` is written correctly.'
- What this solution (achieved 0.22183) has done: 'I fix the TensorFlow import crash caused by an incompatible protobuf runtime by forcing the pure-Python protobuf implementation before importing TensorFlow. Then I keep your inference pipeline and label post-processing identical, but ensure the fallback model uses the same input preprocessing that its ImageNet weights expect (this is a minimal, metric-aligned calibration fix that should move score upward without changing the overall approach). Finally, I make paths robust to either `../input/...` or `/kaggle/input/...` layouts and keep the submission rows aligned to the sorted test filenames, writing a valid `submission.csv`.'
- What this solution (achieved 0.21369) has done: 'The timeout is dominated by slow TensorFlow import/initialization and heavy per-image CPU decoding/resizing at 512×512, plus extra overhead from forcing the pure-Python protobuf implementation (which also triggers your current protobuf error). I remove the protobuf-forcing environment variables (fixing the crash and restoring fast C++ protobuf), ensure deterministic behavior via TF seed and deterministic ops, and speed up the `tf.data` pipeline with `cache()` (safe for this small test set) plus `set_deterministic(True)` and explicit `drop_remainder=False`. I also avoid redundant conversions/copies around `predict` and keep the exact same model, thresholds, and label post-processing logic. These changes preserve inference semantics while cutting constant overhead so the notebook finishes well under 600 seconds.'
- What this solution (achieved 0.24507) has done: 'We fix the immediate TensorFlow import crash (`MessageFactory`/protobuf mismatch) by forcing the pure-Python protobuf implementation *before* importing TensorFlow, which is the minimal stability fix needed to run end-to-end in this environment. Then we keep your inference pipeline and label post-processing logic intact, but make the fallback model’s input preprocessing consistent (it already is) and also correct a key scoring issue: the fallback ImageNet model outputs are not aligned to the 5 competition classes, so we only use the external `.h5` model when available and otherwise fall back to a deterministic constant “healthy” submission (score be low but valid) rather than misleading random class logits. This preserves core semantics for the real intended path (external model) while guaranteeing a valid submission and preventing erroneous predictions from an incompatible fallback model. The script still write `submission.csv` with the correct columns and ordering.'
- What this solution (achieved 0.24507) has done: 'We fix the TensorFlow/protobuf crash by avoiding the forced pure-Python protobuf setting (it’s what’s triggering the `MessageFactory`/`GetPrototype` failure in this environment) and keeping TensorFlow imports clean. Then we make the external model path robust by searching common `/kaggle/input/**` locations so the intended `.h5` is actually found when it exists; this is the minimal change that should move the score up toward your target because the current “all healthy” fallback caps performance. Finally, we keep your dataset creation, prediction loop, thresholds, and submission formatting the same, only adding a safe multi-candidate model-path resolution and ensuring the submission is written as `submission.csv`.'
- What this solution (achieved 0.24507) has done: 'We fix the TensorFlow import crash caused by a protobuf/TensorFlow incompatibility by forcing TensorFlow to use the pure-Python protobuf implementation *before* importing TensorFlow (and keeping the rest of your pipeline unchanged). Then we add a tiny safety fallback: if TensorFlow still can’t import in this environment, the script still write a valid `submission.csv` (all `healthy`) so it always runs end-to-end. No changes are made to your model inference, thresholds, label post-processing, or submission formatting when the external `.h5` model loads successfully. This should unblock execution and allow you to actually use the external model (which is necessary to move the score toward the target).'
- What this solution (achieved 0.24507) has done: 'I fix the TensorFlow import crash by removing the forced pure-Python protobuf environment variables that are triggering the `MessageFactory.GetPrototype` failure in this Kaggle runtime, while keeping the rest of your pipeline unchanged. Then I keep your external-model loading logic but make it slightly more robust by also searching `/kaggle/input/**/2SMResNet50.h5`, so the intended `.h5` is more likely to be found and used (which is necessary to improve score from the current all-healthy fallback). Finally, I preserve your existing thresholding and label string formatting, and ensure the submission is written as `submission.csv` with the correct row alignment and columns.'

# 9. Code solution

## === cell 0
import os, re, random, glob

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", None)
os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", None)

import numpy as np
import pandas as pd

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

TF_AVAILABLE = True
TF_IMPORT_ERROR = None
try:
    import tensorflow as tf
    import tensorflow.keras.backend as K

    tf.keras.utils.set_random_seed(SEED)
    try:
        tf.config.experimental.enable_op_determinism(True)
    except Exception:
        pass

    print("tf:", tf.__version__)
    print("tf.keras:", tf.keras.__version__)
except Exception as e:
    TF_AVAILABLE = False
    TF_IMPORT_ERROR = repr(e)
    print(
        "WARNING: TensorFlow failed to import; will write fallback submission. Error:",
        TF_IMPORT_ERROR,
    )



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
if TF_AVAILABLE:

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
source_candidates = [
    "../input/plant-pathology-2021-fgvc8/test_images",
    "/kaggle/input/plant-pathology-2021-fgvc8/test_images",
]
source = None
for c in source_candidates:
    if os.path.isdir(c):
        source = c
        break
if source is None:
    raise FileNotFoundError(
        f"Could not find test_images directory. Tried: {source_candidates}"
    )

_valid_img = re.compile(r".*\.(jpg|jpeg|png)$", re.IGNORECASE)
test_files = sorted([f for f in os.listdir(source) if _valid_img.match(f)])
IMAGE_PATHS = [os.path.join(source, f) for f in test_files]

print("Test images dir:", source)
print("Num test images:", len(IMAGE_PATHS))
print("First 3:", IMAGE_PATHS[:3])



## === cell 4
IMAGE_PATHS[:10]



## === cell 5
if TF_AVAILABLE:
    AUTO = tf.data.experimental.AUTOTUNE

    options = tf.data.Options()
    options.experimental_deterministic = True

    test_dataset = (
        tf.data.Dataset.from_tensor_slices(IMAGE_PATHS)
        .with_options(options)
        .map(decode_image, num_parallel_calls=AUTO)
        .cache()
        .batch(BATCH_SIZE, drop_remainder=False)
        .prefetch(AUTO)
    )



## === cell 6
if TF_AVAILABLE:
    from tensorflow import keras

    class FixedDropout(tf.keras.layers.Dropout):
        def _get_noise_shape(self, inputs):
            if self.noise_shape is None:
                return self.noise_shape
            symbolic_shape = K.shape(inputs)
            noise_shape = [
                symbolic_shape[axis] if shape is None else shape
                for axis, shape in enumerate(self.noise_shape)
            ]
            return tuple(noise_shape)




## === cell 7
model = None
using_external = False

if TF_AVAILABLE:
    model_path_candidates = [
        "../input/2smresnet50/2SMResNet50.h5",
        "/kaggle/input/2smresnet50/2SMResNet50.h5",
        "../input/2smresnet50/2SMResNet50.h5".replace("../input", "/kaggle/input"),
    ]

    model_path_candidates += sorted(
        glob.glob("/kaggle/input/**/2SMResNet50.h5", recursive=True)
    )

    model_path = None
    for p in model_path_candidates:
        if os.path.exists(p):
            model_path = p
            break

    if model_path is not None:
        model = tf.keras.models.load_model(
            model_path,
            compile=False,
            custom_objects={"FixedDropout": FixedDropout},
        )
        using_external = True
        print("Loaded external model:", model_path)
    else:
        print(
            "WARNING: External model not found; will generate a valid fallback submission predicting 'healthy' for all images."
        )
else:
    print(
        "WARNING: TensorFlow unavailable; will generate a valid fallback submission predicting 'healthy' for all images."
    )



## === cell 8
if TF_AVAILABLE and using_external:
    probs = model.predict(test_dataset, verbose=1)
    temp_probs = np.asarray(probs)
    print("probs shape:", temp_probs.shape)
else:
    temp_probs = None



## === cell 9
name = {
    0: "scab",
    1: "frog_eye_leaf_spot",
    2: "complex",
    3: "rust",
    4: "powdery_mildew",
    6: "healthy",
}

threshold = {0: 0.5, 1: 0.4, 2: 0.5, 3: 0.5, 4: 0.5}

pred_string = []
if TF_AVAILABLE and using_external:
    for line in temp_probs:
        s = ""
        count = 0
        for i in range(5):
            if line[i] > threshold[i]:
                s = s + name[i] + " "
                count += 1
        if count >= 2:
            notComplex = True
            for i in range(5):
                if line[i] > threshold[i] and name[i] == "complex":
                    notComplex = False
                    break
            if notComplex is True:
                s = s + "complex" + " "

        if s == "":
            s = name[6]
        pred_string.append(s.strip())
else:
    pred_string = [name[6]] * len(test_files)

print("Num predictions:", len(pred_string))
print("First 5 labels:", pred_string[:5])



## === cell 10
df = pd.DataFrame({"image": test_files, "labels": pred_string})

assert (
    len(df) == len(test_files) == len(pred_string)
), "Mismatch between filenames and predictions."

df.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", df.shape)
print(df.head())
