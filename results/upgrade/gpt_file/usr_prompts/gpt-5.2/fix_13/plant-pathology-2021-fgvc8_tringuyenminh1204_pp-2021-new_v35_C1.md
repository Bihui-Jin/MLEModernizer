# Goal

I want you to improve my Kaggle competition solution to increase the score toward a target. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (evaluation score improvement); avoid unrelated refactors or stylistic edits.
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

0.7887903970452463

# 6. Current score

0.24731

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.29785) has done: 'Main runtime wins come from eliminating unnecessary work in the input pipeline and reducing per-image compute while keeping the exact same model and thresholds. The biggest bottleneck is resizing every JPEG to 512×512 and running ResNet50 on that size; we keep the same ResNet50 architecture and head but switch to its native 224×224 input resolution (still identical core logic: ResNet50 + Dense(256) + Dropout + Dense(5) sigmoid, same thresholds/label rules). We also remove `cache()` (it caches decoded+resized images in RAM and doesn’t help one-pass inference), and we ensure the dataset uses parallel I/O efficiently with deterministic execution preserved. Finally, we vectorize the label-string building to reduce Python-loop overhead while keeping the exact same semantics (including the “add complex if >=2 labels and complex not already present” rule).'
- What this solution (achieved 0.24113) has done: 'I fix the runtime crash in the first cell by removing/guarding the XLA JIT toggle that can trigger a protobuf `MessageFactory.GetPrototype` incompatibility in some Kaggle TF builds, while keeping determinism seeding intact. I also ensure the test image folder resolution is robust (fallback to the alternate provided path) so the pipeline always finds all test images. Finally, I keep your exact model and thresholding logic but add the correct ResNet50 `preprocess_input` step (this is evaluation-consistent with ImageNet weights and should raise score toward the target without changing architecture/loops), and I keep the submission formatting unchanged.'
- What this solution (achieved 0.24113) has done: 'I fix the crash coming from TensorFlow’s determinism/XLA configuration by fully guarding those calls (they can trigger the protobuf `MessageFactory.GetPrototype` error in some Kaggle TF builds) while keeping seeds and the rest of your pipeline unchanged. I also make test image path discovery more robust by checking the common Kaggle-mounted locations and falling back safely, without changing how images are loaded or how predictions are generated. Finally, I keep your exact model architecture and threshold/label-string rules intact, ensuring the submission is written as a valid `submission.csv` with the required `image,labels` columns.'
- What this solution (achieved 0.24113) has done: 'I fix the TensorFlow import crash (`MessageFactory.GetPrototype`) by safely forcing the pure-Python protobuf implementation before importing TensorFlow, which is a common Kaggle TF/protobuf incompatibility. Then I keep your exact model/inference/thresholding logic intact, but correct a subtle head bug: you currently feed `input_tensor=x_in` while also defining `inputs=inputs`, which creates a graph mismatch; I instead apply `preprocess_input` as a normal layer and pass the original `inputs` into ResNet50. Finally, I ensure test image discovery doesn’t miss nested directories and that the submission is written with the required `image,labels` columns to `submission.csv`.'
- What this solution (achieved 0.24113) has done: 'I fix the TensorFlow import crash by forcing TensorFlow to use the pure-Python protobuf implementation *before the Python protobuf module is ever imported*, since importing `tensorflow` after `google.protobuf` is loaded can still trigger the `MessageFactory.GetPrototype` error. I keep your exact model, input size, preprocessing, thresholds, and label-string rules unchanged (score-impact should be neutral aside from actually running). I also add a small safety fallback for protobuf implementation type to reduce crash risk and ensure the notebook always reaches the CSV write step. The output still be a valid `submission.csv` with `image,labels` columns.'
- What this solution (achieved 0.24113) has done: 'I fix the TensorFlow import crash (`AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'`) by ensuring the protobuf environment variables are set before *any* protobuf/TensorFlow-related import and by forcing the pure-Python protobuf implementation early. I also add a safe fallback that monkey-patches the missing `GetPrototype` method when this specific incompatibility still occurs, so the notebook can run end-to-end reliably. The rest of the pipeline (dataset loading, ResNet50 architecture/head, preprocessing, thresholds, and label post-processing) be kept the same to preserve evaluation semantics while allowing the run to complete and produce `submission.csv`. Finally, I keep test image discovery robust and ensure the submission columns match exactly `image,labels`.'
- What this solution (achieved 0.24113) has done: 'Your score is far below the target, so the smallest “toward-target” improvement is to fix the likely cause: you’re running an ImageNet-pretrained ResNet50 with random (untrained) Dense head weights, which yields near-random multilabel predictions and very low mean F1. I keep the exact same backbone/head architecture and inference-only flow, but add a minimal weight-loading step for the head (and optionally full model) from common Kaggle input locations if a saved model/weights file exists; if not found, it fail loudly rather than silently producing a poor submission. I also keep your exact preprocessing and label post-processing/threshold logic unchanged to preserve evaluation semantics. This should move the score substantially upward toward the target if you intended to use trained weights.'
- What this solution (achieved 0.24731) has done: 'The runtime failure is caused by the hard error when no trained weights file is found; in Kaggle’s environment you typically won’t have extra `/kaggle/input/model/*.h5` unless you add it yourself, so the notebook must be able to run end-to-end without that dependency. I keep your exact model architecture, preprocessing, thresholds, and label post-processing semantics, but change the weight-loading block to be non-fatal: it try to load weights if present and otherwise proceed with ImageNet backbone + randomly initialized head (so it always produces a valid `submission.csv`). Since your current score is far below target, I also add a minimal, metric-consistent calibration step that does not change the model: compute per-class thresholds from the provided `train.csv` label prevalences (a lightweight prior-based calibration) and use those instead of the fixed thresholds only when weights are missing; if weights are found, your original fixed thresholds are used unchanged. This should improve score versus near-random outputs while keeping the core logic intact and ensuring a valid CSV is written.'

# 9. Code solution

## === cell 0
import os

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "2")
os.environ.setdefault("TF_XLA_FLAGS", "--tf_xla_enable_xla_devices=false")

import re, math, random
import numpy as np
import pandas as pd

try:
    from google.protobuf import message_factory as _message_factory

    if hasattr(_message_factory, "MessageFactory") and not hasattr(
        _message_factory.MessageFactory, "GetPrototype"
    ):

        def _GetPrototype(self, descriptor):
            return self.GetMessageClass(descriptor)

        _message_factory.MessageFactory.GetPrototype = _GetPrototype
except Exception as _e:
    pass

import tensorflow as tf
import tensorflow.keras.backend as K

print("TF:", tf.__version__)
print("Keras:", tf.keras.__version__)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

try:
    if hasattr(tf.config.experimental, "enable_op_determinism"):
        tf.config.experimental.enable_op_determinism()
except Exception as e:
    print("enable_op_determinism failed (ignored):", repr(e))

try:
    import multiprocessing as _mp

    _cpu = _mp.cpu_count()
    tf.config.threading.set_intra_op_parallelism_threads(min(8, _cpu))
    tf.config.threading.set_inter_op_parallelism_threads(min(4, _cpu))
except Exception as e:
    print("thread config failed (ignored):", repr(e))



## === cell 1
import pathlib




## === cell 2
@tf.function
def decode_image(filename, label=None, image_size=(224, 224)):
    bits = tf.io.read_file(filename)
    image = tf.image.decode_jpeg(bits, channels=3)
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.image.resize(image, image_size)
    if label is None:
        return image
    else:
        return image, label




## === cell 3
BATCH_SIZE = 32



## === cell 4
candidate_sources = [
    "/kaggle/input/plant-pathology-2021-fgvc8/test_images",
    "/kaggle/input/plant-pathology-2021-fgvc8/plant-pathology-2021-fgvc8/test_images",
    "../input/plant-pathology-2021-fgvc8/test_images",
    "../input/plant-pathology-2021-fgvc8/plant-pathology-2021-fgvc8/test_images",
]
source = None
for s in candidate_sources:
    if os.path.isdir(s):
        source = s
        break
if source is None:
    raise FileNotFoundError(
        f"Could not find test_images in any of: {candidate_sources}"
    )

p = pathlib.Path(source)
valid_ext = {".jpg", ".jpeg", ".png"}

IMAGE_PATHS = sorted(
    str(fp) for fp in p.rglob("*") if fp.is_file() and fp.suffix.lower() in valid_ext
)

TEST_IMAGE_IDS = [os.path.basename(pp) for pp in IMAGE_PATHS]

print("Using test_images:", source)
print("Num test images found:", len(IMAGE_PATHS))
print("First 3:", TEST_IMAGE_IDS[:3])



## === cell 5
IMAGE_PATHS[:5]



## === cell 6
AUTO = tf.data.AUTOTUNE



## === cell 7
_opts = tf.data.Options()
_opts.experimental_deterministic = True
_opts.experimental_optimization.apply_default_optimizations = True
_opts.experimental_optimization.map_parallelization = True

test_dataset = tf.data.Dataset.from_tensor_slices(IMAGE_PATHS).with_options(_opts)
test_dataset = test_dataset.map(
    decode_image, num_parallel_calls=AUTO, deterministic=True
)
test_dataset = test_dataset.batch(
    BATCH_SIZE, drop_remainder=False, num_parallel_calls=AUTO
)
test_dataset = test_dataset.prefetch(AUTO)



## === cell 8
from tensorflow import keras




## === cell 9
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




## === cell 10
inputs = keras.Input(shape=(224, 224, 3))
x = keras.applications.resnet.preprocess_input(inputs * 255.0)

backbone = keras.applications.ResNet50(
    include_top=False,
    weights="imagenet",
    input_tensor=None,
    input_shape=(224, 224, 3),
    pooling="avg",
)
x = backbone(x, training=False)

x = keras.layers.Dense(256, activation="relu")(x)
x = keras.layers.Dropout(0.2)(x)
outputs = keras.layers.Dense(5, activation="sigmoid")(x)

model = keras.Model(inputs=inputs, outputs=outputs)


def _try_load_weights(m):
    candidates = [
        "/kaggle/input/model/model.h5",
        "/kaggle/input/model/best.h5",
        "/kaggle/input/model/weights.h5",
        "/kaggle/input/model/model.weights.h5",
        "/kaggle/input/model/best.weights.h5",
        "/kaggle/input/plant-pathology-2021-fgvc8/model.h5",
        "/kaggle/input/plant-pathology-2021-fgvc8/best.h5",
        "/kaggle/input/plant-pathology-2021-fgvc8/weights.h5",
        "/kaggle/working/model.h5",
        "/kaggle/working/best.h5",
        "/kaggle/working/weights.h5",
        "./model.h5",
        "./best.h5",
        "./weights.h5",
        "./model.weights.h5",
        "./best.weights.h5",
    ]
    for path in candidates:
        if os.path.isfile(path):
            print("Loading model weights from:", path)
            m.load_weights(path)
            return path
    return None


loaded_from = _try_load_weights(model)
if loaded_from is None:
    print(
        "WARNING: No trained weights found. Proceeding with ImageNet backbone and randomly initialized head.\n"
        "A submission.csv will still be generated, but score may be low unless thresholds are calibrated."
    )

model.trainable = False  # inference-only; avoids any accidental training
print(model.output_shape)



## === cell 11
model.compile(run_eagerly=False)

probs = model.predict(test_dataset, verbose=1)
temp_probs = probs

print("probs shape:", probs.shape)



## === cell 12
probs[:2]



## === cell 13
name = {
    0: "scab",
    1: "frog_eye_leaf_spot",
    2: "complex",
    3: "rust",
    4: "powdery_mildew",
    6: "healthy",
}

default_threshold = {0: 0.5, 1: 0.4, 2: 0.35, 3: 0.35, 4: 0.6}

threshold = dict(default_threshold)
if loaded_from is None:
    train_csv_candidates = [
        "/kaggle/input/plant-pathology-2021-fgvc8/train.csv",
        "/kaggle/input/train.csv",
        "../input/plant-pathology-2021-fgvc8/train.csv",
        "../input/train.csv",
    ]
    train_csv_path = None
    for tp in train_csv_candidates:
        if os.path.isfile(tp):
            train_csv_path = tp
            break

    if train_csv_path is not None:
        tr = pd.read_csv(train_csv_path)
        labels_series = tr["labels"].fillna("")
        preval = {}
        for i in range(5):
            cls = name[i]
            preval[i] = labels_series.str.contains(
                rf"(^| )\b{re.escape(cls)}\b( |$)", regex=True
            ).mean()

        for i in range(5):
            p_i = float(preval[i])
            thr_i = 0.5 - 0.25 * (p_i - 0.2)  # centered around 0.2 prevalence
            thr_i = float(np.clip(thr_i, 0.2, 0.8))
            threshold[i] = thr_i

        print("Using prevalence-calibrated thresholds (weights missing):", threshold)
    else:
        print(
            "Train CSV not found for calibration; using default thresholds:", threshold
        )
else:
    print("Using default thresholds (weights loaded):", threshold)

thr = np.array([threshold[i] for i in range(5)], dtype=temp_probs.dtype)
above = temp_probs > thr  # shape (N, 5)

idx_to_name = [name[i] for i in range(5)]
complex_idx = 2  # fixed by mapping above

pred_string = []
for row_mask in above:
    s_parts = [idx_to_name[i] for i in range(5) if row_mask[i]]
    count = len(s_parts)
    if count >= 2 and (not row_mask[complex_idx]):
        s_parts.append("complex")
    pred_string.append(name[6] if not s_parts else " ".join(s_parts))

print("Num predictions:", len(pred_string))
print(pred_string[:5])



## === cell 14
pred_string[:10]



## === cell 15
df = pd.DataFrame({"image": TEST_IMAGE_IDS, "labels": pred_string})

df.to_csv("submission.csv", index=False)
print(df.head())
print("Wrote submission.csv with shape:", df.shape)
print("Submission path:", os.path.abspath("submission.csv"))
