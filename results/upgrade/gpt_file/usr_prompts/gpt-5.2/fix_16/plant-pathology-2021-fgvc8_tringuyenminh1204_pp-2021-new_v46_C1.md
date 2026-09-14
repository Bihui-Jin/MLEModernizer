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

0.8258541089566027

# 6. Current score

0.31852

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.23847) has done: 'I fix the two blockers preventing an end-to-end run: (1) the TensorFlow import crash caused by `kaggle_datasets` (remove that unused import), and (2) the missing pretrained model file by replacing it with an in-notebook TF/Keras model that preserves the same “predict → threshold → submission” semantics. I also make the image list deterministic and ensure the submission `image` order matches the prediction order to fix the length mismatch error. Finally, I keep the post-processing logic intact but ensure label strings are correctly space-delimited (no trailing spaces) and always non-empty, producing a valid `submission.csv`.'
- What this solution (achieved 0.31852) has done: 'The timeout is dominated by heavy per-image CPU preprocessing (JPEG decode + resize to 512²) and slow `model.predict` throughput, amplified by a too-large input resolution for ResNet50 and by forcing XLA/mixed precision settings that can backfire on Kaggle CPUs. To finish within 600 seconds without changing the model architecture or inference semantics, I keep ResNet50 + the same head and thresholds, but switch preprocessing to ResNet50’s native 224×224 input size (exact same model, just correct/cheaper input tensor shape), remove unnecessary `steps=` forcing, and optimize the `tf.data` pipeline with caching, parallelism, and non-blocking prefetch. I also disable mixed precision and XLA for stable/fast CPU inference (they often slow down CPU and increase compile overhead), while keeping determinism/seed behavior intact. Finally, I vectorize the label-string assembly to reduce Python-loop overhead while producing identical label outputs.'
- What this solution (achieved 0.31852) has done: 'We fix the TensorFlow import crash (`MessageFactory.GetPrototype`), which is caused by an incompatible protobuf runtime in this environment, by forcing the pure-Python protobuf implementation *before* importing TensorFlow. This is an execution-blocking bug and is score-neutral. After that, we keep your exact model/inference/post-processing logic intact, ensuring the pipeline runs end-to-end and writes a valid `submission.csv` with the required `image,labels` columns and correct ordering. No architectural/training/threshold changes are introduced, so the score behavior should remain consistent while unblocking submission generation.'
- What this solution (achieved 0.31852) has done: 'I fix the TensorFlow import crash caused by a protobuf incompatibility by forcing the pure-Python protobuf implementation and (crucially) disabling the C++ protobuf backend before TensorFlow loads. This is an execution-blocking bug and should be score-neutral, but it let the notebook run end-to-end again and reliably write `submission.csv`. I keep your model/inference/post-processing logic unchanged, only adding a small import-order guard and a safe fallback so TensorFlow can import in the Kaggle runtime. The rest of the pipeline (data root discovery, tf.data, ResNet50 head, thresholds, and submission ordering) is preserved.'

# 9. Code solution

## === cell 0
import os, re, math, random
import numpy as np
import pandas as pd

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "2")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_DISABLE_CPP_IMPLEMENTATION", "1")

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

import tensorflow as tf
import tensorflow.keras.backend as K

tf.random.set_seed(SEED)

try:
    tf.config.threading.set_intra_op_parallelism_threads(0)
    tf.config.threading.set_inter_op_parallelism_threads(0)
except Exception:
    pass

try:
    tf.config.optimizer.set_jit(False)
except Exception:
    pass

print("tf:", tf.__version__)
print("tf.keras:", tf.keras.__version__)



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
import pathlib

CANDIDATE_ROOTS = [
    "/kaggle/input/plant-pathology-2021-fgvc8",
    "/kaggle/data/plant-pathology-2021-fgvc8",
    "../input/plant-pathology-2021-fgvc8",
    "../data/plant-pathology-2021-fgvc8",
]


def find_data_root(candidates):
    for root in candidates:
        if not os.path.exists(root):
            continue
        for p in [root, os.path.join(root, "plant-pathology-2021-fgvc8")]:
            if os.path.exists(
                os.path.join(p, "sample_submission.csv")
            ) and os.path.isdir(os.path.join(p, "test_images")):
                return p
    for root in candidates:
        if not os.path.exists(root):
            continue
        try:
            for entry in os.listdir(root):
                p = os.path.join(root, entry)
                if (
                    os.path.isdir(p)
                    and os.path.exists(os.path.join(p, "sample_submission.csv"))
                    and os.path.isdir(os.path.join(p, "test_images"))
                ):
                    return p
        except Exception:
            pass
    return None


DATA_ROOT = find_data_root(CANDIDATE_ROOTS)
if DATA_ROOT is None:
    raise FileNotFoundError(
        "Could not find dataset root containing sample_submission.csv and test_images under expected Kaggle paths."
    )

source = os.path.join(DATA_ROOT, "test_images")
sample_sub_path = os.path.join(DATA_ROOT, "sample_submission.csv")

print("DATA_ROOT:", DATA_ROOT)
print("test_images:", source, "exists:", os.path.isdir(source))
print(
    "sample_submission.csv:",
    sample_sub_path,
    "exists:",
    os.path.exists(sample_sub_path),
)



## === cell 2
IMG_SIZE = (224, 224)


@tf.function
def decode_image(filename, label=None):
    bits = tf.io.read_file(filename)
    image = tf.io.decode_jpeg(bits, channels=3, dct_method="INTEGER_FAST")
    image = tf.image.resize(image, IMG_SIZE, method="bilinear", antialias=False)
    image = tf.cast(image, tf.float32) / 255.0
    image.set_shape([IMG_SIZE[0], IMG_SIZE[1], 3])
    if label is None:
        return image
    else:
        return image, label




## === cell 3
BATCH_SIZE = 128



## === cell 4
valid_ext = (".jpg", ".jpeg", ".png")
IMAGE_FILES = sorted([f for f in os.listdir(source) if f.lower().endswith(valid_ext)])

if len(IMAGE_FILES) == 0:
    raise RuntimeError(f"No image files found in {source}")

IMAGE_PATHS = [os.path.join(source, f) for f in IMAGE_FILES]
print("Num test images:", len(IMAGE_PATHS))
print("First 3:", IMAGE_FILES[:3])



## === cell 5
_ = IMAGE_PATHS[:5]
print("Example paths:", _)



## === cell 6
AUTO = tf.data.experimental.AUTOTUNE

options = tf.data.Options()
options.experimental_deterministic = True  # preserve determinism
options.experimental_optimization.apply_default_optimizations = True
options.experimental_optimization.map_parallelization = True

test_dataset = (
    tf.data.Dataset.from_tensor_slices(IMAGE_PATHS)
    .with_options(options)
    .map(decode_image, num_parallel_calls=AUTO, deterministic=True)
    .cache()
    .batch(BATCH_SIZE, drop_remainder=False)
    .prefetch(AUTO)
)



## === cell 7
from tensorflow import keras
from tensorflow.keras import layers


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




## === cell 8
inputs = keras.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
base = keras.applications.ResNet50(
    include_top=False, weights="imagenet", input_tensor=inputs, pooling="avg"
)
base.trainable = False

x = base.output
x = layers.Dense(256, activation="relu")(x)
x = FixedDropout(0.2)(x)
outputs = layers.Dense(6, activation="sigmoid")(x)

model = keras.Model(inputs=inputs, outputs=outputs)

if getattr(model.output, "dtype", None) != "float32":
    model = keras.Model(inputs=model.inputs, outputs=tf.cast(model.outputs, tf.float32))

model.summary()



## === cell 9
probs = model.predict(test_dataset, verbose=1)
temp_probs = probs

print("probs shape:", probs.shape)
print("probs min/max:", float(probs.min()), float(probs.max()))



## === cell 10
name = {
    0: "scab",
    1: "frog_eye_leaf_spot",
    2: "complex",
    3: "rust",
    4: "powdery_mildew",
    5: "healthy",
}

threshold = {0: 0.2, 1: 0.2, 2: 0.2, 3: 0.2, 4: 0.2, 5: 0.2}
threshold2 = {0: 0.2, 1: 0.2, 2: 0.2, 3: 0.2, 4: 0.2, 5: 0.2}

complex_idx = 2  # get_key("complex") == 2 under the fixed mapping

thr = np.array([threshold[i] for i in range(6)], dtype=np.float32)
thr2 = np.array([threshold2[i] for i in range(6)], dtype=np.float32)

above_thr = temp_probs > thr[None, :]
above_thr2 = temp_probs > thr2[None, :]

count2 = above_thr2.sum(axis=1)
has_complex = above_thr[:, complex_idx]
need_add_complex = (count2 >= 2) & (~has_complex)

label_names = np.array([name[i] for i in range(6)], dtype=object)

pred_string = []
for j in range(above_thr.shape[0]):
    idx = np.flatnonzero(above_thr[j])
    if idx.size:
        parts = list(label_names[idx])
        if need_add_complex[j]:
            parts.append("complex")
    else:
        parts = [name[5]]
    pred_string.append(" ".join(parts).strip())

print("Num predictions:", len(pred_string))
print("First 5 preds:", pred_string[:5])



## === cell 11
sample_sub = pd.read_csv(sample_sub_path)
expected_cols = list(sample_sub.columns)
if expected_cols != ["image", "labels"]:
    print("Warning: unexpected sample_submission columns:", expected_cols)

pred_map = dict(zip(IMAGE_FILES, pred_string))
ordered_images = sample_sub["image"].tolist()

missing = [im for im in ordered_images if im not in pred_map]
if len(missing) > 0:
    raise RuntimeError(
        f"Missing predictions for {len(missing)} images. Example: {missing[:3]}"
    )

sub = pd.DataFrame(
    {
        "image": ordered_images,
        "labels": [pred_map[im] for im in ordered_images],
    }
)

sub = sub[["image", "labels"]]
assert len(sub) == len(sample_sub)

sub_path = "submission.csv"
sub.to_csv(sub_path, index=False)
print("Wrote:", sub_path)
print(sub.head())
print(sub.tail())
