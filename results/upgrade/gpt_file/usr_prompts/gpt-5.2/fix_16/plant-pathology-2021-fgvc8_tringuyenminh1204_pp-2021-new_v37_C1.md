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

0.8204801477377661

# 6. Current score

0.13499

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.20811) has done: 'I remove the failing `kaggle_datasets` import that triggers the protobuf `MessageFactory.GetPrototype` error, since it is unused for this filesystem-based notebook. I also fix the missing external model file by building the same kind of backbone model (ResNet50) inside the notebook so inference can run end-to-end without extra inputs. Next, I make the test image list deterministic and align submission rows to `sample_submission.csv` order to avoid the length/order mismatch that caused the DataFrame error. Finally, I keep your existing threshold-to-label logic intact (including the “add complex if >=2 diseases” rule) and ensure the output `submission.csv` has the required `image,labels` columns.'
- What this solution (achieved 0.13499) has done: 'I fix the TensorFlow/protobuf crash by removing the optional determinism/JIT toggles that trigger the `MessageFactory.GetPrototype` issue in this environment, while keeping the same model and preprocessing. Then I fix inference by using `model.predict(test_dataset)` (Keras 3 no longer accepts a `tf.data.Dataset` directly via `model(...)`), which also unblocks downstream thresholding. Finally, I keep your exact threshold→label string logic, ensure prediction length matches `sample_submission.csv`, and always write a valid `submission.csv` with `image,labels`.'

# 9. Code solution

## === cell 0
import os, re, math, random
import numpy as np
import pandas as pd

try:
    import tensorflow as tf
    import tensorflow.keras.backend as K
except Exception as e:
    raise RuntimeError(
        "TensorFlow failed to import in this environment. "
        "This notebook expects the Kaggle runtime with TensorFlow available."
    ) from e

print("tf:", tf.__version__)

try:
    for gpu in tf.config.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(gpu, True)
except Exception:
    pass

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
try:
    tf.random.set_seed(SEED)
except Exception:
    pass



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
import pathlib




## === cell 2
@tf.function
def decode_image(filename, label=None, image_size=(512, 512)):
    bits = tf.io.read_file(filename)
    image = tf.image.decode_jpeg(bits, channels=3, dct_method="INTEGER_FAST", ratio=2)
    image = tf.image.resize(
        image, image_size, method=tf.image.ResizeMethod.BILINEAR, antialias=False
    )
    image = tf.keras.applications.resnet50.preprocess_input(tf.cast(image, tf.float32))
    image.set_shape((image_size[0], image_size[1], 3))
    if label is None:
        return image
    else:
        return image, label




## === cell 3
try:
    gpus = tf.config.list_physical_devices("GPU")
    _HAS_GPU = len(gpus) > 0
except Exception:
    _HAS_GPU = False

BATCH_SIZE = 64 if _HAS_GPU else 32



## === cell 4
source = "../input/plant-pathology-2021-fgvc8/test_images"

sample_path = "../input/plant-pathology-2021-fgvc8/sample_submission.csv"
sample_sub = pd.read_csv(sample_path)

IMAGE_PATHS = [os.path.join(source, img) for img in sample_sub["image"].tolist()]

if len(IMAGE_PATHS) > 0:
    try:
        _probe = IMAGE_PATHS[0]
        if not tf.io.gfile.exists(_probe):
            print(
                "Warning: test image paths not found locally (expected in hidden test at submit time). Example:",
                _probe,
            )
    except Exception:
        pass



## === cell 5
IMAGE_PATHS[:5], len(IMAGE_PATHS)



## === cell 6
AUTO = tf.data.experimental.AUTOTUNE



## === cell 7
options = tf.data.Options()
options.experimental_deterministic = True
try:
    options.experimental_optimization.apply_default_optimizations = True
except Exception:
    pass
try:
    options.threading.private_threadpool_size = 0
except Exception:
    pass

test_dataset = tf.data.Dataset.from_tensor_slices(IMAGE_PATHS).with_options(options)
test_dataset = test_dataset.map(
    decode_image, num_parallel_calls=AUTO, deterministic=True
)
test_dataset = test_dataset.batch(BATCH_SIZE, drop_remainder=False)
test_dataset = test_dataset.prefetch(AUTO)




## === cell 8
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




## === cell 9
def build_inference_model(image_size=(512, 512, 3), n_outputs=5):
    inputs = tf.keras.Input(shape=image_size)
    base = tf.keras.applications.ResNet50(
        include_top=False, weights="imagenet", input_tensor=inputs, pooling="avg"
    )
    x = base.output
    x = tf.keras.layers.Dense(512, activation="relu")(x)
    x = FixedDropout(0.2)(x)
    outputs = tf.keras.layers.Dense(n_outputs, activation="sigmoid")(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model


model = build_inference_model(image_size=(512, 512, 3), n_outputs=5)
model.compile(optimizer="adam", loss="binary_crossentropy")

_ = model(tf.zeros((1, 512, 512, 3), dtype=tf.float32), training=False)



## === cell 10
n_test = len(IMAGE_PATHS)

probs = model.predict(test_dataset, verbose=1)
probs = np.asarray(probs, dtype=np.float32)

if probs.shape[0] != n_test:
    probs = probs[:n_test]

temp_probs = probs
temp_probs.shape



## === cell 11
name = {
    0: "scab",
    1: "frog_eye_leaf_spot",
    2: "complex",
    3: "rust",
    4: "powdery_mildew",
    6: "healthy",
}

threshold = {
    0: 0.25,
    1: 0.4,
    2: 0.25,
    3: 0.4,
    4: 0.4,
}


def get_key(val):
    for key, value in name.items():
        if val == value:
            return key
    return "key doesn't exist"


thr = np.array([threshold[i] for i in range(5)], dtype=temp_probs.dtype)
mask = temp_probs > thr  # (N,5) bool
need_add_complex = (mask.sum(axis=1) >= 2) & (~mask[:, 2])

names5 = np.array([name[i] for i in range(5)], dtype=object)

pred_string = []
append_pred = pred_string.append
for i in range(mask.shape[0]):
    m = mask[i]
    toks = []
    if m[0]:
        toks.append(names5[0])
    if m[1]:
        toks.append(names5[1])
    if m[2]:
        toks.append(names5[2])
    if m[3]:
        toks.append(names5[3])
    if m[4]:
        toks.append(names5[4])
    if need_add_complex[i]:
        toks.append("complex")
    append_pred(name[6] if not toks else " ".join(toks))

if len(pred_string) != len(sample_sub):
    if len(pred_string) == 0:
        print(
            "Warning: No predictions produced. Falling back to all 'healthy' to create a valid submission."
        )
        pred_string = [name[6]] * len(sample_sub)
    elif len(pred_string) < len(sample_sub):
        print(
            f"Warning: Only {len(pred_string)} predictions for {len(sample_sub)} rows. "
            "Padding remaining rows with 'healthy' to match required submission length."
        )
        pred_string = pred_string + [name[6]] * (len(sample_sub) - len(pred_string))
    else:
        print(
            f"Warning: More predictions ({len(pred_string)}) than submission rows ({len(sample_sub)}). Truncating."
        )
        pred_string = pred_string[: len(sample_sub)]

len(pred_string), pred_string[:5]



## === cell 12
sub = sample_sub.copy()
sub["labels"] = pred_string

sub = sub[["image", "labels"]]
sub.to_csv("submission.csv", index=False)

print(sub.head())
print("Wrote submission.csv with shape:", sub.shape)
