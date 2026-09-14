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

0.8214773776546646

# 6. Current score

0.30517

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.30585) has done: 'I fix the TensorFlow import crash by avoiding the incompatible `sklearn` dependency (it triggers a protobuf `MessageFactory` error in this environment) and replace it with a tiny local label parser that preserves the same semantics. Then I fix the `decode_image` pipeline: the current `decode_and_crop_jpeg` call uses Python `max()` on a Tensor, which breaks graph tracing; I switch to a standard `decode_jpeg` + `resize` path that is stable in `tf.data`. Finally, I make sure `test_dataset` is successfully created before inference, and that we always write a valid `submission.csv` with the required `image,labels` columns and space-delimited labels.'
- What this solution (achieved 0.30748) has done: 'We fix the crash happening at import time by avoiding the TensorFlow/protobuf incompatibility that gets triggered by the current TensorFlow import in this environment, while keeping the rest of your pipeline (InceptionV3 model + sigmoid outputs + thresholding + submission formatting) the same. Concretely, we make TensorFlow import lazy and more robust by setting safe environment flags before importing it, and we add a defensive fallback so the notebook can still run end-to-end and write `submission.csv` even if TF cannot be imported (it then output a simple baseline prediction to ensure a valid file). This unblocks execution and allows the intended model inference path to run when TF loads successfully. No changes are made to your model architecture or post-processing semantics when TF is available.'
- What this solution (achieved 0.30567) has done: 'I fix the TensorFlow/protobuf import crash that prevents the model path from running, by ensuring the environment variables are set before *any* TensorFlow-related imports and by avoiding optional imports that can trigger the protobuf `MessageFactory` issue. I also make the class-to-index mapping consistent with the actual training labels (currently it’s hard-coded in an order that can be wrong), which should significantly improve F1 without changing your model architecture or thresholding logic. Finally, I keep the same inference and submission formatting, but add small defensive checks so `submission.csv` is always produced correctly even if TF is unavailable.'
- What this solution (achieved 0.30517) has done: 'I fix the TensorFlow/protobuf import crash by forcing the pure-Python protobuf implementation before any TF import and by removing the `tf.function` decorations that can trigger graph/protobuf interactions in this environment. Then I keep your exact model/inference logic but make the label mapping robust (still derived from `train.csv`) and ensure the Dense head always matches the number of discovered classes (and remains 6 as expected). Finally, I keep the same thresholding/submission formatting, while guaranteeing `submission.csv` is always written with the required `image,labels` columns even if TF still fails to import.'

# 9. Code solution

## === cell 0
import os
import gc
import warnings

import numpy as np
import pandas as pd

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION"] = "2"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = os.environ.get("TF_CPP_MIN_LOG_LEVEL", "2")

warnings.filterwarnings("ignore")

np.random.seed(0)

BATCH_SIZE = 64

IMAGE_PATH = "../input/plant-pathology-2021-fgvc8/train_images/"
TRAIN_PATH = "../input/plant-pathology-2021-fgvc8/train.csv"
SUB_PATH = "../input/plant-pathology-2021-fgvc8/sample_submission.csv"
TEST_IMAGE_DIR = "../input/plant-pathology-2021-fgvc8/test_images/"

sub = pd.read_csv(SUB_PATH)
test_data = sub.copy()
train_data = pd.read_csv(TRAIN_PATH)

train_data["labels"] = train_data["labels"].astype(str).apply(lambda s: s.split(" "))
all_classes = sorted(
    {lab for labs in train_data["labels"].tolist() for lab in labs if lab}
)

print("Loaded train rows:", len(train_data), "test rows:", len(test_data))
print("Classes from train:", all_classes)

if len(all_classes) != 6:
    print("WARNING: Expected 6 classes, got:", len(all_classes), all_classes)

idx2label = {i: c for i, c in enumerate(all_classes)}
label_arr = np.array([idx2label[i] for i in range(len(all_classes))], dtype=object)
print("idx2label mapping:", idx2label)

TF_AVAILABLE = True
TF_IMPORT_ERROR = None
try:
    import tensorflow as tf
    from tensorflow import keras
    import tensorflow.keras.layers as L

    tf.random.set_seed(0)
    print("TF version:", tf.__version__)

    try:
        tf.config.threading.set_intra_op_parallelism_threads(0)
        tf.config.threading.set_inter_op_parallelism_threads(0)
    except Exception:
        pass

    try:
        tf.config.optimizer.set_jit(True)
    except Exception:
        pass

    AUTO = tf.data.experimental.AUTOTUNE
except Exception as e:
    TF_AVAILABLE = False
    TF_IMPORT_ERROR = repr(e)
    print("WARNING: TensorFlow failed to import; will write a fallback submission.")
    print("TF import error:", TF_IMPORT_ERROR)

gc.collect()



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
IMG_SIZE = (299, 299)

if TF_AVAILABLE:
    def decode_image(filename, label=None, image_size=IMG_SIZE):
        bits = tf.io.read_file(filename)
        image = tf.image.decode_jpeg(bits, channels=3)
        image = tf.image.convert_image_dtype(image, tf.float32)
        image = tf.image.resize(image, image_size, antialias=True)
        if label is None:
            return image
        return image, label

    test_paths = (TEST_IMAGE_DIR + test_data["image"].astype(str)).to_numpy()

    options = tf.data.Options()
    options.experimental_deterministic = True

    test_dataset = (
        tf.data.Dataset.from_tensor_slices(test_paths)
        .with_options(options)
        .map(decode_image, num_parallel_calls=AUTO, deterministic=True)
        .batch(BATCH_SIZE, drop_remainder=False)
        .prefetch(AUTO)
    )

    print("Test dataset ready. Num test images:", len(test_data))



## === cell 2
if TF_AVAILABLE:
    inputs = tf.keras.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))

    x = tf.keras.applications.InceptionV3(include_top=False, weights=None)(inputs)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)

    num_classes = len(all_classes)
    outputs = tf.keras.layers.Dense(num_classes, activation="sigmoid")(x)
    model = tf.keras.models.Model(inputs, outputs)

    weights_path = "../input/trail-1-dataset/Inceptionv3.h5"
    if os.path.exists(weights_path):
        model.load_weights(weights_path)
        print("Loaded custom weights:", weights_path)
    else:
        print("Custom weights not found. Falling back to ImageNet backbone weights.")
        backbone = tf.keras.applications.InceptionV3(
            include_top=False, weights="imagenet", input_tensor=inputs
        )
        x = tf.keras.layers.GlobalAveragePooling2D()(backbone.output)
        outputs = tf.keras.layers.Dense(num_classes, activation="sigmoid")(x)
        model = tf.keras.models.Model(inputs, outputs)

    model.summary()



## === cell 3
if TF_AVAILABLE:
    preds = model.predict(test_dataset, verbose=0)

    threshold = 0.25
    preds_np = np.asarray(preds)

    if preds_np.ndim != 2 or preds_np.shape[1] != len(all_classes):
        raise ValueError(
            f"Unexpected prediction shape: {preds_np.shape} for num_classes={len(all_classes)}"
        )

    mask = preds_np >= threshold  # [N,C] boolean
    has_any = mask.any(axis=1)
    argmax_idx = preds_np.argmax(axis=1)

    mask2 = mask.copy()
    mask2[~has_any, :] = False
    mask2[~has_any, argmax_idx[~has_any]] = True

    testlabels = [" ".join(label_arr[m].tolist()) for m in mask2]

    out = sub.copy()
    out["labels"] = testlabels
else:
    out = sub.copy()
    out["labels"] = "healthy"

out["image"] = out["image"].astype(str)
out["labels"] = out["labels"].astype(str).str.strip().replace({"": "healthy"})

out.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", out.shape)
print(out.head())
