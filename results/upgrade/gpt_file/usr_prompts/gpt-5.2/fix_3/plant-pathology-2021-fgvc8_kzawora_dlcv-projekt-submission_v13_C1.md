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

0.6406278855032318

# 6. Current score

0.24507

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved 0.24507) has done: 'I fix the root cause preventing the model from loading (the protobuf/TensorFlow incompatibility that triggers the `MessageFactory.GetPrototype` error) by switching to `tf.keras.models.load_model(..., compile=False)` and, if needed, rebuilding the model from its config and loading weights. Then I ensure the pipeline always defines `model`, runs prediction, and produces `submission.csv` with the exact required columns and correct image-to-prediction alignment. I also make the test image copy step robust (only copy when missing) and add a small fallback so the generator and submission stay consistent even if directory structure differs. These changes are execution/format fixes and should be score-neutral except for restoring the intended predictions.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd
from pathlib import Path
import tensorflow as tf

BASE_INPUT = Path("../input/plant-pathology-2021-fgvc8")
MODEL_PATH = Path("../input/dlcv-projekt/model-best.h5")

print("TensorFlow:", tf.__version__)
print("Base input exists:", BASE_INPUT.exists())
print("Model path exists:", MODEL_PATH.exists())


def safe_load_keras_model(model_path: Path):
    try:
        return tf.keras.models.load_model(model_path, compile=False)
    except Exception as e1:
        print("Primary load_model failed:", repr(e1))
        try:
            with tf.keras.utils.custom_object_scope({}):
                m = tf.keras.models.load_model(model_path, compile=False)
            return m
        except Exception as e2:
            print("Secondary load_model failed:", repr(e2))
        try:
            import h5py

            with h5py.File(model_path, "r") as f:
                if "model_config" in f.attrs:
                    model_config = f.attrs.get("model_config")
                    if isinstance(model_config, bytes):
                        model_config = model_config.decode("utf-8")
                    m = tf.keras.models.model_from_json(model_config)
                    m.load_weights(str(model_path))
                    return m
                else:
                    raise RuntimeError(
                        "No model_config found in H5; cannot reconstruct model safely."
                    )
        except Exception as e3:
            print("Fallback reconstruction failed:", repr(e3))
            raise


model = safe_load_keras_model(MODEL_PATH)
print("Model loaded. Outputs:", getattr(model, "output_shape", None))



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
os.makedirs("/kaggle/tmp/test_dataset/test", exist_ok=True)

src_test_dir = BASE_INPUT / "test_images"
dst_test_dir = Path("/kaggle/tmp/test_dataset/test")

dst_jpgs = list(dst_test_dir.glob("*.jpg"))
if len(dst_jpgs) == 0:
    os.system(f'cp -r "{src_test_dir}/." "{dst_test_dir}/"')
dst_jpgs = list(dst_test_dir.glob("*.jpg"))
print("Copied test images:", len(dst_jpgs), "from:", src_test_dir)



## === cell 2
from tensorflow.keras.preprocessing.image import ImageDataGenerator

test_datagen = ImageDataGenerator()  # keep original semantics (no rescale)

test_generator = test_datagen.flow_from_directory(
    "/kaggle/tmp/test_dataset",
    class_mode=None,
    target_size=(380, 380),
    shuffle=False,
    batch_size=128,
)

sub_path = BASE_INPUT / "sample_submission.csv"
sub = pd.read_csv(sub_path)
print("sample_submission rows:", len(sub))
print("generator files:", len(test_generator.filenames))



## === cell 3
x = model.predict(test_generator, verbose=1)
print("Pred shape:", x.shape)
print("Pred min/max:", float(np.min(x)), float(np.max(x)))



## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3095792972.py in <cell line: 0>()
      1 # Predict
----> 2 x = model.predict(test_generator, verbose=1)
      3 print("Pred shape:", x.shape)
      4 print("Pred min/max:", float(np.min(x)), float(np.max(x)))
      5 

NameError: name 'model' is not defined

## === cell 4
labels = ["complex", "frog_eye_leaf_spot", "powdery_mildew", "rust", "scab"]
threshold = 0.4

z = (x > threshold).astype(np.int32)
print("Binarized shape:", z.shape)



## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1495134795.py in <cell line: 0>()
      3 
      4 # Fix: robust binarization (keeps original decision rule: x > threshold)
----> 5 z = (x > threshold).astype(np.int32)
      6 print("Binarized shape:", z.shape)
      7 

NameError: name 'x' is not defined

## === cell 5
predictions = [[labels[i] for i, v in enumerate(row) if v != 0] for row in z]
predictions_str = [" ".join(p) if len(p) else "healthy" for p in predictions]

pred_filenames = [Path(f).name for f in test_generator.filenames]
pred_map = dict(zip(pred_filenames, predictions_str))

sub["labels"] = sub["image"].map(pred_map).fillna("healthy")

print(sub.head())
print("Submission rows:", len(sub), "Pred rows:", len(predictions_str))

sub = sub[["image", "labels"]]
sub["image"] = sub["image"].astype(str)
sub["labels"] = sub["labels"].astype(str)



## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3656254749.py in <cell line: 0>()
      1 # Build predictions strings in the required "space-delimited list" format
----> 2 predictions = [[labels[i] for i, v in enumerate(row) if v != 0] for row in z]
      3 predictions_str = [" ".join(p) if len(p) else "healthy" for p in predictions]
      4 
      5 # Fix: ensure submission ordering/filenames match Kaggle expectations.

NameError: name 'z' is not defined

## === cell 6
out_file = "submission.csv"
sub.to_csv(out_file, index=False)
print("Wrote:", out_file, "Size(bytes):", os.path.getsize(out_file))
print("Saved columns:", list(sub.columns))
print("Unique label strings (sample):", sub["labels"].value_counts().head())
