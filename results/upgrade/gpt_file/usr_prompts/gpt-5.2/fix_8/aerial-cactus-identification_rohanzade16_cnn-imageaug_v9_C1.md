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
Create a classifier to predict whether an image contains a cactus.

## Metric
Area under the ROC curve.

## Submission Format
For each ID in the test set, you must predict a probability for the `has_cactus` variable. The file should contain a header and have the following format:

```
id,has_cactus
000940378805c44108d287872b2f04ce.jpg,0.5
0017242f54ececa4512b4d7937d1e21e.jpg,0.5
001ee6d8564003107853118ab87df407.jpg,0.5
etc.
```

## Dataset
This dataset contains a large number of 32 x 32 thumbnail images containing aerial photos of a cactus. The file name of an image corresponds to its `id`.

- **train/** - the training set images
- **test/** - the test set images (you must predict the labels of these)
- **train.csv** - the training set labels, indicates whether the image has a cactus (`has_cactus = 1`)
- **sample_submission.csv** - a sample submission file in the correct format

# 2. Python version

3.11

# 3. Installed packages

geopandas==0.14.4
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
numpy==1.26.4
opencv-python==4.12.0.88
opencv-python-headless==4.12.0.88
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
protobuf==6.33.0
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
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
            description.md (56 lines)
            sample_submission.csv (3326 lines)
            sample_submission.csv.zip (67.3 kB)
            test.zip (3.5 MB)
            train.csv (14176 lines)
            train.csv.zip (285.6 kB)
            train.zip (15.0 MB)
            aerial-cactus-identification/
                description.md (56 lines)
                sample_submission.csv (3326 lines)
                ... and 5 other files
                aerial-cactus-identification/
                test/
                    76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                    f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                    ... and 3323 other files
                    test/
                train/
                    775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                    65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                    ... and 14173 other files
                    train/
            test/
                76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                ... and 3323 other files
                test/
            train/
                775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                ... and 14173 other files
                train/
        input/
            description.md (56 lines)
            sample_submission.csv (3326 lines)
            sample_submission.csv.zip (67.3 kB)
            test.zip (3.5 MB)
            train.csv (14176 lines)
            train.csv.zip (285.6 kB)
            train.zip (15.0 MB)
            aerial-cactus-identification/
                description.md (56 lines)
                sample_submission.csv (3326 lines)
                ... and 5 other files
                aerial-cactus-identification/
                test/
                    76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                    f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                    ... and 3323 other files
                    test/
                train/
                    775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                    65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                    ... and 14173 other files
                    train/
            test/
                76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                ... and 3323 other files
                test/
                    76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                    f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                    ... and 3323 other files
                    test/
            train/
                775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                ... and 14173 other files
                train/
                    775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                    65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                    ... and 14173 other files
                    train/
        working/
            aerial-cactus-identification/
                description.md (56 lines)
                sample_submission.csv (3326 lines)
                ... and 5 other files
                aerial-cactus-identification/
                test/
                    76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                    f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                    ... and 3323 other files
                    test/
                train/
                    775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                    65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                    ... and 14173 other files
                    train/
```

-> data/aerial-cactus-identification/sample_submission.csv has 3325 rows and 2 columns.
The columns are: id, has_cactus

-> data/aerial-cactus-identification/train.csv has 14175 rows and 2 columns.
The columns are: id, has_cactus

-> data/sample_submission.csv has 3325 rows and 2 columns.
The columns are: id, has_cactus

-> data/train.csv has 14175 rows and 2 columns.
The columns are: id, has_cactus

-> input/aerial-cactus-identification/sample_submission.csv has 3325 rows and 2 columns.
The columns are: id, has_cactus

-> input/aerial-cactus-identification/train.csv has 14175 rows and 2 columns.
The columns are: id, has_cactus

-> (stopped after 10 files for performance)

# 5. Target score

0.4994

# 6. Current score

0.93916

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.44244) has done: 'I fix the environment/runtime break at import time by pinning protobuf’s pure-Python implementation before TensorFlow loads, which avoids the `MessageFactory.GetPrototype` crash. Then I make Keras 3 checkpointing/loading compatible by saving to a `.keras` file and loading that exact file. Finally, I make unzip destinations deterministic (to `/kaggle/working/train` and `/kaggle/working/test`) and ensure the test generator preserves filename order so predictions align with `id`, producing a valid `submission.csv` with the required columns.'
- What this solution (achieved 0.44436) has done: 'I fix two blockers that prevent this notebook from running end-to-end: the protobuf/TensorFlow import crash and the `class_weight` mapping error caused by string labels with categorical outputs. I keep the same model architecture/training loop, but make labels numeric (0/1) and use `class_mode="binary"` with a single-unit sigmoid output so `class_weight` applies correctly and AUC is computed on the positive class probability. I also make checkpoint saving/loading robust by ensuring the best-model file is actually created before loading, and I keep deterministic test filename ordering so the submission aligns with `id`. These changes are minimal yet should improve AUC toward the 0.4994 target by making the learning objective and AUC metric consistent (binary label + sigmoid probability) instead of a 2-class softmax with string labels.'
- What this solution (achieved 0.94065) has done: 'I fix the TensorFlow/protobuf import crash by forcing the pure-Python protobuf implementation *before any protobuf/TensorFlow-related imports* and by ensuring the env var is set in a way that works reliably in this Kaggle image. Then I fix the `flow_from_dataframe(..., class_mode="binary")` error by converting the label column to the string format that Keras’ legacy generator requires, while keeping the model as a single-unit sigmoid binary classifier (same core approach). Finally, I ensure the training generators are successfully created so the fit call runs, and keep deterministic test file ordering so the written `submission.csv` aligns with the required `id` order and format.'
- What this solution (achieved 0.9419) has done: 'I fix the import-time crash by setting the protobuf implementation env vars before any TensorFlow/protobuf-related imports, and (as a fallback) forcing a compatible protobuf version if the environment still triggers the `MessageFactory.GetPrototype` issue. These changes are purely to unblock runtime and should be score-neutral because they do not alter the model/training logic. I also add a small guard to ensure the unzipped folders contain images (so the generators don’t silently train on 0 files) and keep the deterministic test filename ordering so `id` aligns with predictions in `submission.csv`. No model architecture, loss, or training approach is changed, so your current score behavior should remain essentially the same once it runs end-to-end.'
- What this solution (achieved 0.94025) has done: 'I fix the TensorFlow/protobuf import crash by broadening the exception handling to catch the actual failure you’re seeing and then applying the same “install a compatible protobuf and retry” fallback. Since your current AUC (0.9419) is far above the target (0.4994) and higher-is-better, I not change the model/training logic that affects score; the goal is correctness and stability only. I also make the unzip step idempotent (avoid nested `train/train` or `test/test` folders) so generators always see images. Finally, I keep deterministic test file ordering and ensure the submission is written as `/kaggle/working/submission.csv` with the required columns.'
- What this solution (achieved 0.9419) has done: 'You’re failing immediately at TensorFlow import due to a protobuf compatibility crash, so I make the import guard truly robust by (1) forcing the pure-Python protobuf implementation before any TF/protobuf import and (2) automatically installing a protobuf version that is compatible with TF 2.18 in this environment (protobuf 5.x), then retrying the import after clearing loaded modules. This is a runtime-only fix and is score-neutral: it does not change the model, training loop, data split, or prediction post-processing. I keep all paths, generators, model architecture, epochs, and submission formatting exactly the same so you continue to get a valid `/kaggle/working/submission.csv`. Because your current score is far above the (low) target and higher-is-better, I won’t make any score-improving changes beyond unblocking execution.'
- What this solution (achieved 0.93916) has done: 'I fix the TensorFlow/protobuf import crash by making the protobuf fallback run *before* TensorFlow is ever imported: proactively installing a TF-2.18-compatible protobuf (5.28.3) and forcing the pure-Python implementation, which avoids the `MessageFactory.GetPrototype` failure. This is a runtime-only change and is score-neutral (no changes to model, training loop, split, or prediction logic). I also keep the unzip/generator logic intact and ensure the pipeline always reaches the CSV write step at `/kaggle/working/submission.csv` with the required `id,has_cactus` columns. Since your current score (0.9419) is far above the target (0.4994) and higher is better, I not make any score-improving changes.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "2")

import sys
import subprocess


def _safe_import_tensorflow():
    """
    Robustly handle protobuf/TensorFlow incompatibility that can crash at import time.

    Fix: Proactively install a TF 2.18-compatible protobuf (5.28.3) before importing TF.
    This avoids the observed crash: AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'
    """
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-q", "protobuf==5.28.3"]
    )

    try:
        import tensorflow as tf  # noqa: F401

        return tf
    except Exception as e:
        msg = str(e)
        protobuf_related = (
            ("GetPrototype" in msg)
            or ("MessageFactory" in msg)
            or ("google.protobuf" in msg)
            or ("protobuf" in msg.lower())
        )
        if not protobuf_related:
            raise

        import importlib

        importlib.invalidate_caches()

        for m in list(sys.modules.keys()):
            if m.startswith("google.protobuf") or m.startswith("tensorflow"):
                sys.modules.pop(m, None)

        import tensorflow as tf  # noqa: F401

        return tf


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
import shutil
import zipfile
from pathlib import Path

tf = _safe_import_tensorflow()
print("TF version:", tf.__version__)



## === cell 1
data = pd.read_csv("/kaggle/input/aerial-cactus-identification/train.csv")
data.sample(5)



## === cell 2
data = data.astype({"id": str})
data["has_cactus"] = data["has_cactus"].astype(np.int32).astype(str)



## === cell 3
data.sample(5)




## === cell 4
def unzip_to(zip_path: str, dst_dir: str):
    """
    Fix: Make unzip idempotent and avoid nested folders like /kaggle/working/train/train
    by clearing the destination directory before extraction.
    """
    dst = Path(dst_dir)
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dst)


unzip_to(
    "/kaggle/input/aerial-cactus-identification/train.zip", "/kaggle/working/train"
)
unzip_to("/kaggle/input/aerial-cactus-identification/test.zip", "/kaggle/working/test")

print("Train dir exists:", Path("/kaggle/working/train").exists())
print("Test dir exists:", Path("/kaggle/working/test").exists())

n_train_imgs = len(list(Path("/kaggle/working/train").glob("*.jpg")))
n_test_imgs = len(list(Path("/kaggle/working/test").glob("*.jpg")))
print("Train images:", n_train_imgs, "Test images:", n_test_imgs)
if n_train_imgs == 0 or n_test_imgs == 0:
    raise RuntimeError("Unzipped image folders appear empty; cannot proceed.")



## === cell 5
idg = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1 / 255.0, validation_split=0.1
)



## === cell 6
train_idg = idg.flow_from_dataframe(
    data,
    "/kaggle/working/train",
    x_col="id",
    y_col="has_cactus",
    target_size=(32, 32),
    batch_size=64,
    subset="training",
    seed=42,
    class_mode="binary",
    shuffle=True,
)



## === cell 7
val_idg = idg.flow_from_dataframe(
    data,
    "/kaggle/working/train",
    x_col="id",
    y_col="has_cactus",
    target_size=(32, 32),
    batch_size=64,
    subset="validation",
    seed=42,
    class_mode="binary",
    shuffle=False,
)



## === cell 8
pass



## === cell 9
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Input((32, 32, 3), name="InputLayer"))
model.add(tf.keras.layers.Flatten(name="Flat"))
model.add(tf.keras.layers.Dropout(0.25, name="Drop1"))
model.add(tf.keras.layers.Dense(512, activation="relu", name="D1"))
model.add(tf.keras.layers.Dense(128, activation="relu", name="D2"))
model.add(tf.keras.layers.Dense(1, activation="sigmoid", name="Output"))
model.summary()



## === cell 10
model.compile(
    optimizer=tf.keras.optimizers.SGD(),
    loss=tf.keras.losses.binary_crossentropy,
    metrics=[tf.keras.metrics.AUC(curve="ROC", name="AUC"), "acc"],
)



## === cell 11
from sklearn.utils import class_weight

y_num = data["has_cactus"].astype(np.int32).values
classes = np.array([0, 1], dtype=np.int32)
cw = class_weight.compute_class_weight(
    class_weight="balanced", classes=classes, y=y_num
)
class_weights = {int(c): float(w) for c, w in zip(classes, cw)}



## === cell 12
class_weights



## === cell 13
ckpt_path = "/kaggle/working/BestModelAsPerValAUC.keras"
model_ckpt = tf.keras.callbacks.ModelCheckpoint(
    ckpt_path,
    monitor="val_AUC",
    save_best_only=True,
    mode="max",
    verbose=1,
)



## === cell 14
history = model.fit(
    train_idg,
    epochs=15,
    validation_data=val_idg,
    class_weight=class_weights,
    callbacks=[model_ckpt],
)



## === cell 15
if Path(ckpt_path).exists():
    model = tf.keras.models.load_model(ckpt_path)
else:
    print("Warning: checkpoint not found; using the last trained model:", ckpt_path)



## === cell 16
test_dir = "/kaggle/working/test"
test_files = sorted([f for f in os.listdir(test_dir) if f.lower().endswith(".jpg")])
test_result = pd.DataFrame(test_files, columns=["id"])
test_result.head()



## === cell 17
test_idg = idg.flow_from_dataframe(
    test_result,
    test_dir,
    batch_size=64,
    x_col="id",
    target_size=(32, 32),
    class_mode=None,
    shuffle=False,
)



## === cell 18
test_pred = model.predict(test_idg, verbose=1)



## === cell 19
test_pred.shape



## === cell 20
type(test_pred)



## === cell 21
test_pred[:5]



## === cell 22
test_pred[:, 0].shape



## === cell 23
test_result["has_cactus"] = test_pred[:, 0].astype(np.float32)



## === cell 24
test_result.sample(5)



## === cell 25
pass



## === cell 26
sub_path = "/kaggle/working/submission.csv"
test_result.to_csv(sub_path, index=False)
print("Wrote:", sub_path, "rows:", len(test_result))



## === cell 27
df = pd.read_csv(sub_path)
df.head()
