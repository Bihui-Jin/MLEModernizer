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

3.7

# 3. Installed packages

geopandas==0.14.4
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
pillow==11.3.0
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
tqdm==4.67.1

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

0.5

# 6. Current score

0.997

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.99733) has done: 'Diagnosis: The crash happens before any model code completes because TensorFlow imports protobuf internals that are incompatible with the installed `protobuf==6.33.0`, triggering `AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'` during TensorFlow initialization. This is a known protobuf/TensorFlow version mismatch issue and is not related to your model/training logic. The minimal safe fix is to force protobuf to use the pure-Python implementation via environment variables **before** importing TensorFlow (and ideally before importing any protobuf users). This keeps the rest of the notebook logic unchanged and unblocks execution.

Patch summary: In cell 0, set `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` (and a compatible version flag) at the very top, before importing TensorFlow. No model/training/inference logic is changed; only the runtime configuration is adjusted to avoid the protobuf API mismatch crash.

Updated cells: (cell 0 only)

Compatibility notes for cell k+1: Variables `test_df` and `answers` are still produced with the same types/structures as before, so cell 1 run unchanged.

Assumptions: TensorFlow 2.18.0 is installed and functional when protobuf uses the Python implementation; performance impact is acceptable for this notebook and does not alter evaluation semantics.'
- What this solution (achieved 0.99657) has done: 'Diagnosis: The crash happens before any model work completes, and the traceback (`AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'`) is a known protobuf runtime incompatibility triggered during TensorFlow import/initialization. This environment has `protobuf==6.33.0`, while TF 2.18 expects protobuf 4.x/5.x behavior; forcing `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` does not fix this API mismatch. The minimal fix is to use the upb-backed implementation (default) and avoid pinning the python implementation via env vars.

Patch summary: In cell 0, remove/override the two protobuf environment-variable settings so TensorFlow can initialize with the compatible protobuf backend. No model/data logic is changed; only the env var setup that directly causes the crash is adjusted.

Updated cells: cell 0 only (buggy cell).

Compatibility notes for cell k+1: Variables `test_df` and `answers` are still produced exactly as before, so cell 1 remains unchanged and run.

Assumptions: The runtime supports the default protobuf backend (upb), and no other code relies on the protobuf-python implementation being forced.'
- What this solution (achieved 0.98437) has done: 'Diagnosis: The crash happens before any model code runs because TensorFlow (via protobuf) is hitting an API mismatch: `MessageFactory` in the installed `protobuf==6.33.0` no longer provides `GetPrototype`, which older generated protobuf code inside TF 2.18 can still call. Your cell currently *removes* protobuf-related environment overrides, so TF ends up using the incompatible fast C++ protobuf implementation and crashes on import/initialization.  
Patch summary: In cell 0, set `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` (and version=3) **before importing TensorFlow** so TF uses the pure-Python protobuf runtime that still supports the expected API, avoiding the `GetPrototype` AttributeError. This is the minimal, localized change and does not alter the ML logic/training semantics.  
Updated cells: Only cell 0 is changed, with a short comment explaining why.  
Compatibility notes for cell k+1: Variables `test_df` and `answers` are still produced identically, so cell 1 remains compatible and write the submission as before.  
Assumptions: Environment honors `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION` at TF import time (standard behavior) and using the Python protobuf implementation is acceptable for this notebook’s runtime.'
- What this solution (achieved 0.99731) has done: 'Diagnosis: The crash happens before any training/inference output because TensorFlow imports protobuf internals and hits an API mismatch: `MessageFactory.GetPrototype` was removed/changed in newer `protobuf` (you have protobuf==6.33.0), while TensorFlow 2.18 expects an older protobuf API. The two environment variables set at the top don’t resolve this incompatibility in this environment, so importing/using TensorFlow triggers the AttributeError.  
Patch summary: In cell 0, enforce a protobuf/TensorFlow compatibility workaround by setting `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` *and* importing `google.protobuf` early, plus pinning protobuf’s pure-Python backend before TensorFlow is imported. This keeps the model/training logic unchanged and simply prevents the protobuf factory call path that triggers the error.  
Updated cells: Only cell 0 is changed, and only at the very top around the environment/protobuf initialization; all model/data logic remains identical.  
Compatibility notes for cell k+1: Variables `test_df` and `answers` are still produced with the same types/shapes, so cell 1 continues to work unchanged.  
Assumptions: `google.protobuf` is available (it is, because protobuf is installed) and using the pure-Python protobuf implementation is sufficient to avoid the removed `GetPrototype` call path in this runtime.'
- What this solution (achieved 0.99771) has done: 'Diagnosis: The crash happens during importing TensorFlow (or a dependency it loads) because your cell forces `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python`, which makes TensorFlow use the pure-Python protobuf runtime. With protobuf==6.33.0 this runtime no longer exposes `MessageFactory.GetPrototype`, triggering `AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'`.  
Patch summary: Remove the forced pure-Python protobuf implementation so TensorFlow can use the compatible C++/upb protobuf runtime. Keep everything else (model, training loop, data paths, outputs) unchanged.  
Updated cells: Only cell 0 is modified, and only at the environment-variable lines causing the incompatibility.  
Compatibility notes for cell k+1: Variables `test_df` and `answers` are still created exactly as before, so cell 1 run unchanged.  
Assumptions: The environment has a working protobuf binary runtime (default) compatible with TensorFlow 2.18.0, and we don’t need to force the Python protobuf implementation.'
- What this solution (achieved 0.99374) has done: 'Diagnosis: The crash happens before any training/inference logic because importing TensorFlow triggers a protobuf runtime incompatibility: the installed `protobuf==6.33.0` is too new for the TensorFlow 2.18 stack in this environment, leading to `AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'` during protobuf descriptor/message construction. The existing environment-variable workaround is incomplete for protobuf 6.x, so TensorFlow ends up using the incompatible C++ protobuf implementation. The minimal, deterministic fix is to force protobuf to use the pure-Python implementation (which remains compatible) by setting `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` before importing TensorFlow/protobuf-dependent libraries.

Patch summary: In cell 0 only, replace the `os.environ.pop(...)` lines with setting `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION` to `"python"` (and remove any version override) prior to importing `google.protobuf`/`tensorflow`. No model/training/inference logic is changed.

Updated cells: cell 0 only (buggy cell).

Compatibility notes for cell k+1: All variables created in cell 0 (`test_df`, `answers`) are unchanged in name/type/shape; cell 1 run as-is.

Assumptions: This environment allows selecting the pure-Python protobuf implementation via environment variables at runtime, and performance impact is acceptable for this notebook.'
- What this solution (achieved 0.99692) has done: 'Diagnosis: The crash happens at `import tensorflow as tf` because cell 0 forces `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION="cpp"`, which makes protobuf try to load the C++ extension (`google.protobuf.pyext._message`). In this environment that extension is not available/compatible (ImportError), so TensorFlow fails to import. TensorFlow 2.18.0 works with the pure-Python protobuf implementation here, so we should not force the C++ protobuf backend.  
Patch summary: In cell 0, remove the forced `"cpp"` protobuf setting and explicitly set `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION="python"` (and clear the version override) before importing TensorFlow/protobuf. This is the smallest change that unblocks TensorFlow import while leaving the rest of the training/inference logic unchanged.  
Updated cells: Only cell 0 is modified.  
Compatibility notes for cell k+1: `test_df` and `answers` are still created with the same types/contents, so cell 1 run unchanged.  
Assumptions: The environment’s TensorFlow build is compatible with protobuf’s Python implementation (common in many notebook/Kaggle-like runtimes).'
- What this solution (achieved 0.99483) has done: 'Diagnosis: The crash happens while importing TensorFlow because the notebook forces `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION="cpp"`, which makes `google.protobuf` try to load the C++/pyext backend (`google.protobuf.pyext._message`). In this environment that backend is not available/compatible, producing `ImportError: cannot import name '_message'`. TensorFlow works with the pure-Python protobuf runtime here, so we should not force the `cpp` implementation.  
Patch summary: In the failing cell, change the protobuf runtime selection to `"python"` (or leave it unset) before importing TensorFlow, ensuring protobuf doesn’t try to import the missing C++ extension. Keep everything else identical to preserve training/inference semantics.  
Updated cells: Only cell 0 is modified.  
Compatibility notes for cell k+1: `test_df` and `answers` are still produced with the same types/structure, so cell 1 run unchanged.  
Assumptions: The environment’s installed `protobuf==6.33.0` supports the pure-Python implementation and TensorFlow 2.18.0 can run with it (common in such environments).'
- What this solution (achieved 0.997) has done: 'Diagnosis: The crash happens during TensorFlow import/initialization because the runtime has `protobuf==6.33.0`, which is incompatible with the TensorFlow 2.18 stack for this notebook and triggers `AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'`. The existing environment-variable workaround forces the pure-Python protobuf implementation, but that does not fix the API mismatch introduced by protobuf 6. The minimal deterministic fix is to monkey-patch `google.protobuf.message_factory.MessageFactory.GetPrototype` to call the newer `GetMessageClass` when needed, before TensorFlow is imported. This keeps the rest of the notebook logic unchanged and unblocks execution.

Patch summary: In cell 0, add a small compatibility patch right after importing `google.protobuf` and before importing TensorFlow: if `MessageFactory.GetPrototype` is missing, define it in terms of `GetMessageClass` (or fall back to the internal symbol). No other logic, paths, model code, or training/inference behavior is changed.

Updated cells: Only cell 0 is modified as below.

Compatibility notes for cell k+1: All variables created in cell 0 (`test_df`, `answers`, etc.) remain identical in type/shape/meaning, so cell 1 continues to run unchanged.

Assumptions: `google.protobuf.message_factory.MessageFactory` exists (it does in protobuf 6.x), and either `GetMessageClass` or the internal `_InternalCreateMessageClass` is available to derive a prototype class.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", None)

import google.protobuf  # noqa: F401

from google.protobuf import message_factory as _message_factory

if not hasattr(_message_factory.MessageFactory, "GetPrototype"):

    def _GetPrototype(self, descriptor):
        if hasattr(self, "GetMessageClass"):
            return self.GetMessageClass(descriptor)
        return self._InternalCreateMessageClass(descriptor)

    _message_factory.MessageFactory.GetPrototype = _GetPrototype

from glob import glob
import shutil
from tqdm import tqdm

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_dir = "../input/train/train/"
test_dir = "../input/test/test"


input_shape = (32, 32, 3)
batch_size = 32
num_classes = 2
num_epochs = 1
data_augmentation = True

learning_rate = 0.001


def build_model(input_shape):
    inputs = layers.Input(input_shape)
    net = layers.Conv2D(32, (3, 3), padding="same")(inputs)
    net = layers.Activation("relu")(net)
    net = layers.Conv2D(32, (3, 3))(net)
    net = layers.Activation("relu")(net)
    net = layers.MaxPooling2D(pool_size=(2, 2))(net)
    net = layers.Dropout(0.25)(net)

    net = layers.Conv2D(64, (3, 3), padding="same")(net)
    net = layers.Activation("relu")(net)
    net = layers.Conv2D(64, (3, 3))(net)
    net = layers.Activation("relu")(net)
    net = layers.MaxPooling2D(pool_size=(2, 2))(net)
    net = layers.Dropout(0.25)(net)

    net = layers.Flatten()(net)
    net = layers.Dense(512)(net)
    net = layers.Activation("relu")(net)
    net = layers.Dropout(0.5)(net)
    net = layers.Dense(num_classes)(net)
    net = layers.Activation("softmax")(net)

    return tf.keras.Model(inputs=inputs, outputs=net)


model = build_model(input_shape)
model.compile(
    loss="categorical_crossentropy",
    optimizer=tf.keras.optimizers.Adam(learning_rate),
    metrics=["accuracy"],
)


df = pd.read_csv("../input/train.csv")
df.head()

datasets = []
for fname, cls in zip(df["id"], df["has_cactus"]):
    datasets.append((fname, cls))

np.random.shuffle(datasets)
train_paths = datasets[: int(len(datasets) * 0.8)]
test_paths = datasets[int(len(datasets) * 0.8) :]


def batch_dataset(batch_paths):
    batch_images = []
    batch_labels = []

    for fname, cls in batch_paths:
        img_path = os.path.join(train_dir, fname)
        image = np.array(Image.open(img_path))
        label = np.array(np.array([0, 1]) == cls).astype(np.uint8)
        batch_images.append(image)
        batch_labels.append(label)

    batch_images = np.array(batch_images)
    batch_labels = np.array(batch_labels)
    return batch_images, batch_labels


images, labels = batch_dataset(train_paths[:4])
images.shape, labels.shape

for epoch in range(num_epochs):
    np.random.shuffle(train_paths)
    batches_per_epoch = len(train_paths) // batch_size
    for step in tqdm(range(batches_per_epoch)):
        images, labels = batch_dataset(
            train_paths[step * batch_size : (step + batch_size) * batch_size]
        )
        model.fit(images, labels, batch_size=batch_size, verbose=0)


test_df = pd.read_csv("../input/sample_submission.csv")
test_df["id"][0]

answers = []
for fname in test_df["id"]:
    path = os.path.join(test_dir, fname)
    image = np.array(Image.open(path))
    image = np.expand_dims(image, 0)
    logit = model.predict(image)
    ans = logit[0, 1]
    answers.append(ans)


## === cell 1
submit_data = {'id': test_df['id'],
               'has_cactus': answers}

submit_df = pd.DataFrame(submit_data)
submit_df.to_csv('samplesubmission.csv', index=False)
