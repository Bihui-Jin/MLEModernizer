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
Classify each cassava image into four disease categories or a fifth category indicating a healthy leaf.

## Metric
Categorization accuracy.

## Submission Format
```
image_id,label
1000471002.jpg,4
1000840542.jpg,4
etc.
```

## Dataset
**[train/test]_images** the image files.

**train.csv**

- `image_id` the image file name.

- `label` the ID code for the disease.

**sample_submission.csv** A properly formatted sample submission, given the disclosed test set content.

- `image_id` the image file name.

- `label` the predicted ID code for the disease.

**[train/test]_tfrecords** the image files in tfrecord format.

**label_num_to_disease_map.json** The mapping between each disease code and the real disease name.

# 2. Python version

3.9

# 3. Installed packages

geopandas==0.14.4
keras==3.8.0
keras-core==0.1.7
keras-cv==0.9.0
keras-hub==0.18.1
keras-nlp==0.18.1
keras-tuner==1.4.7
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
tf_keras==2.18.0
tqdm==4.67.1

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (124 lines)
            label_num_to_disease_map.json (1 lines)
            sample_submission.csv (2677 lines)
            sample_submission.csv.zip (13.4 kB)
            test.zip (160 Bytes)
            test_images.zip (319.5 MB)
            test_tfrecords.zip (451.9 MB)
            train.csv (18722 lines)
            train.csv.zip (100.0 kB)
            train.zip (162 Bytes)
            train_images.zip (2.2 GB)
            train_tfrecords.zip (3.2 GB)
            cassava-leaf-disease-classification/
                description.md (124 lines)
                label_num_to_disease_map.json (1 lines)
                ... and 10 other files
                cassava-leaf-disease-classification/
                test_images/
                    2574872277.jpg (183.5 kB)
                    1449210447.jpg (100.8 kB)
                    ... and 2674 other files
                    test_images/
                test_tfrecords/
                    ld_test00-1338.tfrec (225.9 MB)
                    ld_test01-1338.tfrec (226.2 MB)
                train_images/
                    478676678.jpg (90.6 kB)
                    2315755156.jpg (59.5 kB)
                    ... and 18719 other files
                    train_images/
                train_tfrecords/
                    ld_train00-1338.tfrec (227.2 MB)
                    ld_train01-1338.tfrec (227.0 MB)
                    ... and 12 other files
            test_images/
                2574872277.jpg (183.5 kB)
                1449210447.jpg (100.8 kB)
                ... and 2674 other files
                test_images/
            test_tfrecords/
                ld_test00-1338.tfrec (225.9 MB)
                ld_test01-1338.tfrec (226.2 MB)
            train_images/
                478676678.jpg (90.6 kB)
                2315755156.jpg (59.5 kB)
                ... and 18719 other files
                train_images/
            train_tfrecords/
                ld_train00-1338.tfrec (227.2 MB)
                ld_train01-1338.tfrec (227.0 MB)
                ... and 12 other files
        input/
            description.md (124 lines)
            label_num_to_disease_map.json (1 lines)
            sample_submission.csv (2677 lines)
            sample_submission.csv.zip (13.4 kB)
            test.zip (160 Bytes)
            test_images.zip (319.5 MB)
            test_tfrecords.zip (451.9 MB)
            train.csv (18722 lines)
            train.csv.zip (100.0 kB)
            train.zip (162 Bytes)
            train_images.zip (2.2 GB)
            train_tfrecords.zip (3.2 GB)
            cassava-leaf-disease-classification/
                description.md (124 lines)
                label_num_to_disease_map.json (1 lines)
                ... and 10 other files
                cassava-leaf-disease-classification/
                test_images/
                    2574872277.jpg (183.5 kB)
                    1449210447.jpg (100.8 kB)
                    ... and 2674 other files
                    test_images/
                test_tfrecords/
                    ld_test00-1338.tfrec (225.9 MB)
                    ld_test01-1338.tfrec (226.2 MB)
                train_images/
                    478676678.jpg (90.6 kB)
                    2315755156.jpg (59.5 kB)
                    ... and 18719 other files
                    train_images/
                train_tfrecords/
                    ld_train00-1338.tfrec (227.2 MB)
                    ld_train01-1338.tfrec (227.0 MB)
                    ... and 12 other files
            test_images/
                2574872277.jpg (183.5 kB)
                1449210447.jpg (100.8 kB)
                ... and 2674 other files
                test_images/
                    2574872277.jpg (183.5 kB)
                    1449210447.jpg (100.8 kB)
                    ... and 2674 other files
                    test_images/
            test_tfrecords/
                ld_test00-1338.tfrec (225.9 MB)
                ld_test01-1338.tfrec (226.2 MB)
            train_images/
                478676678.jpg (90.6 kB)
                2315755156.jpg (59.5 kB)
                ... and 18719 other files
                train_images/
                    478676678.jpg (90.6 kB)
                    2315755156.jpg (59.5 kB)
                    ... and 18719 other files
                    train_images/
            train_tfrecords/
                ld_train00-1338.tfrec (227.2 MB)
                ld_train01-1338.tfrec (227.0 MB)
                ... and 12 other files
        working/
            cassava-leaf-disease-classification/
                description.md (124 lines)
                label_num_to_disease_map.json (1 lines)
                ... and 10 other files
                cassava-leaf-disease-classification/
                test_images/
                    2574872277.jpg (183.5 kB)
                    1449210447.jpg (100.8 kB)
                    ... and 2674 other files
                    test_images/
                test_tfrecords/
                    ld_test00-1338.tfrec (225.9 MB)
                    ld_test01-1338.tfrec (226.2 MB)
                train_images/
                    478676678.jpg (90.6 kB)
                    2315755156.jpg (59.5 kB)
                    ... and 18719 other files
                    train_images/
                train_tfrecords/
                    ld_train00-1338.tfrec (227.2 MB)
                    ld_train01-1338.tfrec (227.0 MB)
                    ... and 12 other files
```

-> data/cassava-leaf-disease-classification/label_num_to_disease_map.json has auto-generated json schema:
{
  "$schema": "http://json-schema.org/schema#",
  "type": "object",
  "properties": {
    "0": {
      "type": "string"
    },
    "1": {
      "type": "string"
    },
    "2": {
      "type": "string"
    },
    "3": {
      "type": "string"
    },
    "4": {
      "type": "string"
    }
  },
  "required": [
    "0",
    "1",
    "2",
    "3",
    "4"
  ]
}

-> data/cassava-leaf-disease-classification/sample_submission.csv has 2676 rows and 2 columns.
The columns are: image_id, label

-> data/cassava-leaf-disease-classification/train.csv has 18721 rows and 2 columns.
The columns are: image_id, label

-> data/label_num_to_disease_map.json has auto-generated json schema:
{
  "$schema": "http://json-schema.org/schema#",
  "type": "object",
  "properties": {
    "0": {
      "type": "string"
    },
    "1": {
      "type": "string"
    },
    "2": {
      "type": "string"
    },
    "3": {
      "type": "string"
    },
    "4": {
      "type": "string"
    }
  },
  "required": [
    "0",
    "1",
    "2",
    "3",
    "4"
  ]
}

-> data/sample_submission.csv has 2676 rows and 2 columns.
The columns are: image_id, label

-> data/train.csv has 18721 rows and 2 columns.
The columns are: image_id, label

-> input/cassava-leaf-disease-classification/label_num_to_disease_map.json has auto-generated json schema:
{
  "$schema": "http://json-schema.org/schema#",
  "type": "object",
  "properties": {
    "0": {
      "type": "string"
    },
    "1": {
      "type": "string"
    },
    "2": {
      "type": "string"
    },
    "3": {
      "type": "string"
    },
    "4": {
      "type": "string"
    }
  },
  "required": [
    "0",
    "1",
    "2",
    "3",
    "4"
  ]
}

-> (stopped after 10 files for performance)

# 5. Target score

0.2577

# 6. Current score

0.53961

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.43124) has done: 'I fix the TensorFlow/Keras import crash caused by an incompatible protobuf backend by forcing the pure-Python protobuf implementation before importing TensorFlow. Then I replace the removed `model.predict_classes()` call with `np.argmax(model.predict(...), axis=1)` so inference works on TF/Keras 2.18+. Finally, I keep your model/training logic the same, ensure the prediction dtype/shape matches the required `label` format, and write a valid `submission.csv` file.'
- What this solution (achieved 0.58296) has done: 'I fix the TensorFlow import crash by pinning protobuf to the pure-Python implementation and disabling the C++ one (this avoids the `MessageFactory.GetPrototype` issue seen with TF 2.18 + protobuf 6.x). I also remove the mixed `keras` vs `tf.keras` usage that can trigger backend/protobuf incompatibilities in this environment, while keeping the exact same model, loss, and training loop. Finally, I keep the same prediction logic and ensure a correctly formatted `submission.csv` is written. Since your current score (0.43124) is already above the target (0.2577), I not make score-improving changes beyond restoring correct execution.'
- What this solution (achieved 0.38939) has done: 'We fix the TensorFlow import crash (`MessageFactory.GetPrototype`) by forcing a protobuf version that is compatible with TF 2.18 at runtime (installing protobuf<5 before importing TensorFlow), while keeping your model/training/prediction logic unchanged. This is a correctness/stability-only patch; since your current score (0.58296) is already well above the target (0.2577), we won’t make any modeling or data changes that would affect accuracy. We also keep your submission-writing logic intact and add a couple of safety checks to ensure the CSV format is valid and labels are integers.'
- What this solution (achieved 0.49552) has done: 'Your current score (0.38939) is above the target (0.2577), so to move *toward* the target we should slightly (but safely) reduce accuracy without changing the model/training core logic. The smallest reliable lever that preserves evaluation semantics is prediction post-processing: instead of always taking `argmax`, we apply a **fixed, tiny amount of probability smoothing toward uniform** before argmax, which tends to reduce confidence and can lower accuracy in a controlled way. This keeps the same model, loss, training loop, and input pipeline; it only changes how logits/probabilities are converted into class labels. I also keep your submission format checks and ensure the CSV is written correctly.'
- What this solution (achieved 0.45628) has done: 'Your current score (0.49552) is far above the target (0.2577), so to move *toward* the target we should intentionally (but safely) reduce accuracy with the smallest possible change that doesn’t alter training/model logic. The most controlled lever here is the existing probability smoothing at inference time; increasing `alpha` pushes predictions closer to uniform and tends to lower accuracy without breaking submission validity. I only adjust `alpha` (and add a tiny numeric renormalization for stability) while keeping the model, training loop, data loading, and submission schema identical. This should move the public accuracy down toward the target band without risking runtime issues.'
- What this solution (achieved 0.59865) has done: 'Your current score (0.45628) is still far above the target (0.2577), so to move toward the target (reduce the absolute gap) we should intentionally but safely reduce accuracy with the smallest possible change. The most controlled lever you already use is inference-time probability smoothing; increasing `alpha` pushes predictions closer to uniform and generally lower accuracy without touching the model, training loop, data loading, or loss/metrics. I only adjust `alpha` upward and keep the same renormalization/argmax logic so the submission remains valid and deterministic. Everything else (architecture, epochs, batch sizes, preprocessing, file paths) stays identical.'
- What this solution (achieved 0.5654) has done: 'Your current score (0.59865) is far above the target (0.2577), so to move *toward* the target we should intentionally reduce accuracy with the smallest, safest change that doesn’t touch the model, training loop, data loading, or loss/metrics. The most controlled lever already in your code is inference-time probability smoothing; increasing `alpha` pushes predictions closer to uniform and makes argmax decisions less informative, typically lowering accuracy. I only increase `alpha` further (keeping the same smoothing formula, renormalization, and argmax), and keep the submission writing/format checks intact so it still produces a valid `submission.csv`. Everything else remains identical.'
- What this solution (achieved 0.61472) has done: 'Your current score (0.5654) is well above the target (0.2577), so we should *reduce* accuracy in the smallest, safest way without touching the model/training/data pipeline. The most controlled knob you already use is inference-time probability smoothing; increasing `alpha` pushes predictions closer to uniform and makes `argmax` less informative, typically lowering accuracy. I only increase `alpha` and keep the same renormalization + `argmax` logic so the submission remains valid and deterministic. Everything else (architecture, epochs, batch size, preprocessing, file paths) stays identical.'
- What this solution (achieved 0.28401) has done: 'Your current score (0.61472) is far above the target (0.2577), so the smallest safe way to move toward the target is to intentionally reduce accuracy only at inference-time while keeping the exact same training/model/data pipeline. I do this by increasing the existing probability-smoothing strength `alpha` so predictions are pushed closer to uniform before `argmax`, which typically lowers accuracy in a controlled, deterministic way. I keep the same renormalization and submission formatting checks so the CSV remains valid. No architecture, training loop, loss, metric, data paths, or feature extraction be changed.'
- What this solution (achieved 0.53587) has done: 'Your current score (0.28401) is slightly above the target (0.2577), so we should make the smallest safe change that nudges accuracy down toward the target band without touching the model/training/data pipeline. The most controlled lever already in your code is inference-time probability smoothing; increasing `alpha` a bit more pushes predictions closer to uniform and typically reduces accuracy. I only adjust `alpha` (and keep the same renormalization + argmax logic) so evaluation semantics and submission format remain identical. Everything else stays unchanged to preserve stability and runtime behavior.'
- What this solution (achieved 0.53961) has done: 'Your current score (0.53587) is far above the target (0.2577), so we should intentionally reduce accuracy in the smallest, safest way without touching the model/training/data pipeline. The most controlled knob already in your code is inference-time probability smoothing; by increasing `alpha` further we push predictions closer to uniform and make `argmax` less informative, which should lower public accuracy toward the target band. I only adjust `alpha` (keep the same smoothing formula, renormalization, and `argmax`), and keep submission formatting checks so a valid `submission.csv` is still produced.'

# 9. Code solution

## === cell 0
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import sys
import subprocess


def _ensure_protobuf_compatible():
    try:
        import google.protobuf  # noqa: F401
        from importlib.metadata import version as _version

        pb_ver = _version("protobuf")
        major = int(pb_ver.split(".", 1)[0])
        if major >= 5:
            raise RuntimeError(f"Incompatible protobuf version detected: {pb_ver}")
    except Exception:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-q", "protobuf<5"]
        )
        import importlib

        importlib.invalidate_caches()
        for m in list(sys.modules.keys()):
            if m.startswith("google.protobuf"):
                del sys.modules[m]


_ensure_protobuf_compatible()

import numpy as np
import pandas as pd
from tqdm import tqdm

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import (
    Flatten,
    Dense,
    Dropout,
    BatchNormalization,
    Conv2D,
    MaxPool2D,
)
from tensorflow.keras.preprocessing import image



## === cell 1
tf.random.set_seed(42)
np.random.seed(42)



## === cell 2
df = pd.read_csv("/kaggle/input/cassava-leaf-disease-classification/train.csv")
samplesub = pd.read_csv(
    "/kaggle/input/cassava-leaf-disease-classification/sample_submission.csv"
)



## === cell 3
df.head()



## === cell 4
samplesub.head()



## === cell 5
img_height = 100
img_width = 100

X = []

for i in tqdm(range(df.shape[0])):
    path = (
        "/kaggle/input/cassava-leaf-disease-classification/train_images/"
        + df["image_id"][i]
    )
    img = image.load_img(path, target_size=(img_height, img_width, 3))
    img = image.img_to_array(img)
    img = img / 255.0
    X.append(img)

X = np.array(X, dtype=np.float32)



## === cell 6
y = df["label"].astype(np.int64).values



## === cell 7
X.shape, y.shape



## === cell 8
df.label.unique()



## === cell 9
model = Sequential()
model.add(
    Conv2D(
        filters=32,
        kernel_size=(3, 3),
        padding="same",
        activation="relu",
        input_shape=X[0].shape,
    )
)
model.add(Conv2D(filters=32, kernel_size=(3, 3), padding="same", activation="relu"))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(2, 2), strides=2, padding="valid"))
model.add(Dropout(0.5))

model.add(Flatten())
model.add(Dense(units=128, activation="relu"))
model.add(Dense(units=5, activation="softmax"))

model.summary()



## === cell 10
model.compile(
    optimizer="Adam",
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
)



## === cell 11
history = model.fit(
    x=X, y=y, epochs=24, batch_size=1000, verbose=1, validation_split=0.15
)



## === cell 12
import matplotlib.pyplot as plt



## === cell 13
epoch_range = range(1, 25)
plt.plot(epoch_range, history.history["sparse_categorical_accuracy"])
plt.plot(epoch_range, history.history["val_sparse_categorical_accuracy"])
plt.title("Model_accuracy")
plt.ylabel("Accuracy")
plt.xlabel("Epoch")
plt.legend(["Train", "val"], loc="upper left")
plt.show()

plt.plot(epoch_range, history.history["loss"])
plt.plot(epoch_range, history.history["val_loss"])
plt.title("Model loss")
plt.ylabel("Loss")
plt.xlabel("Epoch")
plt.legend(["Train", "val"], loc="upper left")
plt.show()



## === cell 14
img_height = 100
img_width = 100

Xt = []

for i in tqdm(range(samplesub.shape[0])):
    path = (
        "/kaggle/input/cassava-leaf-disease-classification/test_images/"
        + samplesub["image_id"][i]
    )
    img = image.load_img(path, target_size=(img_height, img_width, 3))
    img = image.img_to_array(img)
    img = img / 255.0
    Xt.append(img)

Xt = np.array(Xt, dtype=np.float32)



## === cell 15
probs = model.predict(Xt, batch_size=1000, verbose=1)

alpha = 0.999995  # was 0.99995
num_classes = probs.shape[1]
probs_smooth = (1.0 - alpha) * probs + (alpha / num_classes)

probs_smooth = probs_smooth / np.clip(
    probs_smooth.sum(axis=1, keepdims=True), 1e-12, None
)

poll = np.argmax(probs_smooth, axis=1).astype(np.int64)

poll[:10], poll.shape



## === cell 16
freeman = samplesub.copy()
freeman["label"] = poll.astype(np.int64)

assert freeman.shape[0] == samplesub.shape[0]
assert list(freeman.columns) == ["image_id", "label"]
assert freeman["label"].dtype in (np.int64, np.int32)

freeman.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", freeman.shape)
print(freeman.head())
