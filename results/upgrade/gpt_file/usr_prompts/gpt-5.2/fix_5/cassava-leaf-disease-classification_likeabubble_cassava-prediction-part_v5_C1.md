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

3.13

# 3. Installed packages

numpy==1.26.4
protobuf==6.33.0
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

0.8399818676337262

# 6. Current score

0.09268

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.15919) has done: 'I fix the protobuf/TensorFlow import crash by forcing the pure-Python protobuf implementation early, which avoids the `MessageFactory.GetPrototype` attribute error in this environment. Since your external pre-trained model path doesn’t exist, I replace only that failing dependency with a small built-in Keras application model (keeping the same “load model → predict → argmax” pipeline) so the notebook runs end-to-end. I also ensure the TFRecord parsing matches the Cassava dataset keys (`image`, `image_name`), keep deterministic ordering, and write a valid `submission.csv` with the exact required header/columns. These changes are aimed at correctness and producing a valid submission; score is expected to be reasonable but not guaranteed without the original checkpoint.'
- What this solution (achieved 0.09268) has done: 'I fix the TensorFlow/protobuf crash by setting the protobuf environment variables before TensorFlow is imported (your current cell ordering imports TF too early). Then I correct the Xception construction error by using the required 299×299 input size when `include_top=True` with ImageNet weights, and resize TFRecord images to match so inference runs. Finally, I keep your existing “predict → map 1000→5 → argmax → write submission.csv” core logic intact, ensuring `image_id` alignment with `sample_submission.csv` and that a valid `submission.csv` is always produced.'
- What this solution (achieved 0.09268) has done: 'We fix the protobuf/TensorFlow import crash by forcing the pure-Python protobuf implementation at process start (via `sitecustomize.py`) and by importing TensorFlow only after verifying those env vars are actually set, with a safe fallback that restarts the interpreter if needed. Then we keep your exact inference core logic (Xception → predict 1000 → map to 5 → argmax) but correct a major submission-alignment bug: the TFRecord names are `bytes` and must be decoded and normalized to match `sample_submission.csv` keys, otherwise almost everything becomes NaN and gets filled with a single class (causing the very low score). Finally, we make the TFRecord item counting robust (no reliance on filename regex) and ensure we always write a valid `submission.csv` with correct columns and row count.'

# 9. Code solution

## === cell 0
import os, sys, textwrap, pathlib

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION"] = "2"
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

sitecustomize_path = pathlib.Path.cwd() / "sitecustomize.py"
sitecustomize_path.write_text(
    textwrap.dedent(
        """
        import os
        os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
        os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "2")
        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
        """
    ).lstrip(),
    encoding="utf-8",
)

if str(pathlib.Path.cwd()) not in sys.path:
    sys.path.insert(0, str(pathlib.Path.cwd()))

print("Wrote sitecustomize to:", str(sitecustomize_path))
print(
    "Env PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION =",
    os.environ.get("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"),
)
print(
    "Env PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION =",
    os.environ.get("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION"),
)




## === cell 1
import re
import numpy as np
import pandas as pd

try:
    import tensorflow as tf
    from tensorflow import keras
except AttributeError as e:
    if "MessageFactory" in str(e) and "GetPrototype" in str(e):
        print(
            "Detected protobuf/TensorFlow import crash; restarting Python with pure-Python protobuf..."
        )
        os.execv(sys.executable, [sys.executable] + sys.argv)
    raise

print("TF:", tf.__version__)
print("Keras:", keras.__version__)

tf.random.set_seed(42)
np.random.seed(42)




## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 2
NUM_CLASSES = 5
IMG_SIZE = 299

base = keras.applications.Xception(
    include_top=True,
    weights="imagenet",
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
)

inputs = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
x = keras.applications.xception.preprocess_input(inputs * 255.0)
outputs = base(x, training=False)  # (None, 1000)
model = keras.Model(inputs, outputs)
model.trainable = False




## === cell 3
model.summary()




## === cell 4
DATA_DIR = "/kaggle/input/cassava-leaf-disease-classification"
test_filenames = tf.io.gfile.glob(f"{DATA_DIR}/test_tfrecords/*.tfrec")
test_filenames = sorted(test_filenames)
print("Test tfrecords:", len(test_filenames))
assert len(test_filenames) > 0, "No TFRecord files found; check the path."

sample_path = f"{DATA_DIR}/sample_submission.csv"
sample = pd.read_csv(sample_path)
print("Sample rows:", len(sample), "cols:", list(sample.columns))
assert list(sample.columns) == [
    "image_id",
    "label",
], "Unexpected sample submission format."




## === cell 5
def read_tfrec(example):
    fmt = {
        "image": tf.io.FixedLenFeature([], tf.string),
        "image_name": tf.io.FixedLenFeature([], tf.string, default_value=""),
        "image_id": tf.io.FixedLenFeature([], tf.string, default_value=""),
    }
    ex = tf.io.parse_single_example(example, fmt)
    image = tf.image.decode_jpeg(ex["image"], channels=3)

    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE), antialias=True)
    image = tf.cast(image, tf.float32) / 255.0

    name = ex["image_name"]
    name = tf.where(tf.equal(name, ""), ex["image_id"], name)
    return image, name




## === cell 6
options = tf.data.Options()
options.experimental_deterministic = True

dset = tf.data.TFRecordDataset(test_filenames, num_parallel_reads=tf.data.AUTOTUNE)
dset = dset.with_options(options)
dset = dset.map(read_tfrec, num_parallel_calls=tf.data.AUTOTUNE)
dset = dset.batch(32)
dset = dset.prefetch(tf.data.AUTOTUNE)

image_ds = dset.map(lambda image, name: image, num_parallel_calls=tf.data.AUTOTUNE)
names_ds = dset.map(
    lambda image, name: name, num_parallel_calls=tf.data.AUTOTUNE
).unbatch()




## === cell 7
prob_1000 = model.predict(image_ds, verbose=1)
print(
    "prob_1000 shape:",
    prob_1000.shape,
    "min/max:",
    float(prob_1000.min()),
    float(prob_1000.max()),
)




## === cell 8
cassava_imagenet_map = {
    0: [948, 949, 950, 951, 952, 953, 954, 955],  # fruit/produce-ish
    1: [985, 986, 987, 988, 989],  # plant/flower-ish
    2: [309, 310, 311, 312, 313],  # insects-ish (proxy for pest damage)
    3: [986, 987, 988, 689, 690],  # fungus/mold-ish proxies
    4: [999, 998, 997, 996],  # fallback "other/healthy-ish" (very weak)
}

num_test = prob_1000.shape[0]
prob_5 = np.zeros((num_test, NUM_CLASSES), dtype=np.float32)

for lbl in range(NUM_CLASSES):
    idxs = [i for i in cassava_imagenet_map.get(lbl, []) if 0 <= i < prob_1000.shape[1]]
    if len(idxs) == 0:
        continue
    prob_5[:, lbl] = prob_1000[:, idxs].sum(axis=1)

row_sums = prob_5.sum(axis=1, keepdims=True)
zero_rows = row_sums[:, 0] == 0
if np.any(zero_rows):
    prob_5[zero_rows] = 1.0 / NUM_CLASSES
    row_sums = prob_5.sum(axis=1, keepdims=True)

prob_5 = prob_5 / row_sums
pred = np.argmax(prob_5, axis=-1).astype(np.int64)
print("Pred shape:", pred.shape, "Pred unique:", np.unique(pred))




## === cell 9
def _normalize_image_id(x: str) -> str:
    x = x.strip()
    if x and x.isdigit():
        return x + ".jpg"
    return x


test_ids_raw = next(iter(names_ds.batch(len(pred)))).numpy()

test_ids = []
for v in test_ids_raw:
    if isinstance(v, (bytes, bytearray, np.bytes_)):
        s = bytes(v).decode("utf-8", errors="ignore")
    else:
        s = str(v)
    test_ids.append(_normalize_image_id(s))

print("IDs read:", len(test_ids), "Preds:", len(pred))

n = min(len(test_ids), len(pred))
id2pred = {test_ids[i]: int(pred[i]) for i in range(n)}

if len(id2pred) == 0:
    fill_value = 0
else:
    vals, counts = np.unique(list(id2pred.values()), return_counts=True)
    fill_value = int(vals[np.argmax(counts)])

sub = sample.copy()
sub["label"] = sub["image_id"].map(id2pred).fillna(fill_value).astype(np.int64)

assert len(sub) == len(sample), "Submission row count mismatch."
assert (
    sub["label"].between(0, NUM_CLASSES - 1).all()
), "Predicted labels out of range 0-4."

sub_path = "submission.csv"
sub.to_csv(sub_path, index=False)
print("Wrote:", sub_path, "rows:", len(sub))

with open(sub_path, "r", encoding="utf-8") as f:
    for _ in range(5):
        print(f.readline().rstrip("\n"))
