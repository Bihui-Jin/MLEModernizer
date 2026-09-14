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

3.9

# 3. Installed packages

No external packages required in the script and installed.

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

0.6317618615896041

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.10613) has done: 'I fix the early TensorFlow import crash (the protobuf `MessageFactory.GetPrototype` issue) by forcing TensorFlow to use the pure-Python protobuf implementation before importing it, which is a common Kaggle runtime incompatibility fix. Then, because your external weights file `../input/model-v11/clf_new_26.h5` is not available, I keep the same inference pipeline but add a safe fallback model (simple Keras CNN) that lets the notebook run end-to-end and still produce a valid `submission.csv`. I also make the data path robust to either `../input/...` or `/kaggle/input/...` layouts without changing the core generator/prediction semantics. Finally, I ensure `image_id` extraction and row ordering match `sample_submission.csv` so the submission is valid.'
- What this solution (achieved 0.14611) has done: 'I fix the TensorFlow/protobuf import crash by switching to the supported workaround that also forces the pure-Python protobuf backend (`PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python`) *and* the “python” protobuf API version, and by importing TensorFlow only after those env vars are set. To move accuracy up toward your target (since the current score is far below), I keep your end-to-end inference pipeline but replace the untrained fallback CNN with an ImageNet-pretrained EfficientNetB0 head (same single-pass predict flow, no extra training loops), which is a minimal change that substantially improves predictions when the external `.h5` weights are missing. I also make the missing-model path robust by searching for the weights file in `/kaggle/input/**` and using the best available option (provided `.h5`, else pretrained backbone). Submission formatting/order alignment with `sample_submission.csv` be preserved.'
- What this solution (achieved 0.23318) has done: 'I fix the TensorFlow import crash caused by an incompatible protobuf runtime by forcing a safe protobuf version before importing TensorFlow (this is the root error in cell 0). I keep your same inference pipeline and fallback model logic, but make it robust to missing optional imports (`tensorflow_hub`, `efficientnet.tfkeras`) so it cannot crash when those packages aren’t installed. I also ensure the test image discovery prefers the `sample_submission.csv` ordering (so the submission aligns deterministically) while keeping the same prediction semantics. These changes are execution-stability focused and should also help score by preventing misalignment-related label assignment issues.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION"] = "2"
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

try:
    import google.protobuf  # noqa: F401
except Exception:
    pass

import glob
import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

SEED = 42
DEBUG = False

tf.random.set_seed(SEED)
np.random.seed(SEED)

CANDIDATE_DATA_DIRS = [
    "../input/cassava-leaf-disease-classification",
    "/kaggle/input/cassava-leaf-disease-classification",
    "../input/cassava-leaf-disease-classification/cassava-leaf-disease-classification",
    "/kaggle/input/cassava-leaf-disease-classification/cassava-leaf-disease-classification",
]
DATA_DIR = next((p for p in CANDIDATE_DATA_DIRS if os.path.exists(p)), None)
if DATA_DIR is None:
    raise FileNotFoundError(f"Could not find dataset dir. Tried: {CANDIDATE_DATA_DIRS}")

TEST_IMG_GLOB = os.path.join(DATA_DIR, "test_images", "*.jpg")
SAMPLE_SUB_PATH = os.path.join(DATA_DIR, "sample_submission.csv")

print("Using DATA_DIR:", DATA_DIR)
print("Sample submission exists:", os.path.exists(SAMPLE_SUB_PATH))
print("Test glob:", TEST_IMG_GLOB)



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
weight_path = "../input/model-v11/clf_new_26.h5"

candidate_h5 = []
if os.path.exists("/kaggle/input"):
    candidate_h5 = glob.glob("/kaggle/input/**/clf_new_26.h5", recursive=True)
if len(candidate_h5) > 0:
    weight_path = candidate_h5[0]

custom_objects = {}
try:
    import tensorflow_hub as hub  # type: ignore

    custom_objects["KerasLayer"] = hub.KerasLayer
except Exception:
    pass

try:
    import efficientnet.tfkeras as efn  # type: ignore

    custom_objects.update(
        {
            "EfficientNetB0": getattr(efn, "EfficientNetB0", None),
            "EfficientNetB1": getattr(efn, "EfficientNetB1", None),
            "EfficientNetB2": getattr(efn, "EfficientNetB2", None),
            "EfficientNetB3": getattr(efn, "EfficientNetB3", None),
            "EfficientNetB4": getattr(efn, "EfficientNetB4", None),
            "EfficientNetB5": getattr(efn, "EfficientNetB5", None),
            "EfficientNetB6": getattr(efn, "EfficientNetB6", None),
            "EfficientNetB7": getattr(efn, "EfficientNetB7", None),
            "Swish": getattr(efn, "Swish", None),
            "FixedDropout": getattr(efn, "FixedDropout", None),
        }
    )
    custom_objects = {k: v for k, v in custom_objects.items() if v is not None}
except Exception:
    pass

my_model = None
if os.path.exists(weight_path):
    my_model = load_model(weight_path, custom_objects=custom_objects, compile=False)
    print("Loaded model from:", weight_path)
else:
    print(
        f"WARNING: Model file not found at {weight_path}. Using pretrained EfficientNetB0 (ImageNet) mapped to 5 classes."
    )

    inputs = keras.Input(shape=(300, 300, 3))
    x = keras.applications.efficientnet.preprocess_input(inputs)
    base = keras.applications.EfficientNetB0(
        include_top=True,  # 1000-way ImageNet head
        weights="imagenet",
        input_tensor=x,
    )
    base.trainable = False

    logits_1000 = base.output  # after softmax (shape: [None, 1000])
    eps = tf.constant(1e-7, dtype=logits_1000.dtype)
    logp = tf.math.log(tf.clip_by_value(logits_1000, eps, 1.0))

    rng = np.random.default_rng(SEED)
    W = rng.normal(loc=0.0, scale=0.01, size=(1000, 5)).astype("float32")
    b = np.zeros((5,), dtype="float32")

    mapped = keras.layers.Dense(
        5,
        activation="softmax",
        use_bias=True,
        kernel_initializer=keras.initializers.Constant(W),
        bias_initializer=keras.initializers.Constant(b),
        name="imagenet_to_cassava_map",
        trainable=False,
    )(logp)

    my_model = keras.Model(inputs, mapped)

print("Model ready. Output shape:", my_model.output_shape)



## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/1759696631.py in <cell line: 0>()
     62     # Convert to "logits-like" space for linear mixing stability, then softmax to 5
     63     eps = tf.constant(1e-7, dtype=logits_1000.dtype)
---> 64     logp = tf.math.log(tf.clip_by_value(logits_1000, eps, 1.0))
     65 
     66     # Deterministic fixed weights (seeded) for mapping, improves over random per-run initialization.

/usr/local/lib/python3.11/dist-packages/tensorflow/python/ops/weak_tensor_ops.py in wrapper(*args, **kwargs)
     86   def wrapper(*args, **kwargs):
     87     if not ops.is_auto_dtype_conversion_enabled():
---> 88       return op(*args, **kwargs)
     89     bound_arguments = signature.bind(*args, **kwargs)
     90     bound_arguments.apply_defaults()

/usr/local/lib/python3.11/dist-packages/tensorflow/python/util/traceback_utils.py in error_handler(*args, **kwargs)
    151     except Exception as e:
    152       filtered_tb = _process_traceback_frames(e.__traceback__)
--> 153       raise e.with_traceback(filtered_tb) from None
    154     finally:
    155       del filtered_tb

/usr/local/lib/python3.11/dist-packages/keras/src/backend/common/keras_tensor.py in __tf_tensor__(self, dtype, name)
    136 
    137     def __tf_tensor__(self, dtype=None, name=None):
--> 138         raise ValueError(
    139             "A KerasTensor cannot be used as input to a TensorFlow function. "
    140             "A KerasTensor is a symbolic placeholder for a shape and dtype, "

ValueError: A KerasTensor cannot be used as input to a TensorFlow function. A KerasTensor is a symbolic placeholder for a shape and dtype, used when constructing Keras Functional models or Keras Functions. You can only use it as input to a Keras layer or a Keras operation (from the namespaces `keras.layers` and `keras.operations`). You are likely doing something like:

```
x = Input(...)
...
tf_fn(x)  # Invalid.
```

What you should do instead is wrap `tf_fn` in a layer:

```
class MyLayer(Layer):
    def call(self, x):
        return tf_fn(x)

x = MyLayer()(x)
```


## === cell 2
if os.path.exists(SAMPLE_SUB_PATH):
    sample = pd.read_csv(SAMPLE_SUB_PATH)
    img_paths = [
        os.path.join(DATA_DIR, "test_images", fname)
        for fname in sample["image_id"].tolist()
    ]
    missing = [p for p in img_paths if not os.path.exists(p)]
    if len(missing) > 0:
        print(
            f"WARNING: {len(missing)} test image paths missing; falling back to glob ordering."
        )
        test_images = sorted(glob.glob(TEST_IMG_GLOB))
        if len(test_images) == 0:
            raise FileNotFoundError(f"No test images found via glob: {TEST_IMG_GLOB}")
        df_test = pd.DataFrame(test_images, columns=["path"])
    else:
        df_test = pd.DataFrame(img_paths, columns=["path"])
else:
    test_images = sorted(glob.glob(TEST_IMG_GLOB))
    if len(test_images) == 0:
        raise FileNotFoundError(f"No test images found via glob: {TEST_IMG_GLOB}")
    df_test = pd.DataFrame(test_images, columns=["path"])


def make_test_gen(batch_size=64):
    my_test_idg = ImageDataGenerator()
    test_gen = my_test_idg.flow_from_dataframe(
        dataframe=df_test,
        x_col="path",
        y_col=None,
        batch_size=batch_size,
        seed=SEED,
        shuffle=False,
        class_mode=None,
        target_size=(300, 300),
    )
    return test_gen




## === cell 3
test_gen = make_test_gen(batch_size=128)

steps = int(np.ceil(len(df_test) / test_gen.batch_size))
pred_test = my_model.predict(test_gen, steps=steps, verbose=1)

pred_test_labels = np.argmax(pred_test, axis=-1).astype(int)

final_submission = df_test.copy()
final_submission["image_id"] = (
    final_submission["path"].str.replace("\\", "/", regex=False).str.split("/").str[-1]
)
final_submission["label"] = pred_test_labels[: len(final_submission)]

if os.path.exists(SAMPLE_SUB_PATH):
    sample = pd.read_csv(SAMPLE_SUB_PATH)
    final_csv = sample[["image_id"]].merge(
        final_submission[["image_id", "label"]], on="image_id", how="left"
    )
    if final_csv["label"].isna().any():
        fill_label = int(pd.Series(pred_test_labels).mode().iloc[0])
        final_csv["label"] = final_csv["label"].fillna(fill_label).astype(int)
else:
    final_csv = final_submission[["image_id", "label"]]

final_csv.to_csv("submission.csv", index=False)

print(final_csv.head())
print("Wrote submission.csv with rows:", len(final_csv))



## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/2583092690.py in <cell line: 0>()
      2 
      3 steps = int(np.ceil(len(df_test) / test_gen.batch_size))
----> 4 pred_test = my_model.predict(test_gen, steps=steps, verbose=1)
      5 
      6 pred_test_labels = np.argmax(pred_test, axis=-1).astype(int)

AttributeError: 'NoneType' object has no attribute 'predict'

## === cell 4
assert os.path.exists("submission.csv"), "submission.csv was not created."
chk = pd.read_csv("submission.csv")
print(chk.shape, chk.columns.tolist())
print(chk["label"].value_counts().head())
chk.head()

## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
AssertionError                            Traceback (most recent call last)
/tmp/ipykernel_11/871185145.py in <cell line: 0>()
----> 1 assert os.path.exists("submission.csv"), "submission.csv was not created."
      2 chk = pd.read_csv("submission.csv")
      3 print(chk.shape, chk.columns.tolist())
      4 print(chk["label"].value_counts().head())
      5 chk.head()

AssertionError: submission.csv was not created.
