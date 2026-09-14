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

0.56465

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.10613) has done: 'I fix the early TensorFlow import crash (the protobuf `MessageFactory.GetPrototype` issue) by forcing TensorFlow to use the pure-Python protobuf implementation before importing it, which is a common Kaggle runtime incompatibility fix. Then, because your external weights file `../input/model-v11/clf_new_26.h5` is not available, I keep the same inference pipeline but add a safe fallback model (simple Keras CNN) that lets the notebook run end-to-end and still produce a valid `submission.csv`. I also make the data path robust to either `../input/...` or `/kaggle/input/...` layouts without changing the core generator/prediction semantics. Finally, I ensure `image_id` extraction and row ordering match `sample_submission.csv` so the submission is valid.'
- What this solution (achieved 0.14611) has done: 'I fix the TensorFlow/protobuf import crash by switching to the supported workaround that also forces the pure-Python protobuf backend (`PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python`) *and* the “python” protobuf API version, and by importing TensorFlow only after those env vars are set. To move accuracy up toward your target (since the current score is far below), I keep your end-to-end inference pipeline but replace the untrained fallback CNN with an ImageNet-pretrained EfficientNetB0 head (same single-pass predict flow, no extra training loops), which is a minimal change that substantially improves predictions when the external `.h5` weights are missing. I also make the missing-model path robust by searching for the weights file in `/kaggle/input/**` and using the best available option (provided `.h5`, else pretrained backbone). Submission formatting/order alignment with `sample_submission.csv` be preserved.'
- What this solution (achieved 0.23318) has done: 'I fix the TensorFlow import crash caused by an incompatible protobuf runtime by forcing a safe protobuf version before importing TensorFlow (this is the root error in cell 0). I keep your same inference pipeline and fallback model logic, but make it robust to missing optional imports (`tensorflow_hub`, `efficientnet.tfkeras`) so it cannot crash when those packages aren’t installed. I also ensure the test image discovery prefers the `sample_submission.csv` ordering (so the submission aligns deterministically) while keeping the same prediction semantics. These changes are execution-stability focused and should also help score by preventing misalignment-related label assignment issues.'
- What this solution (achieved 0.06465) has done: 'I fix the TensorFlow/protobuf crash by switching to the more robust Kaggle workaround that prefers the pure-Python protobuf backend and (when available) uses the legacy Keras package to avoid the `MessageFactory.GetPrototype` incompatibility. Then I fix the EfficientNet fallback model construction error by replacing raw `tf.*` ops on a `KerasTensor` with equivalent Keras layers (`Lambda`) so the model can be built and `predict()` works. Finally, I make `my_model` always defined (never `None`) so submission writing is guaranteed, while keeping your existing inference/generator/submission ordering logic unchanged.'
- What this solution (achieved 0.56465) has done: 'The immediate blocker is the protobuf/TensorFlow import crash (`MessageFactory.GetPrototype`), so I harden the environment setup to force the pure-Python protobuf backend and ensure TensorFlow/Keras import happens only after those variables are set. Then I keep your exact inference pipeline (ImageDataGenerator → predict → argmax → merge into sample_submission order), but make the fallback model materially better (still single-pass, no training loops) by mapping ImageNet logits to 5 classes using a fixed, semantically meaningful projection instead of random weights, which should move accuracy much closer to your target. I also make data/weights path discovery more robust across Kaggle directory layouts without changing I/O semantics. Finally, I keep the submission formatting and ordering aligned to `sample_submission.csv` and guarantee `submission.csv` is always written.'
- What this solution (achieved 0.56465) has done: 'You’re hitting the known TensorFlow/protobuf incompatibility (`MessageFactory.GetPrototype`) before the notebook can even reach model loading/prediction, so the key fix is to force the pure-Python protobuf backend *and* ensure the C++ implementation is not used at runtime by setting the env vars early and disabling TF’s use of the C++ protobuf bindings when possible. I also make the TF/Keras import path deterministic (avoid the `tf_keras` branch if it triggers the protobuf crash in this environment) while keeping your exact inference pipeline and fallback EfficientNetB0 mapping unchanged. Finally, I keep submission ordering aligned to `sample_submission.csv` and ensure the CSV is always written with the required columns.'
- What this solution (achieved 0.56465) has done: 'The TensorFlow import crash is happening before any model/inference logic runs, so the primary fix is to harden the protobuf/TensorFlow import order by force-uninstalling the C++ protobuf path (pure-Python protobuf) and using the safer TF import sequence. Then, to move accuracy up toward your target with minimal semantic change, I keep the same inference pipeline and EfficientNetB0 fallback, but fix the “healthy vs leafy” duplication by making the 5th class (“healthy”) draw from a broader set of green-plant ImageNet classes rather than reusing the exact same indices as “leafy”. Finally, I keep submission ordering aligned to `sample_submission.csv` and ensure `submission.csv` is always written with correct columns and row count.'
- What this solution (achieved 0.56465) has done: 'We fix the immediate runtime blocker by hardening the protobuf/TensorFlow import so it avoids the `MessageFactory.GetPrototype` crash in this Kaggle image (this is purely an environment/import-order fix and score-neutral). Then we keep your exact inference pipeline and fallback EfficientNetB0 semantic mapping, but make it more robust by forcing legacy `tf.keras` usage and disabling XLA to avoid additional graph/protobuf edge crashes. Finally, we add a small safety fix to ensure `predict()` output length always matches the test generator length, guaranteeing a valid `submission.csv` with correct row count and required columns.'
- What this solution (achieved 0.56465) has done: 'We fix the TensorFlow/protobuf `MessageFactory.GetPrototype` crash by setting the additional environment variables that force the pure-Python protobuf runtime and disable the C++ implementation before importing TensorFlow, which is the true root cause of your current failure. This is execution-only and should be score-neutral, but it’s required to run end-to-end and generate `submission.csv`. I also harden the TensorFlow import path slightly (set a safe Keras backend env var) without changing your model/inference logic. Finally, we keep your exact data ordering (sample_submission-aligned) and prediction → argmax → CSV formatting unchanged.'
- What this solution (achieved 0.56465) has done: 'We fix the TensorFlow/protobuf import crash by forcing a compatible protobuf runtime *before* any TensorFlow/Keras import, and by proactively removing the incompatible `google._upb` module path if present (this is the root cause of `MessageFactory.GetPrototype`). This change is execution-only and keeps your model/inference logic identical. Then we keep the rest of your pipeline unchanged (same EfficientNetB0 fallback mapping, same ImageDataGenerator inference, same sample-submission ordering) while ensuring imports happen only after the environment is made safe. The result should run end-to-end and write a valid `submission.csv` with the required columns and row count, and it should preserve (or improve) your current score rather than degrade it.'

# 9. Code solution

## === cell 0
import os

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "2")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_DISABLE_CPP", "1")

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_XLA_FLAGS", "--tf_xla_enable_xla_devices=false")
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

import sys

sys.modules.pop("google._upb", None)
sys.modules.pop("google._upb._message", None)

import glob
import numpy as np
import pandas as pd

SEED = 42
DEBUG = False

import tensorflow as tf  # type: ignore
from tensorflow import keras  # type: ignore
from tensorflow.keras.models import load_model  # type: ignore
from tensorflow.keras.preprocessing.image import ImageDataGenerator  # type: ignore

tf.random.set_seed(SEED)
np.random.seed(SEED)

try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

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
print("TensorFlow version:", tf.__version__)



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
        f"WARNING: Model file not found at {weight_path}. Using pretrained EfficientNetB0 (ImageNet) with semantic 5-class mapping."
    )

    inputs = keras.Input(shape=(300, 300, 3))
    x = keras.applications.efficientnet.preprocess_input(inputs)

    base = keras.applications.EfficientNetB0(
        include_top=True,  # 1000-way ImageNet head
        weights="imagenet",
        input_tensor=x,
    )
    base.trainable = False

    probs_1000 = base.output  # (None, 1000), softmax

    LEAFY = [
        985,
        986,
        987,
        988,
        989,
        990,
        930,
        931,
        932,
        933,
        934,
        935,
        948,
        949,
        950,
        951,
        967,
        968,
        969,
        970,
    ]
    INSECTS = list(range(300, 320))
    FUNGI = [992, 993, 994, 995, 996, 997]

    HEALTHY = [
        947,
        951,
        953,
        954,
        955,
        956,
        957,
        958,
        959,
        960,
        964,
        965,
        966,
        971,
        972,
        973,
        974,
        975,
        976,
        980,
    ]
    HEALTHY = sorted({i for i in HEALTHY if 0 <= i < 1000})

    leaf_score = keras.layers.Lambda(
        lambda t: tf.reduce_sum(tf.gather(t, LEAFY, axis=-1), axis=-1, keepdims=True),
        name="leaf_score",
    )(probs_1000)
    insect_score = keras.layers.Lambda(
        lambda t: tf.reduce_sum(tf.gather(t, INSECTS, axis=-1), axis=-1, keepdims=True),
        name="insect_score",
    )(probs_1000)
    fungi_score = keras.layers.Lambda(
        lambda t: tf.reduce_sum(tf.gather(t, FUNGI, axis=-1), axis=-1, keepdims=True),
        name="fungi_score",
    )(probs_1000)
    healthy_score = keras.layers.Lambda(
        lambda t: tf.reduce_sum(tf.gather(t, HEALTHY, axis=-1), axis=-1, keepdims=True),
        name="healthy_score",
    )(probs_1000)

    stacked = keras.layers.Concatenate(name="stacked_scores")(
        [leaf_score, insect_score, fungi_score, healthy_score]
    )  # (None,4)

    max_known = keras.layers.Lambda(
        lambda t: tf.reduce_max(t, axis=-1, keepdims=True), name="max_known"
    )(stacked)
    other_score = keras.layers.Lambda(
        lambda t: tf.clip_by_value(1.0 - t, 0.0, 1.0), name="other_score"
    )(max_known)

    logits5 = keras.layers.Concatenate(name="cassava_logits5")(
        [leaf_score, insect_score, fungi_score, other_score, healthy_score]
    )

    probs5 = keras.layers.Lambda(
        lambda t: tf.nn.softmax(tf.math.log(tf.clip_by_value(t, 1e-6, 1.0)), axis=-1),
        name="cassava_probs5",
    )(logits5)

    my_model = keras.Model(inputs, probs5)

if my_model is None:
    raise RuntimeError("Model was not created/loaded; cannot proceed to prediction.")

print("Model ready. Output shape:", my_model.output_shape)



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

pred_test = np.asarray(pred_test)[: len(df_test)]

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



## === cell 4
assert os.path.exists("submission.csv"), "submission.csv was not created."
chk = pd.read_csv("submission.csv")
print(chk.shape, chk.columns.tolist())
print(chk["label"].value_counts().head())
chk.head()
