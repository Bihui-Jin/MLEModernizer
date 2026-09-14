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

0.8068902991840435

# 6. Current score

0.05493

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.07997) has done: 'I remove the `tensorflow_hub` import that is triggering the protobuf `MessageFactory.GetPrototype` crash in this environment, since it is unused by your pipeline. I also eliminate the notebook-style `!pip install` cell (it won’t run in a pure `.py` Kaggle submission context) and instead rely on `tf.keras.applications.EfficientNetB3` to rebuild the same expected architecture, then load your provided `.h5` weights so inference can proceed. Finally, I fix the missing model definition (`my_model`) and make the test image path robust to the actual Kaggle `/kaggle/input/...` location, ensuring a valid `submission.csv` with the correct `image_id,label` columns is always written.'
- What this solution (achieved 0.11584) has done: 'I remove the import path that triggers the `MessageFactory.GetPrototype` protobuf crash by avoiding Keras/TensorFlow submodules that pull in the problematic dependency chain, and I add a safe fallback so the script can run even if the external `.h5` weights dataset is not attached. I fix the Keras 3 weight-loading failure by using `tf.keras` consistently and (when weights exist) loading them via legacy H5 compatibility; otherwise the model run with random initialization to still generate a valid `submission.csv`. I also ensure the test image discovery and `image_id` extraction are correct and stable, and that the submission is written with the exact required columns. These are minimal changes focused on unblocking execution and improving accuracy when the weights file is available (which is necessary to reach the target score band).'
- What this solution (achieved 0.61099) has done: 'The crash happens before your code runs any training/inference: importing TensorFlow triggers a known protobuf incompatibility (`MessageFactory.GetPrototype`) in this runtime. The minimal reliable fix is to force TensorFlow to use the pure-Python protobuf implementation *before* importing `tensorflow`, which avoids the broken compiled-protobuf path. After that, the rest of your pipeline (EfficientNetB3 + optional `.h5` weights + ImageDataGenerator inference) can run unchanged and produce a valid `submission.csv`. This should also move your score upward toward the target because it restores proper model execution with weights loading (when the weights file is available).'
- What this solution (achieved 0.06764) has done: 'The timeout is dominated by slow input decoding/resize in `ImageDataGenerator.flow_from_dataframe` plus conservative TensorFlow runtime settings; the model forward pass itself is relatively fast. I keep the exact same model and prediction logic, but switch the test input pipeline to `tf.data` (parallel JPEG decode + resize + prefetch), which is equivalent for inference and avoids Python-side generators. I also enable XLA JIT and set TensorFlow threading to better utilize CPU cores, without changing outputs beyond negligible floating-point differences. Finally, I remove the forced pure-Python protobuf implementation which slows TF graph loading/execution.'
- What this solution (achieved 0.11024) has done: 'The runtime crash happens at `import tensorflow` due to a protobuf implementation mismatch (`MessageFactory.GetPrototype`). The minimal reliable fix is to force TensorFlow to use the pure-Python protobuf implementation *before* importing TensorFlow, which avoids the broken compiled-protobuf path in this environment. I keep your model/inference pipeline unchanged (EfficientNetB3, weight loading, tf.data decode/resize/predict) and only adjust the environment setup order so the script runs end-to-end and writes a valid `submission.csv`. This should also restore proper weight loading/execution, moving accuracy back up toward your target band.'
- What this solution (achieved 0.05605) has done: 'The immediate blocker is the protobuf crash that happens during `import tensorflow`; setting `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` is not sufficient in this Kaggle image because TensorFlow still ends up importing an incompatible `google.protobuf` runtime. The minimal reliable fix is to force-install a compatible protobuf version at runtime *before* importing TensorFlow, and to keep the environment variables in place as a fallback. After TensorFlow imports cleanly, I keep your model, weights loading, tf.data preprocessing, and submission-writing logic unchanged so behavior/score remains driven by the same pipeline (and should move upward toward the target if weights are present and correctly loaded). I also add a tiny guard to avoid double-import issues and ensure the protobuf pin happens first.'
- What this solution (achieved 0.08857) has done: 'Your current score (0.05605) is far below the target (0.80689), and the most likely reason is that inference is being run on *unnormalized* images (raw 0–255 float32) instead of the `tf.keras.applications.efficientnet.preprocess_input` that the saved EfficientNetB3 weights expect. I keep the exact same model and inference flow, but add the correct EfficientNet preprocessing in the `tf.data` pipeline so the loaded `.h5` weights are used in the intended input space. I also set `my_model.trainable = False` (no behavior change for predict, but avoids any accidental training-mode surprises) and explicitly use `training=False` during prediction for stability. These are minimal changes that should move accuracy substantially upward toward your target without changing architecture or training logic.'
- What this solution (achieved 0.57885) has done: 'The timeout is dominated by test-set input pipeline overhead: every image is decoded twice (once to get shape, again to crop), plus `Dataset.cache()` retains ~2,676 full 512×512×3 float tensors in memory and costs extra time upfront. I keep the exact same model and preprocessing semantics, but refactor the test loader to decode once and do the center-square crop via `tf.image.crop_to_bounding_box`, which is equivalent to the current crop window logic. I also remove the in-memory caching for the test dataset (it isn’t reused) and enable non-blocking prefetching to overlap CPU decode with GPU/CPU inference. These changes preserve determinism and output semantics while substantially reducing per-image CPU work and memory pressure.'
- What this solution (achieved 0.05493) has done: 'I fix the TensorFlow import crash (`MessageFactory.GetPrototype`) by forcing the pure-Python protobuf implementation *before* any TensorFlow import and removing the “try TF then pip install protobuf” logic that still crashes during the initial import attempt. I also fix the external weights discovery so it can find `fineTuned_v0.38.h5` in any attached Kaggle dataset (not just the single hardcoded path), which should restore the intended fine-tuned model performance and move accuracy upward toward your target. Finally, I keep the model/inference pipeline the same (EfficientNetB3 + preprocess_input + tf.data) and ensure a valid `submission.csv` is always written with the required `image_id,label` columns in the sample submission order.'

# 9. Code solution

## === cell 0
import os
import glob
import sys

import numpy as np
import pandas as pd

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION"] = "2"

SEED = 42
DEBUG = False
os.environ["PYTHONHASHSEED"] = str(SEED)

import tensorflow as tf  # noqa: E402

try:
    tf.config.optimizer.set_jit(True)
except Exception:
    pass

try:
    _n_cpu = os.cpu_count() or 4
    tf.config.threading.set_intra_op_parallelism_threads(_n_cpu)
    tf.config.threading.set_inter_op_parallelism_threads(max(1, _n_cpu // 2))
except Exception:
    pass

tf.random.set_seed(SEED)
np.random.seed(SEED)

try:
    tf.data.experimental.enable_debug_mode(False)
except Exception:
    pass

try:
    tf.config.experimental.set_synchronous_execution(False)
except Exception:
    pass



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
INPUT_ROOT_CANDIDATES = [
    "/kaggle/input/cassava-leaf-disease-classification",
    "../input/cassava-leaf-disease-classification",
    "/kaggle/input/cassava-leaf-disease-classification/cassava-leaf-disease-classification",
    "../input/cassava-leaf-disease-classification/cassava-leaf-disease-classification",
]

INPUT_ROOT = None
for p in INPUT_ROOT_CANDIDATES:
    if os.path.exists(p) and os.path.exists(os.path.join(p, "sample_submission.csv")):
        INPUT_ROOT = p
        break

if INPUT_ROOT is None:
    raise FileNotFoundError(
        "Could not locate cassava-leaf-disease-classification dataset folder in expected locations."
    )

TRAIN_CSV_PATH = os.path.join(INPUT_ROOT, "train.csv")
SAMPLE_SUB_PATH = os.path.join(INPUT_ROOT, "sample_submission.csv")
TRAIN_IMG_DIR = os.path.join(INPUT_ROOT, "train_images")
TEST_IMG_DIR = os.path.join(INPUT_ROOT, "test_images")

if not os.path.exists(TRAIN_CSV_PATH):
    raise FileNotFoundError(f"Missing train.csv at: {TRAIN_CSV_PATH}")
if not os.path.exists(SAMPLE_SUB_PATH):
    raise FileNotFoundError(f"Missing sample_submission.csv at: {SAMPLE_SUB_PATH}")
if not os.path.exists(TRAIN_IMG_DIR):
    raise FileNotFoundError(f"Missing train_images dir at: {TRAIN_IMG_DIR}")
if not os.path.exists(TEST_IMG_DIR):
    raise FileNotFoundError(f"Missing test_images dir at: {TEST_IMG_DIR}")

train_df = pd.read_csv(TRAIN_CSV_PATH)
SAMPLE_SUB = pd.read_csv(SAMPLE_SUB_PATH)

WEIGHT_PATH_CANDIDATES = [
    "/kaggle/input/experiment-with-models-using-keras-with-updates/fineTuned_v0.38.h5",
    "../input/experiment-with-models-using-keras-with-updates/fineTuned_v0.38.h5",
]
WEIGHT_PATH = None
for wp in WEIGHT_PATH_CANDIDATES:
    if os.path.exists(wp):
        WEIGHT_PATH = wp
        break

if WEIGHT_PATH is None:
    for wp in glob.glob("/kaggle/input/**/fineTuned_v0.38.h5", recursive=True):
        if os.path.exists(wp):
            WEIGHT_PATH = wp
            break

print("INPUT_ROOT:", INPUT_ROOT)
print("Found external weights:", WEIGHT_PATH is not None)
if WEIGHT_PATH is not None:
    print("WEIGHT_PATH:", WEIGHT_PATH)



## === cell 2
layers = tf.keras.layers
Model = tf.keras.Model
EfficientNetB3 = tf.keras.applications.EfficientNetB3
preprocess_input = tf.keras.applications.efficientnet.preprocess_input


def build_model(input_size=512, n_classes=5):
    inputs = layers.Input(shape=(input_size, input_size, 3))
    base = EfficientNetB3(include_top=False, weights=None, input_tensor=inputs)
    x = layers.GlobalAveragePooling2D()(base.output)
    outputs = layers.Dense(n_classes, activation="softmax")(x)
    return Model(inputs=inputs, outputs=outputs)


IMG_SIZE = 512
N_CLASSES = 5

my_model = build_model(input_size=IMG_SIZE, n_classes=N_CLASSES)

_loaded = False
if WEIGHT_PATH is not None:
    try:
        my_model.load_weights(WEIGHT_PATH)
        _loaded = True
        print(f"Loaded weights from: {WEIGHT_PATH} (standard load)")
    except Exception as e1:
        print(f"WARNING: Standard load_weights failed: {repr(e1)}")
        try:
            my_model.load_weights(WEIGHT_PATH, by_name=True, skip_mismatch=True)
            _loaded = True
            print(
                f"Loaded weights from: {WEIGHT_PATH} (by_name=True, skip_mismatch=True)"
            )
        except Exception as e2:
            print(f"WARNING: Fallback load_weights also failed: {repr(e2)}")
            _loaded = False




## === cell 3
def make_train_val_splits(df, seed=42, val_frac=0.1):
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    n_val = int(len(df) * val_frac)
    val_df = df.iloc[:n_val].copy()
    trn_df = df.iloc[n_val:].copy()
    return trn_df, val_df


def make_labeled_ds(df, img_dir, batch_size=16, image_size=512, training=True):
    paths = tf.constant([os.path.join(img_dir, x) for x in df["image_id"].values])
    labels = tf.constant(df["label"].values, dtype=tf.int32)

    @tf.function
    def _load_and_preprocess(path, label):
        img_bytes = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img_bytes, channels=3)
        img = tf.image.resize(
            img, [image_size, image_size], method=tf.image.ResizeMethod.BILINEAR
        )
        img = tf.cast(img, tf.float32)
        img = preprocess_input(img)
        return img, label

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))

    opts = tf.data.Options()
    try:
        opts.deterministic = True
    except Exception:
        pass
    try:
        opts.experimental_optimization.apply_default_optimizations = True
        opts.experimental_optimization.autotune_buffers = True
        opts.experimental_optimization.map_parallelization = True
    except Exception:
        pass
    ds = ds.with_options(opts)

    if training:
        ds = ds.shuffle(2048, seed=SEED, reshuffle_each_iteration=True)

    ds = ds.map(_load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)

    if not training:
        ds = ds.cache()

    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


if not _loaded:
    raise FileNotFoundError(
        "External weights were not found/loaded; this solution expects fineTuned_v0.38.h5 "
        "to be attached in /kaggle/input (any dataset folder). "
        "Please attach the weights dataset containing fineTuned_v0.38.h5."
    )

my_model.trainable = False



## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/1098743114.py in <cell line: 0>()
     51 
     52 if not _loaded:
---> 53     raise FileNotFoundError(
     54         "External weights were not found/loaded; this solution expects fineTuned_v0.38.h5 "
     55         "to be attached in /kaggle/input (any dataset folder). "

FileNotFoundError: External weights were not found/loaded; this solution expects fineTuned_v0.38.h5 to be attached in /kaggle/input (any dataset folder). Please attach the weights dataset containing fineTuned_v0.38.h5.

## === cell 4
test_images = sorted(glob.glob(os.path.join(TEST_IMG_DIR, "*.jpg")))
if len(test_images) == 0:
    raise FileNotFoundError(f"No .jpg files found under: {TEST_IMG_DIR}")

df_test = pd.DataFrame({"path": test_images})
df_test["image_id"] = df_test["path"].map(os.path.basename)


def make_test_ds(df_paths, batch_size=64, image_size=512):
    paths = tf.constant(df_paths["path"].values)

    @tf.function
    def _load_and_preprocess(path):
        img_bytes = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img_bytes, channels=3)

        shape = tf.shape(img)  # [h,w,3]
        h = shape[0]
        w = shape[1]
        side = tf.minimum(h, w)
        offset_y = (h - side) // 2
        offset_x = (w - side) // 2
        img = tf.image.crop_to_bounding_box(img, offset_y, offset_x, side, side)

        img = tf.image.resize(
            img, [image_size, image_size], method=tf.image.ResizeMethod.BILINEAR
        )
        img = tf.cast(img, tf.float32)
        img = preprocess_input(img)
        return img

    ds = tf.data.Dataset.from_tensor_slices(paths)

    opts = tf.data.Options()
    try:
        opts.deterministic = True
    except Exception:
        pass
    try:
        opts.experimental_optimization.apply_default_optimizations = True
        opts.experimental_optimization.autotune_buffers = True
        opts.experimental_optimization.map_parallelization = True
    except Exception:
        pass
    ds = ds.with_options(opts)

    ds = ds.map(_load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds




## === cell 5
id_to_path = dict(zip(df_test["image_id"].values, df_test["path"].values))
ordered_paths = SAMPLE_SUB["image_id"].map(id_to_path)

if ordered_paths.isna().any():
    missing_ids = SAMPLE_SUB.loc[ordered_paths.isna(), "image_id"].tolist()
    raise RuntimeError(
        f"Some sample_submission image_id not found in test_images dir: {missing_ids[:5]}"
    )

df_test_ordered = pd.DataFrame(
    {"image_id": SAMPLE_SUB["image_id"].values, "path": ordered_paths.values}
)

test_ds = make_test_ds(df_test_ordered, batch_size=128, image_size=IMG_SIZE)
pred_test = my_model.predict(test_ds, verbose=1)

pred_test_labels = np.argmax(pred_test, axis=-1).astype(int)

final_csv = pd.DataFrame(
    {"image_id": df_test_ordered["image_id"].values, "label": pred_test_labels}
)
final_csv.to_csv("submission.csv", index=False)

if final_csv.shape[0] != SAMPLE_SUB.shape[0]:
    raise RuntimeError(
        "Submission row count does not match sample_submission row count."
    )
if final_csv["image_id"].isna().any():
    raise RuntimeError("Found NaN image_id values in submission.")
if final_csv["label"].isna().any():
    raise RuntimeError("Found NaN label values in submission.")
if not os.path.exists("submission.csv"):
    raise RuntimeError("submission.csv was not written as expected.")

print("Wrote submission.csv with shape:", final_csv.shape)



## === cell 6
final_csv.head()
