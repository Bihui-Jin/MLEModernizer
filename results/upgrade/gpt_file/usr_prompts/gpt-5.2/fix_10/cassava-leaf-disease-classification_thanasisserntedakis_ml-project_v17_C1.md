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

0.819431852523421

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.50785) has done: 'I fix the import/runtime crash by removing the unnecessary `sklearn` top-level import (it can trigger protobuf incompatibilities in Kaggle) and using `tf.keras` consistently so `ImageDataGenerator` is always defined. I also remove the dependency on missing external model files under `../input/*` (those datasets are not present in your environment) and replace them with equivalent on-the-fly `tf.keras.applications` backbones to preserve the same ensemble→stacking structure. Finally, I ensure the meta-model input shape matches the stacked predictions (4 models × 5 classes = 20 features) and write a valid `submission.csv` with the required `image_id,label` columns.'
- What this solution (achieved 0.11584) has done: 'The crash happens before your code runs because TensorFlow’s protobuf dependency is incompatible with the environment’s currently loaded protobuf runtime, triggering `MessageFactory.GetPrototype` errors at `import tensorflow as tf`. The minimal stable fix is to force the pure-Python protobuf implementation *before* importing TensorFlow (this is a known workaround on Kaggle for this exact stacktrace). After that, the pipeline run end-to-end and write a valid `submission.csv` with the required `image_id,label` columns. I’m keeping the model/stacking logic intact and only making execution-stability fixes (no score-tuning changes beyond making it run reliably).'

# 9. Code solution

## === cell 0
import os

os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", None)
os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", None)

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
try:
    import tensorflow as tf

    tf.config.optimizer.set_jit(True)
except Exception:
    pass

import glob
import json
import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input

tf.random.set_seed(42)
np.random.seed(42)

print("TF:", tf.__version__)

DATA_DIR = "/kaggle/input/cassava-leaf-disease-classification"
ALT_DATA_DIR = "../input/cassava-leaf-disease-classification"

if not os.path.exists(DATA_DIR) and os.path.exists(ALT_DATA_DIR):
    DATA_DIR = ALT_DATA_DIR

TRAIN_CSV = os.path.join(DATA_DIR, "train.csv")
TEST_GLOB = os.path.join(DATA_DIR, "test_images", "*.jpg")
TRAIN_IMG_DIR = os.path.join(DATA_DIR, "train_images")

test_images = glob.glob(TEST_GLOB)
if len(test_images) == 0:
    raise FileNotFoundError(
        f"Could not find test images under expected path: {TEST_GLOB}"
    )

df_test = pd.DataFrame(test_images, columns=["path"])
df_test["image_id"] = df_test["path"].str.rsplit("/", n=1).str[-1]

df_train = pd.read_csv(TRAIN_CSV)
df_train["path"] = (
    TRAIN_IMG_DIR.rstrip("/") + "/" + df_train["image_id"].astype(str)
).values

if not os.path.isdir(TRAIN_IMG_DIR):
    raise FileNotFoundError(f"Training image directory not found: {TRAIN_IMG_DIR}")

IMG_SIZE = 224
SIZE = (IMG_SIZE, IMG_SIZE)
BATCH_SIZE = 32

print("Train rows:", len(df_train), "Test rows:", len(df_test))




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
def build_base_model(app_model_fn, preprocess_fn, name):
    inp = Input(shape=(IMG_SIZE, IMG_SIZE, 3), name=f"{name}_input")
    x = tf.keras.layers.Lambda(lambda z: preprocess_fn(z), name=f"{name}_preprocess")(
        inp
    )
    backbone = app_model_fn(
        include_top=False,
        weights="imagenet",
        input_tensor=x,
        pooling="avg",
    )
    backbone.trainable = False  # inference only
    out = Dense(5, activation="softmax", name=f"{name}_cls")(backbone.output)
    model = Model(inp, out, name=name)
    return model


resnet = build_base_model(
    tf.keras.applications.ResNet50,
    tf.keras.applications.resnet.preprocess_input,
    "resnet50",
)
mobilenet = build_base_model(
    tf.keras.applications.MobileNetV2,
    tf.keras.applications.mobilenet_v2.preprocess_input,
    "mobilenetv2",
)
densenet = build_base_model(
    tf.keras.applications.DenseNet121,
    tf.keras.applications.densenet.preprocess_input,
    "densenet121",
)
efficientnet = build_base_model(
    tf.keras.applications.EfficientNetB0,
    tf.keras.applications.efficientnet.preprocess_input,
    "efficientnetb0",
)

models = [resnet, mobilenet, densenet, efficientnet]

final_model = Sequential(
    [
        Input(shape=(20,)),
        Dense(64, activation="relu"),
        Dense(5, activation="softmax"),
    ],
    name="final_stacker",
)

final_model.compile(
    loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
)




## === cell 2
AUTOTUNE = tf.data.AUTOTUNE

_DS_OPTIONS = tf.data.Options()
_DS_OPTIONS.experimental_deterministic = True
try:
    _DS_OPTIONS.experimental_optimization.map_parallelization = True
    _DS_OPTIONS.experimental_optimization.parallel_batch = True
except Exception:
    pass

CACHE_DIR = "/kaggle/working/tfdata_cache"
os.makedirs(CACHE_DIR, exist_ok=True)


def decode_image_only(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3, dct_method="INTEGER_FAST")
    img = tf.image.resize(img, SIZE, method="bilinear")
    img = tf.cast(img, tf.float32)
    return img


def make_base_image_ds_from_paths(
    paths, batch_size=BATCH_SIZE, cache=False, cache_id=""
):
    ds = tf.data.Dataset.from_tensor_slices(paths)
    ds = ds.map(decode_image_only, num_parallel_calls=AUTOTUNE)

    if cache:
        cache_path = os.path.join(CACHE_DIR, f"{cache_id}.cache")
        ds = ds.cache(cache_path)

    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    ds = ds.with_options(_DS_OPTIONS)
    return ds


def make_preprocessed_ds_from_paths(
    paths, preprocess_fn, batch_size=BATCH_SIZE, cache=False, cache_id=""
):
    ds = tf.data.Dataset.from_tensor_slices(paths)

    def _decode_and_preprocess(p):
        x = decode_image_only(p)
        x = preprocess_fn(x)
        return x

    ds = ds.map(_decode_and_preprocess, num_parallel_calls=AUTOTUNE)

    if cache:
        cache_path = os.path.join(CACHE_DIR, f"{cache_id}.cache")
        ds = ds.cache(cache_path)

    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    ds = ds.with_options(_DS_OPTIONS)
    return ds


def build_stacked_features_from_preds(pred_list):
    return np.concatenate(pred_list, axis=1).astype(np.float32)




## === cell 3
from sklearn.model_selection import StratifiedKFold

try:
    _ = StratifiedKFold
    HAVE_SKLEARN = True
except Exception:
    HAVE_SKLEARN = False


def get_model_preds_for_paths(
    model, paths, preprocess_fn, batch_size, cache_id, cache_decoded_ds=None
):
    if cache_decoded_ds is not None:
        ds = cache_decoded_ds.map(
            lambda x: preprocess_fn(x), num_parallel_calls=AUTOTUNE
        )
        ds = (
            ds.batch(batch_size, drop_remainder=False)
            .prefetch(AUTOTUNE)
            .with_options(_DS_OPTIONS)
        )
    else:
        ds = make_preprocessed_ds_from_paths(
            paths,
            preprocess_fn=preprocess_fn,
            batch_size=batch_size,
            cache=False,
            cache_id=cache_id,
        )

    steps = int(np.ceil(len(paths) / batch_size))

    try:
        for xb in ds.take(1):
            _ = model(xb, training=False)
    except Exception:
        pass

    return model.predict(ds, verbose=0, steps=steps)


NUM_CLASSES = 5
y = df_train["label"].values.astype(np.int32)

train_paths = df_train["path"].values

print(
    "\nBuilding cached decoded training dataset once (decoded+resized cached on disk, reused across all base models)..."
)
train_base_ds = make_base_image_ds_from_paths(
    train_paths, batch_size=BATCH_SIZE, cache=True, cache_id="train_decoded_224_b32"
)

print(
    "\nGenerating full-train predictions once per base model (same semantics; frozen models)..."
)
preprocess_fns = [
    tf.keras.applications.resnet.preprocess_input,
    tf.keras.applications.mobilenet_v2.preprocess_input,
    tf.keras.applications.densenet.preprocess_input,
    tf.keras.applications.efficientnet.preprocess_input,
]

full_pred_per_model = []
for m, pp, name in zip(
    models, preprocess_fns, ["resnet", "mobilenet", "densenet", "effnet"]
):
    full_pred_per_model.append(
        get_model_preds_for_paths(
            m,
            train_paths,
            pp,
            batch_size=BATCH_SIZE,
            cache_id=f"train_{name}",
            cache_decoded_ds=train_base_ds,
        ).astype(np.float32)
    )

if HAVE_SKLEARN:
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_pred_per_model = [
        np.zeros((len(df_train), NUM_CLASSES), dtype=np.float32) for _ in models
    ]

    for fold, (_, va_idx) in enumerate(skf.split(np.zeros(len(df_train)), y), start=1):
        print(
            f"Fold {fold}: assigning OOF predictions for {len(va_idx)} validation samples"
        )
        for m_i in range(len(models)):
            oof_pred_per_model[m_i][va_idx] = full_pred_per_model[m_i][va_idx]

    X_stack = build_stacked_features_from_preds(oof_pred_per_model)
    Y_onehot = tf.keras.utils.to_categorical(y, NUM_CLASSES).astype(np.float32)

    print("\nTraining stacker on OOF stacked features:", X_stack.shape, Y_onehot.shape)
    final_model.fit(X_stack, Y_onehot, epochs=20, batch_size=128, verbose=1)
else:
    idx = np.arange(len(df_train))
    np.random.RandomState(42).shuffle(idx)
    cut = int(0.8 * len(idx))
    tr_idx, va_idx = idx[:cut], idx[cut:]

    pred_tr_per_model = [p[tr_idx] for p in full_pred_per_model]
    pred_va_per_model = [p[va_idx] for p in full_pred_per_model]

    X_tr = build_stacked_features_from_preds(pred_tr_per_model)
    X_va = build_stacked_features_from_preds(pred_va_per_model)
    y_tr = tf.keras.utils.to_categorical(
        df_train.iloc[tr_idx]["label"].values, NUM_CLASSES
    )
    y_va = tf.keras.utils.to_categorical(
        df_train.iloc[va_idx]["label"].values, NUM_CLASSES
    )

    final_model.fit(
        X_tr, y_tr, validation_data=(X_va, y_va), epochs=20, batch_size=128, verbose=1
    )




## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_12/3868490031.py in <cell line: 0>()
     74 ):
     75     full_pred_per_model.append(
---> 76         get_model_preds_for_paths(
     77             m,
     78             train_paths,

/tmp/ipykernel_12/3868490031.py in get_model_preds_for_paths(model, paths, preprocess_fn, batch_size, cache_id, cache_decoded_ds)
     44         pass
     45 
---> 46     return model.predict(ds, verbose=0, steps=steps)
     47 
     48 

/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py in error_handler(*args, **kwargs)
    120             # To get the full stack trace, call:
    121             # `keras.config.disable_traceback_filtering()`
--> 122             raise e.with_traceback(filtered_tb) from None
    123         finally:
    124             del filtered_tb

/usr/local/lib/python3.11/dist-packages/keras/src/layers/input_spec.py in assert_input_compatibility(input_spec, inputs, layer_name)
    243                 if spec_dim is not None and dim is not None:
    244                     if spec_dim != dim:
--> 245                         raise ValueError(
    246                             f'Input {input_index} of layer "{layer_name}" is '
    247                             "incompatible with the layer: "

ValueError: Input 0 of layer "resnet50" is incompatible with the layer: expected shape=(None, 224, 224, 3), found shape=(32, 32, 224, 224, 3)

## === cell 4
print(
    "\nBuilding cached decoded test dataset once (decoded+resized cached on disk, reused across all base models)..."
)
test_paths = df_test["path"].values
test_base_ds = make_base_image_ds_from_paths(
    test_paths, batch_size=BATCH_SIZE, cache=True, cache_id="test_decoded_224_b32"
)

test_pred_per_model = []
for m, pp, name in zip(
    models, preprocess_fns, ["resnet", "mobilenet", "densenet", "effnet"]
):
    preds_test = get_model_preds_for_paths(
        m,
        test_paths,
        pp,
        batch_size=BATCH_SIZE,
        cache_id=f"test_{name}",
        cache_decoded_ds=test_base_ds,
    )
    test_pred_per_model.append(preds_test.astype(np.float32))

x_flat = build_stacked_features_from_preds(test_pred_per_model)
print("Test stacked feature shape:", x_flat.shape)

pred_test = final_model.predict(x_flat, verbose=0)
pred_test_labels = np.argmax(pred_test, axis=-1).astype(int)

final_csv = df_test[["image_id"]].copy()
final_csv["label"] = pred_test_labels
final_csv.to_csv("submission.csv", index=False)

print(final_csv.head())
print("Wrote submission.csv with", len(final_csv), "rows")

## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_12/1267629630.py in <cell line: 0>()
     11     models, preprocess_fns, ["resnet", "mobilenet", "densenet", "effnet"]
     12 ):
---> 13     preds_test = get_model_preds_for_paths(
     14         m,
     15         test_paths,

/tmp/ipykernel_12/3868490031.py in get_model_preds_for_paths(model, paths, preprocess_fn, batch_size, cache_id, cache_decoded_ds)
     44         pass
     45 
---> 46     return model.predict(ds, verbose=0, steps=steps)
     47 
     48 

/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py in error_handler(*args, **kwargs)
    120             # To get the full stack trace, call:
    121             # `keras.config.disable_traceback_filtering()`
--> 122             raise e.with_traceback(filtered_tb) from None
    123         finally:
    124             del filtered_tb

/usr/local/lib/python3.11/dist-packages/keras/src/layers/input_spec.py in assert_input_compatibility(input_spec, inputs, layer_name)
    243                 if spec_dim is not None and dim is not None:
    244                     if spec_dim != dim:
--> 245                         raise ValueError(
    246                             f'Input {input_index} of layer "{layer_name}" is '
    247                             "incompatible with the layer: "

ValueError: Input 0 of layer "resnet50" is incompatible with the layer: expected shape=(None, 224, 224, 3), found shape=(32, 32, 224, 224, 3)
