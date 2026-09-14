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

geopandas==0.14.4
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
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

0.2809

# 6. Current score

0.61435

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.11024) has done: 'I fix the `AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'` by forcing TensorFlow to use the pure-Python protobuf implementation, which is the most reliable workaround for the protobuf 6.x / TF ecosystem mismatch in Kaggle images. Then I fix the data generators so they produce correctly-shaped batches for ResNet50 (explicit target size, categorical labels for training, and no labels/shuffling for test), and I ensure the submission matches `sample_submission.csv` ordering and required columns. These changes keep the same core model/training approach (ResNet50 from scratch, 1 epoch) but make the pipeline run end-to-end and write a valid `submission.csv`.'
- What this solution (achieved 0.10987) has done: 'I fix the TensorFlow/protobuf crash by ensuring the protobuf environment variables are set before TensorFlow import (and by importing `google.protobuf` early to lock in the pure-Python implementation). Then I fix the ResNet50 ImageNet initialization error by keeping the same ResNet50 architecture/training loop but switching to `include_top=False` with a small 5-class classification head, which is the minimal change needed to legally use ImageNet weights. Finally, I ensure inference runs after training and that `submission.csv` is written with exactly the required columns and ordering from `sample_submission.csv`, so you get a valid Kaggle submission file end-to-end.'
- What this solution (achieved 0.61435) has done: 'I fix the protobuf/TensorFlow crash by ensuring the pure-Python protobuf implementation is locked in *before* TensorFlow (and anything that transitively imports compiled protobuf) is imported, and I add a safe fallback to prevent the `MessageFactory.GetPrototype` error on TF 2.18 + protobuf 6.x. Then I make one minimal, score-improving change that preserves your exact training approach (same ResNet50 + 1 epoch loop): freeze the ImageNet backbone during the single epoch so the small head can learn without destroying pretrained features, which should move accuracy upward toward your target. Finally, I keep submission formatting identical but add a small guard to ensure predicted labels map correctly to the 0–4 class indices expected by the competition.'
- What this solution (achieved 0.61435) has done: 'We fix the TensorFlow/protobuf crash by forcing the pure-Python protobuf runtime *before* any TensorFlow import and by adding a safe monkey-patch for the missing `MessageFactory.GetPrototype` method that triggers on protobuf 6.x. This is a correctness/stability fix only and should not materially change your model’s training/inference logic. Then we keep the exact same data pipeline, frozen ResNet50 backbone, 1-epoch training loop, and submission formatting, but ensure everything runs end-to-end and writes a valid `submission.csv` with the required columns and ordering.'
- What this solution (achieved 0.61435) has done: 'We fix the protobuf/TensorFlow crash by forcing the pure-Python protobuf runtime before importing TensorFlow and by adding a more robust monkey-patch that covers both `google.protobuf.message_factory.MessageFactory` and the internal C++-backed factory class used by protobuf 6.x (where the `GetPrototype` attribute error is being raised). This is a stability-only change and should not affect your model/training logic or score behavior. Then we keep your exact data pipeline, frozen ResNet50+head, 1-epoch training, and submission formatting unchanged, ensuring `submission.csv` is always written with the required columns and order.'
- What this solution (achieved 0.61435) has done: 'Your crash happens before any of your monkey-patches can take effect because TensorFlow (via TFDS/metadata/protobuf) triggers a `GetPrototype` call on a *factory instance* that still lacks that method under protobuf 6.x. I fix this by (1) locking the pure-Python protobuf implementation earlier, and (2) patching `GetPrototype` onto both the relevant factory *classes* and the already-created *default factory instance* (`message_factory._DEFAULT_FACTORY`) before importing TensorFlow. This is a stability-only change (no model/data/training logic changes), so your score behavior should remain essentially the same while the notebook runs end-to-end and writes a valid `submission.csv`.'
- What this solution (achieved 0.61435) has done: 'I fix the TensorFlow/protobuf crash that happens before your patch takes effect by forcing pure-Python protobuf earlier and patching both the factory classes and any already-instantiated default factory, plus a last-resort safe wrapper around `GetPrototype` resolution. This is a stability-only change and keeps your ResNet50 (frozen backbone) + 1 epoch training loop and data generators intact, so it should not intentionally improve the score further (your current score is already well above the target). I also keep submission formatting identical, ensuring the output is `submission.csv` with `image_id,label` and the same row order as `sample_submission.csv`. The rest of the pipeline (data loading, training, inference) remains unchanged.'
- What this solution (achieved 0.61435) has done: 'The crash happens before your patch can reliably take effect because protobuf 6.x sometimes triggers `GetPrototype` on additional factory classes/instances (including internal `symbol_database` factories) during TensorFlow import. I strengthen the protobuf compatibility patch by (1) forcing pure-Python protobuf before any TF-related import, and (2) patching `GetPrototype` on all known factory classes plus the default/active factory instances early. This is a stability-only change (no model/data/training changes), so it should keep your current score behavior while making the notebook run end-to-end. I also keep the submission writing exactly as required (`submission.csv` with `image_id,label` in sample order).'
- What this solution (achieved 0.61435) has done: 'The crash happens during TensorFlow import because protobuf 6.x can still call `GetPrototype` on a factory object/class that hasn’t been patched early enough (including `symbol_database`/descriptor pool factories). I strengthen the protobuf compatibility patch by forcing the pure-Python implementation before any TF import and patching `GetPrototype` onto all relevant factory classes/instances (including descriptor-pool related ones) in a more comprehensive way. I not change the model, data generators, training loop, or submission formatting (so score behavior should remain essentially the same and still above your target). The script then run end-to-end and write a valid `submission.csv`.'
- What this solution (achieved 0.61435) has done: 'Your script is crashing during TensorFlow import because protobuf 6.x can still invoke `GetPrototype` on a factory/pool instance that isn’t reliably patched by the current shim. I make the smallest stability fix by (1) forcing the pure-Python protobuf implementation early, and (2) applying a broader, early patch that covers the default `MessageFactory` class, the default factory instance, and the symbol database factory, ensuring `GetPrototype` resolves to `GetMessageClass` everywhere it’s used. I not change your model, training loop (frozen ResNet50 + 1 epoch), data generators, or submission formatting, so score behavior should remain essentially the same (and still above your target). The result run end-to-end and always write a valid `submission.csv` with `image_id,label` in `sample_submission.csv` order.'
- What this solution (achieved 0.61435) has done: 'We fix the TensorFlow/protobuf import crash by applying a broader and earlier compatibility shim that guarantees `GetPrototype` exists on all protobuf factory classes and on already-instantiated default factories/symbol-database factories before importing TensorFlow. This is a stability/correctness-only change and won’t change your model architecture, training loop (frozen ResNet50 + 1 epoch), data pipeline, or submission formatting/ordering, so your score should remain essentially the same (currently already above the target). We also keep the submission write-out exactly as required (`submission.csv` with `image_id,label`) and ensure it always gets created.'
- What this solution (achieved 0.61435) has done: 'We fix the TensorFlow import crash caused by the protobuf 6.x `MessageFactory.GetPrototype` mismatch by applying a broader shim: enforce pure-Python protobuf early and patch `GetPrototype` not only on the factory classes/instances but also via `symbol_database.Default().GetPrototype`, which is where TensorFlow often trips. This is a stability-only change and does not alter your model architecture, training loop, or data pipeline, so your (already above-target) score behavior should remain essentially unchanged. Then we keep the same ResNet50 frozen-backbone 1-epoch training and the same submission formatting/order, ensuring `submission.csv` is always written correctly.'
- What this solution (achieved 0.61435) has done: 'I fix the TensorFlow import crash caused by the protobuf 6.x `MessageFactory.GetPrototype` mismatch by applying a more robust, earlier patch that covers the symbol database’s internal factory/pool and descriptor-pool message-class resolution paths before importing TensorFlow. This is a stability-only change (no model/data/training logic changes), so your score behavior should remain essentially the same (and still above the target). Then I keep your existing ResNet50 frozen-backbone 1-epoch training and the exact same submission formatting/order, ensuring a valid `submission.csv` is always written.'

# 9. Code solution

## === cell 0
import os

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "2")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_DISABLE_CPP", "1")
os.environ.setdefault("PROTOCOL_BUFFERS_DISABLE_CPP_IMPLEMENTATION", "1")

import google.protobuf  # noqa: F401

from google.protobuf import message_factory as _message_factory
from google.protobuf import symbol_database as _symbol_database


def _patch_getprototype_on_obj(obj):
    """
    Ensure obj.GetPrototype exists by delegating to GetMessageClass when available.
    Works for both classes and instances (best-effort).
    """
    if obj is None:
        return
    if hasattr(obj, "GetPrototype"):
        return
    if hasattr(obj, "GetMessageClass"):

        def GetPrototype(self_or_descriptor, maybe_descriptor=None, _obj=obj):
            descriptor = (
                maybe_descriptor if maybe_descriptor is not None else self_or_descriptor
            )
            return _obj.GetMessageClass(descriptor)

        try:
            setattr(obj, "GetPrototype", GetPrototype)
        except Exception:
            pass


def _patch_symbol_database_and_factories():
    """
    Patch symbol_database.Default() and anything it references (factory/pool),
    because TF often triggers GetPrototype via symbol_database/descriptor pool code paths.
    """
    try:
        _sym_db = _symbol_database.Default()
    except Exception:
        return

    if not hasattr(_sym_db, "GetPrototype"):

        def _symdb_GetPrototype(descriptor, _db=_sym_db):
            fac = getattr(_db, "_factory", None) or getattr(_db, "factory", None)
            if fac is not None:
                if hasattr(fac, "GetMessageClass"):
                    return fac.GetMessageClass(descriptor)
                if hasattr(fac, "GetPrototype"):
                    return fac.GetPrototype(descriptor)
            pool = getattr(_db, "pool", None) or getattr(_db, "_pool", None)
            if pool is not None and hasattr(pool, "FindMessageTypeByName"):
                pass
            raise AttributeError(
                "symbol_database has no available GetPrototype/GetMessageClass."
            )

        try:
            setattr(_sym_db, "GetPrototype", _symdb_GetPrototype)
        except Exception:
            pass

    for attr in ("_factory", "factory", "pool", "_pool"):
        _patch_getprototype_on_obj(getattr(_sym_db, attr, None))


_patch_getprototype_on_obj(getattr(_message_factory, "MessageFactory", None))
_patch_getprototype_on_obj(getattr(_message_factory, "_DEFAULT_FACTORY", None))

_patch_symbol_database_and_factories()

try:
    import google.protobuf.pyext._message as _cpp_message  # type: ignore

    _patch_getprototype_on_obj(getattr(_cpp_message, "MessageFactory", None))
except Exception:
    pass

try:
    from google.protobuf import descriptor_pool as _descriptor_pool

    _patch_getprototype_on_obj(getattr(_descriptor_pool, "DescriptorPool", None))
    try:
        _patch_getprototype_on_obj(_descriptor_pool.Default())
    except Exception:
        pass
except Exception:
    pass

import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, Model

tf.keras.utils.set_random_seed(42)

INPUT_DIR = "/kaggle/input/cassava-leaf-disease-classification"



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
train_df = pd.read_csv(os.path.join(INPUT_DIR, "train.csv"))
train_df["label"] = train_df["label"].astype(str)

train_datagen = ImageDataGenerator(rescale=1.0 / 255.0)

train_gen = train_datagen.flow_from_dataframe(
    dataframe=train_df,
    directory=os.path.join(INPUT_DIR, "train_images"),
    x_col="image_id",
    y_col="label",
    target_size=(224, 224),
    class_mode="categorical",
    batch_size=32,
    shuffle=True,
    seed=42,
)



## === cell 2
base = ResNet50(include_top=False, weights="imagenet", input_shape=(224, 224, 3))

base.trainable = False

x = layers.GlobalAveragePooling2D()(base.output)
x = layers.Dense(5, activation="softmax")(x)
model = Model(inputs=base.input, outputs=x)

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
model.fit(train_gen, epochs=1)



## === cell 3
sample_sub = pd.read_csv(os.path.join(INPUT_DIR, "sample_submission.csv"))
test_df = sample_sub.copy()
test_df["label"] = "0"  # dummy placeholder; unused because class_mode=None

test_datagen = ImageDataGenerator(rescale=1.0 / 255.0)

test_gen = test_datagen.flow_from_dataframe(
    dataframe=test_df,
    directory=os.path.join(INPUT_DIR, "test_images"),
    x_col="image_id",
    y_col=None,
    target_size=(224, 224),
    class_mode=None,
    batch_size=32,
    shuffle=False,
)

y_pred = model.predict(test_gen, verbose=1)

pred_labels = np.argmax(y_pred, axis=1).astype(np.int64)
pred_labels = np.clip(pred_labels, 0, 4)

test_df["label"] = pred_labels.astype(int)

sub_path = "submission.csv"
test_df[["image_id", "label"]].to_csv(sub_path, index=False)
print(f"Done. Wrote {sub_path} with {len(test_df)} rows.")
