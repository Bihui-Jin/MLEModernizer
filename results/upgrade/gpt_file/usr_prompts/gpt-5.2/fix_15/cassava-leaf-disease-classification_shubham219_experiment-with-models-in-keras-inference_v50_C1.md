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

0.6177092777274101

# 6. Current score

0.15845

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.17339) has done: 'I remove the incompatible `tensorflow_hub` import that is triggering the protobuf `MessageFactory.GetPrototype` crash, since it’s not used by your pipeline. I also replace the notebook `!pip install` cells with a safe in-notebook install fallback and, more importantly, add a robust import/load path: if the external `efficientnet` package or the provided weight file is unavailable, we fall back to a standard `tf.keras.applications.EfficientNetB4` model so `my_model` is always defined. Finally, I ensure the test image path is correct for this environment, fix generator sizing, and write a valid `submission.csv` with `image_id,label` columns.'
- What this solution (achieved 0.0938) has done: 'We need to fix the TensorFlow import crash happening immediately in cell 0 (`MessageFactory` protobuf incompatibility). The most reliable minimal fix in Kaggle is to force the pure-Python protobuf implementation before importing TensorFlow, which avoids the failing C++/upb path. After that, we keep your existing inference-only pipeline intact, but improve score toward the 0.617 target by ensuring the intended pretrained `.h5` model loads when available and, if it doesn’t, compiling the fallback model and loading ImageNet weights correctly (otherwise predictions are effectively random, matching your very low 0.173 score). Finally, we keep the submission formatting checks and guarantee `submission.csv` is written.'
- What this solution (achieved 0.05531) has done: 'We fix the immediate TensorFlow import crash (`MessageFactory.GetPrototype`) by forcing TensorFlow to use the pure-Python protobuf runtime *before* TensorFlow is imported, and by explicitly setting `TF_USE_LEGACY_KERAS=1` (a common Kaggle TF compatibility requirement) plus avoiding any TF/keras imports until after those env vars are set. Then we keep your inference-only pipeline intact, but ensure the fallback model’s head is not random (which is why accuracy is near chance): if the custom `.h5` isn’t available, we load the base EfficientNetB4 ImageNet weights and use the standard EfficientNet preprocessing; this preserves the architecture while making predictions meaningful. Finally, we keep the submission alignment with `sample_submission.csv` and guarantee `submission.csv` is created with the exact required columns and row count.'
- What this solution (achieved 0.17377) has done: 'We fix the TensorFlow import crash (`MessageFactory.GetPrototype`) by setting additional protobuf-related environment variables *before* importing TensorFlow, which is the root cause preventing the pipeline from running at all. Since your current score (0.05531) is far below target, we also remove the “zeros-initialized” random/untrained classification head in the fallback model and instead use the standard ImageNet-pretrained EfficientNetB4 top classifier (1000 classes) and map its predictions deterministically into 5 labels; this is a minimal change that makes fallback predictions non-degenerate without altering your inference-only flow. We keep all paths and submission formatting logic intact and still write `submission.csv` with the required columns and row count. Finally, we add a tiny safety fallback for preprocessing to ensure it matches the actual model used.'
- What this solution (achieved 0.31203) has done: 'We fix the TensorFlow import crash (`MessageFactory` / protobuf mismatch) by setting the protobuf/TF environment variables *as early as possible* and adding a safe fallback to force the pure-Python protobuf backend before importing TensorFlow. Then we remove the currently-harmful “ImageNet-top then mod-5” mapping (it produces near-random labels and matches your low score) by ensuring the intended 5-class `.h5` model loads when available, and otherwise building a 5-class EfficientNetB4 head (same backbone) so predictions are at least aligned to the cassava label space. Finally, we make preprocessing conditional on which EfficientNet implementation is actually used (efficientnet.tfkeras vs tf.keras.applications) to avoid silent input scaling mismatches, and keep the exact required `submission.csv` format and ordering.'
- What this solution (achieved 0.11921) has done: 'The TensorFlow import is failing because the script uninstalls `protobuf`, which removes `google.protobuf` that TensorFlow requires; I stop uninstalling protobuf and instead (optionally) reinstall a compatible protobuf if it’s missing. I also make the TF import more robust by ensuring the env vars are set before import and by attempting a lightweight `pip install protobuf` only when `google.protobuf` cannot be imported. These changes are execution-unblocking and score-neutral; they preserve your exact model/inference logic and keep the same submission formatting/paths so a valid `submission.csv` is always written. Once TF imports, the rest of your pipeline (loading the `.h5` if present, else EfficientNetB4 fallback) run end-to-end.'
- What this solution (achieved 0.08931) has done: 'We fix the TensorFlow import crash caused by the protobuf runtime mismatch (`MessageFactory.GetPrototype`) by forcing the pure-Python protobuf implementation early and (critically) monkey-patching `MessageFactory.GetPrototype` to `GetMessageClass` when missing, which is a compatibility shim that unblocks TF import without changing your model/inference logic. We also stop uninstalling packages at runtime (uninstalling `tensorflow-hub` is unnecessary here and can destabilize the environment), keeping the rest of your pipeline intact. After TF imports successfully, everything else remains the same: load the provided `.h5` model if available, otherwise build the EfficientNetB4 fallback, run prediction, and write a valid `submission.csv` with the required columns and ordering. These changes are execution-unblocking and should improve score versus the current near-random output by ensuring the intended model actually runs.'
- What this solution (achieved 0.16442) has done: 'The immediate blocker is that the protobuf shim is patching the `MessageFactory` *class* but TensorFlow triggers the error on the *instance* method lookup, so the patch must be applied to the `google.protobuf.message_factory.MessageFactory` module-level class (and safely on both class/instance) before importing TensorFlow. I adjust the patch to robustly add `GetPrototype` whenever it’s missing, and also set `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` as early as possible (already done) without uninstalling anything. These changes are execution-unblocking and score-neutral; the rest of your model loading/inference/submission logic remains unchanged. Once TF imports, the same `.h5` loading (or EfficientNetB4 fallback) and CSV writing run end-to-end.'
- What this solution (achieved 0.19768) has done: 'I fix the TensorFlow/protobuf crash by patching the *instance-level* `MessageFactory` (not just the class) so `GetPrototype` exists when TensorFlow calls it, and I apply this patch before any TensorFlow import attempts. I also make the TF import retry path consistent by fully clearing `google.protobuf` modules before re-importing and re-patching, which prevents the stale unpatched factory instance from persisting. These changes are execution-unblocking and score-neutral (they don’t change your model/inference logic). Once TF imports, the rest of your existing pipeline (model load/fallback, generator, predict, and `submission.csv`) runs end-to-end as-is.'
- What this solution (achieved 0.12631) has done: 'The crash happens before any training/inference because TensorFlow imports a protobuf `MessageFactory` instance that lacks `GetPrototype`; your current patch only handles a few cases and can miss the specific factory object TF touches. I make the protobuf shim more robust by patching `google.protobuf.message_factory.MessageFactory.GetPrototype` and also patching the concrete `message_factory._GENERATED_MESSAGE_FACTORY` instance when present, and I apply this before (and on retry before) importing TensorFlow. These changes are execution-unblocking and score-neutral; once TF imports, your existing model-load/fallback, preprocessing, prediction, and submission-writing logic remains the same. After this, the script should run end-to-end and produce `submission.csv` correctly.'
- What this solution (achieved 0.48655) has done: 'The pipeline is currently blocked at TensorFlow import due to a protobuf `MessageFactory.GetPrototype` mismatch; your patch doesn’t cover the exact factory instance TensorFlow touches early in import. I make the protobuf shim robust by (1) forcing pure-Python protobuf before any TF import, and (2) patching both the `MessageFactory` class and any existing singleton/instances (including `_DEFAULT_FACTORY`-style objects) by adding a compatible `GetPrototype` method wrapper. This is execution-unblocking and score-neutral, but it also improve score in practice because it lets the intended pretrained `.h5` (or at least a meaningful EfficientNet fallback) actually run instead of failing. Everything else (model selection logic, preprocessing, prediction, and submission formatting) is kept the same.'
- What this solution (achieved 0.09791) has done: 'The crash happens during TensorFlow import because TF touches a specific protobuf `MessageFactory` singleton that still lacks `GetPrototype`; your current patch misses the right object. I make the protobuf shim more robust by patching both the class and the module-level singleton factories that TF commonly uses (`_GENERATED_MESSAGE_FACTORY`, `_DEFAULT_FACTORY`, and `symbol_database.Default()`’s factory) before importing TensorFlow, and re-apply the patch on the retry path as well. This is execution-unblocking and score-neutral (it doesn’t change your model/prediction logic) but is necessary so the pipeline can run end-to-end. I also keep the rest of your inference/submission logic unchanged so your score should at least match (and likely improve from) the current 0.48655 once the model can actually run.'
- What this solution (achieved 0.15845) has done: 'We fix the TensorFlow import crash by applying the protobuf `GetPrototype` shim earlier and more comprehensively: patch the class, patch known singleton instances, and (crucially) patch the C++ `message_factory` module instances that TensorFlow may actually use. We also add a final retry that forces the pure-Python protobuf backend via `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` and clears protobuf/TF modules before re-importing. These changes are execution-unblocking and score-improving because your current score indicates you’re effectively not running the intended model/inference reliably. The rest of the pipeline (model loading/fallback, preprocessing, prediction, and submission formatting) is kept the same.'

# 9. Code solution

## === cell 0
import os

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "3")
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

import glob
import numpy as np
import pandas as pd

SEED = 42
DEBUG = False

os.environ["PYTHONHASHSEED"] = str(SEED)
np.random.seed(SEED)



## === cell 1
import sys
import subprocess
import importlib


def _ensure_protobuf_importable():
    """
    Bugfix: ensure google.protobuf can be imported (but do NOT uninstall anything).
    """
    try:
        import google.protobuf  # noqa: F401

        return True
    except Exception as e:
        print(
            "google.protobuf not importable, attempting to install protobuf:", repr(e)
        )
        try:
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "-q", "protobuf"]
            )
            importlib.invalidate_caches()
            import google.protobuf  # noqa: F401

            print("protobuf installed and importable.")
            return True
        except Exception as e2:
            print("Warning: protobuf install failed; continuing:", repr(e2))
            return False


def _add_getprototype_to_class(cls):
    """
    Add GetPrototype to a MessageFactory-like class if missing, mapping to GetMessageClass.
    """
    try:
        if cls is None or hasattr(cls, "GetPrototype"):
            return False
        if hasattr(cls, "GetMessageClass") and callable(
            getattr(cls, "GetMessageClass")
        ):

            def _class_getproto(self, descriptor):
                return self.GetMessageClass(descriptor)

            setattr(cls, "GetPrototype", _class_getproto)
            return True
        return False
    except Exception:
        return False


def _patch_one_factory_instance(inst):
    """
    Adds a GetPrototype method to a MessageFactory-like instance if it's missing.
    Uses a wrapper (not just aliasing) to handle signature differences safely.
    """
    try:
        if inst is None or hasattr(inst, "GetPrototype"):
            return False

        cls = inst.__class__
        changed = _add_getprototype_to_class(cls)

        if hasattr(inst, "GetMessageClass") and callable(
            getattr(inst, "GetMessageClass")
        ):

            def _getproto(self, descriptor):
                return self.GetMessageClass(descriptor)

            try:
                setattr(inst, "GetPrototype", _getproto.__get__(inst, cls))
                changed = True
            except Exception:
                pass

        return changed
    except Exception:
        return False


def _patch_protobuf_messagefactory():
    """
    Bugfix: robustly patch protobuf MessageFactory to provide GetPrototype.
    TensorFlow may hit different MessageFactory singletons depending on protobuf version.
    This must run BEFORE importing tensorflow.

    Key fix vs previous version: also patch google.protobuf.pyext.cpp_message_factory
    (C++ implementation) where TensorFlow often obtains its MessageFactory instance.
    """
    patched_any = False

    try:
        import google.protobuf.message_factory as mf  # type: ignore

        MF = getattr(mf, "MessageFactory", None)
        if _add_getprototype_to_class(MF):
            patched_any = True
            print(
                "Patched protobuf: message_factory.MessageFactory.GetPrototype (class wrapper)"
            )

        candidate_names = [
            "_GENERATED_MESSAGE_FACTORY",
            "_DEFAULT_FACTORY",
            "default_factory",
            "message_factory",
        ]
        for name in candidate_names:
            inst = getattr(mf, name, None)
            if _patch_one_factory_instance(inst):
                patched_any = True
                print(
                    f"Patched protobuf: message_factory.{name}.GetPrototype (instance wrapper)"
                )

        try:
            import google.protobuf.symbol_database as _sym_db  # type: ignore

            db = _sym_db.Default()
            for attr in ("_factory", "pool", "message_factory", "_message_factory"):
                inst = getattr(db, attr, None)
                if _patch_one_factory_instance(inst):
                    patched_any = True
                    print(
                        f"Patched protobuf: symbol_database.Default().{attr}.GetPrototype"
                    )
        except Exception as e:
            if DEBUG:
                print("symbol_database patch skipped:", repr(e))

        try:
            if MF is not None:
                for name, val in list(mf.__dict__.items()):
                    if val is None:
                        continue
                    try:
                        if isinstance(val, MF) and _patch_one_factory_instance(val):
                            patched_any = True
                            print(
                                f"Patched protobuf: scanned message_factory.{name}.GetPrototype"
                            )
                    except Exception:
                        continue
        except Exception:
            pass

    except Exception as e:
        print("Warning: could not patch google.protobuf.message_factory:", repr(e))

    try:
        from google.protobuf.pyext import cpp_message_factory as cmf  # type: ignore

        for name in (
            "_GENERATED_MESSAGE_FACTORY",
            "default_factory",
            "_DEFAULT_FACTORY",
        ):
            inst = getattr(cmf, name, None)
            if _patch_one_factory_instance(inst):
                patched_any = True
                print(
                    f"Patched protobuf: cpp_message_factory.{name}.GetPrototype (instance wrapper)"
                )

        CMF = getattr(cmf, "MessageFactory", None)
        if _add_getprototype_to_class(CMF):
            patched_any = True
            print(
                "Patched protobuf: cpp_message_factory.MessageFactory.GetPrototype (class wrapper)"
            )
    except Exception as e:
        if DEBUG:
            print("cpp_message_factory patch skipped:", repr(e))

    return patched_any


_ensure_protobuf_importable()
_patch_protobuf_messagefactory()




## === cell 2
def _import_tensorflow_with_retries():
    """
    Bugfix: TF import can fail depending on which protobuf backend is picked up.
    We retry after clearing modules and re-applying patches.
    """
    last_err = None
    for attempt in range(3):
        try:
            import tensorflow as tf  # noqa: F401

            return tf
        except Exception as e:
            last_err = e
            print(f"TensorFlow import failed (attempt {attempt+1}/3):", repr(e))

            os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
            os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION"] = "3"
            os.environ["TF_USE_LEGACY_KERAS"] = "1"

            for m in list(sys.modules.keys()):
                if m.startswith(("tensorflow", "google.protobuf", "protobuf", "keras")):
                    sys.modules.pop(m, None)
            importlib.invalidate_caches()

            _ensure_protobuf_importable()
            _patch_protobuf_messagefactory()

    raise RuntimeError(f"TensorFlow import failed after retries: {repr(last_err)}")


tf = _import_tensorflow_with_retries()
tf.random.set_seed(SEED)

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

print("TF version:", tf.__version__)




## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 3
def _maybe_pip_install(wheel_path: str):
    """
    Keep original behavior: optionally install provided wheels if present.
    """
    if os.path.exists(wheel_path):
        try:
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "-q", wheel_path]
            )
            print(f"Installed: {wheel_path}")
        except Exception as e:
            print(f"Warning: failed to install {wheel_path}: {e}")
    else:
        print(f"Wheel not found (skipping): {wheel_path}")


_maybe_pip_install(
    "/kaggle/input/kerasapplication/Keras_Applications-1.0.8-py3-none-any.whl"
)
_maybe_pip_install("/kaggle/input/efficientnet/efficientnet-1.1.1-py3-none-any.whl")



## === cell 4
my_model = None
USING_EFFICIENTNET_TFKERAS = False  # controls preprocessing choice

try:
    from efficientnet.tfkeras import EfficientNetB4 as EffNetB4_tfk  # noqa: F401

    USING_EFFICIENTNET_TFKERAS = True
    print("Imported efficientnet.tfkeras")
except Exception as e:
    print("Warning: could not import efficientnet.tfkeras:", repr(e))

candidate_weight_paths = [
    "../input/experiment-with-models-using-keras-with-updates/effnetB4_v0.25.h5",
    "/kaggle/input/experiment-with-models-using-keras-with-updates/effnetB4_v0.25.h5",
]

weight_path = None
for p in candidate_weight_paths:
    if os.path.exists(p):
        weight_path = p
        break

if weight_path is not None:
    try:
        my_model = load_model(weight_path, compile=False)
        print("Loaded model from:", weight_path)
    except Exception as e1:
        print("Warning: first load_model failed:", repr(e1))
        try:
            custom_objects = {}
            if USING_EFFICIENTNET_TFKERAS:
                from efficientnet.tfkeras import EfficientNetB4 as EffB4_custom

                custom_objects["EfficientNetB4"] = EffB4_custom
            my_model = load_model(
                weight_path, compile=False, custom_objects=custom_objects
            )
            print("Loaded model from with custom_objects:", weight_path)
        except Exception as e2:
            print(
                "Warning: failed to load .h5 model, will fall back to base model:",
                repr(e2),
            )
            my_model = None

if my_model is None:
    from tensorflow.keras import Model, layers

    if USING_EFFICIENTNET_TFKERAS:
        from efficientnet.tfkeras import EfficientNetB4

        base = EfficientNetB4(
            include_top=False, weights="imagenet", input_shape=(300, 300, 3)
        )
    else:
        from tensorflow.keras.applications import EfficientNetB4

        base = EfficientNetB4(
            include_top=False, weights="imagenet", input_shape=(300, 300, 3)
        )

    x = layers.GlobalAveragePooling2D()(base.output)
    x = layers.Dropout(0.2)(x)
    out = layers.Dense(5, activation="softmax")(x)
    my_model = Model(inputs=base.input, outputs=out)
    print(
        "Built fallback EfficientNetB4(include_top=False, ImageNet weights) + 5-class head model."
    )



## === cell 5
test_globs = [
    "../input/cassava-leaf-disease-classification/test_images/*.jpg",
    "/kaggle/input/cassava-leaf-disease-classification/test_images/*.jpg",
    "/kaggle/input/cassava-leaf-disease-classification/cassava-leaf-disease-classification/test_images/*.jpg",
]
test_images = []
for g in test_globs:
    test_images = glob.glob(g)
    if len(test_images) > 0:
        print("Using test glob:", g, "count:", len(test_images))
        break

if len(test_images) == 0:
    raise FileNotFoundError(
        "Could not find any test images under expected /kaggle/input paths."
    )

df_test = pd.DataFrame(test_images, columns=["path"])


def make_test_gen(batch_size=64):
    if USING_EFFICIENTNET_TFKERAS:
        from efficientnet.tfkeras import preprocess_input
    else:
        from tensorflow.keras.applications.efficientnet import preprocess_input

    my_test_idg = ImageDataGenerator(preprocessing_function=preprocess_input)
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




## === cell 6
test_gen = make_test_gen(batch_size=128)
steps = int(np.ceil(test_gen.n / test_gen.batch_size))

pred_test = my_model.predict(test_gen, steps=steps, verbose=1)
pred_test_labels = np.argmax(pred_test, axis=-1).astype(int)

final_submission = df_test.copy()
final_submission["image_id"] = final_submission["path"].str.split("/").str[-1]
final_submission["label"] = pred_test_labels

final_csv = final_submission[["image_id", "label"]]

sample_path_candidates = [
    "../input/cassava-leaf-disease-classification/sample_submission.csv",
    "/kaggle/input/cassava-leaf-disease-classification/sample_submission.csv",
    "/kaggle/input/cassava-leaf-disease-classification/cassava-leaf-disease-classification/sample_submission.csv",
]
sample_path = None
for p in sample_path_candidates:
    if os.path.exists(p):
        sample_path = p
        break

if sample_path is not None:
    sample_sub = pd.read_csv(sample_path)
    final_csv = sample_sub[["image_id"]].merge(final_csv, on="image_id", how="left")
    if final_csv["label"].isna().any():
        fill_label = int(pd.Series(pred_test_labels).mode().iloc[0])
        final_csv["label"] = final_csv["label"].fillna(fill_label).astype(int)

final_csv["label"] = final_csv["label"].astype(int)
final_csv.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", final_csv.shape)
print(final_csv.head())



## === cell 7
assert os.path.exists("submission.csv"), "submission.csv was not created."
check = pd.read_csv("submission.csv")
assert list(check.columns) == [
    "image_id",
    "label",
], f"Bad columns: {check.columns.tolist()}"
assert len(check) == len(
    df_test
), f"Row count mismatch: submission={len(check)} test={len(df_test)}"
assert check["label"].between(0, 4).all(), "Labels out of range 0-4."
check.head()
