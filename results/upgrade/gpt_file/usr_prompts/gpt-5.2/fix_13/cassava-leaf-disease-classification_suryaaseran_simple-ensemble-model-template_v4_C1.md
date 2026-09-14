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

0.3715624055605923

# 6. Current score

0.30419

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.31913) has done: 'We need to (1) fix the TensorFlow import crash caused by an incompatible protobuf runtime by pinning the pure-Python protobuf implementation via environment variables set before importing TensorFlow, and (2) remove the hard dependency on an unattached external dataset (`/kaggle/input/cassavamodels`) by adding a minimal fallback model that still produces valid 5-class predictions. To keep core prediction semantics the same when the pretrained `.h5` files exist, we still load and average the two models exactly as before; only when they are missing we build a lightweight Keras model to generate predictions so a valid `submission.csv` is always written. Finally, we keep the same ImageDataGenerator pipeline and ensure output aligns with `sample_submission.csv` ordering and required columns.'
- What this solution (achieved 0.12519) has done: 'We fix the TensorFlow import crash (`MessageFactory.GetPrototype`) by forcing the pure-Python protobuf runtime *and* ensuring TensorFlow is imported only after those environment variables are set, plus providing a safe fallback path that doesn’t require TensorFlow at all if import still fails in this environment. To move accuracy upward toward the target (0.3716) with minimal semantic change, we also remove the weak “randomly-initialized fallback CNN” (which drags score down) and instead use a deterministic, simple heuristic fallback based on image brightness/green-ness computed with PIL (no extra packages), which typically beats random guessing on this dataset. When the two pretrained `.h5` models exist, we keep the exact original behavior: load both, average predictions, argmax, and write `submission.csv`. Finally, we keep submission ordering aligned to `sample_submission.csv` and always write a valid `submission.csv`.'
- What this solution (achieved 0.12519) has done: 'We fix the TensorFlow import crash (`MessageFactory.GetPrototype`) by force-removing any preloaded `google.protobuf` modules and setting the protobuf environment variables *before* attempting TensorFlow import, which is the root cause of the current runtime failure. We also make the TF import block more robust by retrying once after the protobuf cleanup, so the pipeline can use the pretrained `.h5` models when available (highest impact toward your target score) instead of always falling back to the heuristic. Finally, we keep the existing prediction logic (average of two models → argmax) unchanged, preserve the current heuristic fallback, and ensure `submission.csv` is always written with the correct columns and ordering.'
- What this solution (achieved 0.12519) has done: 'We fix the TensorFlow/protobuf crash by forcing the pure-Python protobuf runtime *and* starting the Python process with the required protobuf flags (via `sitecustomize.py`) before TensorFlow is imported, which prevents the `MessageFactory.GetPrototype` failure in Kaggle. We keep your existing two-model averaging logic unchanged when the `.h5` models exist, but also make the TF import more robust by cleaning any preloaded protobuf/tensorflow modules and retrying once. Since your current score (0.12519) is far below the target (0.37156), the main improvement is enabling the pretrained-model path to actually run (instead of falling back to the weak heuristic). Finally, we ensure `submission.csv` is always written with correct columns and in the same order as `sample_submission.csv`.'
- What this solution (achieved 0.12519) has done: 'We fix the TensorFlow/protobuf crash that prevents your pretrained-model path from running by ensuring the protobuf implementation is forced to pure-Python *before any protobuf code is imported*, and by defensively purging any already-imported protobuf modules (including `google.protobuf.*`) plus clearing `protobuf`/`google` caches before retrying TF import. This is a correctness/runtime fix (not a modeling change) and should also move your score up toward the target because it re-enables the original two-model averaging inference instead of the weak heuristic fallback. We also make the TF import block more robust by applying the env vars again immediately before import and by importing protobuf once (pure-python) to “lock” the runtime implementation prior to importing TensorFlow. Submission formatting, ordering, and the two-model averaging logic remain unchanged.'
- What this solution (achieved 0.08595) has done: 'The crash happens before your fallback can run because importing TensorFlow triggers a protobuf API mismatch (`MessageFactory.GetPrototype`), and your current “retry” logic can’t recover from that. I make the TensorFlow import fully optional by catching this specific failure and immediately switching to the non-TF heuristic path, so the notebook always completes and writes a valid `submission.csv`. To move accuracy up toward your target with minimal semantic change, I also strengthen the heuristic slightly by using a few additional simple color statistics (still PIL+NumPy only) and ensuring deterministic, robust handling of corrupt/missing images. No changes are made to your pretrained-model averaging logic when TensorFlow and the `.h5` files are actually available.'
- What this solution (achieved 0.10164) has done: 'I make TensorFlow import truly optional by preventing the protobuf crash from aborting the kernel: we move the TF import attempt into a safe subprocess probe, and only import TF in-process if the probe succeeds (otherwise we immediately use the existing PIL+NumPy heuristic). This directly fixes the current runtime error (`MessageFactory.GetPrototype`) that currently stops execution before a submission is written. To move accuracy upward toward your target with minimal semantic change, I also improve the fallback’s calibration slightly by using per-image color statistics to pick the closest prototype class (still deterministic, PIL+NumPy only, no new dependencies), which should beat the current hand-threshold rules. Submission formatting/order remain exactly aligned to `sample_submission.csv`, and we always write `submission.csv`.'
- What this solution (achieved 0.21001) has done: 'We prevent the protobuf `MessageFactory.GetPrototype` crash from aborting the run by ensuring TensorFlow is never imported in-process unless a subprocess probe succeeds, and by also catching any unexpected exception at the top level of the TF import attempt. This fixes the current runtime error so the notebook always reaches the submission-writing cell. Because your current score (0.10164) is far below the target (0.37156), we also make the non-TF fallback stronger (still deterministic, still PIL+NumPy only) by using a tiny k-NN over per-image color/texture features computed from the *training set* (no leakage) instead of fixed hand prototypes; this typically improves over the current heuristic while keeping the overall pipeline semantics unchanged (produce a single label per test image). The pretrained-model averaging path remains exactly the same when TF and the `.h5` files are available.'
- What this solution (achieved 0.27803) has done: 'I fix the TensorFlow/protobuf crash by ensuring the pure-Python protobuf setting is applied as early as possible and by isolating the TF import probe into a subprocess that does not inherit the current environment’s problematic protobuf state. If TF still can’t be imported safely, the code reliably fall back to the existing non-TF kNN heuristic so it always completes and writes a valid `submission.csv`. To improve score toward your target (current 0.21001 vs target 0.37156), I keep the same heuristic core but modestly strengthen it by (a) increasing the per-class feature-bank cap within the time budget and (b) adding a couple of lightweight texture features (still PIL+NumPy) while keeping the same kNN majority-vote semantics. Submission formatting, ordering, and file path outputs remain unchanged.'
- What this solution (achieved 0.27803) has done: 'I make the TensorFlow import truly non-fatal by moving the TF probe into a subprocess that runs with a clean environment (no inherited `sitecustomize` / protobuf state) and by ensuring we never import TensorFlow in-process unless that probe succeeds. This directly fixes the current hard crash (`MessageFactory` missing `GetPrototype`) so the notebook always reaches the submission-writing code path. I also defensively wrap the TF probe/import with broader exception handling and keep your existing non-TF kNN fallback unchanged (so score should at least match your current 0.278 and may improve only if TF+models become usable). Finally, I keep paths and submission formatting exactly as required and always write `submission.csv`.'
- What this solution (achieved 0.30419) has done: 'Your current score (0.27803) is well below the target (0.37156), so we should improve accuracy with the smallest change that preserves the existing inference semantics. Since your TensorFlow + pretrained-model path is unavailable, the only way to move toward the target is to strengthen the existing non-TF kNN fallback without changing its overall approach (still: build a small train feature bank → kNN vote → labels). I make two minimal, directly-relevant tweaks: (1) use a slightly larger per-class feature bank (more representative neighbors) and (2) switch from unweighted majority vote to distance-weighted voting (still kNN, just better calibrated), keeping everything deterministic and within the time budget. Submission formatting/order and all paths remain unchanged, and it still always write a valid `submission.csv`.'

# 9. Code solution

## === cell 0
import os
import sys
import importlib
import warnings
import subprocess
import textwrap

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION"] = "2"
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

sitecustomize_code = """\
import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION"] = "2"
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
"""
try:
    with open("sitecustomize.py", "w", encoding="utf-8") as f:
        f.write(sitecustomize_code)
except Exception as e:
    print("Warning: could not write sitecustomize.py:", repr(e))

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

SEED = 42
np.random.seed(SEED)

DATA_DIR = "/kaggle/input/cassava-leaf-disease-classification"
SAMPLE_PATH = os.path.join(DATA_DIR, "sample_submission.csv")
TEST_IMG_DIR = os.path.join(DATA_DIR, "test_images")
TRAIN_CSV = os.path.join(DATA_DIR, "train.csv")
TRAIN_IMG_DIR = os.path.join(DATA_DIR, "train_images")

print("Sample path exists:", os.path.exists(SAMPLE_PATH))
print("Test image dir exists:", os.path.isdir(TEST_IMG_DIR))
print("Train csv exists:", os.path.exists(TRAIN_CSV))
print("Train image dir exists:", os.path.isdir(TRAIN_IMG_DIR))

TF_AVAILABLE = False
tf = None
load_model = None
ImageDataGenerator = None


def _purge_modules(prefixes):
    """Remove already-imported modules that can pin an incompatible protobuf runtime."""
    for mod in list(sys.modules.keys()):
        if any(mod == p or mod.startswith(p + ".") for p in prefixes):
            del sys.modules[mod]


def _tf_subprocess_probe():
    """
    Probe TF import in a fresh subprocess so a protobuf crash cannot kill this process.

    Bugfix: run Python with -S and -s to avoid importing site/sitecustomize/user-site which can
    pre-import protobuf in a bad state and trigger MessageFactory.GetPrototype issues.
    """
    code = textwrap.dedent(
        """
        import os
        os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
        os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION"] = "2"
        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
        import tensorflow as tf
        print(tf.__version__)
        """
    ).strip()
    try:
        env = os.environ.copy()
        env["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
        env["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION"] = "2"
        env.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

        res = subprocess.run(
            [sys.executable, "-S", "-s", "-c", code],
            capture_output=True,
            text=True,
            check=False,
            timeout=60,
            env=env,
        )
        if res.returncode == 0:
            return True, (res.stdout.strip() or "ok")
        return False, (
            res.stderr.strip() or res.stdout.strip() or f"rc={res.returncode}"
        )
    except Exception as e:
        return False, repr(e)


def try_import_tf():
    """
    Import TF only if a subprocess probe confirms it can import cleanly.
    This prevents protobuf AttributeError from aborting the whole run.
    """
    global TF_AVAILABLE, tf, load_model, ImageDataGenerator

    os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
    os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION"] = "2"
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

    ok, msg = _tf_subprocess_probe()
    if not ok:
        TF_AVAILABLE = False
        print("TensorFlow probe failed; will fall back to non-TF inference.")
        print("TF probe error:", msg)
        return False

    _purge_modules(
        ["google", "google.protobuf", "protobuf", "tensorflow", "tensorboard"]
    )
    importlib.invalidate_caches()

    try:
        import tensorflow as tf_  # noqa: F401
        from tensorflow.keras.models import load_model as load_model_  # noqa: F401
        from tensorflow.keras.preprocessing.image import (
            ImageDataGenerator as ImageDataGenerator_,
        )  # noqa: F401

        tf = tf_
        load_model = load_model_
        ImageDataGenerator = ImageDataGenerator_
        TF_AVAILABLE = True
        try:
            tf.random.set_seed(SEED)
        except Exception:
            pass
        print("TensorFlow imported successfully:", tf.__version__)
        return True
    except Exception as e:
        TF_AVAILABLE = False
        print(
            "TensorFlow import failed in-process; will fall back to non-TF inference."
        )
        print("TF import error:", repr(e))
        return False


try:
    ok = try_import_tf()
except Exception as e:
    ok = False
    TF_AVAILABLE = False
    print("TensorFlow import attempt raised; will fall back to non-TF inference.")
    print("TF import raised:", repr(e))

if not ok:
    print("Will use non-TF fallback inference.")



## === cell 1
MODEL_DIR = "/kaggle/input/cassavamodels"
m1_path = os.path.join(MODEL_DIR, "model (1).h5")
m2_path = os.path.join(MODEL_DIR, "model (2).h5")

models_available = os.path.exists(m1_path) and os.path.exists(m2_path) and TF_AVAILABLE
print("Pretrained models available (and TF import ok):", models_available)
print("Expected:", m1_path, "(exists=", os.path.exists(m1_path), ")", sep="")
print("Expected:", m2_path, "(exists=", os.path.exists(m2_path), ")", sep="")

inceptionres = None
res50 = None

if models_available:
    inceptionres = load_model(m1_path, compile=False)
    res50 = load_model(m2_path, compile=False)
    print("Loaded models:", os.path.basename(m1_path), "and", os.path.basename(m2_path))
else:
    print("Will use heuristic fallback (no pretrained .h5 and/or TF unavailable).")



## === cell 2
sample = pd.read_csv(SAMPLE_PATH)
print(sample.head())
print("Rows:", len(sample), "Cols:", list(sample.columns))

if "image_id" not in sample.columns:
    raise ValueError("sample_submission.csv must contain an 'image_id' column")



## === cell 3
submission = None

if models_available:
    TARGET_SIZE = 512

    test_datagen = ImageDataGenerator(rescale=1.0 / 255.0)

    test_generator = test_datagen.flow_from_dataframe(
        dataframe=sample,
        directory=TEST_IMG_DIR,
        x_col="image_id",
        y_col=None,
        target_size=(TARGET_SIZE, TARGET_SIZE),
        class_mode=None,
        batch_size=16,
        shuffle=False,
    )

    pred1 = inceptionres.predict(test_generator, verbose=1)
    pred2 = res50.predict(test_generator, verbose=1)

    print("pred1 shape:", pred1.shape, "pred2 shape:", pred2.shape)

    if pred1.shape != pred2.shape:
        raise ValueError(
            f"Model outputs have different shapes: {pred1.shape} vs {pred2.shape}"
        )

    preds = (pred1 + pred2) / 2.0
    final = np.argmax(preds, axis=1).astype(int)

    if len(final) != len(sample):
        raise ValueError(
            f"Prediction length mismatch: len(final)={len(final)} vs len(sample)={len(sample)}"
        )

    submission = pd.DataFrame({"image_id": sample["image_id"].values, "label": final})

else:
    from PIL import Image

    def extract_feats(img: Image.Image) -> np.ndarray:
        im = img.convert("RGB").resize((192, 192))
        arr = np.asarray(im, dtype=np.float32) / 255.0

        mean_rgb = arr.mean(axis=(0, 1))
        r, g, b = mean_rgb.tolist()
        brightness = (r + g + b) / 3.0

        std_rgb = arr.std(axis=(0, 1))
        sr, sg, sb = std_rgb.tolist()
        colorfulness = (sr + sg + sb) / 3.0

        green_excess = g - (r + b) / 2.0
        yellowish = (r + g) / 2.0 - b
        rb_diff = abs(r - b)

        max_rgb = arr.max(axis=(0, 1))
        min_rgb = arr.min(axis=(0, 1))
        contrast = float(((max_rgb - min_rgb).mean()))

        gray = (
            0.2989 * arr[..., 0] + 0.5870 * arr[..., 1] + 0.1140 * arr[..., 2]
        ).astype(np.float32)
        dx = gray[:, 1:] - gray[:, :-1]
        dy = gray[1:, :] - gray[:-1, :]
        edge_energy = float((dx * dx).mean() + (dy * dy).mean())
        texture_std = float(gray.std())

        return np.array(
            [
                brightness,
                green_excess,
                yellowish,
                colorfulness,
                rb_diff,
                contrast,
                edge_energy,
                texture_std,
            ],
            dtype=np.float32,
        )

    train_df = pd.read_csv(TRAIN_CSV)
    if not {"image_id", "label"}.issubset(train_df.columns):
        raise ValueError("train.csv must contain 'image_id' and 'label' columns")

    rng = np.random.RandomState(SEED)

    per_class_cap = 900

    feats_list = []
    y_list = []

    for cls in sorted(train_df["label"].unique().tolist()):
        cls_df = train_df[train_df["label"] == cls]
        take_n = min(per_class_cap, len(cls_df))
        idx = rng.choice(len(cls_df), size=take_n, replace=False)
        subset = cls_df.iloc[idx]

        for fname in subset["image_id"].values:
            fpath = os.path.join(TRAIN_IMG_DIR, fname)
            try:
                with Image.open(fpath) as img:
                    feats_list.append(extract_feats(img))
                    y_list.append(int(cls))
            except Exception:
                continue

    if len(feats_list) == 0:
        print(
            "Warning: could not build training feature bank; falling back to constant label 0."
        )
        labels = np.zeros(len(sample), dtype=int)
        submission = pd.DataFrame(
            {"image_id": sample["image_id"].values, "label": labels}
        )
    else:
        Xtr = np.stack(feats_list, axis=0).astype(np.float32)
        ytr = np.array(y_list, dtype=int)

        mu = Xtr.mean(axis=0, keepdims=True)
        sigma = Xtr.std(axis=0, keepdims=True) + 1e-6
        Xtrn = (Xtr - mu) / sigma

        K = 11
        labels = []
        missing = 0

        eps = 1e-8

        for fname in sample["image_id"].values:
            fpath = os.path.join(TEST_IMG_DIR, fname)
            try:
                with Image.open(fpath) as img:
                    x = extract_feats(img).reshape(1, -1).astype(np.float32)
                xn = (x - mu) / sigma
                d2 = np.sum((Xtrn - xn) ** 2, axis=1)
                nn = np.argpartition(d2, K - 1)[:K]
                votes = ytr[nn]
                w = 1.0 / (d2[nn] + eps)

                scores = np.zeros(5, dtype=np.float64)
                for cls, ww in zip(votes, w):
                    scores[int(cls)] += float(ww)
                pred = int(scores.argmax())
                labels.append(pred)
            except Exception:
                labels.append(0)
                missing += 1

        if missing:
            print(
                "Warning: failed to open",
                missing,
                "images; defaulted their labels to 0.",
            )

        submission = pd.DataFrame(
            {
                "image_id": sample["image_id"].values,
                "label": np.array(labels, dtype=int),
            }
        )

submission.to_csv("submission.csv", index=False)
print(submission.head())
print("Wrote submission.csv with", len(submission), "rows")



## === cell 4
print(submission.tail())
print("Submission columns:", list(submission.columns))
print("submission.csv exists:", os.path.exists("submission.csv"))
assert list(submission.columns) == ["image_id", "label"]
assert len(submission) == len(sample)
assert submission["label"].dtype.kind in ("i", "u")
