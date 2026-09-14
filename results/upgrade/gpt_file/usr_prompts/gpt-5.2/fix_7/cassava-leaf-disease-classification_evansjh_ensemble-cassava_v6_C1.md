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

0.8924146267754609

# 6. Current score

0.61099

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.61099) has done: 'I fix the import-time TensorFlow/protobuf crash by switching to the Kaggle-stable `tf.keras` import path and forcing the pure-Python protobuf implementation early, which avoids the `MessageFactory.GetPrototype` error. Then I make model loading robust: only load models whose files actually exist and fall back to a simple majority-class submission if none are available, so a valid `submission.csv` is always produced. I also ensure predictions cover exactly the images from `sample_submission.csv` (correct order/coverage), preventing missing/extra rows that can invalidate submissions. These changes preserve your ensemble-voting core logic when models exist, while making the notebook run end-to-end reliably.'
- What this solution (achieved 0.61099) has done: 'I fix the TensorFlow/protobuf crash that happens at import time by switching the protobuf implementation to the one TensorFlow expects in Kaggle and setting the required env vars before importing TensorFlow. Then I make model discovery robust by automatically loading any available `.h5` models from the provided `/kaggle/input/...` locations (instead of only three hardcoded paths), which preserves your ensemble-voting logic but increases the chance you actually ensemble multiple strong models and improve accuracy toward the target. Finally, I keep the prediction loop and submission formatting the same while ensuring all paths exist and a valid `submission.csv` is always written.'
- What this solution (achieved 0.61099) has done: 'You’re failing at the very first step because TensorFlow can’t import due to a protobuf C-extension mismatch; setting `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION="cpp"` makes it worse in this environment. I fix this by forcing the pure-Python protobuf backend *before* importing TensorFlow and by adding safe fallbacks if TensorFlow is unavailable so the pipeline still writes a valid `submission.csv`. To improve score toward your target (and keep core logic intact), I keep your existing ensemble voting when models load, but add a minimal “soft voting” (average probabilities then argmax) which is typically more accurate than majority vote and still the same ensemble semantics. I also speed up inference safely by batching per-model predictions (same outputs, fewer overheads), helping stay within the time limit.'
- What this solution (achieved 0.61099) has done: 'I fix the TensorFlow/protobuf import crash by setting the protobuf environment variables in the safest way for Kaggle (and doing it before any TensorFlow-related imports), then importing TensorFlow via `tf.keras` only. I also make the inference loop robust to missing test images (so they don’t silently become all-zero inputs that skew predictions) by only predicting on successfully loaded images and filling any missing ones with a fallback label. These changes keep your ensemble “average probabilities then argmax” core logic intact, but ensure the notebook runs end-to-end and that the submission is valid and aligned to `sample_submission.csv`. With TensorFlow reliably importing and models loading, the score should move up toward your target because you actually be using the provided trained models rather than falling back.'
- What this solution (achieved 0.61099) has done: 'The immediate blocker is the TensorFlow import crash caused by an incompatible protobuf runtime; the current env var setup forces the Python protobuf backend but still hits `MessageFactory.GetPrototype`. I fix this by setting the safer protobuf env configuration *before* importing TensorFlow and by explicitly disabling TF’s use of the C++ protobuf implementation, which is the typical root cause of this exact attribute error on Kaggle. Then, to improve accuracy toward your target while preserving the same ensemble “average probabilities then argmax” logic, I ensure inputs are fed only for images that actually loaded (so unloaded images don’t contribute zeros that distort soft-voting) and I apply a minimal, standard per-model input normalization fallback if the model expects `[-1,1]` scaling. The script still always write `/kaggle/working/submission.csv` with the required columns and row order.'
- What this solution (achieved 0.61099) has done: 'You’re blocked by a TensorFlow↔protobuf incompatibility that crashes at import time (`MessageFactory.GetPrototype`), so the first fix is to force the pure-Python protobuf backend *and* prevent TensorFlow from trying to use the C++ protobuf implementation before importing TF. Next, because the current 0.61099 indicates you’re often falling back (no models loaded), I keep your exact soft-voting ensemble logic but broaden model discovery to load any `.h5` models found under `/kaggle/input/` (still only using models that actually exist), which should legitimately raise accuracy toward your target. Finally, I keep the submission alignment to `sample_submission.csv` and add small robustness checks so predictions always cover every test image and produce a valid `/kaggle/working/submission.csv`.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "3")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_PROTOBUF_DISABLE_CPP_IMPLEMENTATION", "1")

import pandas as pd
import numpy as np

TF_AVAILABLE = True
TF_IMPORT_ERROR = None
try:
    import tensorflow as tf
    from tensorflow.keras.models import load_model
    from tensorflow.keras.preprocessing.image import load_img, img_to_array
except Exception as e:
    TF_AVAILABLE = False
    TF_IMPORT_ERROR = repr(e)

print("Python:", os.sys.version)
print("TensorFlow available:", TF_AVAILABLE)
if TF_AVAILABLE:
    print("TensorFlow:", tf.__version__)
else:
    print("TensorFlow import error:", TF_IMPORT_ERROR)



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
model_path_1 = (
    "/kaggle/input/combinedmodel3/tensorflow2/default/1/BestModel_3454_8937.h5"
)
model_path_2 = (
    "/kaggle/input/combinedmodel3/tensorflow2/default/1/best_model_0.37458707.h5"
)
model_path_3 = (
    "/kaggle/input/combinedmodel3/tensorflow2/default/1/googlenet_inceptionv3.h5"
)
model_path_4 = (
    "/kaggle/input/bestmodel_550_2/tensorflow2/default/1/BestModel_3577_8940.h5"
)
model_path_5 = (
    "/kaggle/input/bestmodel_8878/tensorflow2/default/1/BestModel_8878_0358.h5"
)

test_image_dir = "/kaggle/input/cassava-leaf-disease-classification/test_images"
sample = "/kaggle/input/cassava-leaf-disease-classification/sample_submission.csv"



## === cell 2
sample_csv = pd.read_csv(sample)
sample_csv.head(), sample_csv.shape




## === cell 3
def _discover_h5_paths():
    explicit = [model_path_1, model_path_2, model_path_3, model_path_4, model_path_5]

    roots = [
        "/kaggle/input/combinedmodel3",
        "/kaggle/input/bestmodel_550_2",
        "/kaggle/input/bestmodel_8878",
        "/kaggle/input",
    ]

    found = []
    for p in explicit:
        if os.path.isfile(p) and p.lower().endswith(".h5"):
            found.append(p)

    for r in roots:
        if os.path.isdir(r):
            for dirpath, _, filenames in os.walk(r):
                for fn in filenames:
                    if fn.lower().endswith(".h5"):
                        found.append(os.path.join(dirpath, fn))

    seen = set()
    out = []
    for p in found:
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out


def _infer_input_size(model):
    try:
        shp = model.input_shape
        if isinstance(shp, list):
            shp = shp[0]
        if len(shp) == 4 and shp[1] and shp[2]:
            return (int(shp[1]), int(shp[2]))
    except Exception:
        pass
    return (550, 550)


def _infer_input_range(model):
    """
    Preserve ensemble semantics; only infer whether model expects [-1,1] vs [0,1]
    when it contains an explicit Rescaling layer with known params.
    """
    try:
        for layer in getattr(model, "layers", []):
            cfg = getattr(layer, "get_config", lambda: {})()
            name = str(cfg.get("name", "")).lower()
            if "rescaling" in name:
                scale = cfg.get("scale", None)
                offset = cfg.get("offset", None)
                if scale is not None and offset is not None:
                    if (
                        abs(float(scale) - (1.0 / 127.5)) < 1e-6
                        and abs(float(offset) + 1.0) < 1e-6
                    ):
                        return "minus_one_to_one"
    except Exception:
        pass
    return "zero_to_one"


h5_paths = _discover_h5_paths()

models = []
missing = []
if TF_AVAILABLE:
    for path in h5_paths:
        try:
            m = load_model(path, compile=False)
            input_size = _infer_input_size(m)
            input_range = _infer_input_range(m)
            models.append((m, input_size, input_range, path))
        except Exception as e:
            missing.append((path, f"load_error: {repr(e)}"))
else:
    missing = [(p, "tensorflow_unavailable") for p in h5_paths]

print(
    f"Discovered {len(h5_paths)} .h5 file(s). Loaded {len(models)} model(s). Failed: {len(missing)}"
)
for _, _, _, p in models[:10]:
    print(" - loaded:", p)
for p, why in missing[:10]:
    print(" - failed:", p, "=>", why)



## === cell 4
class_labels = {
    0: "Cassava Bacterial Blight (CBB)",
    1: "Cassava Brown Streak Disease (CBSD)",
    2: "Cassava Green Mottle (CGM)",
    3: "Cassava Mosaic Disease (CMD)",
    4: "Healthy",
}



## === cell 5
image_ids = sample_csv["image_id"].tolist()

no_model_fallback_label = 3  # used only if zero models load OR image failed to load
submission_path = "/kaggle/working/submission.csv"

if (not TF_AVAILABLE) or (len(models) == 0):
    submission_df = pd.DataFrame(
        {
            "image_id": image_ids,
            "label": [int(no_model_fallback_label)] * len(image_ids),
        }
    )
else:
    size_to_models = {}
    for m, sz, rng, p in models:
        size_to_models.setdefault(tuple(sz), []).append((m, rng, p))

    preds_sum = np.zeros((len(image_ids), 5), dtype=np.float32)
    preds_count = 0
    loaded_any = np.zeros((len(image_ids),), dtype=bool)

    for sz, model_list in size_to_models.items():
        batch = np.zeros((len(image_ids), sz[0], sz[1], 3), dtype=np.float32)
        loaded = np.zeros((len(image_ids),), dtype=bool)

        for i, image_id in enumerate(image_ids):
            image_path = os.path.join(test_image_dir, image_id)
            if not os.path.exists(image_path):
                continue
            try:
                img = load_img(image_path, target_size=sz)
                arr = img_to_array(img).astype(np.float32) / 255.0
                batch[i] = arr
                loaded[i] = True
            except Exception:
                continue

        if not loaded.any():
            continue

        loaded_any |= loaded

        idx = np.where(loaded)[0]
        batch_loaded = batch[idx]

        for m, rng, _p in model_list:
            x = batch_loaded
            if rng == "minus_one_to_one":
                x = x * 2.0 - 1.0

            pr = m.predict(x, verbose=0)
            pr = np.asarray(pr, dtype=np.float32)

            if pr.ndim != 2 or pr.shape[1] != 5:
                pr = pr.reshape((len(idx), 5))

            preds_sum[idx] += pr
            preds_count += 1

    if preds_count == 0:
        labels = [int(no_model_fallback_label)] * len(image_ids)
    else:
        avg_preds = preds_sum / float(preds_count)
        labels = avg_preds.argmax(axis=1).astype(int)
        labels[~loaded_any] = int(no_model_fallback_label)
        labels = labels.tolist()

    submission_df = pd.DataFrame({"image_id": image_ids, "label": labels})

submission_df.to_csv(submission_path, index=False)

print(f"Submission file saved at: {submission_path}")
print(submission_df.head())
print("Rows:", len(submission_df), "Expected:", len(sample_csv))



## === cell 6
assert (
    submission_df.shape[0] == sample_csv.shape[0]
), "Row count mismatch vs sample_submission.csv"
assert list(submission_df.columns) == ["image_id", "label"], "Wrong submission columns"
assert submission_df["label"].between(0, 4).all(), "Labels must be in [0, 4]"

submission_df
