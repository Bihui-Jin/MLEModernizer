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
Detect the presence and position of catheters and lines on chest x-rays.

## Metric
Area under the ROC curve for each label, with the final score being the average of the individual AUCs of each predicted column.

## Submission Format
For each ID in the test set, you must predict a probability for all target variables. The file should contain a header and have the following format:
```
StudyInstanceUID,ETT - Abnormal,ETT - Borderline,ETT - Normal,NGT - Abnormal,NGT - Borderline,NGT - Incompletely Imaged,NGT - Normal,CVC - Abnormal,CVC - Borderline,CVC - Normal,Swan Ganz Catheter Present
1.2.826.0.1.3680043.8.498.62451881164053375557257228990443168843,0,0,0,0,0,0,0,0,0,0,0
1.2.826.0.1.3680043.8.498.83721761279899623084220697845011427274,0,0,0,0,0,0,0,0,0,0,0
1.2.826.0.1.3680043.8.498.12732270010839808189235995393981377825,0,0,0,0,0,0,0,0,0,0,0
1.2.826.0.1.3680043.8.498.11769539755086084996287023095028033598,0,0,0,0,0,0,0,0,0,0,0
1.2.826.0.1.3680043.8.498.87838627504097587943394933987052577153,0,0,0,0,0,0,0,0,0,0,0
1.2.826.0.1.3680043.8.498.53211840524738036417560823327351887819,0,0,0,0,0,0,0,0,0,0,0
1.2.826.0.1.3680043.8.498.93555795394184819372299157360228027866,0,0,0,0,0,0,0,0,0,0,0
1.2.826.0.1.3680043.8.498.52241894131170494723503100795076463919,0,0,0,0,0,0,0,0,0,0,0
1.2.826.0.1.3680043.8.498.36500167484503936720548852591033878284,0,0,0,0,0,0,0,0,0,0,0
1.2.826.0.1.3680043.8.498.86199852603457900780565655267977637728,0,0,0,0,0,0,0,0,0,0,0
```

## Dataset
`train.csv` contains image IDs, binary labels, and patient IDs.

TFRecords are available for both train and test.

We've also included `train_annotations.csv`. These are segmentation annotations for training samples that have them. They are included solely as additional information for competitors.

- train.csv - contains image IDs, binary labels, and patient IDs.
- sample_submission.csv - a sample submission file in the correct format
- test - test images
- train - training images

### Columns
- `StudyInstanceUID` - unique ID for each image
- `ETT - Abnormal` - endotracheal tube placement abnormal
- `ETT - Borderline` - endotracheal tube placement borderline abnormal
- `ETT - Normal` - endotracheal tube placement normal
- `NGT - Abnormal` - nasogastric tube placement abnormal
- `NGT - Borderline` - nasogastric tube placement borderline abnormal
- `NGT - Incompletely Imaged` - nasogastric tube placement inconclusive due to imaging
- `NGT - Normal` - nasogastric tube placement borderline normal
- `CVC - Abnormal` - central venous catheter placement abnormal
- `CVC - Borderline` - central venous catheter placement borderline abnormal
- `CVC - Normal` - central venous catheter placement normal
- `Swan Ganz Catheter Present`
- `PatientID` - unique ID for each patient in the dataset

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

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (172 lines)
            sample_submission.csv (3010 lines)
            sample_submission.csv.zip (64.2 kB)
            test.zip (642.8 MB)
            train.csv (27075 lines)
            train.csv.zip (798.6 kB)
            train.zip (5.8 GB)
            train_annotations.csv (16262 lines)
            train_annotations.csv.zip (1.4 MB)
            ranzcr-clip-catheter-line-classification/
                description.md (172 lines)
                sample_submission.csv (3010 lines)
                ... and 7 other files
                ranzcr-clip-catheter-line-classification/
                test/
                    1.2.826.0.1.3680043.8.498.57477258980718966370268402373568359767.jpg (201.5 kB)
                    1.2.826.0.1.3680043.8.498.70369997506092680321332747830211112877.jpg (176.2 kB)
                    ... and 3007 other files
                    test/
                train/
                    1.2.826.0.1.3680043.8.498.22695482406757766723043400617436591679.jpg (306.8 kB)
                    1.2.826.0.1.3680043.8.498.67485046061813043086989609654834659261.jpg (315.8 kB)
                    ... and 27072 other files
                    train/
            test/
                1.2.826.0.1.3680043.8.498.57477258980718966370268402373568359767.jpg (201.5 kB)
                1.2.826.0.1.3680043.8.498.70369997506092680321332747830211112877.jpg (176.2 kB)
                ... and 3007 other files
                test/
            train/
                1.2.826.0.1.3680043.8.498.22695482406757766723043400617436591679.jpg (306.8 kB)
                1.2.826.0.1.3680043.8.498.67485046061813043086989609654834659261.jpg (315.8 kB)
                ... and 27072 other files
                train/
        input/
            description.md (172 lines)
            sample_submission.csv (3010 lines)
            sample_submission.csv.zip (64.2 kB)
            test.zip (642.8 MB)
            train.csv (27075 lines)
            train.csv.zip (798.6 kB)
            train.zip (5.8 GB)
            train_annotations.csv (16262 lines)
            train_annotations.csv.zip (1.4 MB)
            ranzcr-clip-catheter-line-classification/
                description.md (172 lines)
                sample_submission.csv (3010 lines)
                ... and 7 other files
                ranzcr-clip-catheter-line-classification/
                test/
                    1.2.826.0.1.3680043.8.498.57477258980718966370268402373568359767.jpg (201.5 kB)
                    1.2.826.0.1.3680043.8.498.70369997506092680321332747830211112877.jpg (176.2 kB)
                    ... and 3007 other files
                    test/
                train/
                    1.2.826.0.1.3680043.8.498.22695482406757766723043400617436591679.jpg (306.8 kB)
                    1.2.826.0.1.3680043.8.498.67485046061813043086989609654834659261.jpg (315.8 kB)
                    ... and 27072 other files
                    train/
            test/
                1.2.826.0.1.3680043.8.498.57477258980718966370268402373568359767.jpg (201.5 kB)
                1.2.826.0.1.3680043.8.498.70369997506092680321332747830211112877.jpg (176.2 kB)
                ... and 3007 other files
                test/
                    1.2.826.0.1.3680043.8.498.57477258980718966370268402373568359767.jpg (201.5 kB)
                    1.2.826.0.1.3680043.8.498.70369997506092680321332747830211112877.jpg (176.2 kB)
                    ... and 3007 other files
                    test/
            train/
                1.2.826.0.1.3680043.8.498.22695482406757766723043400617436591679.jpg (306.8 kB)
                1.2.826.0.1.3680043.8.498.67485046061813043086989609654834659261.jpg (315.8 kB)
                ... and 27072 other files
                train/
                    1.2.826.0.1.3680043.8.498.22695482406757766723043400617436591679.jpg (306.8 kB)
                    1.2.826.0.1.3680043.8.498.67485046061813043086989609654834659261.jpg (315.8 kB)
                    ... and 27072 other files
                    train/
        working/
            ranzcr-clip-catheter-line-classification/
                description.md (172 lines)
                sample_submission.csv (3010 lines)
                ... and 7 other files
                ranzcr-clip-catheter-line-classification/
                test/
                    1.2.826.0.1.3680043.8.498.57477258980718966370268402373568359767.jpg (201.5 kB)
                    1.2.826.0.1.3680043.8.498.70369997506092680321332747830211112877.jpg (176.2 kB)
                    ... and 3007 other files
                    test/
                train/
                    1.2.826.0.1.3680043.8.498.22695482406757766723043400617436591679.jpg (306.8 kB)
                    1.2.826.0.1.3680043.8.498.67485046061813043086989609654834659261.jpg (315.8 kB)
                    ... and 27072 other files
                    train/
```

-> data/ranzcr-clip-catheter-line-classification/sample_submission.csv has 3009 rows and 10 columns.
The columns are: StudyInstanceUID, ETT - Abnormal, ETT - Borderline, ETT - Normal, NGT - Abnormal, NGT - Borderline, NGT - Incompletely Imaged, NGT - Normal, CVC - Abnormal, CVC - Borderline

-> data/ranzcr-clip-catheter-line-classification/train.csv has 27074 rows and 13 columns.
The columns are: StudyInstanceUID, ETT - Abnormal, ETT - Borderline, ETT - Normal, NGT - Abnormal, NGT - Borderline, NGT - Incompletely Imaged, NGT - Normal, CVC - Abnormal, CVC - Borderline, CVC - Normal, Swan Ganz Catheter Present, PatientID

-> data/ranzcr-clip-catheter-line-classification/train_annotations.csv has 16261 rows and 3 columns.
The columns are: StudyInstanceUID, label, data

-> data/sample_submission.csv has 3009 rows and 10 columns.
The columns are: StudyInstanceUID, ETT - Abnormal, ETT - Borderline, ETT - Normal, NGT - Abnormal, NGT - Borderline, NGT - Incompletely Imaged, NGT - Normal, CVC - Abnormal, CVC - Borderline

-> data/train.csv has 27074 rows and 13 columns.
The columns are: StudyInstanceUID, ETT - Abnormal, ETT - Borderline, ETT - Normal, NGT - Abnormal, NGT - Borderline, NGT - Incompletely Imaged, NGT - Normal, CVC - Abnormal, CVC - Borderline, CVC - Normal, Swan Ganz Catheter Present, PatientID

-> data/train_annotations.csv has 16261 rows and 3 columns.
The columns are: StudyInstanceUID, label, data

-> (stopped after 10 files for performance)

# 5. Target score

0.8483315316302348

# 6. Current score

0.5

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.5) has done: 'I fix the environment/runtime crash caused by an incompatible `kaggle_datasets` import (it isn’t used by your pipeline anyway) by removing that import so TensorFlow can initialize cleanly. Then I make the model-loading step robust: instead of hard-failing on a missing external `.h5`, the code search common Kaggle input locations for a saved Keras model and, if none is found, fall back to a simple baseline that outputs per-class priors from `train.csv` so a valid `submission.csv` is always produced. Finally, I fix the submission columns mismatch by ensuring all 11 required target columns exist (adding any missing ones) and by writing predictions with correct shape/order aligned to `sample_submission.csv`.'
- What this solution (achieved 0.5) has done: 'The timeout is dominated by the fallback training path: it builds cached tf.data pipelines for ~27k JPGs at a large resolution (600x600) and then trains for 2 epochs, which is far too much work under 600 seconds. To preserve core logic and accuracy semantics while eliminating the slow path, the script below makes model loading deterministic and robust (including SavedModel directories) and hard-fails early if no pretrained model is found, instead of training a new CNN. It also removes the extremely expensive on-disk dataset caching (which forces full dataset materialization) and replaces it with safe in-memory prefetching + deterministic parallel decode to speed up inference without changing predictions. Finally, it ensures `steps` for predict are set to avoid any extra dataset bookkeeping overhead.'
- What this solution (achieved 0.5) has done: 'I fix the runtime crash happening before your pipeline runs by pinning protobuf to the version TensorFlow 2.18 expects (this avoids the `MessageFactory.GetPrototype` AttributeError). Then I ensure the submission format is always valid for this competition by aligning the output columns to the full required 11-label set (the provided sample file in your environment is missing some columns). Finally, I keep the existing model-loading/inference logic intact so scores can improve when a pretrained model is present, while preserving the safe priors fallback to always generate `submission.csv`.'
- What this solution (achieved 0.49111) has done: 'Your current 0.5 score strongly suggests you’re usually hitting the “priors fallback” (constant predictions), which yields AUC≈0.5. To move toward the 0.848 target with minimal core-logic change, I (1) ensure the script uses the correct on-disk sample submission (the one in this environment is missing required columns), (2) make model discovery actually find models nested inside Kaggle datasets (recursive scan up to a small limit) so inference runs when a pretrained model exists, and (3) keep the priors fallback but make it slightly non-constant per row by using a deterministic, tiny UID-hash jitter around priors (keeps expected value near priors, but breaks ties so AUC can rise above 0.5 if you still fall back). These are small, targeted changes that preserve your pipeline’s architecture/training semantics (no new training) and still always produce a valid `submission.csv`. The dataset/inference paths remain unchanged and runtime stays within the 600s budget.'
- What this solution (achieved 0.5) has done: 'Your current score (0.49111) is far below the target (0.84833), which strongly indicates you are still often using the “priors fallback”, whose per-row hash jitter cannot legitimately improve AUC because it’s independent of the true labels. To move toward the target with minimal core-logic change, I (1) make model discovery more reliable by scanning for common “best/weights/fold” filenames and preferring multi-output (11-unit) classification models, and (2) add a safe, deterministic test-time augmentation (horizontal flip) *only when a real model is found*, averaging predictions to raise AUC without changing architecture/training/loss. The fallback path remain, but I remove the jitter so it doesn’t add noise (it won’t help AUC) and keep priors as a stable baseline. Submission column alignment to the required 11 labels is kept intact and we still always write a valid `submission.csv`.'

# 9. Code solution

## === cell 0
import os
import sys
import subprocess

try:
    import google.protobuf  # noqa: F401
    from google.protobuf import __version__ as _pb_ver

    if _pb_ver.startswith("6."):
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-q", "protobuf==4.25.3"]
        )
        import importlib

        importlib.invalidate_caches()
except Exception as _e:
    print("protobuf pin attempt skipped/failed:", repr(_e))

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit  # kept (part of original imports)
import tensorflow as tf

tf.keras.utils.set_random_seed(42)
try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

try:
    tf.config.optimizer.set_jit(True)
except Exception:
    pass

try:
    tf.config.threading.set_intra_op_parallelism_threads(0)
    tf.config.threading.set_inter_op_parallelism_threads(0)
except Exception:
    pass

try:
    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
except Exception:
    pass




## === cell 1
def auto_select_accelerator():
    """
    Reference:
        * https://www.kaggle.com/mgornergoogle/getting-started-with-100-flowers-on-tpu
        * https://www.kaggle.com/xhlulu/ranzcr-efficientnet-tpu-training
    """
    try:
        tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
        tf.config.experimental_connect_to_cluster(tpu)
        tf.tpu.experimental.initialize_tpu_system(tpu)
        strategy = tf.distribute.experimental.TPUStrategy(tpu)
        print("Running on TPU:", tpu.master())
    except (ValueError, tf.errors.NotFoundError):
        strategy = tf.distribute.get_strategy()
    print(f"Running on {strategy.num_replicas_in_sync} replicas")

    return strategy


def build_decoder(with_labels=True, target_size=(256, 256), ext="jpg"):
    def decode(path):
        file_bytes = tf.io.read_file(path)
        if ext == "png":
            img = tf.image.decode_png(file_bytes, channels=3)
        elif ext in ["jpg", "jpeg"]:
            img = tf.io.decode_jpeg(file_bytes, channels=3, dct_method="INTEGER_FAST")
        else:
            raise ValueError("Image extension not supported")

        img = tf.image.convert_image_dtype(img, tf.float32)  # == cast/255 but faster
        img = tf.image.resize(img, target_size)
        return img

    def decode_with_labels(path, label):
        return decode(path), label

    return decode_with_labels if with_labels else decode


def build_augmenter(with_labels=True):
    def augment(img):
        img = tf.image.random_flip_left_right(img)
        img = tf.image.random_flip_up_down(img)
        return img

    def augment_with_labels(img, label):
        return augment(img), label

    return augment_with_labels if with_labels else augment


def build_dataset(
    paths,
    labels=None,
    bsize=32,
    cache=True,
    decode_fn=None,
    augment_fn=None,
    augment=True,
    repeat=True,
    shuffle=1024,
    cache_dir="",
):
    options = tf.data.Options()
    options.experimental_deterministic = True
    try:
        options.experimental_optimization.map_parallelization = True
        options.experimental_optimization.parallel_batch = True
        options.experimental_optimization.apply_default_optimizations = True
        options.experimental_optimization.autotune_buffers = True
        options.experimental_optimization.autotune_cpu_budget = True
        options.experimental_slack = True
    except Exception:
        pass

    if decode_fn is None:
        decode_fn = build_decoder(labels is not None)

    if augment_fn is None:
        augment_fn = build_augmenter(labels is not None)

    AUTO = tf.data.AUTOTUNE
    slices = paths if labels is None else (paths, labels)

    dset = tf.data.Dataset.from_tensor_slices(slices).with_options(options)
    dset = dset.map(decode_fn, num_parallel_calls=AUTO)

    if cache:
        if not cache_dir:
            cache_dir = os.path.join("/kaggle/working", "tfdata_cache")
        dset = dset.cache()

    dset = dset.map(augment_fn, num_parallel_calls=AUTO) if augment else dset

    if shuffle:
        dset = dset.shuffle(shuffle, reshuffle_each_iteration=True)

    dset = dset.repeat() if repeat else dset
    dset = dset.batch(bsize, drop_remainder=False)
    dset = dset.prefetch(AUTO)

    return dset




## === cell 2
COMPETITION_NAME = "ranzcr-clip-catheter-line-classification"
strategy = auto_select_accelerator()
BATCH_SIZE = strategy.num_replicas_in_sync * 16




## === cell 3
load_dir = f"/kaggle/input/{COMPETITION_NAME}/"


def _load_best_sample_submission():
    candidates = [
        os.path.join(load_dir, "sample_submission.csv"),
        "/kaggle/input/sample_submission.csv",
    ]
    best_df = None
    best_path = None

    for p in candidates:
        if not os.path.exists(p):
            continue
        try:
            df = pd.read_csv(p)
        except Exception:
            continue
        ncols = df.shape[1]
        if best_df is None or ncols > best_df.shape[1]:
            best_df, best_path = df, p

    if best_df is None:
        best_path = os.path.join(load_dir, "sample_submission.csv")
        best_df = pd.read_csv(best_path)

    print("Using sample_submission:", best_path, "with columns:", list(best_df.columns))
    return best_df


sub_df = _load_best_sample_submission()

REQUIRED_LABELS = [
    "ETT - Abnormal",
    "ETT - Borderline",
    "ETT - Normal",
    "NGT - Abnormal",
    "NGT - Borderline",
    "NGT - Incompletely Imaged",
    "NGT - Normal",
    "CVC - Abnormal",
    "CVC - Borderline",
    "CVC - Normal",
    "Swan Ganz Catheter Present",
]

for c in REQUIRED_LABELS:
    if c not in sub_df.columns:
        sub_df[c] = 0.0

label_cols = REQUIRED_LABELS

test_paths = (
    os.path.join(load_dir, "test")
    + "/"
    + sub_df["StudyInstanceUID"].astype(str)
    + ".jpg"
)

IMSIZE = (224, 240, 260, 300, 380, 456, 528, 600)
IMG_SIZE = IMSIZE[7]

test_decoder = build_decoder(
    with_labels=False, target_size=(IMG_SIZE, IMG_SIZE), ext="jpg"
)

dtest = build_dataset(
    test_paths.values,
    bsize=BATCH_SIZE,
    repeat=False,
    shuffle=False,
    augment=False,
    cache=False,
    decode_fn=test_decoder,
)


def _hflip_only(img):
    return tf.image.flip_left_right(img)


dtest_flip = dtest.map(_hflip_only, num_parallel_calls=tf.data.AUTOTUNE).prefetch(
    tf.data.AUTOTUNE
)




## === cell 4
def find_any_keras_model_path(prefer_n_outputs=None):
    orig = "/kaggle/input/modeltpu2/bestmodel_tpu (2).h5"
    if tf.io.gfile.exists(orig):
        return orig

    candidate_dirs = [
        "/kaggle/input/modeltpu2",
        "/kaggle/input/model",
        "/kaggle/input/models",
        "/kaggle/input",
        f"/kaggle/input/{COMPETITION_NAME}",
    ]

    exts = (".keras", ".h5", ".hdf5")

    def _is_savedmodel_dir(p):
        try:
            return tf.io.gfile.isdir(p) and tf.io.gfile.exists(
                os.path.join(p, "saved_model.pb")
            )
        except Exception:
            return False

    def _score_name(name: str) -> int:
        n = name.lower()
        score = 0
        if "best" in n:
            score += 5
        if "fold" in n:
            score += 3
        if "final" in n:
            score += 2
        if "model" in n:
            score += 1
        if "weight" in n or "weights" in n:
            score += 1
        return score

    def _try_load_and_check(p):
        try:
            m = tf.keras.models.load_model(p, compile=False)
        except Exception:
            return None
        try:
            out_shape = m.output_shape
            if isinstance(out_shape, (list, tuple)) and len(out_shape) >= 2:
                nout = int(out_shape[-1])
            else:
                nout = None
        except Exception:
            nout = None
        return (p, nout)

    found = []

    for d in candidate_dirs:
        try:
            if not tf.io.gfile.exists(d):
                continue
            for f in tf.io.gfile.listdir(d):
                p = os.path.join(d, f)
                if f.endswith(exts) and tf.io.gfile.exists(p):
                    found.append(("file", p, _score_name(f)))
                elif _is_savedmodel_dir(p):
                    found.append(("savedmodel", p, _score_name(f)))
        except Exception:
            pass

    def _bounded_walk(root, max_dirs=2500, max_depth=6):
        stack = [(root, 0)]
        seen = 0
        while stack and seen < max_dirs:
            d, depth = stack.pop()
            seen += 1
            try:
                entries = tf.io.gfile.listdir(d)
            except Exception:
                continue
            for name in entries:
                p = os.path.join(d, name)
                if name.endswith(exts):
                    try:
                        if tf.io.gfile.exists(p):
                            yield ("file", p, _score_name(name))
                    except Exception:
                        pass
                else:
                    try:
                        if tf.io.gfile.isdir(p):
                            if _is_savedmodel_dir(p):
                                yield ("savedmodel", p, _score_name(name))
                            if depth + 1 <= max_depth:
                                stack.append((p, depth + 1))
                    except Exception:
                        pass

    for d in candidate_dirs:
        try:
            if not tf.io.gfile.exists(d) or not tf.io.gfile.isdir(d):
                continue
        except Exception:
            continue
        for kind, p, sc in _bounded_walk(d):
            found.append((kind, p, sc))

    if not found:
        return None

    found.sort(key=lambda x: x[2], reverse=True)
    top = found[:30]  # keep bounded for runtime

    best_path = None
    best_match = -1
    for _, p, sc in top:
        chk = _try_load_and_check(p)
        if chk is None:
            continue
        _, nout = chk
        match = sc
        if prefer_n_outputs is not None and nout == prefer_n_outputs:
            match += 100  # strongly prefer correct output width
        if match > best_match:
            best_match = match
            best_path = p

    return best_path


_PRIORS_CACHE = None


def compute_label_priors(train_csv_path, label_columns):
    global _PRIORS_CACHE
    if (
        _PRIORS_CACHE is not None
        and tuple(_PRIORS_CACHE[0]) == tuple(label_columns)
        and _PRIORS_CACHE[1] == train_csv_path
    ):
        return _PRIORS_CACHE[2].copy()

    train_df = pd.read_csv(train_csv_path)
    priors = []
    for c in label_columns:
        if c in train_df.columns:
            priors.append(float(train_df[c].mean()))
        else:
            priors.append(0.0)
    priors = np.array(priors, dtype=np.float32)
    _PRIORS_CACHE = (list(label_columns), train_csv_path, priors)
    return priors.copy()


def build_simple_cnn(img_size, n_classes):
    inp = tf.keras.Input(shape=(img_size, img_size, 3))
    x = tf.keras.layers.Conv2D(16, 3, padding="same", activation="relu")(inp)
    x = tf.keras.layers.MaxPool2D()(x)
    x = tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.MaxPool2D()(x)
    x = tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.MaxPool2D()(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    out = tf.keras.layers.Dense(n_classes, activation="sigmoid")(x)
    model = tf.keras.Model(inp, out)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=tf.keras.losses.BinaryCrossentropy(),
    )
    return model


model_path = find_any_keras_model_path(prefer_n_outputs=len(label_cols))
print("Resolved model path:", model_path)

model_p = None
if model_path is not None:
    with strategy.scope():
        try:
            model_p = tf.keras.models.load_model(model_path, compile=False)
        except Exception as e:
            print("Model load failed. Error:", repr(e))
            model_p = None




## === cell 5
if model_p is None:
    train_csv_path = os.path.join(load_dir, "train.csv")
    priors = compute_label_priors(train_csv_path, label_cols).astype(np.float32)
    preds = np.tile(priors, (len(sub_df), 1)).astype(np.float32)
else:
    try:
        model_p.run_eagerly = False
    except Exception:
        pass

    n_test = len(sub_df)
    pred_steps = int(np.ceil(n_test / BATCH_SIZE))

    preds_a = model_p.predict(dtest, steps=pred_steps, verbose=1)
    preds_b = model_p.predict(dtest_flip, steps=pred_steps, verbose=1)
    preds = (np.asarray(preds_a) + np.asarray(preds_b)) * 0.5

    preds = np.asarray(preds)

    if preds.ndim == 1:
        preds = preds.reshape(-1, 1)

    if preds.shape[0] != n_test:
        preds = preds[:n_test]

    if preds.shape[1] != len(label_cols):
        train_csv_path = os.path.join(load_dir, "train.csv")
        priors = compute_label_priors(train_csv_path, label_cols)

        if preds.shape[1] < len(label_cols):
            pad = np.tile(priors[preds.shape[1] :], (preds.shape[0], 1))
            preds = np.concatenate([preds, pad], axis=1)
        else:
            preds = preds[:, : len(label_cols)]

preds = np.clip(preds, 0.0, 1.0)

sub_df[label_cols] = preds.astype(np.float32)

sub_out = sub_df[["StudyInstanceUID"] + label_cols].copy()
sub_out.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", sub_out.shape)
print("Submission columns:", list(sub_out.columns))

sub_out.head()
