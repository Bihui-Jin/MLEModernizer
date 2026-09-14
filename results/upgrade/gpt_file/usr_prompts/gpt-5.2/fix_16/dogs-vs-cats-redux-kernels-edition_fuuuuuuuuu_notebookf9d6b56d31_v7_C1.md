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
Given a dataset of images of dogs and cats, predict if an image is a dog or a cat.

## Metric
Log loss.

## Submission Format
For each image in the test set, you must submit a probability that image is a dog. The file should have a header and be in the following format:

```
id,label
1,0.5
2,0.5
3,0.5
...
```

## Dataset
The train folder contains 25,000 images of dogs and cats. Each image in this folder has the label as part of the filename. The test folder contains 12,500 images, named according to a numeric id.

# 2. Python version

3.12

# 3. Installed packages

geopandas==0.14.4
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
numpy==1.26.4
opencv-python==4.12.0.88
opencv-python-headless==4.12.0.88
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
            cat.1714.jpg (7.8 kB)
            cat.10025.jpg (18.4 kB)
            ... and 24998 other files
            description.md (50 lines)
            sample_submission.csv (2501 lines)
            sample_submission.csv.zip (6.0 kB)
            test.zip (56.6 MB)
            train.zip (513.0 MB)
            dogs-vs-cats-redux-kernels-edition/
                cat.1714.jpg (7.8 kB)
                cat.10025.jpg (18.4 kB)
                ... and 24998 other files
                description.md (50 lines)
                sample_submission.csv (2501 lines)
                sample_submission.csv.zip (6.0 kB)
                test.zip (56.6 MB)
                train.zip (513.0 MB)
                dogs-vs-cats-redux-kernels-edition/
                test/
                    test/
                    unknown/
                        900.jpg (42.3 kB)
                        572.jpg (30.6 kB)
                        ... and 2498 other files
                train/
                    cat/
                        cat.4838.jpg (20.2 kB)
                        cat.1314.jpg (21.7 kB)
                        ... and 11240 other files
                    dog/
                        dog.6712.jpg (35.3 kB)
                        dog.7152.jpg (36.1 kB)
                        ... and 11256 other files
                    train/
            test/
                test/
                unknown/
                    900.jpg (42.3 kB)
                    572.jpg (30.6 kB)
                    ... and 2498 other files
            train/
                cat/
                    cat.4838.jpg (20.2 kB)
                    cat.1314.jpg (21.7 kB)
                    ... and 11240 other files
                dog/
                    dog.6712.jpg (35.3 kB)
                    dog.7152.jpg (36.1 kB)
                    ... and 11256 other files
                train/
        input/
            cat.1714.jpg (7.8 kB)
            cat.10025.jpg (18.4 kB)
            ... and 24998 other files
            description.md (50 lines)
            sample_submission.csv (2501 lines)
            sample_submission.csv.zip (6.0 kB)
            test.zip (56.6 MB)
            train.zip (513.0 MB)
            dogs-vs-cats-redux-kernels-edition/
                cat.1714.jpg (7.8 kB)
                cat.10025.jpg (18.4 kB)
                ... and 24998 other files
                description.md (50 lines)
                sample_submission.csv (2501 lines)
                sample_submission.csv.zip (6.0 kB)
                test.zip (56.6 MB)
                train.zip (513.0 MB)
                dogs-vs-cats-redux-kernels-edition/
                test/
                    test/
                    unknown/
                        900.jpg (42.3 kB)
                        572.jpg (30.6 kB)
                        ... and 2498 other files
                train/
                    cat/
                        cat.4838.jpg (20.2 kB)
                        cat.1314.jpg (21.7 kB)
                        ... and 11240 other files
                    dog/
                        dog.6712.jpg (35.3 kB)
                        dog.7152.jpg (36.1 kB)
                        ... and 11256 other files
                    train/
            test/
                test/
                    test/
                    unknown/
                        900.jpg (42.3 kB)
                        572.jpg (30.6 kB)
                        ... and 2498 other files
                unknown/
                    900.jpg (42.3 kB)
                    572.jpg (30.6 kB)
                    ... and 2498 other files
            train/
                cat/
                    cat.4838.jpg (20.2 kB)
                    cat.1314.jpg (21.7 kB)
                    ... and 11240 other files
                dog/
                    dog.6712.jpg (35.3 kB)
                    dog.7152.jpg (36.1 kB)
                    ... and 11256 other files
                train/
                    cat/
                        cat.4838.jpg (20.2 kB)
                        cat.1314.jpg (21.7 kB)
                        ... and 11240 other files
                    dog/
                        dog.6712.jpg (35.3 kB)
                        dog.7152.jpg (36.1 kB)
                        ... and 11256 other files
                    train/
        working/
            dogs-vs-cats-redux-kernels-edition/
                cat.1714.jpg (7.8 kB)
                cat.10025.jpg (18.4 kB)
                ... and 24998 other files
                description.md (50 lines)
                sample_submission.csv (2501 lines)
                sample_submission.csv.zip (6.0 kB)
                test.zip (56.6 MB)
                train.zip (513.0 MB)
                dogs-vs-cats-redux-kernels-edition/
                test/
                    test/
                    unknown/
                        900.jpg (42.3 kB)
                        572.jpg (30.6 kB)
                        ... and 2498 other files
                train/
                    cat/
                        cat.4838.jpg (20.2 kB)
                        cat.1314.jpg (21.7 kB)
                        ... and 11240 other files
                    dog/
                        dog.6712.jpg (35.3 kB)
                        dog.7152.jpg (36.1 kB)
                        ... and 11256 other files
                    train/
```

-> data/dogs-vs-cats-redux-kernels-edition/sample_submission.csv has 2500 rows and 2 columns.
The columns are: id, label

-> data/sample_submission.csv has 2500 rows and 2 columns.
The columns are: id, label

-> input/dogs-vs-cats-redux-kernels-edition/sample_submission.csv has 2500 rows and 2 columns.
The columns are: id, label

-> input/sample_submission.csv has 2500 rows and 2 columns.
The columns are: id, label

-> working/dogs-vs-cats-redux-kernels-edition/sample_submission.csv has 2500 rows and 2 columns.
The columns are: id, label

# 5. Target score

12.75741

# 6. Current score

0.63494

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 0.64817) has done: 'I fix the root cause of the FileNotFoundError by making the train/test directory discovery robust to the actual zip extraction structure, so `load_data()` and `load_test_data()` can always find images. I also ensure the training split cell runs before any training calls so `X_train/y_train` exist, eliminating the downstream NameErrors without changing the model/training logic. Finally, I guarantee a valid `submission.csv` is written with the required `id,label` columns by aligning predictions to `sample_submission.csv` and filling any missing ids with 0.5.'
- What this solution (achieved 0.65638) has done: 'The crash comes from a protobuf/TensorFlow incompatibility triggered by the protobuf “MessageFactory.GetPrototype” API change; to keep the core training/prediction logic unchanged, I fix it by forcing the pure-Python protobuf implementation before importing TensorFlow. I also make the zip extraction idempotent and ensure we always locate the correct flat `train/` and `test/` image folders, so data loading can’t silently point at the wrong nested directory. Finally, I keep your existing model/training exactly as-is but make submission generation robust: always align to `sample_submission.csv`, ensure `id` parsing works, and write `/kaggle/working/submission.csv` with the required `id,label` columns.'
- What this solution (achieved 0.63453) has done: 'I fix the TensorFlow import crash by ensuring the protobuf runtime is forced to the pure-Python implementation *before* TensorFlow is imported, which avoids the `MessageFactory.GetPrototype` AttributeError in this Kaggle environment. I also make the environment setting robust by setting (not just defaulting) the protobuf env vars early, and I keep your model, training loop, augmentation, and submission logic unchanged. Finally, I preserve your existing data extraction and directory discovery behavior while making sure the notebook runs end-to-end and writes `/kaggle/working/submission.csv` with `id,label`.'
- What this solution (achieved 0.6376) has done: 'I fix the TensorFlow import crash by forcing the pure-Python protobuf implementation **before any TensorFlow-related import happens** (your current order still imports TensorFlow after some other imports, but the env var must be set at process start, so we move it to the very first cell). I keep your model, training loop, augmentation, and submission generation logic the same, only adjusting the cell order/contents to ensure the environment is set correctly and the notebook runs end-to-end. I also make the train/test directory detection slightly more robust to the extracted zip structure without changing what images are used. These changes are score-neutral; they primarily unblock execution and ensure a valid `/kaggle/working/submission.csv` is always written with `id,label`.'
- What this solution (achieved 0.67069) has done: 'I fix the TensorFlow/protobuf crash by forcing the pure-Python protobuf implementation at the very start of the process (before any protobuf/TensorFlow-related import), which is the root cause of the `MessageFactory.GetPrototype` error. I also make the zip extraction and train/test directory discovery robust but keep the same data usage and model/training logic. Finally, I keep submission generation identical in semantics while guaranteeing correct `id,label` formatting and alignment to `sample_submission.csv` so a valid `/kaggle/working/submission.csv` is always produced.'
- What this solution (achieved 0.65206) has done: 'We fix the TensorFlow import crash by ensuring the protobuf pure-Python implementation is forced at the very start of the process (before any protobuf/TensorFlow-related imports) and by performing the TensorFlow import immediately after that, so later cells can’t accidentally trigger protobuf first. We also make the test directory discovery stricter so it points to the actual flat folder containing numeric `*.jpg` files (avoiding nested `unknown/` layouts that can break filename-to-id parsing). These changes are execution/stability fixes and should be score-neutral (they don’t alter the model/training core logic), while guaranteeing the notebook runs end-to-end and writes a valid `/kaggle/working/submission.csv`. Finally, we keep the existing training loop/model exactly the same and preserve the submission alignment to `sample_submission.csv`.'
- What this solution (achieved 0.61934) has done: 'The current failure is a TensorFlow import crash caused by a protobuf API incompatibility, and your environment variables alone aren’t reliably preventing it in this Kaggle runtime. I fix this by forcing the pure-Python protobuf implementation *and* monkey-patching the missing `MessageFactory.GetPrototype` method before importing TensorFlow, which keeps your model/training logic unchanged and unblocks execution. I also keep the existing zip extraction, directory discovery, training, and submission generation the same, only adding small robustness guards so the pipeline always completes and writes `/kaggle/working/submission.csv` with `id,label`. These changes are score-neutral (they don’t change the model or training behavior) and should allow your current ~0.65 score to be reproduced.'
- What this solution (achieved 0.68495) has done: 'I fix the TensorFlow import crash by ensuring the protobuf compatibility patch is applied at process start and also patching the specific `MessageFactory` class instance path TensorFlow hits, before importing `tensorflow`. I also make the extraction and directory discovery run before any TensorFlow usage, but without changing your model architecture, training loop, augmentation, or submission semantics. These are execution/stability fixes and should be score-neutral (your current score is already far better than the target, so we avoid any score-changing “improvements”). Finally, I keep the submission generation identical while guaranteeing `/kaggle/working/submission.csv` is always written with `id,label`.'
- What this solution (achieved 0.67557) has done: 'I fix the TensorFlow import crash by moving the protobuf compatibility environment variables and monkey-patch to happen before any protobuf/TensorFlow-dependent imports, and by patching both the public and internal `MessageFactory` paths that TensorFlow can hit in this environment. I also make the import cell resilient by applying the patch immediately before importing `tensorflow`, without changing your model, training loop, augmentation, or submission semantics (so the score behavior stays essentially the same). Finally, I keep the existing data discovery and submission alignment, ensuring the notebook always completes and writes `/kaggle/working/submission.csv` with `id,label`.'
- What this solution (achieved 0.64748) has done: 'I fix the TensorFlow import crash by forcing the pure-Python protobuf runtime before any protobuf/TensorFlow-dependent imports and by patching the specific `MessageFactory` paths that TensorFlow can touch (including `google.protobuf.symbol_database.Default().pool._message_factory`). This is the root cause of the current `AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'` in cell 5, and the patch is execution-only (score-neutral). I also keep your model/training/inference logic unchanged, only making the import order safe and adding a couple of guards so directory discovery and submission writing always complete. Since your current score (0.67557, lower-is-better) is already far better than the target (12.75741), I won’t make any score-improving changes—just stability fixes.'
- What this solution (achieved 0.61987) has done: 'I fix the TensorFlow import crash by making the protobuf compatibility patch happen at process start and explicitly patching both the class and the singleton factory instance that TensorFlow triggers during import. I keep your model, training loop, and submission logic unchanged, only reorganizing the import order so TensorFlow can’t be imported before the patch applies. I also add a small safety guard so the script still produces a valid `/kaggle/working/submission.csv` even if training is skipped due to an unexpected runtime issue, without changing normal behavior. These changes are execution/stability-focused and should be score-neutral relative to your current ~0.65 logloss.'
- What this solution (achieved 0.65079) has done: 'I fix the TensorFlow import crash by applying the protobuf compatibility patch at process start (before *any* TensorFlow-related imports) and by importing TensorFlow immediately after that, which prevents the `MessageFactory.GetPrototype` error in this environment. I also keep the rest of your pipeline (data loading, model, training loop, and submission generation) the same, only making the minimum reorder/guard changes needed for end-to-end execution. Since your current score (0.61987, lower-is-better) is already far better than the target (12.75741), I not introduce any score-improving changes—only stability/correctness fixes to ensure a valid `submission.csv` is produced.'
- What this solution (achieved 0.64854) has done: 'I fix the TensorFlow import crash by ensuring the protobuf compatibility patch is applied in the same process before TensorFlow is imported, including patching the specific C++/python factory instances TensorFlow can touch. This is an execution/stability fix and should be score-neutral relative to your current ~0.65 logloss, while allowing the notebook to run end-to-end. I also keep your model/training/inference logic unchanged, only adjusting import order and making the protobuf patch more comprehensive. The pipeline still write a valid `/kaggle/working/submission.csv` with the required `id,label` columns aligned to `sample_submission.csv`.'
- What this solution (achieved 0.63494) has done: 'I fix the TensorFlow import crash by applying the protobuf `MessageFactory.GetPrototype` compatibility patch at the very start of the process and importing TensorFlow immediately after, so TensorFlow can’t touch an unpatched factory during import. This is an execution/stability fix and should be score-neutral (your current score is already far better than the target, and lower is better). I also keep your extraction, directory discovery, model, training loop, and submission logic the same, only adding small guards to ensure the paths exist and the submission always aligns to `sample_submission.csv`. The resulting notebook run end-to-end and always write `/kaggle/working/submission.csv` with `id,label`.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION"] = "2"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import pandas as pd

for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames[:5]:
        print(os.path.join(dirname, filename))



## === cell 1
import zipfile

os.makedirs("/kaggle/working/train", exist_ok=True)
os.makedirs("/kaggle/working/test", exist_ok=True)

train_zip = "/kaggle/input/dogs-vs-cats-redux-kernels-edition/train.zip"
test_zip = "/kaggle/input/dogs-vs-cats-redux-kernels-edition/test.zip"

if len(os.listdir("/kaggle/working/train")) == 0:
    with zipfile.ZipFile(train_zip, "r") as zip_ref:
        zip_ref.extractall("/kaggle/working/train")

if len(os.listdir("/kaggle/working/test")) == 0:
    with zipfile.ZipFile(test_zip, "r") as zip_ref:
        zip_ref.extractall("/kaggle/working/test")

print("Extracted train dirs:", os.listdir("/kaggle/working/train")[:10])
print("Extracted test dirs:", os.listdir("/kaggle/working/test")[:10])




## === cell 2
def _ensure_protobuf_compat():
    try:
        import google.protobuf  # noqa: F401
        from google.protobuf import __version__ as pb_ver

        print(
            "protobuf version:",
            pb_ver,
            "implementation env:",
            os.environ.get("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"),
        )
    except Exception as e:
        print("Warning: protobuf import issue:", e)


_ensure_protobuf_compat()




## === cell 3
def _patch_protobuf_message_factory():
    """
    Fix for protobuf>=5 API changes where some MessageFactory instances don't expose GetPrototype,
    but TensorFlow still calls it during import in some environments.

    This patch must run BEFORE importing tensorflow.
    """

    def _add_getprototype_on_type(target_type, name_hint=""):
        try:
            if target_type is None:
                return False
            if hasattr(target_type, "GetPrototype"):
                return False
            if not hasattr(target_type, "GetMessageClass"):
                return False

            def _GetPrototype(self, descriptor):
                return self.GetMessageClass(descriptor)

            target_type.GetPrototype = _GetPrototype  # type: ignore[attr-defined]
            print(f"Patched GetPrototype on type: {target_type} via {name_hint}")
            return True
        except Exception as e:
            print(f"Warning: could not patch type {name_hint}:", e)
            return False

    def _add_getprototype_on_instance(inst, name_hint=""):
        try:
            if inst is None:
                return False
            if hasattr(inst, "GetPrototype"):
                return False
            if not hasattr(inst, "GetMessageClass"):
                return False

            def _GetPrototype(descriptor):
                return inst.GetMessageClass(descriptor)

            setattr(inst, "GetPrototype", _GetPrototype)
            print(f"Patched GetPrototype on instance via {name_hint}")
            return True
        except Exception as e:
            print(f"Warning: could not patch instance {name_hint}:", e)
            return False

    try:
        from google.protobuf.message_factory import (
            MessageFactory as PublicMessageFactory,
        )

        _add_getprototype_on_type(
            PublicMessageFactory, "google.protobuf.message_factory.MessageFactory"
        )
    except Exception as e:
        print("Warning: could not patch public protobuf MessageFactory:", e)

    try:
        from google.protobuf.internal import python_message as _python_message

        InternalMessageFactory = getattr(_python_message, "MessageFactory", None)
        _add_getprototype_on_type(
            InternalMessageFactory,
            "google.protobuf.internal.python_message.MessageFactory",
        )
    except Exception as e:
        print("Warning: could not patch internal protobuf MessageFactory:", e)

    try:
        import google.protobuf.message_factory as _mf

        mf_singleton = getattr(_mf, "message_factory", None)
        if mf_singleton is not None:
            _add_getprototype_on_instance(
                mf_singleton,
                "google.protobuf.message_factory.message_factory singleton",
            )
            _add_getprototype_on_type(
                mf_singleton.__class__,
                "google.protobuf.message_factory.message_factory singleton type",
            )
    except Exception as e:
        print("Warning: could not patch module-level message_factory singleton:", e)

    try:
        from google.protobuf import symbol_database as _symbol_database

        db = _symbol_database.Default()
        pool = getattr(db, "pool", None)
        mf_inst = getattr(pool, "_message_factory", None) if pool is not None else None
        _add_getprototype_on_instance(
            mf_inst, "google.protobuf.symbol_database.Default().pool._message_factory"
        )
        if mf_inst is not None:
            _add_getprototype_on_type(
                mf_inst.__class__,
                "google.protobuf.symbol_database.Default().pool._message_factory type",
            )
    except Exception as e:
        print("Warning: could not patch symbol_database Default message factory:", e)

    try:
        from google.protobuf import descriptor_pool as _descriptor_pool

        default_pool = _descriptor_pool.Default()
        mf_inst = getattr(default_pool, "_message_factory", None)
        _add_getprototype_on_instance(
            mf_inst, "google.protobuf.descriptor_pool.Default()._message_factory"
        )
        if mf_inst is not None:
            _add_getprototype_on_type(
                mf_inst.__class__,
                "google.protobuf.descriptor_pool.Default()._message_factory type",
            )
    except Exception as e:
        print("Warning: could not patch descriptor_pool Default message factory:", e)


_patch_protobuf_message_factory()

import tensorflow as tf
import tensorflow.keras as keras

print("TensorFlow:", tf.__version__)




## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 4
def _find_dir_with_jpgs(
    root, must_contain_subdirs=None, require_numeric_filenames=False
):
    """
    Find the first directory under `root` that contains .jpg files.
    If must_contain_subdirs is provided (list), require those subdirs exist under the directory.
    If require_numeric_filenames is True, require at least one .jpg with an int stem (e.g., '123.jpg').
    """
    root = os.path.abspath(root)
    for dirpath, dirnames, filenames in os.walk(root):
        if must_contain_subdirs is not None:
            ok = all(
                os.path.isdir(os.path.join(dirpath, d)) for d in must_contain_subdirs
            )
            if not ok:
                continue

        jpgs = [f for f in filenames if f.lower().endswith(".jpg")]
        if not jpgs:
            continue

        if require_numeric_filenames:
            ok_num = False
            for f in jpgs[:50]:
                stem = os.path.splitext(f)[0]
                if stem.isdigit():
                    ok_num = True
                    break
            if not ok_num:
                continue

        return dirpath
    return None


preferred_train = "/kaggle/working/train/train"
preferred_test = "/kaggle/working/test/test"

train_dir = (
    preferred_train
    if os.path.isdir(preferred_train)
    else _find_dir_with_jpgs("/kaggle/working/train")
)
test_dir = (
    preferred_test
    if os.path.isdir(preferred_test)
    else _find_dir_with_jpgs("/kaggle/working/test", require_numeric_filenames=True)
)

print(
    "Using train_dir:",
    train_dir,
    "exists:",
    os.path.isdir(train_dir) if train_dir else None,
)
print(
    "Using test_dir:",
    test_dir,
    "exists:",
    os.path.isdir(test_dir) if test_dir else None,
)

if train_dir is None:
    raise FileNotFoundError(
        "Could not locate any training .jpg files under /kaggle/working/train"
    )
if test_dir is None:
    raise FileNotFoundError(
        "Could not locate numeric test .jpg files under /kaggle/working/test"
    )



## === cell 5
import cv2
import matplotlib.pyplot as plt

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model



## === cell 6
IMG_SIZE = 64




## === cell 7
def load_data(data_dir, sample_size=1000):
    images = []
    labels = []
    files = [f for f in os.listdir(data_dir) if f.lower().endswith(".jpg")]
    files = files[:sample_size]
    for file in files:
        img_path = os.path.join(data_dir, file)
        img = cv2.imread(img_path)
        if img is not None:
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            images.append(img)
            label = 1 if "dog" in file.lower() else 0
            labels.append(label)
        else:
            print(f"error {img_path}")
    return np.array(images, dtype=np.float32) / 255.0, np.array(labels, dtype=np.int64)


def load_test_data(data_dir):
    images = []
    filenames = [f for f in os.listdir(data_dir) if f.lower().endswith(".jpg")]
    filenames = sorted(filenames, key=lambda x: int(os.path.splitext(x)[0]))
    for file in filenames:
        img_path = os.path.join(data_dir, file)
        img = cv2.imread(img_path)
        if img is not None:
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            images.append(img)
        else:
            print(f"Error reading {img_path}")
    return np.array(images, dtype=np.float32) / 255.0, filenames




## === cell 8
X_all, y_all = load_data(train_dir, sample_size=1000)

rng = np.random.RandomState(42)
idx = np.arange(len(X_all))
rng.shuffle(idx)
split = int(0.8 * len(idx))
train_idx, val_idx = idx[:split], idx[split:]

X_train, y_train = X_all[train_idx], y_all[train_idx]
X_test, y_test = X_all[val_idx], y_all[val_idx]

print("Train:", X_train.shape, y_train.shape, "Val:", X_test.shape, y_test.shape)



## === cell 9
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode="nearest",
)




## === cell 10
def create_model(neuron):
    Dense = keras.layers.Dense
    Conv2D = keras.layers.Conv2D
    MaxPooling2D = keras.layers.MaxPooling2D
    Flatten = keras.layers.Flatten
    Dropout = keras.layers.Dropout

    model = keras.models.Sequential()
    model.add(
        Conv2D(32, (3, 3), activation="relu", input_shape=(IMG_SIZE, IMG_SIZE, 3))
    )
    model.add(MaxPooling2D((2, 2)))

    model.add(Conv2D(64, (3, 3), activation="relu"))
    model.add(MaxPooling2D((2, 2)))

    model.add(Conv2D(128, (3, 3), activation="relu"))
    model.add(MaxPooling2D((2, 2)))

    model.add(Flatten())
    model.add(Dense(neuron, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation="sigmoid"))

    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
    return model




## === cell 11
def save_model(model, filename):
    model.save(filename)




## === cell 12
def load_existing_model(filename):
    return load_model(filename)




## === cell 13
def fit_epoch(neuron, batch, epochs, initial_epoch=0, model_filename="model.keras"):
    if initial_epoch == 0 or (not os.path.exists(model_filename)):
        model = create_model(neuron)
    else:
        model = load_existing_model(model_filename)

    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

    steps = max(1, len(X_train) // batch)
    hist = model.fit(
        datagen.flow(X_train, y_train, batch_size=batch, shuffle=True),
        steps_per_epoch=steps,
        validation_data=(X_test, y_test),
        epochs=epochs,
        initial_epoch=initial_epoch,
        verbose=1,
    )

    score = model.evaluate(X_test, y_test, verbose=1)
    print("正解率=", score[1], "loss=", score[0])

    save_model(model, model_filename)

    plt.plot(hist.history["accuracy"])
    plt.plot(hist.history["val_accuracy"])
    plt.title("Accuracy")
    plt.legend(["train", "val"], loc="upper left")
    plt.show()

    plt.plot(hist.history["loss"])
    plt.plot(hist.history["val_loss"])
    plt.title("Loss")
    plt.legend(["train", "val"], loc="upper left")
    plt.show()

    return model




## === cell 14
neuron = 512
batch = 8
model_filename = "/kaggle/working/model.keras"



## === cell 15
model = fit_epoch(
    neuron=neuron,
    batch=batch,
    epochs=10,
    initial_epoch=0,
    model_filename=model_filename,
)



## === cell 16
model = fit_epoch(
    neuron=neuron,
    batch=batch,
    epochs=20,
    initial_epoch=10,
    model_filename=model_filename,
)



## === cell 17
model = fit_epoch(
    neuron=neuron,
    batch=batch,
    epochs=30,
    initial_epoch=20,
    model_filename=model_filename,
)



## === cell 18
output_csv = "/kaggle/working/submission.csv"

X_submit, submit_filenames = load_test_data(test_dir)

if os.path.exists(model_filename):
    model = load_model(model_filename)
    pred = model.predict(X_submit, verbose=1).reshape(-1)
    pred = np.clip(pred, 1e-7, 1 - 1e-7)
else:
    print(
        "Warning: model file not found; using 0.5 fallback predictions to create a valid submission."
    )
    pred = np.full((len(submit_filenames),), 0.5, dtype=np.float64)

sub = (
    pd.DataFrame(
        {
            "id": [int(os.path.splitext(f)[0]) for f in submit_filenames],
            "label": pred.astype(np.float64),
        }
    )
    .sort_values("id")
    .reset_index(drop=True)
)

sample_path = "/kaggle/input/dogs-vs-cats-redux-kernels-edition/sample_submission.csv"
if not os.path.exists(sample_path):
    sample_path = "/kaggle/input/sample_submission.csv"

if os.path.exists(sample_path):
    sample = pd.read_csv(sample_path)
    sub = sample[["id"]].merge(sub, on="id", how="left")
    sub["label"] = sub["label"].fillna(0.5)

sub.to_csv(output_csv, index=False)
print(f"CSVファイル {output_csv} を作成しました")
print(sub.head())
print("Rows:", len(sub), "Cols:", list(sub.columns))
print("label range:", float(sub["label"].min()), float(sub["label"].max()))
