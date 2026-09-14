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
Create a classifier to predict the severity of diabetic retinopathy.

## Metric
Quadratic weighted kappa, which measures the agreement between two ratings. This metric typically varies from 0 (random agreement between raters) to 1 (complete agreement between raters). In the event that there is less agreement between the raters than expected by chance, this metric may go below 0. The quadratic weighted kappa is calculated between the scores assigned by the human rater and the predicted scores.

Images have five possible ratings, 0,1,2,3,4.  Each image is characterized by a tuple *(e*,*e)*, which corresponds to its scores by *Rater A* (human) and *Rater B* (predicted).  The quadratic weighted kappa is calculated as follows. First, an N x N histogram matrix *O* is constructed, such that *O* corresponds to the number of images that received a rating *i* by *A* and a rating *j* by *B*. An *N-by-N* matrix of weights, *w*, is calculated based on the difference between raters' scores:

An *N-by-N* histogram matrix of expected ratings, *E*, is calculated, assuming that there is no correlation between rating scores.  This is calculated as the outer product between each rater's histogram vector of ratings, normalized such that *E* and *O* have the same sum.

## Submission Format
```
id_code,diagnosis
0005cfc8afb6,0
003f0afdcd15,0
etc.
```

## Dataset
You are provided with a large set of retina images taken using [fundus photography](https://en.wikipedia.org/wiki/Fundus_photography) under a variety of imaging conditions.

Labels are on a scale of 0 to 4:

> 0 - No DR
> 1 - Mild
> 2 - Moderate
> 3 - Severe
> 4 - Proliferative DR

Images may contain artifacts, be out of focus, underexposed, or overexposed. The images were gathered from multiple clinics using a variety of cameras over an extended period of time, which will introduce further variation.

- **train.csv** - the training labels
- **test.csv** - the test set (you must predict the `diagnosis` value for these variables)
- **sample_submission.csv** - a sample submission file in the correct format
- **train.zip** - the training set images
- **test.zip** - the public test set images

# 2. Python version

3.10

# 3. Installed packages

No external packages required in the script and installed.

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (118 lines)
            sample_submission.csv (368 lines)
            sample_submission.csv.zip (3.2 kB)
            test.csv (368 lines)
            test.csv.zip (2.9 kB)
            test.zip (160 Bytes)
            test_images.zip (902.9 MB)
            train.csv (3296 lines)
            train.csv.zip (27.5 kB)
            train.zip (162 Bytes)
            train_images.zip (7.7 GB)
            aptos2019-blindness-detection/
                description.md (118 lines)
                sample_submission.csv (368 lines)
                ... and 9 other files
                aptos2019-blindness-detection/
                test_images/
                    218c822a3dd9.png (5.7 MB)
                    0e82bcacc475.png (5.2 MB)
                    ... and 365 other files
                    test_images/
                train_images/
                    184a185e7447.png (337.5 kB)
                    c4aef0d88d1b.png (876.6 kB)
                    ... and 3293 other files
                    train_images/
            test_images/
                218c822a3dd9.png (5.7 MB)
                0e82bcacc475.png (5.2 MB)
                ... and 365 other files
                test_images/
            train_images/
                184a185e7447.png (337.5 kB)
                c4aef0d88d1b.png (876.6 kB)
                ... and 3293 other files
                train_images/
        input/
            description.md (118 lines)
            sample_submission.csv (368 lines)
            sample_submission.csv.zip (3.2 kB)
            test.csv (368 lines)
            test.csv.zip (2.9 kB)
            test.zip (160 Bytes)
            test_images.zip (902.9 MB)
            train.csv (3296 lines)
            train.csv.zip (27.5 kB)
            train.zip (162 Bytes)
            train_images.zip (7.7 GB)
            aptos2019-blindness-detection/
                description.md (118 lines)
                sample_submission.csv (368 lines)
                ... and 9 other files
                aptos2019-blindness-detection/
                test_images/
                    218c822a3dd9.png (5.7 MB)
                    0e82bcacc475.png (5.2 MB)
                    ... and 365 other files
                    test_images/
                train_images/
                    184a185e7447.png (337.5 kB)
                    c4aef0d88d1b.png (876.6 kB)
                    ... and 3293 other files
                    train_images/
            test_images/
                218c822a3dd9.png (5.7 MB)
                0e82bcacc475.png (5.2 MB)
                ... and 365 other files
                test_images/
                    218c822a3dd9.png (5.7 MB)
                    0e82bcacc475.png (5.2 MB)
                    ... and 365 other files
                    test_images/
            train_images/
                184a185e7447.png (337.5 kB)
                c4aef0d88d1b.png (876.6 kB)
                ... and 3293 other files
                train_images/
                    184a185e7447.png (337.5 kB)
                    c4aef0d88d1b.png (876.6 kB)
                    ... and 3293 other files
                    train_images/
        working/
            aptos2019-blindness-detection/
                description.md (118 lines)
                sample_submission.csv (368 lines)
                ... and 9 other files
                aptos2019-blindness-detection/
                test_images/
                    218c822a3dd9.png (5.7 MB)
                    0e82bcacc475.png (5.2 MB)
                    ... and 365 other files
                    test_images/
                train_images/
                    184a185e7447.png (337.5 kB)
                    c4aef0d88d1b.png (876.6 kB)
                    ... and 3293 other files
                    train_images/
```

-> data/aptos2019-blindness-detection/sample_submission.csv has 367 rows and 2 columns.
The columns are: id_code, diagnosis

-> data/aptos2019-blindness-detection/test.csv has 367 rows and 1 columns.
The columns are: id_code

-> data/aptos2019-blindness-detection/train.csv has 3295 rows and 2 columns.
The columns are: id_code, diagnosis

-> data/sample_submission.csv has 367 rows and 2 columns.
The columns are: id_code, diagnosis

-> data/test.csv has 367 rows and 1 columns.
The columns are: id_code

-> data/train.csv has 3295 rows and 2 columns.
The columns are: id_code, diagnosis

-> input/aptos2019-blindness-detection/sample_submission.csv has 367 rows and 2 columns.
The columns are: id_code, diagnosis

-> (stopped after 10 files for performance)

# 5. Target score

0.8508778503083017

# 6. Current score

0.05524

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.46472) has done: 'I remove the failing `pip install tensorflow-addons` step and the `tensorflow_addons` import, since it triggers the protobuf `MessageFactory.GetPrototype` error in this environment and is not used by the inference-only pipeline. I also fix the cell numbering/order and ensure all required imports (notably `pandas` and `keras`) are available when used, eliminating the `NameError`s. Since the referenced external model path `../input/none-eff-b0-model/none_eff_b0_model` is not present in your provided dataset, I keep the same “load a saved model and run predictions” core logic but add a safe fallback that trains a small CNN on the provided `train_images/` so the notebook always runs end-to-end and produces `submission.csv`. Finally, I ensure correct preprocessing (including scaling to [0,1]) and that predictions are valid integer classes 0–4 in the required submission format.'
- What this solution (achieved 0.44641) has done: 'The crash happens before any training/inference because importing TensorFlow triggers a protobuf incompatibility (`MessageFactory.GetPrototype`). The smallest reliable fix in this Kaggle environment is to force TensorFlow to use the pure-Python protobuf implementation *before* importing TensorFlow/keras, which avoids the failing C++/binary protobuf path. After that, the rest of your pipeline can run unchanged: it still try to load the external model if present and otherwise train the same fallback CNN and write `submission.csv` in the required format. No score-oriented changes are made beyond restoring execution (your current score is low mainly because the external model is missing; improving that materially would require larger modeling changes than allowed).'
- What this solution (achieved 0.54304) has done: 'The immediate blocker is the protobuf/TensorFlow incompatibility: setting `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` isn’t sufficient in this environment because TensorFlow still hits a `MessageFactory.GetPrototype` path. The minimal reliable fix is to pin protobuf to the compatible implementation *before importing TensorFlow* by forcing the pure-Python backend and importing `google.protobuf` first, plus defensively falling back to `tf.keras` (and only importing standalone `keras` if it works). Once TensorFlow imports cleanly, your existing logic (load external model if present else train the fallback CNN) run end-to-end and write a valid `submission.csv`. I’m not changing the model/training loop beyond these compatibility/import fixes, so score changes come only from restoring proper execution (and any minor determinism differences).'
- What this solution (achieved 0.42185) has done: 'We fix the crash that happens before any training/inference by addressing the TensorFlow↔protobuf incompatibility that triggers `MessageFactory.GetPrototype` during `import tensorflow`. The most reliable minimal fix in this environment is to (1) force the pure-Python protobuf backend, and (2) monkey-patch the missing `GetPrototype` method to call `GetMessageClass` *before* importing TensorFlow so TF can import successfully. After TF imports, the rest of your pipeline (external model load if present, otherwise the same fallback CNN training + prediction loop) remains unchanged and produce a valid `submission.csv`. This should also improve score versus the current run only insofar as it restores correct execution (no score-harming logic changes beyond import compatibility).'
- What this solution (achieved 0.23879) has done: 'The immediate crash is caused by a protobuf API mismatch: in this environment `google.protobuf.message_factory.MessageFactory` may exist, but the TensorFlow import path calls `GetPrototype` on an *instance* that doesn’t have it. I fix this by applying a safer monkey-patch that adds `GetPrototype` to both `MessageFactory` and `message_factory.MessageFactory`, and also patching the specific default factory instance if present, all *before* importing TensorFlow. I keep the rest of your pipeline (path resolution, external model loading, fallback CNN training, and submission writing) unchanged so behavior/score changes only come from restoring correct execution. This allow the notebook to run end-to-end and reliably write a valid `submission.csv`.'
- What this solution (achieved 0.4477) has done: 'The pipeline is currently blocked at `import tensorflow` due to a protobuf `MessageFactory.GetPrototype` mismatch; your patch doesn’t reliably attach `GetPrototype` to the exact class instance TensorFlow uses. I apply a more robust pre-TF monkey-patch that (1) forces pure-Python protobuf and (2) adds `GetPrototype` to both the C++ and Python `MessageFactory` classes as well as the default factory instance when present, so TensorFlow can import consistently. After TensorFlow imports, I keep your existing training/inference logic unchanged and ensure `submission.csv` is written with the correct columns and integer labels 0–4. No score-tuning changes are introduced beyond restoring correct execution (your low score is mainly from relying on the small fallback model).'
- What this solution (achieved 0.53093) has done: 'The current crash happens at `import tensorflow` because TensorFlow calls `GetPrototype()` on a *specific* protobuf `MessageFactory` instance/class that still lacks that method in this environment. I apply a more robust protobuf patch **before** importing TensorFlow by adding `GetPrototype` to both Python and (if present) C++ message factory classes *and* to the default/global factory object(s) that TensorFlow commonly uses. This is a correctness/stability fix only (no model/training logic changes), so your pipeline should run end-to-end and write a valid `submission.csv`. After TensorFlow imports, everything else (fallback CNN, training loop, inference, submission formatting) stays the same.'
- What this solution (achieved 0.38516) has done: 'The immediate blocker is still the protobuf/TensorFlow incompatibility: TensorFlow is calling `GetPrototype()` on a protobuf `MessageFactory` instance that your patch doesn’t reliably modify, so the import fails before any training/inference can run. I make the protobuf patch more robust by (1) forcing the Python protobuf implementation, and (2) monkey-patching `GetPrototype` onto both the python and C++ message factory classes *and* onto the global/default factory instance returned by `message_factory.GetMessages()`/`symbol_database.Default()` paths, before importing TensorFlow. After TensorFlow imports, I keep your model/training/inference logic unchanged so score changes only come from restoring correct execution (no intentional score tuning). The script then run end-to-end and write a valid `submission.csv` with the required columns.'
- What this solution (achieved 0.50021) has done: 'We fix the TensorFlow import crash by applying a more direct protobuf compatibility patch: ensure the pure-Python protobuf backend is used and add a `GetPrototype` method to the exact `MessageFactory` class/instances TensorFlow touches (including the global default factory and symbol database factory), before importing TensorFlow. This is a runtime/stability fix only and keeps your model/training/inference logic unchanged. After TensorFlow imports successfully, the rest of the pipeline run end-to-end and write `submission.csv` in the required format. No intentional score-tuning changes are introduced beyond restoring a working run.'
- What this solution (achieved 0.39317) has done: 'The failure is happening before any training/inference because TensorFlow is still encountering a protobuf `MessageFactory.GetPrototype` call on a factory instance that our previous patch didn’t reliably touch. I make the protobuf patch more robust by also patching the global factory returned by `google.protobuf.message_factory._InternalCreateMessageClass`/`GetMessageClass` pathways and the default symbol database’s factory, then import TensorFlow only after that succeeds. The rest of the pipeline (path resolution, optional external model load, fallback CNN training, per-image test inference, and writing `submission.csv`) is kept the same so behavior is unchanged beyond fixing the runtime import crash. This should run end-to-end and produce a valid submission CSV.'
- What this solution (achieved 0.48771) has done: 'The run is currently blocked because TensorFlow import still hits a protobuf `MessageFactory.GetPrototype` mismatch; the existing monkey-patch only installs `GetPrototype` when `GetMessageClass` exists, but in this environment TF ends up using a factory that has neither (so the patch never applies). I replace that patch with a simpler, more robust one that unconditionally adds `GetPrototype` to protobuf `MessageFactory` classes/instances using `GetMessageClass` *or* `GetMessages` when available, and I also patch the default symbol database factory before importing TensorFlow. After TensorFlow imports, I keep your model/training/inference core logic unchanged so the pipeline runs end-to-end and writes a valid `submission.csv`. No score-tuning changes are introduced beyond restoring correct execution.'
- What this solution (achieved 0.20591) has done: 'The immediate runtime blocker is that TensorFlow import still triggers a protobuf `MessageFactory.GetPrototype` call on a factory instance that our patch didn’t actually modify (because some objects don’t expose `GetMessageClass`/`GetMessages`, so we never install anything). I make the protobuf patch unconditional and more direct by attaching a safe `GetPrototype` implementation to the concrete factory class *and* the specific default/global factory instance that TensorFlow uses, before importing TensorFlow. This is a pure compatibility fix (no model/training/prediction logic changes), so it should run end-to-end and still write a valid `submission.csv`. With the pipeline reliably running, any score change comes only from restored execution stability (the model logic remains the same).'
- What this solution (achieved 0.0943) has done: 'Your current gap to the target is large (0.20591 → 0.85088), and the main reason is that the fallback model is too weak and (more importantly) its training preprocessing differs from your intended Ben Graham-style `preprocessing()` while inference uses only a simple crop+resize. I keep the exact same fallback CNN architecture/loss/training loop, but make preprocessing consistent between training and inference by using the same `preprocessing()` pipeline for both. I also switch inference to batched prediction (same model outputs, just faster) so we can afford a slightly larger training subset within the same time budget, which should improve kappa without changing the core approach. Finally, I ensure labels remain integer 0–4 and the submission format stays identical.'
- What this solution (achieved 0.05524) has done: 'Your current score (0.0943) is far below the target (0.8509), so we should make a small but meaningful improvement without changing the CNN architecture or the training loop structure. The biggest score drag in your fallback pipeline is train/test mismatch plus double-rescaling: you preprocess to [0,1] and then apply a `Rescaling(1/255)` again, effectively shrinking inputs to ~[0,0.0039]. I fix this by keeping your Ben Graham preprocessing but feeding the model true float32 [0,1] and removing the extra rescaling layer (no architecture change in layers beyond that single redundant normalization). I also keep the same split/epochs/optimizer/loss, but make the training data array dtype consistent (float32) so the model actually learns from the intended input scale, which should move QWK substantially upward toward your target.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "2")

import gc
import numpy as np
import pandas as pd
import cv2

import google.protobuf  # noqa: F401


def _patch_protobuf_getprototype():
    """
    TensorFlow (in some Kaggle images) expects protobuf MessageFactory instances to have GetPrototype.
    Some protobuf builds here don't expose it, causing: AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'
    We patch both the class and the default/global factory instance(s) so the exact object TF calls is covered.
    """
    try:
        from google.protobuf import message_factory as mf
        from google.protobuf import symbol_database as _symbol_database

        def _fallback_getprototype(self, descriptor):
            gmc = getattr(self, "GetMessageClass", None)
            if callable(gmc):
                return gmc(descriptor)

            gms = getattr(self, "GetMessages", None)
            if callable(gms):
                try:
                    msgs = gms([descriptor.file])
                    cls = msgs.get(descriptor.full_name)
                    if cls is not None:
                        return cls
                except Exception:
                    pass

            try:
                db = _symbol_database.Default()
                pool = getattr(db, "pool", None)
                if pool is not None and hasattr(pool, "FindMessageTypeByName"):
                    _ = pool.FindMessageTypeByName(descriptor.full_name)
            except Exception:
                pass
            raise AttributeError(
                "GetPrototype fallback could not resolve message class"
            )

        def _install_on_obj(obj):
            if obj is None:
                return
            if getattr(obj, "GetPrototype", None) is not None:
                return
            try:
                setattr(obj, "GetPrototype", _fallback_getprototype)
            except Exception:
                try:
                    obj.GetPrototype = _fallback_getprototype.__get__(
                        obj, obj.__class__
                    )
                except Exception:
                    pass

        try:
            from google.protobuf.message_factory import (
                MessageFactory as PyMessageFactory,
            )

            _install_on_obj(PyMessageFactory)
        except Exception:
            pass

        _install_on_obj(getattr(mf, "MessageFactory", None))

        try:
            from google.protobuf.pyext import _message  # type: ignore

            _install_on_obj(getattr(_message, "MessageFactory", None))
        except Exception:
            pass

        for attr in (
            "_DEFAULT_MESSAGE_FACTORY",
            "default_factory",
            "_message_factory",
            "message_factory",
        ):
            _install_on_obj(getattr(mf, attr, None))

        try:
            db = _symbol_database.Default()
            _install_on_obj(getattr(db, "_factory", None))
        except Exception:
            pass

    except Exception:
        pass


_patch_protobuf_getprototype()

import tensorflow as tf

try:
    import keras as _standalone_keras  # noqa: F401

    keras = tf.keras
except Exception:
    keras = tf.keras

layers = keras.layers

SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

print("TensorFlow:", tf.__version__)
print("Using keras from:", keras.__name__ if hasattr(keras, "__name__") else str(keras))



## === cell 1
"""
    Config
"""
IMG_SIZE = 224
BATCH_SIZE = 16


def crop_image_from_gray(img, tol=7):
    if img is None:
        return None
    if img.ndim == 2:
        mask = img > tol
        if mask.any():
            return img[np.ix_(mask.any(1), mask.any(0))]
        return img
    elif img.ndim == 3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        mask = gray_img > tol

        check_shape = img[:, :, 0][np.ix_(mask.any(1), mask.any(0))].shape[0]
        if check_shape == 0:
            return img
        img1 = img[:, :, 0][np.ix_(mask.any(1), mask.any(0))]
        img2 = img[:, :, 1][np.ix_(mask.any(1), mask.any(0))]
        img3 = img[:, :, 2][np.ix_(mask.any(1), mask.any(0))]
        return np.stack([img1, img2, img3], axis=-1)
    return img


def load_ben_color(image, sigmaX=10):
    image = crop_image_from_gray(image)
    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    image = cv2.addWeighted(image, 4, cv2.GaussianBlur(image, (0, 0), sigmaX), -4, 128)
    return image.astype("float32")


"""
    Preprocessing for ImageDataGenerator since ImageDataGenerator reads images in rgb mode, while opencv in bgr
"""


def preprocessing(image, sigmaX=10):
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    image = crop_image_from_gray(image).astype("uint8")
    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    image = cv2.addWeighted(image, 4, cv2.GaussianBlur(image, (0, 0), sigmaX), -4, 128)
    return image.astype("float32") / 255.0




## === cell 2
BASE_CANDIDATES = [
    "/kaggle/input/aptos2019-blindness-detection",
    "/kaggle/data/aptos2019-blindness-detection",
    "/kaggle/input",
    "/kaggle/data",
]


def resolve_path(*parts):
    for base in BASE_CANDIDATES:
        p = os.path.join(base, *parts)
        if os.path.exists(p):
            return p
    p = os.path.join(*parts)
    if os.path.exists(p):
        return p
    return None


train_csv_path = resolve_path("train.csv") or resolve_path(
    "aptos2019-blindness-detection", "train.csv"
)
test_csv_path = resolve_path("test.csv") or resolve_path(
    "aptos2019-blindness-detection", "test.csv"
)
sample_sub_path = resolve_path("sample_submission.csv") or resolve_path(
    "aptos2019-blindness-detection", "sample_submission.csv"
)

train_img_dir = resolve_path("train_images") or resolve_path(
    "aptos2019-blindness-detection", "train_images"
)
test_img_dir = resolve_path("test_images") or resolve_path(
    "aptos2019-blindness-detection", "test_images"
)

assert (
    train_csv_path and test_csv_path and sample_sub_path
), "Could not resolve CSV paths."
assert train_img_dir and test_img_dir, "Could not resolve image directories."

print("train_csv_path:", train_csv_path)
print("test_csv_path:", test_csv_path)
print("sample_sub_path:", sample_sub_path)
print("train_img_dir:", train_img_dir)
print("test_img_dir:", test_img_dir)

train_df = pd.read_csv(train_csv_path)
test_df = pd.read_csv(test_csv_path)
sub_df = pd.read_csv(sample_sub_path)

print(train_df.shape, test_df.shape, sub_df.shape)
print(train_df.head(2))



## === cell 3
external_model_path = "/kaggle/input/none-eff-b0-model/none_eff_b0_model"
model = None

if os.path.exists(external_model_path):
    try:
        model = keras.models.load_model(external_model_path)
        print("Loaded external model:", external_model_path)
    except Exception as e:
        print(
            "Failed to load external model; will train fallback model. Error:", repr(e)
        )
        model = None
else:
    print(
        "External model not found at",
        external_model_path,
        "; will train fallback model.",
    )




## === cell 4
def read_image_rgb(path):
    img = cv2.imread(path)
    if img is None:
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = crop_image_from_gray(img).astype("uint8")
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    return img


def read_image_preprocessed_float(path):
    """
    Score-relevant correctness fix: this function already returns float32 in [0,1].
    We will feed this directly to the model as float32 (and avoid any second Rescaling),
    preventing an unintended double-normalization that severely hurts learning.
    """
    img = cv2.imread(path)
    if img is None:
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return preprocessing(img)  # float32 [0,1]


def build_fallback_model():
    inputs = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))

    x = inputs

    x = layers.Conv2D(16, 3, padding="same", activation="relu")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(32, 3, padding="same", activation="relu")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(5, activation="softmax")(x)
    m = keras.Model(inputs, outputs)
    m.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return m


if model is None:
    max_train_samples = 2000  # keep as-is (already increased previously)
    df_shuf = train_df.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    df_use = df_shuf.iloc[: min(max_train_samples, len(df_shuf))].copy()

    X = np.zeros((len(df_use), IMG_SIZE, IMG_SIZE, 3), dtype=np.float32)
    y = df_use["diagnosis"].astype(np.int64).values

    for i, idc in enumerate(df_use["id_code"].values):
        p = os.path.join(train_img_dir, f"{idc}.png")
        img_f = read_image_preprocessed_float(p)
        if img_f is None:
            img_f = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.float32)
        X[i] = img_f

    n = len(df_use)
    split = int(n * 0.9)
    X_train, y_train = X[:split], y[:split]
    X_val, y_val = X[split:], y[split:]

    model = build_fallback_model()
    model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=3,
        batch_size=BATCH_SIZE,
        verbose=2,
    )

    del X, X_train, X_val, df_shuf, df_use
    gc.collect()



## === cell 5
id_code = test_df["id_code"].values

X_test = np.zeros((len(id_code), IMG_SIZE, IMG_SIZE, 3), dtype=np.float32)
for i, idc in enumerate(id_code):
    img_path = os.path.join(test_img_dir, f"{idc}.png")
    img_f = read_image_preprocessed_float(img_path)
    if img_f is None:
        img_f = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.float32)
    X_test[i] = img_f

pred = model.predict(X_test, batch_size=BATCH_SIZE, verbose=0)
test_prediction = np.argmax(pred, axis=1).astype(np.int64)
test_prediction = np.clip(test_prediction, 0, 4).astype(np.int64)

del X_test, pred
gc.collect()



## === cell 6
submission = pd.DataFrame({"id_code": id_code, "diagnosis": test_prediction})
submission.to_csv("submission.csv", index=False)

unique, counts = np.unique(test_prediction, return_counts=True)
print(dict(zip(unique.tolist(), counts.tolist())))
print("Wrote submission.csv with shape:", submission.shape)
print(submission.head())
