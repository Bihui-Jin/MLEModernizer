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
Develop a model to classify paddy leaf images into one of the nine disease categories or normal leaf.

## Metric
Categorization accuracy.

## Submission Format
```
image_id,label
200001.jpg,normal
200002.jpg,blast
etc.
```

## Dataset
**train.csv** - The training set

- `image_id` - Unique image identifier corresponds to image file names (.jpg) found in the train_images directory.
- `label` - Type of paddy disease, also the target class. There are ten categories, including the normal leaf.
- `variety` - The name of the paddy variety.
- `age` - Age of the paddy in days.

**sample_submission.csv** - Sample submission file.

**train_images** - Training images stored under different sub-directories corresponding to ten target classes. Filename corresponds to the `image_id` column of `train.csv`.

**test_images** - Test set images.

# 2. Python version

3.10

# 3. Installed packages

No external packages required in the script and installed.

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (72 lines)
            sample_submission.csv (2603 lines)
            sample_submission.csv.zip (7.4 kB)
            test.zip (160 Bytes)
            test_images.zip (205.2 MB)
            train.csv (7806 lines)
            train.csv.zip (40.1 kB)
            train.zip (162 Bytes)
            train_images.zip (614.5 MB)
            paddy-disease-classification/
                description.md (72 lines)
                sample_submission.csv (2603 lines)
                ... and 7 other files
                paddy-disease-classification/
                test_images/
                    102916.jpg (92.1 kB)
                    100596.jpg (83.9 kB)
                    ... and 2600 other files
                    test_images/
                train_images/
                    bacterial_leaf_blight/
                        109831.jpg (91.1 kB)
                        109785.jpg (81.8 kB)
                        ... and 356 other files
                    bacterial_leaf_streak/
                        100394.jpg (99.7 kB)
                        103308.jpg (104.3 kB)
                        ... and 295 other files
                    ... and 9 other folders
            test_images/
                102916.jpg (92.1 kB)
                100596.jpg (83.9 kB)
                ... and 2600 other files
                test_images/
            train_images/
                bacterial_leaf_blight/
                    109831.jpg (91.1 kB)
                    109785.jpg (81.8 kB)
                    ... and 356 other files
                bacterial_leaf_streak/
                    100394.jpg (99.7 kB)
                    103308.jpg (104.3 kB)
                    ... and 295 other files
                ... and 9 other folders
        input/
            description.md (72 lines)
            sample_submission.csv (2603 lines)
            sample_submission.csv.zip (7.4 kB)
            test.zip (160 Bytes)
            test_images.zip (205.2 MB)
            train.csv (7806 lines)
            train.csv.zip (40.1 kB)
            train.zip (162 Bytes)
            train_images.zip (614.5 MB)
            paddy-disease-classification/
                description.md (72 lines)
                sample_submission.csv (2603 lines)
                ... and 7 other files
                paddy-disease-classification/
                test_images/
                    102916.jpg (92.1 kB)
                    100596.jpg (83.9 kB)
                    ... and 2600 other files
                    test_images/
                train_images/
                    bacterial_leaf_blight/
                        109831.jpg (91.1 kB)
                        109785.jpg (81.8 kB)
                        ... and 356 other files
                    bacterial_leaf_streak/
                        100394.jpg (99.7 kB)
                        103308.jpg (104.3 kB)
                        ... and 295 other files
                    ... and 9 other folders
            test_images/
                102916.jpg (92.1 kB)
                100596.jpg (83.9 kB)
                ... and 2600 other files
                test_images/
                    102916.jpg (92.1 kB)
                    100596.jpg (83.9 kB)
                    ... and 2600 other files
                    test_images/
            train_images/
                bacterial_leaf_blight/
                    109831.jpg (91.1 kB)
                    109785.jpg (81.8 kB)
                    ... and 356 other files
                bacterial_leaf_streak/
                    100394.jpg (99.7 kB)
                    103308.jpg (104.3 kB)
                    ... and 295 other files
                ... and 9 other folders
        working/
            paddy-disease-classification/
                description.md (72 lines)
                sample_submission.csv (2603 lines)
                ... and 7 other files
                paddy-disease-classification/
                test_images/
                    102916.jpg (92.1 kB)
                    100596.jpg (83.9 kB)
                    ... and 2600 other files
                    test_images/
                train_images/
                    bacterial_leaf_blight/
                        109831.jpg (91.1 kB)
                        109785.jpg (81.8 kB)
                        ... and 356 other files
                    bacterial_leaf_streak/
                        100394.jpg (99.7 kB)
                        103308.jpg (104.3 kB)
                        ... and 295 other files
                    ... and 9 other folders
```

-> data/paddy-disease-classification/sample_submission.csv has 2602 rows and 2 columns.
The columns are: image_id, label

-> data/paddy-disease-classification/train.csv has 7805 rows and 4 columns.
The columns are: image_id, label, variety, age

-> data/sample_submission.csv has 2602 rows and 2 columns.
The columns are: image_id, label

-> data/train.csv has 7805 rows and 4 columns.
The columns are: image_id, label, variety, age

-> input/paddy-disease-classification/sample_submission.csv has 2602 rows and 2 columns.
The columns are: image_id, label

-> input/paddy-disease-classification/train.csv has 7805 rows and 4 columns.
The columns are: image_id, label, variety, age

-> (stopped after 10 files for performance)

# 5. Target score

0.9827188940092166

# 6. Current score

0.17487

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.17487) has done: 'I fix the immediate runtime crash caused by importing `tensorflow_hub` (it triggers a protobuf incompatibility in this environment) by disabling that optional import and loading models without hub custom objects. Then I fix the SciPy `mode` failure (SciPy 1.11+ no longer accepts non-numeric arrays) by replacing it with a safe per-row majority vote using `pandas.Series.mode`, preserving the ensemble “mode of predicted labels” core logic. Finally, I make submission creation robust (even when no external models load) and ensure the output CSV has the required columns and ordering matching `sample_submission.csv`.'
- What this solution (achieved 0.17487) has done: 'I fix the immediate runtime crash in the first cell by removing the protobuf-incompatible import path (caused by `tensorflow_hub`/protobuf internals) and keeping model loading strictly via `tf.keras.models.load_model`. Then I make the test-time augmentation deterministic per TTA pass (seeded) so it behaves consistently and avoids accidental randomness-induced instability, without changing the overall “TTA then ensemble by majority vote” core logic. Finally, I make submission alignment more robust by ensuring `image_id` formatting matches `sample_submission.csv` exactly and always writing a valid `.csv` file.'
- What this solution (achieved 0.17487) has done: 'I fix the immediate protobuf crash by removing the unused hub-related code path and keeping imports limited to TensorFlow/Keras, so the notebook runs in this Kaggle environment. Then I fix the test directory structure used by `flow_from_directory`: it currently points to `/kaggle/working/test_images` while the images are actually placed in `/kaggle/working/test_images/.`, which makes the generator see 0 images and forces the “all normal” fallback (hence the very low score). Finally, I keep the existing TTA + ensemble-by-majority-vote core logic intact, but ensure predictions align to `sample_submission.csv` and always write a valid `.csv` submission.'
- What this solution (achieved 0.17487) has done: 'I fix the immediate runtime crash by forcing TensorFlow to use the Python protobuf implementation before importing `tensorflow` (this addresses the `MessageFactory.GetPrototype` incompatibility). Then I make the test directory symlink/copy step robust and ensure `flow_from_directory` always sees the intended 2602 images (preventing the “0 images → all normal” fallback that tanks accuracy). Finally, I keep your existing TTA + multi-model majority-vote ensemble logic intact, but add a safe fallback to correctly locate the dataset under `/kaggle/input` and always write a valid `submission.csv`-suffix file.'
- What this solution (achieved 0.17487) has done: 'We fix the protobuf-related crash by removing the unsafe environment forcing and instead importing TensorFlow in the normal Kaggle runtime order, which avoids the `MessageFactory.GetPrototype` incompatibility. Then we make the dataset base-path resolution more robust (supporting both `/kaggle/input/...` and `/kaggle/data/...`) so the test images are always found and `flow_from_directory` sees all 2602 images (preventing the low-score “all normal” fallback). Finally, we keep your existing TTA + multi-model majority-vote ensemble logic intact, but add a small safety fix to ensure predictions are aligned to the sample submission ordering and always write a valid `.csv`.'
- What this solution (achieved 0.17487) has done: 'I fix the runtime crash happening before any data/model code runs by forcing TensorFlow to use the Python protobuf implementation (this specifically addresses the `MessageFactory.GetPrototype` AttributeError seen in this environment). Then I keep the rest of your pipeline intact (same TTA, same generators, same majority-vote ensembling), only adding a small safety check to ensure the test generator actually sees all images and that we always produce a correctly ordered `submission.csv`. These changes are directly targeted at restoring correct inference (instead of falling back/terminating), which should move your accuracy sharply upward toward the target.'
- What this solution (achieved 0.17487) has done: 'We fix the protobuf/TensorFlow import crash by removing the forced `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` override, which is what triggers the `MessageFactory.GetPrototype` error in this Kaggle runtime. Then we keep your inference core logic intact (same generators, same TTA loops, same majority-vote ensemble) but add a minimal safety fix: dynamically derive `class_indices/inverse_map` from the first successfully loaded model (fallback to your static mapping) to avoid label-index mismatches that can catastrophically drop accuracy. Finally, we make `predict_with_tta` robust to models that output logits or wrong class dimension by applying a softmax only when needed and skipping incompatible models, ensuring the pipeline always produces a valid submission CSV.'
- What this solution (achieved 0.17487) has done: 'The crash happens immediately when importing TensorFlow due to an incompatible protobuf runtime; the minimal fix is to force TensorFlow to use the pure-Python protobuf implementation *before* importing `tensorflow`. Your very low score is also consistent with falling back to the “all normal” baseline when no external models are available, so this patch keeps your same TTA + majority-vote ensemble logic but makes model loading more robust by also searching under `/kaggle/input/**` for the `.hdf5` files you already reference (no new models/logic, just finding them). Finally, it hardens the class-index mapping using the Keras `ImageDataGenerator(...).flow_from_directory` class ordering (the standard mapping for these datasets) to avoid label-index mismatches that can catastrophically reduce accuracy while keeping the same argmax+vote semantics. The script still writes a valid `submission.csv` with the exact required columns and sample_submission ordering.'
- What this solution (achieved 0.17487) has done: 'I fix the immediate TensorFlow/protobuf import crash by removing the forced `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` environment override (that setting is what triggers the `MessageFactory.GetPrototype` error in this Kaggle runtime). Then I keep your exact inference core logic (same generators, same TTA loop, same majority-vote ensemble) but make the class-index mapping consistent with the dataset folder/class order by deriving it from `train_images/` (falling back to `train.csv` only if needed), which should prevent catastrophic label-index mismatches and move accuracy sharply upward toward the target. Finally, I keep the existing robust submission alignment to `sample_submission.csv` and ensure a valid `.csv` is written end-to-end.'
- What this solution (achieved 0.17487) has done: 'We fix the TensorFlow/protobuf import crash (`MessageFactory` missing `GetPrototype`) by forcing the pure-Python protobuf implementation *before* importing TensorFlow, which is the minimal environment change that unblocks the entire pipeline. Then we keep your existing TTA + multi-model majority-vote ensemble logic intact, but add a tiny robustness tweak: ensure `flow_from_directory` uses the same deterministic class-name list and that the derived `class_indices` is always used consistently for `inverse_map`/`NUM_CLASSES`. Finally, we keep the submission alignment to `sample_submission.csv` and always write a valid `.csv` submission file.'
- What this solution (achieved 0.17487) has done: 'I fix the immediate TensorFlow/protobuf crash by removing the forced `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` environment override (it is the direct cause of the `MessageFactory.GetPrototype` error in this Kaggle runtime). Then, to move accuracy up toward the target while preserving your same TTA + multi-model ensemble core logic, I make `class_indices` match the dataset’s canonical class-folder order (from `train_images/`) instead of alphabetical sorting, avoiding catastrophic label-index mismatches. Finally, I keep your generator/TTA/majority-vote submission flow intact, only hardening a couple of spots so it always finds 2602 test images and always writes a valid `*.csv` submission aligned to `sample_submission.csv`.'
- What this solution (achieved 0.17487) has done: 'I fix the TensorFlow/protobuf crash by removing the environment override that forces the Python protobuf implementation (it triggers the `MessageFactory.GetPrototype` error here), letting TensorFlow import normally. Then I fix the class-index derivation bug that accidentally includes a nested `train_images` directory as an extra “class” by filtering to only real class folders (those containing images), which restores the expected 10-class mapping and prevents label-index mismatch. Finally, I ensure downstream variables (`FLOW_CLASSES`, `n_test`, `files`, etc.) are always defined by unblocking the earlier failures, so the pipeline runs end-to-end and writes a valid `*.csv` submission aligned to `sample_submission.csv`.'
- What this solution (achieved 0.17487) has done: 'You’re currently crashing on `import tensorflow` due to a protobuf/TensorFlow incompatibility (`MessageFactory.GetPrototype`), so the first change is to force the pure-Python protobuf implementation **before** importing TensorFlow (this is the minimal unblocker). Next, to raise accuracy toward the target (and avoid catastrophic label-index mismatch), we derive `class_indices` directly from `train.csv` label order (stable and correct) instead of relying on folder sorting. Finally, we keep your same TTA + majority-vote ensemble logic, but harden model loading by filtering to only models whose output dimension matches `NUM_CLASSES` so incompatible models don’t silently degrade predictions or trigger fallback behavior, and we still always write a valid `*.csv` submission.'

# 9. Code solution

## === cell 0
import os
import random
import numpy as np
import pandas as pd

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "3")

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import load_img, img_to_array

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

BASE_INPUT_CANDIDATES = [
    "/kaggle/input/paddy-disease-classification",
    "/kaggle/input/paddy-disease-classification/paddy-disease-classification",
    "/kaggle/data/paddy-disease-classification",
    "/kaggle/data/paddy-disease-classification/paddy-disease-classification",
]
BASE_INPUT = None
for p in BASE_INPUT_CANDIDATES:
    if os.path.exists(p):
        BASE_INPUT = p
        break
if BASE_INPUT is None:
    BASE_INPUT = "/kaggle/input/paddy-disease-classification"

TEST_IMAGES_DIR = os.path.join(BASE_INPUT, "test_images")
TRAIN_IMAGES_DIR = os.path.join(BASE_INPUT, "train_images")

print("BASE_INPUT:", BASE_INPUT)
print("TEST_IMAGES_DIR:", TEST_IMAGES_DIR, "exists:", os.path.exists(TEST_IMAGES_DIR))
print(
    "TRAIN_IMAGES_DIR:", TRAIN_IMAGES_DIR, "exists:", os.path.exists(TRAIN_IMAGES_DIR)
)

if not os.path.exists(TEST_IMAGES_DIR):
    raise FileNotFoundError(f"Could not find test_images at: {TEST_IMAGES_DIR}")



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
WORK_TEST_ROOT = "/kaggle/working/test_images"
WORK_TEST_CLASSNAME = "test"
WORK_TEST_CLASSDIR = os.path.join(WORK_TEST_ROOT, WORK_TEST_CLASSNAME)
os.makedirs(WORK_TEST_CLASSDIR, exist_ok=True)

test_imgs = sorted(
    [f for f in os.listdir(TEST_IMAGES_DIR) if f.lower().endswith(".jpg")]
)
existing = set(os.listdir(WORK_TEST_CLASSDIR))

import shutil

copied = 0
linked = 0
for f in test_imgs:
    dst = os.path.join(WORK_TEST_CLASSDIR, f)
    if f in existing:
        if os.path.islink(dst) and (not os.path.exists(dst)):
            os.unlink(dst)
        else:
            continue

    src = os.path.join(TEST_IMAGES_DIR, f)
    try:
        os.symlink(src, dst)
        linked += 1
    except FileExistsError:
        pass
    except OSError:
        shutil.copy2(src, dst)
        copied += 1

print("Test images:", len(test_imgs), "linked:", linked, "copied:", copied)
print("WORK_TEST_CLASSDIR sample:", sorted(os.listdir(WORK_TEST_CLASSDIR))[:5])

if len(test_imgs) == 0:
    raise RuntimeError(f"No .jpg files found in {TEST_IMAGES_DIR}")



## === cell 2
CANDIDATE_MODEL_PATHS = [
    "../input/paddy-doc-ensemble-models/ensemble_estimators/resnet_150.hdf5",
    "../input/paddy-doc-ensemble-models/ensemble_estimators/effnet_v2_m.hdf5",
    "../input/k/kasunpramodya/paddy-doctor-training/model_effnet_s.hdf5",
    "../input/notebooka9ca40495e/model_effnet_s.hdf5",
    "../input/paddy-doctor-training/model_effnet_b4.hdf5",
    "../input/paddy-doc-ensemble-models/ensemble_estimators/effnet_v2_m_na.hdf5",
    "../input/paddy-doc-ensemble-models/ensemble_estimators/resinc_v2.hdf5",
]


def _resolve_under_kaggle_input(path: str) -> str | None:
    if os.path.exists(path):
        return path
    base = os.path.basename(path)
    if not base.lower().endswith((".hdf5", ".h5", ".keras")):
        return None
    for root, _, files in os.walk("/kaggle/input"):
        if ".ipynb_checkpoints" in root:
            continue
        if base in files:
            return os.path.join(root, base)
    return None


def load_keras_model_if_exists(path):
    resolved = _resolve_under_kaggle_input(path)
    if resolved is None or (not os.path.exists(resolved)):
        return None
    try:
        m = tf.keras.models.load_model(resolved, custom_objects={}, compile=False)
        return m, resolved
    except Exception as e:
        print(f"Warning: failed to load model at {resolved}: {e}")
        return None


loaded_models = []
loaded_model_paths = []
for p in CANDIDATE_MODEL_PATHS:
    out = load_keras_model_if_exists(p)
    if out is not None:
        m, rp = out
        loaded_models.append(m)
        loaded_model_paths.append(rp)

print("Loaded models:", len(loaded_models))
for p in loaded_model_paths:
    print(" -", p)



## === cell 3
train_csv_path = os.path.join(BASE_INPUT, "train.csv")
if os.path.exists(train_csv_path):
    _train_df = pd.read_csv(train_csv_path)
    _classes_sorted = sorted(_train_df["label"].unique().tolist())
    class_indices = {c: i for i, c in enumerate(_classes_sorted)}
    print("Using class_indices from train.csv sorted labels:", class_indices)
else:
    class_indices = {
        "bacterial_leaf_blight": 0,
        "bacterial_leaf_streak": 1,
        "bacterial_panicle_blight": 2,
        "blast": 3,
        "brown_spot": 4,
        "dead_heart": 5,
        "downy_mildew": 6,
        "hispa": 7,
        "normal": 8,
        "tungro": 9,
    }
    print("Using static class_indices mapping (train.csv not found).")

if len(class_indices) != 10:
    raise RuntimeError(
        f"Expected 10 classes but got {len(class_indices)}: {class_indices}"
    )

inverse_map = {v: k for k, v in class_indices.items()}
NUM_CLASSES = len(class_indices)
print("NUM_CLASSES:", NUM_CLASSES)

FLOW_CLASSES = [WORK_TEST_CLASSNAME]




## === cell 4
def _stateless_aug_common(image, crop_h, crop_w, seed_pair):
    image = tf.convert_to_tensor(image)
    image = tf.cast(image, tf.float32)
    image = tf.image.stateless_random_crop(
        image, size=(crop_h, crop_w, 3), seed=seed_pair
    )
    image = tf.image.stateless_random_brightness(
        image, max_delta=0.2, seed=seed_pair + tf.constant([1, 0], tf.int32)
    )
    image = tf.image.stateless_random_contrast(
        image, lower=0.5, upper=2.0, seed=seed_pair + tf.constant([2, 0], tf.int32)
    )
    return image.numpy()


_TTA_SEED_PAIR = tf.constant([SEED, 0], dtype=tf.int32)


def test_time_augmentation_fn_1(image):
    return _stateless_aug_common(image, 256, 256, _TTA_SEED_PAIR)


def test_time_augmentation_fn_2(image):
    return _stateless_aug_common(image, 384, 384, _TTA_SEED_PAIR)


def test_time_augmentation_fn_3(image):
    return _stateless_aug_common(image, 300, 300, _TTA_SEED_PAIR)




## === cell 5
generator_1 = ImageDataGenerator(
    rescale=1.0 / 255,
    horizontal_flip=True,
    vertical_flip=True,
    preprocessing_function=test_time_augmentation_fn_1,
)

generator_2 = ImageDataGenerator(
    rescale=1.0 / 255,
    featurewise_center=True,
    featurewise_std_normalization=True,
    horizontal_flip=True,
    vertical_flip=True,
    preprocessing_function=test_time_augmentation_fn_2,
)

generator_3 = ImageDataGenerator(
    rescale=1.0 / 255,
    horizontal_flip=True,
    vertical_flip=True,
    preprocessing_function=test_time_augmentation_fn_3,
)



## === cell 6
files_to_fit = [
    os.path.join(TRAIN_IMAGES_DIR, "bacterial_leaf_blight", "100049.jpg"),
    os.path.join(TRAIN_IMAGES_DIR, "bacterial_leaf_streak", "100042.jpg"),
    os.path.join(TRAIN_IMAGES_DIR, "bacterial_panicle_blight", "100068.jpg"),
    os.path.join(TRAIN_IMAGES_DIR, "blast", "100012.jpg"),
    os.path.join(TRAIN_IMAGES_DIR, "brown_spot", "100022.jpg"),
    os.path.join(TRAIN_IMAGES_DIR, "dead_heart", "100020.jpg"),
    os.path.join(TRAIN_IMAGES_DIR, "downy_mildew", "100059.jpg"),
    os.path.join(TRAIN_IMAGES_DIR, "hispa", "100139.jpg"),
    os.path.join(TRAIN_IMAGES_DIR, "normal", "100111.jpg"),
    os.path.join(TRAIN_IMAGES_DIR, "tungro", "100134.jpg"),
]

to_gen_fit = []
for file in files_to_fit:
    if os.path.exists(file):
        to_gen_fit.append(img_to_array(load_img(file), dtype="uint8"))

if len(to_gen_fit) > 0:
    generator_2.fit(np.array(to_gen_fit))
else:
    print(
        "Warning: could not fit generator_2 (no fit images found). Proceeding without fit."
    )

print("Fit images used for generator_2:", len(to_gen_fit))



## === cell 7
test_data_256 = generator_1.flow_from_directory(
    directory=WORK_TEST_ROOT,
    target_size=(256, 256),
    batch_size=32,
    classes=FLOW_CLASSES,
    shuffle=False,
)

test_data_300 = generator_3.flow_from_directory(
    directory=WORK_TEST_ROOT,
    target_size=(300, 300),
    batch_size=16,
    classes=FLOW_CLASSES,
    shuffle=False,
)

test_data_384 = generator_2.flow_from_directory(
    directory=WORK_TEST_ROOT,
    target_size=(384, 384),
    batch_size=16,
    classes=FLOW_CLASSES,
    shuffle=False,
)

n_test = test_data_256.n
print("n_test:", n_test, "filenames:", len(test_data_256.filenames))

if n_test == 0:
    raise RuntimeError(
        f"flow_from_directory found 0 images under {WORK_TEST_ROOT}. "
        f"Expected ~2602. Check symlink/copy step."
    )




## === cell 8
def predict_with_tta(model, generator, tta=5):
    global _TTA_SEED_PAIR
    preds = np.zeros((generator.n, NUM_CLASSES), dtype=np.float32)

    for t in range(int(tta)):
        _TTA_SEED_PAIR = tf.constant([SEED, t], dtype=tf.int32)
        generator.reset()
        p = model.predict(generator, verbose=1)

        p = np.asarray(p)
        if p.ndim > 2:
            p = np.reshape(p, (p.shape[0], -1))
        if p.shape[1] != NUM_CLASSES:
            raise ValueError(
                f"Model output classes {p.shape[1]} != NUM_CLASSES {NUM_CLASSES}"
            )

        row_sums = p.sum(axis=1)
        if (
            (p.min() < -1e-3)
            or (p.max() > 1.0 + 1e-3)
            or (np.nanmean(np.abs(row_sums - 1.0)) > 0.05)
        ):
            p = tf.nn.softmax(p, axis=1).numpy()

        preds += p.astype(np.float32)

    preds /= float(tta)
    return preds


compatible_models = []
compatible_model_paths = []
for m, mp in zip(loaded_models, loaded_model_paths):
    try:
        out_shape = m.output_shape
        if isinstance(out_shape, list):
            out_shape = out_shape[0]
        if out_shape is None or len(out_shape) < 2 or out_shape[-1] != NUM_CLASSES:
            print(
                f"Skipping model (output_shape={m.output_shape}) != NUM_CLASSES={NUM_CLASSES}: {mp}"
            )
            continue
        compatible_models.append(m)
        compatible_model_paths.append(mp)
    except Exception as e:
        print(f"Skipping model due to output_shape inspection failure: {mp} ({e})")

loaded_models = compatible_models
loaded_model_paths = compatible_model_paths
print("Compatible loaded models:", len(loaded_models))
for p in loaded_model_paths:
    print(" -", p)

test_encodings = []
if len(loaded_models) > 0:
    for m in loaded_models:
        in_shape = m.input_shape
        if isinstance(in_shape, list):
            in_shape = in_shape[0]
        h, w = in_shape[1], in_shape[2]
        try:
            if (h, w) == (256, 256):
                enc = predict_with_tta(m, test_data_256, tta=5)
            elif (h, w) == (300, 300):
                enc = predict_with_tta(m, test_data_300, tta=5)
            elif (h, w) == (384, 384):
                enc = predict_with_tta(m, test_data_384, tta=5)
            else:
                enc = predict_with_tta(m, test_data_256, tta=5)
            test_encodings.append(enc)
        except Exception as e:
            print(
                f"Skipping model due to incompatibility (input_shape={in_shape}): {e}"
            )
            continue
else:
    test_encodings = []

print("Num model encodings:", len(test_encodings))



## === cell 9
if len(test_encodings) == 0:
    baseline = np.zeros((n_test, NUM_CLASSES), dtype=np.float32)
    baseline[:, class_indices.get("normal", 0)] = 1.0
    test_encodings = [baseline]

predict_max = [np.argmax(te, axis=1) for te in test_encodings]
enc_max = [np.max(te, axis=1) for te in test_encodings]

print("Num predictors:", len(predict_max), "shape:", predict_max[0].shape)



## === cell 10
predictions = []
for enc in predict_max:
    predictions.append([inverse_map[int(k)] for k in enc])

files = test_data_256.filenames
files = [os.path.basename(f.replace("\\", "/")) for f in files]

print(files[:5], predictions[0][:5])



## === cell 11
s_full = pd.DataFrame({"image_id": files})
for i in range(len(predictions)):
    temp = pd.DataFrame(
        {"image_id": files, f"label_{i}": predictions[i], f"conf_{i}": enc_max[i]}
    )
    s_full = pd.merge(s_full, temp, on="image_id", how="left")

label_cols = [c for c in s_full.columns if c.startswith("label_")]
labels_df = s_full[label_cols]


def row_majority_vote(row: pd.Series):
    m = row.mode(dropna=True)
    if len(m) == 0:
        return "normal", 0
    chosen = m.iloc[0]  # deterministic tie-break
    count = int((row == chosen).sum())
    return chosen, count


mv = labels_df.apply(row_majority_vote, axis=1, result_type="expand")
mv.columns = ["label", "count"]

final = pd.DataFrame(
    {
        "image_id": s_full["image_id"].values,
        "label": mv["label"].values,
        "count": mv["count"].values,
    }
)
print(final.head())



## === cell 12
submission = final[["image_id", "label"]].copy()

sample_path = os.path.join(BASE_INPUT, "sample_submission.csv")
if os.path.exists(sample_path):
    sample = pd.read_csv(sample_path)
    submission = sample[["image_id"]].merge(submission, on="image_id", how="left")
    submission["label"] = submission["label"].fillna("normal")
else:
    print("Warning: sample_submission.csv not found at", sample_path)
    submission["label"] = submission["label"].fillna("normal")

print(submission.head(), submission.shape)

out_path = "model_submission_v23.csv"
submission.to_csv(out_path, index=False)

print("Wrote:", out_path)
print("Columns:", submission.columns.tolist())
print("NA counts:\n", submission.isna().sum())
print("Top labels:\n", submission["label"].value_counts().head())

assert out_path.endswith(".csv")
assert list(submission.columns) == ["image_id", "label"]
