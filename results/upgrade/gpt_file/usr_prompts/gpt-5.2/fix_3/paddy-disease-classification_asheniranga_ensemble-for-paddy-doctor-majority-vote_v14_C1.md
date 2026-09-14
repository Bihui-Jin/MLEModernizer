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

Higher is better.

# 8. Previous improvement plan

- What this solution (achieved 0.17487) has done: 'I fix two execution blockers so the notebook runs end-to-end and always writes a valid `submission.csv`. First, the TensorFlow import is failing due to a protobuf incompatibility in this environment, so I pin a compatible protobuf version at runtime (before importing TF) and then import TF safely. Second, `scipy.stats.mode` no longer supports non-numeric arrays in SciPy>=1.11, so I replace that with a deterministic row-wise majority-vote implementation that works on string labels. These fixes are score-neutral in intent (they restore the same ensemble voting logic) and ensure the submission has the required columns and row alignment with `sample_submission.csv`.'

# 9. Code solution

## === cell 0
import os
import sys
import random
import subprocess
import numpy as np
import pandas as pd


def _ensure_compatible_protobuf():
    try:
        import google.protobuf  # noqa: F401
        from google.protobuf import __version__ as pb_ver

        major = int(pb_ver.split(".")[0])
        if major >= 5:
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "-q", "protobuf==4.25.3"]
            )
            import importlib

            importlib.invalidate_caches()
    except Exception as e:
        print("WARNING: protobuf compatibility check failed:", repr(e))


_ensure_compatible_protobuf()

import tensorflow as tf  # noqa: E402

try:
    import tensorflow_hub as hub  # used only as a custom object placeholder in load_model
except Exception:
    hub = None

try:
    import tensorflow_addons as tfa  # not used in inference pipeline below
except Exception:
    tfa = None

try:
    import cv2  # used by augmentations; optional
except Exception:
    cv2 = None

try:
    import albumentations as A  # unused; optional
except Exception:
    A = None

from tensorflow.keras.preprocessing.image import ImageDataGenerator  # noqa: E402
from tensorflow.keras.utils import load_img, img_to_array  # noqa: E402

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

print("TF:", tf.__version__)
print("hub available:", hub is not None)
print("cv2 available:", cv2 is not None)



## === cell 1
BASE = "../input/paddy-disease-classification"
TEST_DIR = os.path.join(BASE, "test_images")
SAMPLE_SUB = os.path.join(BASE, "sample_submission.csv")

assert os.path.exists(TEST_DIR), f"Missing test dir: {TEST_DIR}"
assert os.path.exists(SAMPLE_SUB), f"Missing sample submission: {SAMPLE_SUB}"

sample_df = pd.read_csv(SAMPLE_SUB)
print(sample_df.head())
print("sample rows:", len(sample_df))



## === cell 2
if hub is not None:
    keraslayer_obj = hub.KerasLayer
else:

    class _MissingHubKerasLayer(tf.keras.layers.Layer):
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "tensorflow_hub is not available but is required to load these models."
            )

    keraslayer_obj = _MissingHubKerasLayer

CUSTOM_OBJECTS = {"KerasLayer": keraslayer_obj}

model_paths = [
    (
        "m1",
        "../input/paddy-doc-ensemble-models/ensemble_estimators/resnet_150.hdf5",
        "256",
    ),
    (
        "m2",
        "../input/paddy-doc-ensemble-models/ensemble_estimators/effnet_v2_m.hdf5",
        "300",
    ),
    ("m3", "../input/k/kasunpramodya/paddy-doctor-training/model_effnet_s.hdf5", "256"),
    ("m4", "../input/notebooka9ca40495e/model_effnet_s.hdf5", "256"),
    ("m5", "../input/paddy-doctor-training/model_effnet_b4.hdf5", "384"),
    (
        "m6",
        "../input/paddy-doc-ensemble-models/ensemble_estimators/effnet_v2_m_na.hdf5",
        "256",
    ),
    (
        "m7",
        "../input/paddy-doc-ensemble-models/ensemble_estimators/resinc_v2.hdf5",
        "256",
    ),
]

models = []
for name, path, imgsize in model_paths:
    if os.path.exists(path):
        try:
            m = tf.keras.models.load_model(
                path, custom_objects=CUSTOM_OBJECTS, compile=False
            )
            models.append((name, m, int(imgsize)))
            print(f"Loaded {name} from {path} (expects {imgsize}x{imgsize})")
        except Exception as e:
            print(f"WARNING: failed to load {name} from {path}: {repr(e)}")
    else:
        print(f"WARNING: model file not found for {name}: {path}")

if len(models) == 0:
    print(
        "WARNING: No models loaded. Will output a valid submission using constant prediction 'normal'."
    )



## === cell 3
input_size = 256  # used only by center_crop_and_random_augmentations_fn; not used in test generators below.


def random_cutout(image, patch_size=16, patches=16):
    if random.choice([True, False]):
        anchors_x, anchors_y = [], []
        for _ in range(patches):
            rv = np.random.randint(0, image.shape[0])
            if rv not in anchors_x:
                anchors_x.append(rv)
        for _ in range(patches):
            rv = np.random.randint(0, image.shape[1])
            if rv not in anchors_y:
                anchors_y.append(rv)
        for x, y in zip(anchors_x, anchors_y):
            image[x : x + patch_size, y : y + patch_size, :] = 0
    return image


def random_gaus_blur(image):
    if cv2 is None:
        return image
    if random.choice([True, False]):
        return cv2.GaussianBlur(image, (7, 7), 0)
    return image


def random_displacment(image):
    if random.choice([True, False]):
        ax = random.choice([0, 1])
        if image.shape[ax] % 6 != 0:
            return image
        slices = np.split(image, 6, axis=ax)
        np.random.shuffle(slices)
        return np.row_stack(slices) if ax == 0 else np.column_stack(slices)
    return image


def center_crop_and_random_augmentations_fn(image):
    image = tf.image.random_crop(image, (input_size, input_size, 3)).numpy()
    image = random_cutout(image, 16, 16)
    image = random_displacment(image)
    image = random_gaus_blur(image)
    image = tf.image.random_brightness(image, 0.2)
    image = tf.image.random_contrast(image, 0.5, 2.0)
    image = tf.image.random_saturation(image, 0.75, 1.25)
    image = tf.image.random_hue(image, 0.1).numpy()
    return image


def test_time_augmentation_fn_1(image):
    image = tf.image.random_crop(image, (256, 256, 3)).numpy()
    image = tf.image.random_brightness(image, 0.2)
    image = tf.image.random_contrast(image, 0.5, 2.0)
    return image


def test_time_augmentation_fn_2(image):
    image = tf.image.random_crop(image, (384, 384, 3)).numpy()
    image = tf.image.random_brightness(image, 0.2)
    image = tf.image.random_contrast(image, 0.5, 2.0)
    return image


def test_time_augmentation_fn_3(image):
    image = tf.image.random_crop(image, (300, 300, 3)).numpy()
    image = tf.image.random_brightness(image, 0.2)
    image = tf.image.random_contrast(image, 0.5, 2.0)
    return image




## === cell 4
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

files_to_fit = [
    f"{BASE}/train_images/bacterial_leaf_blight/100049.jpg",
    f"{BASE}/train_images/bacterial_leaf_streak/100042.jpg",
    f"{BASE}/train_images/bacterial_panicle_blight/100068.jpg",
    f"{BASE}/train_images/blast/100012.jpg",
    f"{BASE}/train_images/brown_spot/100022.jpg",
    f"{BASE}/train_images/dead_heart/100020.jpg",
    f"{BASE}/train_images/downy_mildew/100059.jpg",
    f"{BASE}/train_images/hispa/100139.jpg",
    f"{BASE}/train_images/normal/100111.jpg",
    f"{BASE}/train_images/tungro/100134.jpg",
]
to_gen_fit = []
for fp in files_to_fit:
    if os.path.exists(fp):
        to_gen_fit.append(img_to_array(load_img(fp), dtype="uint8"))
    else:
        print("WARNING: missing fit image:", fp)

if len(to_gen_fit) > 0:
    generator_2.fit(np.array(to_gen_fit), augment=False)
else:
    print(
        "WARNING: generator_2.fit() skipped; disabling featurewise centering/normalization."
    )
    generator_2 = ImageDataGenerator(
        rescale=1.0 / 255,
        horizontal_flip=True,
        vertical_flip=True,
        preprocessing_function=test_time_augmentation_fn_2,
    )




## === cell 5
def resolve_flow_dir(test_dir: str) -> str:
    jpgs = [f for f in os.listdir(test_dir) if f.lower().endswith(".jpg")]
    if len(jpgs) > 0:
        return test_dir
    nested = os.path.join(test_dir, "test_images")
    if os.path.isdir(nested):
        jpgs2 = [f for f in os.listdir(nested) if f.lower().endswith(".jpg")]
        if len(jpgs2) > 0:
            return nested
    return test_dir


test_loc = resolve_flow_dir(TEST_DIR)
print("Using test_loc:", test_loc)

work_root = "/kaggle/working/test_flow"
os.makedirs(work_root, exist_ok=True)
class_dir = os.path.join(work_root, ".")
os.makedirs(class_dir, exist_ok=True)

test_files = sorted([f for f in os.listdir(test_loc) if f.lower().endswith(".jpg")])
if len(test_files) == 0:
    raise RuntimeError(f"No .jpg files found in {test_loc}")

existing = set(os.listdir(class_dir))
for fn in test_files:
    if fn in existing:
        continue
    src = os.path.join(test_loc, fn)
    dst = os.path.join(class_dir, fn)
    try:
        os.symlink(src, dst)
    except Exception:
        import shutil

        shutil.copy2(src, dst)

test_data_256 = generator_1.flow_from_directory(
    directory=work_root,
    target_size=(256, 256),
    batch_size=32,
    classes=["."],
    shuffle=False,
)

test_data_300 = generator_3.flow_from_directory(
    directory=work_root,
    target_size=(300, 300),
    batch_size=16,
    classes=["."],
    shuffle=False,
)

test_data_384 = generator_2.flow_from_directory(
    directory=work_root,
    target_size=(384, 384),
    batch_size=16,
    classes=["."],
    shuffle=False,
)

n_test = test_data_256.samples
print("n_test:", n_test)



## === cell 6
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
inverse_map = {v: k for k, v in class_indices.items()}




## === cell 7
def predict_with_tta(model, gen, n_samples, tta=5):
    enc = np.zeros((n_samples, 10), dtype=np.float32)
    for _ in range(tta):
        gen.reset()
        pred = model.predict(gen, verbose=1)
        enc += pred.astype(np.float32)
    enc /= float(tta)
    return enc


test_encodings = []

if len(models) > 0:
    for name, m, sz in models:
        if sz == 256:
            gen = test_data_256
        elif sz == 300:
            gen = test_data_300
        elif sz == 384:
            gen = test_data_384
        else:
            raise ValueError(f"Unknown expected size {sz} for model {name}")
        print(f"Predicting {name} with input {sz}x{sz}")
        test_encodings.append(predict_with_tta(m, gen, n_test, tta=5))



## === cell 8
if len(models) > 0:
    predict_max = [np.argmax(test_enc, axis=1) for test_enc in test_encodings]
    predictions = []
    for enc in predict_max:
        predictions.append([inverse_map[int(k)] for k in enc])
else:
    predictions = [["normal"] * n_test]

files = test_data_256.filenames
files = [f.replace("\\", "/") for f in files]
files = [f.replace("./", "").replace(".//", "").replace("././", "") for f in files]
files = [os.path.basename(f) for f in files]

print(files[:5], len(files), len(predictions[0]))




## === cell 9
def rowwise_string_mode(df: pd.DataFrame) -> np.ndarray:
    arr = df.to_numpy(dtype=object)
    out = np.empty((arr.shape[0],), dtype=object)
    for i in range(arr.shape[0]):
        row = arr[i, :]
        counts = {}
        best_label = None
        best_count = -1
        for lab in row:
            counts[lab] = counts.get(lab, 0) + 1
            c = counts[lab]
            if c > best_count:
                best_count = c
                best_label = lab
        out[i] = best_label
    return out


s_full = pd.DataFrame({"image_id": files})

for pres in predictions:
    temp = pd.DataFrame({"image_id": files, "label": pres})
    s_full = pd.merge(s_full, temp, on="image_id", how="left")

label_cols = [c for c in s_full.columns if c != "image_id"]
s_full = s_full.set_index("image_id")[label_cols]

final_labels = rowwise_string_mode(s_full)
final = pd.DataFrame({"image_id": s_full.index.values, "label": final_labels})

final = sample_df[["image_id"]].merge(final, on="image_id", how="left")
final["label"] = final["label"].fillna("normal")

print(final.head())
print("final rows:", len(final))
print(final["label"].value_counts().head())



## === cell 10
out_path = "submission.csv"
final.to_csv(out_path, index=False)
print("Wrote:", out_path)
print("Columns:", list(final.columns))
print("File size:", os.path.getsize(out_path), "bytes")
