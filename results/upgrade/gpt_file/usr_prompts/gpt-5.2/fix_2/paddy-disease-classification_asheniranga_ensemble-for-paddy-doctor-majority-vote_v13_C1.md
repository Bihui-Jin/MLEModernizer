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

0.9815668202764976

# 6. Current score

0.17487

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved 0.17487) has done: 'I fix the early import/runtime crash by removing incompatible/unused heavy imports (notably `tensorflow_addons`, `tensorflow_hub`, `albumentations`, etc.) and by loading the `.hdf5` models without requiring `hub.KerasLayer` (since these saved models typically don’t need it at inference time). I also correct the test directory path and the hard-coded test size (your code uses `3469`, but the competition test set is 2602), so predictions and filenames align. Finally, I fix the submission-writing step to ensure the output CSV has exactly `image_id,label` columns (your current file writes the index as `Unnamed: 0`, causing the invalid submission error). These changes are execution/format fixes and preserve the ensemble + TTA voting logic.'

# 9. Code solution

## === cell 0
import os
import random
import numpy as np
import pandas as pd
import scipy.stats as ss

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import load_img, img_to_array


random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
DATA_ROOT = "../input/paddy-disease-classification"
TEST_DIR = os.path.join(DATA_ROOT, "test_images")

if not os.path.isdir(TEST_DIR):
    DATA_ROOT = "../input/paddy-disease-classification/paddy-disease-classification"
    TEST_DIR = os.path.join(DATA_ROOT, "test_images")

assert os.path.isdir(TEST_DIR), f"Could not find test_images directory at {TEST_DIR}"




## === cell 2
def safe_load_model(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found: {path}")
    return tf.keras.models.load_model(path, compile=False)


m1 = safe_load_model(
    "../input/paddy-doc-ensemble-models/ensemble_estimators/resnet_150.hdf5"
)
m2 = safe_load_model(
    "../input/paddy-doc-ensemble-models/ensemble_estimators/effnet_v2_m.hdf5"
)



## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/4290086970.py in <cell line: 0>()
      9 
     10 # These model paths are from your original notebook; they must exist in /input for this to work.
---> 11 m1 = safe_load_model(
     12     "../input/paddy-doc-ensemble-models/ensemble_estimators/resnet_150.hdf5"
     13 )

/tmp/ipykernel_11/4290086970.py in safe_load_model(path)
      4 def safe_load_model(path):
      5     if not os.path.exists(path):
----> 6         raise FileNotFoundError(f"Model file not found: {path}")
      7     return tf.keras.models.load_model(path, compile=False)
      8 

FileNotFoundError: Model file not found: ../input/paddy-doc-ensemble-models/ensemble_estimators/resnet_150.hdf5

## === cell 3
m3 = safe_load_model(
    "../input/k/kasunpramodya/paddy-doctor-training/model_effnet_s.hdf5"
)
m4 = safe_load_model("../input/notebooka9ca40495e/model_effnet_s.hdf5")
m5 = safe_load_model("../input/paddy-doctor-training/model_effnet_b4.hdf5")



## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/2342887452.py in <cell line: 0>()
----> 1 m3 = safe_load_model(
      2     "../input/k/kasunpramodya/paddy-doctor-training/model_effnet_s.hdf5"
      3 )
      4 m4 = safe_load_model("../input/notebooka9ca40495e/model_effnet_s.hdf5")
      5 m5 = safe_load_model("../input/paddy-doctor-training/model_effnet_b4.hdf5")

/tmp/ipykernel_11/4290086970.py in safe_load_model(path)
      4 def safe_load_model(path):
      5     if not os.path.exists(path):
----> 6         raise FileNotFoundError(f"Model file not found: {path}")
      7     return tf.keras.models.load_model(path, compile=False)
      8 

FileNotFoundError: Model file not found: ../input/k/kasunpramodya/paddy-doctor-training/model_effnet_s.hdf5

## === cell 4
input_size = (
    256  # used in center_crop_and_random_augmentations_fn; kept for compatibility
)


def random_cutout(image, patch_size=16, patches=16):
    if random.choice([True, False]):
        anchors_x = []
        anchors_y = []

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
    else:
        return image


def random_gaus_blur(image):
    if random.choice([True, False]):
        x = tf.convert_to_tensor(image[None, ...], dtype=tf.float32)
        x = tf.nn.avg_pool2d(x, ksize=7, strides=1, padding="SAME")
        return x[0].numpy().astype(image.dtype)
    else:
        return image


def random_displacment(image):
    if random.choice([True, False]):
        ax = random.choice([0, 1])
        if ax == 0:
            slices = np.array_split(image, 6, axis=ax)
            np.random.shuffle(slices)
            return np.vstack(slices)
        else:
            slices = np.array_split(image, 6, axis=ax)
            np.random.shuffle(slices)
            return np.hstack(slices)
    else:
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
TRAIN_DIR = os.path.join(DATA_ROOT, "train_images")
assert os.path.isdir(TRAIN_DIR), f"Could not find train_images directory at {TRAIN_DIR}"

files_to_fit = [
    os.path.join(TRAIN_DIR, "bacterial_leaf_blight", "100049.jpg"),
    os.path.join(TRAIN_DIR, "bacterial_leaf_streak", "100042.jpg"),
    os.path.join(TRAIN_DIR, "bacterial_panicle_blight", "100068.jpg"),
    os.path.join(TRAIN_DIR, "blast", "100012.jpg"),
    os.path.join(TRAIN_DIR, "brown_spot", "100022.jpg"),
    os.path.join(TRAIN_DIR, "dead_heart", "100020.jpg"),
    os.path.join(TRAIN_DIR, "downy_mildew", "100059.jpg"),
    os.path.join(TRAIN_DIR, "hispa", "100139.jpg"),
    os.path.join(TRAIN_DIR, "normal", "100111.jpg"),
    os.path.join(TRAIN_DIR, "tungro", "100134.jpg"),
]

to_gen_fit = []
for file in files_to_fit:
    if os.path.exists(file):
        to_gen_fit.append(img_to_array(load_img(file), dtype="uint8"))
    else:
        pass

if len(to_gen_fit) == 0:
    raise RuntimeError(
        "No images found to fit generator_2 statistics; cannot continue."
    )

generator_2.fit(np.asarray(to_gen_fit))



## === cell 7
test_parent = os.path.dirname(TEST_DIR)  # .../paddy-disease-classification
test_subfolder = os.path.basename(TEST_DIR)  # test_images

test_data_256 = generator_1.flow_from_directory(
    directory=test_parent,
    target_size=(256, 256),
    batch_size=32,
    classes=[test_subfolder],
    shuffle=False,
)

test_data_300 = generator_3.flow_from_directory(
    directory=test_parent,
    target_size=(300, 300),
    batch_size=16,
    classes=[test_subfolder],
    shuffle=False,
)

test_data_384 = generator_2.flow_from_directory(
    directory=test_parent,
    target_size=(384, 384),
    batch_size=16,
    classes=[test_subfolder],
    shuffle=False,
)

n_test = test_data_256.n
assert (
    test_data_300.n == n_test and test_data_384.n == n_test
), "Mismatch in test generator sizes"



## === cell 8
test_encodings = []

for m in [m1, m3, m4]:
    enc = np.zeros((n_test, 10), dtype=np.float32)
    for _ in range(5):
        enc_ = m.predict(test_data_256, verbose=1)
        enc += enc_.astype(np.float32)
    enc /= 5.0
    test_encodings.append(enc)

test_encodings2 = np.zeros((n_test, 10), dtype=np.float32)
for _ in range(5):
    encodings2 = m2.predict(test_data_300, verbose=1)
    test_encodings2 += encodings2.astype(np.float32)
test_encodings2 /= 5.0
test_encodings.append(test_encodings2)

test_encodings5 = np.zeros((n_test, 10), dtype=np.float32)
for _ in range(5):
    encodings5 = m5.predict(test_data_384, verbose=1)
    test_encodings5 += encodings5.astype(np.float32)
test_encodings5 /= 5.0
test_encodings.append(test_encodings5)



## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3605660007.py in <cell line: 0>()
      2 test_encodings = []
      3 
----> 4 for m in [m1, m3, m4]:
      5     enc = np.zeros((n_test, 10), dtype=np.float32)
      6     for _ in range(5):

NameError: name 'm1' is not defined

## === cell 9
predict_max = [np.argmax(test_enc, axis=1) for test_enc in test_encodings]

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

predictions = []
for enc in predict_max:
    predictions.append([inverse_map[int(k)] for k in enc])



## === cell 10
files = test_data_256.filenames
files = [f.replace("\\", "/") for f in files]
image_ids = [f.split("/", 1)[-1] for f in files]  # keep basename

s_full = pd.DataFrame({"image_id": image_ids})
for i, pres in enumerate(predictions):
    s_full[f"m{i+1}"] = pres

s_full = s_full.set_index("image_id")

s_full_mode = ss.mode(s_full, axis=1, keepdims=False)
final = pd.DataFrame({"image_id": s_full.index.values, "label": s_full_mode.mode})



## === cell 11
sample_path = os.path.join(DATA_ROOT, "sample_submission.csv")
sample = pd.read_csv(sample_path)

final = sample[["image_id"]].merge(final, on="image_id", how="left")

final["label"] = final["label"].fillna("normal")

final.to_csv("model_submission_v21.csv", index=False)
print(final.head())
print("Saved submission to model_submission_v21.csv with shape:", final.shape)
