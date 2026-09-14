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

# 8. Previous improvement plan

- What this solution (achieved 0.17487) has done: 'I fix the immediate runtime crash caused by importing `tensorflow_hub` (it triggers a protobuf incompatibility in this environment) by disabling that optional import and loading models without hub custom objects. Then I fix the SciPy `mode` failure (SciPy 1.11+ no longer accepts non-numeric arrays) by replacing it with a safe per-row majority vote using `pandas.Series.mode`, preserving the ensemble “mode of predicted labels” core logic. Finally, I make submission creation robust (even when no external models load) and ensure the output CSV has the required columns and ordering matching `sample_submission.csv`.'

# 9. Code solution

## === cell 0
import os
import random
import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import load_img, img_to_array

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

BASE_INPUT = "/kaggle/input/paddy-disease-classification"
TEST_IMAGES_DIR = os.path.join(BASE_INPUT, "test_images")

HUB_KERAS_LAYER = None



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
WORK_TEST_ROOT = "/kaggle/working/test_images"
WORK_TEST_CLASSDIR = os.path.join(WORK_TEST_ROOT, ".")
os.makedirs(WORK_TEST_CLASSDIR, exist_ok=True)

test_imgs = sorted(
    [f for f in os.listdir(TEST_IMAGES_DIR) if f.lower().endswith(".jpg")]
)
existing = set(os.listdir(WORK_TEST_CLASSDIR))
for f in test_imgs:
    if f in existing:
        continue
    src = os.path.join(TEST_IMAGES_DIR, f)
    dst = os.path.join(WORK_TEST_CLASSDIR, f)
    try:
        os.symlink(src, dst)
    except FileExistsError:
        pass
    except OSError:
        import shutil

        shutil.copy2(src, dst)

len(test_imgs)



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


def load_keras_model_if_exists(path):
    if not os.path.exists(path):
        return None
    custom_objects = {}
    try:
        m = tf.keras.models.load_model(
            path, custom_objects=custom_objects, compile=False
        )
        return m
    except Exception as e:
        print(f"Warning: failed to load model at {path}: {e}")
        return None


loaded_models = []
loaded_model_paths = []
for p in CANDIDATE_MODEL_PATHS:
    m = load_keras_model_if_exists(p)
    if m is not None:
        loaded_models.append(m)
        loaded_model_paths.append(p)

print("Loaded models:", len(loaded_models))
for p in loaded_model_paths:
    print(" -", p)



## === cell 3
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

NUM_CLASSES = 10




## === cell 4
def test_time_augmentation_fn_1(image):
    image = tf.convert_to_tensor(image)
    image = tf.image.random_crop(image, (256, 256, 3))
    image = tf.image.random_brightness(image, 0.2)
    image = tf.image.random_contrast(image, 0.5, 2.0)
    return image.numpy()


def test_time_augmentation_fn_2(image):
    image = tf.convert_to_tensor(image)
    image = tf.image.random_crop(image, (384, 384, 3))
    image = tf.image.random_brightness(image, 0.2)
    image = tf.image.random_contrast(image, 0.5, 2.0)
    return image.numpy()


def test_time_augmentation_fn_3(image):
    image = tf.convert_to_tensor(image)
    image = tf.image.random_crop(image, (300, 300, 3))
    image = tf.image.random_brightness(image, 0.2)
    image = tf.image.random_contrast(image, 0.5, 2.0)
    return image.numpy()




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
TRAIN_IMAGES_DIR = os.path.join(BASE_INPUT, "train_images")

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

len(to_gen_fit)



## === cell 7
test_data_256 = generator_1.flow_from_directory(
    directory=WORK_TEST_ROOT,
    target_size=(256, 256),
    batch_size=32,
    classes=["."],
    shuffle=False,
)

test_data_300 = generator_3.flow_from_directory(
    directory=WORK_TEST_ROOT,
    target_size=(300, 300),
    batch_size=16,
    classes=["."],
    shuffle=False,
)

test_data_384 = generator_2.flow_from_directory(
    directory=WORK_TEST_ROOT,
    target_size=(384, 384),
    batch_size=16,
    classes=["."],
    shuffle=False,
)

n_test = test_data_256.n
n_test, len(test_data_256.filenames)




## === cell 8
def predict_with_tta(model, generator, tta=5):
    preds = np.zeros((generator.n, NUM_CLASSES), dtype=np.float32)
    for _ in range(tta):
        generator.reset()
        p = model.predict(generator, verbose=1)
        preds += p.astype(np.float32)
    preds /= float(tta)
    return preds


test_encodings = []

if len(loaded_models) > 0:
    for m in loaded_models:
        in_shape = m.input_shape
        if isinstance(in_shape, list):
            in_shape = in_shape[0]
        h = in_shape[1]
        w = in_shape[2]
        if (h, w) == (256, 256):
            enc = predict_with_tta(m, test_data_256, tta=5)
        elif (h, w) == (300, 300):
            enc = predict_with_tta(m, test_data_300, tta=5)
        elif (h, w) == (384, 384):
            enc = predict_with_tta(m, test_data_384, tta=5)
        else:
            try:
                enc = predict_with_tta(m, test_data_256, tta=5)
            except Exception as e:
                print(f"Skipping model due to incompatible input shape {in_shape}: {e}")
                continue
        test_encodings.append(enc)
else:
    test_encodings = []

len(test_encodings)



## === cell 9
if len(test_encodings) == 0:
    baseline = np.zeros((n_test, NUM_CLASSES), dtype=np.float32)
    baseline[:, class_indices["normal"]] = 1.0
    test_encodings = [baseline]

predict_max = [np.argmax(te, axis=1) for te in test_encodings]
enc_max = [np.max(te, axis=1) for te in test_encodings]

len(predict_max), predict_max[0].shape



## === cell 10
predictions = []
for enc in predict_max:
    predictions.append([inverse_map[int(k)] for k in enc])

files = test_data_256.filenames
files = [f.replace("./", "") for f in files]

files[:5], predictions[0][:5]



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
    chosen = m.iloc[0]  # deterministic tie-break: lexicographically first among modes
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
final.head()



## === cell 12
submission = final[["image_id", "label"]].copy()

sample_path = os.path.join(BASE_INPUT, "sample_submission.csv")
if os.path.exists(sample_path):
    sample = pd.read_csv(sample_path)
    submission = sample[["image_id"]].merge(submission, on="image_id", how="left")
    submission["label"] = submission["label"].fillna("normal")

submission.head(), submission.shape



## === cell 13
out_path = "model_submission_v23.csv"
submission.to_csv(out_path, index=False)

print("Wrote:", out_path)
print(submission.columns.tolist())
print(submission.isna().sum())
print(submission["label"].value_counts().head())
