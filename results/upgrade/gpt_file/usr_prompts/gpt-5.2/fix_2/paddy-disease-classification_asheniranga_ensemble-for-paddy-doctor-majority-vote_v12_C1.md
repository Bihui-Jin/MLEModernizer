# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


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

# 5. Code solution

## === cell 0
import os
import random
import numpy as np
import pandas as pd
import tensorflow as tf

try:
    import tensorflow_hub as hub  # used only for custom_objects when loading certain models
except Exception:
    hub = None

try:
    import tensorflow_addons as tfa
except Exception:
    tfa = None

try:
    import scipy.stats as ss
except Exception:
    ss = None

from tensorflow.keras.preprocessing.image import ImageDataGenerator

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

DATA_DIR = "../input/paddy-disease-classification"
TRAIN_DIR = os.path.join(DATA_DIR, "train_images")
TEST_DIR = os.path.join(DATA_DIR, "test_images")
SAMPLE_SUB_PATH = os.path.join(DATA_DIR, "sample_submission.csv")

print("TF version:", tf.__version__)
print("Train dir exists:", os.path.isdir(TRAIN_DIR))
print("Test dir exists:", os.path.isdir(TEST_DIR))
print("Sample submission exists:", os.path.isfile(SAMPLE_SUB_PATH))



## === cell 1


def random_cut_out(images):
    if tfa is None:
        return images
    return tfa.image.random_cutout(images, (32, 32), constant_values=0)


def center_crop_and_random_augmentations_fn(image):
    image = tf.convert_to_tensor(image)
    image = tf.image.random_crop(image, (256, 256, 3))
    image = tf.image.random_brightness(image, 0.2)
    image = tf.image.random_contrast(image, 0.5, 2.0)
    image = tf.image.random_saturation(image, 0.75, 1.25)
    image = tf.image.random_hue(image, 0.1)
    return image.numpy()




## === cell 2

generator = ImageDataGenerator(
    rescale=1 / 255.0,
    rotation_range=5,
    width_shift_range=0.25,
    height_shift_range=0.25,
    shear_range=0.2,
    zoom_range=0.1,
    horizontal_flip=True,
    vertical_flip=True,
    validation_split=0.2,
    preprocessing_function=center_crop_and_random_augmentations_fn,
)

train_datagen = generator.flow_from_directory(
    TRAIN_DIR,
    target_size=(256, 256),
    batch_size=16,
    subset="training",
    seed=SEED,
)

valid_datagen = generator.flow_from_directory(
    TRAIN_DIR,
    target_size=(256, 256),
    batch_size=16,
    subset="validation",
    seed=SEED,
)

train_datagen_300 = generator.flow_from_directory(
    TRAIN_DIR,
    target_size=(300, 300),
    batch_size=16,
    subset="training",
    seed=SEED,
)

valid_datagen_300 = generator.flow_from_directory(
    TRAIN_DIR,
    target_size=(300, 300),
    batch_size=16,
    subset="validation",
    seed=SEED,
)



## === cell 3
test_loc = TEST_DIR

test_data_256 = ImageDataGenerator(rescale=1.0 / 255.0).flow_from_directory(
    directory=test_loc,
    target_size=(256, 256),
    batch_size=16,
    classes=["."],
    shuffle=False,
)

test_data_300 = ImageDataGenerator(rescale=1.0 / 255.0).flow_from_directory(
    directory=test_loc,
    target_size=(300, 300),
    batch_size=16,
    classes=["."],
    shuffle=False,
)

print("Num test files (256):", test_data_256.n)
print("Num test files (300):", test_data_300.n)



## === cell 4


def _custom_objects_for_load():
    if hub is not None:
        return {"KerasLayer": hub.KerasLayer}
    return {}


def safe_load_model(path):
    if os.path.exists(path):
        return tf.keras.models.load_model(
            path, custom_objects=_custom_objects_for_load()
        )
    return None


m1 = safe_load_model(
    "../input/paddy-doc-ensemble-models/ensemble_estimators/effnet_b3.hdf5"
)
m2 = safe_load_model(
    "../input/paddy-doc-ensemble-models/ensemble_estimators/effnet_v2_m.hdf5"
)
m3 = safe_load_model(
    "../input/paddy-doc-ensemble-models/ensemble_estimators/xception.hdf5"
)
m4 = safe_load_model("../input/paddydocoutputs/model_resnet150.hdf5")
m5 = safe_load_model(
    "../input/k/kasunpramodya/paddy-doctor-training/model_effnet_s.hdf5"
)
m6 = safe_load_model("../input/notebooka9ca40495e/model_effnet_s.hdf5")

loaded = [m is not None for m in [m1, m2, m3, m4, m5, m6]]
print("Loaded ensemble models:", loaded, "count:", sum(loaded))

if sum(loaded) == 0:
    num_classes = train_datagen.num_classes

    base = tf.keras.applications.EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_shape=(256, 256, 3),
    )
    base.trainable = False

    inp = tf.keras.Input(shape=(256, 256, 3))
    x = base(inp, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    out = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
    fallback_model = tf.keras.Model(inp, out)

    fallback_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    EPOCHS = 3
    fallback_model.fit(
        train_datagen,
        validation_data=valid_datagen,
        epochs=EPOCHS,
        verbose=1,
    )

    m1, m2, m3, m4, m5, m6 = fallback_model, None, None, None, None, None



## === cell 5
model_train_test_score = []

for model in [m1, m2, m3, m4, m5, m6]:
    if model is None:
        model_train_test_score.append(None)
        continue
    try:
        model_train_test_score.append(model.evaluate(valid_datagen, verbose=0))
    except Exception:
        model_train_test_score.append(model.evaluate(valid_datagen_300, verbose=0))

model_train_test_score




## === cell 6
def predict_model(model, gen256, gen300):
    if model is None:
        return None
    try:
        return model.predict(gen256, verbose=1)
    except Exception:
        return model.predict(gen300, verbose=1)


m1_p = predict_model(m1, test_data_256, test_data_300)
m2_p = predict_model(m2, test_data_256, test_data_300)
m3_p = predict_model(m3, test_data_256, test_data_300)
m4_p = predict_model(m4, test_data_256, test_data_300)
m5_p = predict_model(m5, test_data_256, test_data_300)
m6_p = predict_model(m6, test_data_256, test_data_300)

pred_probas = [p for p in [m1_p, m2_p, m3_p, m4_p, m5_p, m6_p] if p is not None]
print("Num prediction arrays:", len(pred_probas))
print("Prediction shape example:", pred_probas[0].shape if pred_probas else None)



## === cell 7
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



## === cell 8

if len(pred_probas) == 0:
    raise RuntimeError("No model predictions available; cannot create submission.")

avg_proba = np.mean(np.stack(pred_probas, axis=0), axis=0)
pred_idx = np.argmax(avg_proba, axis=1)
pred_label = [inverse_map[int(i)] for i in pred_idx]

files = test_data_256.filenames
image_ids = pd.Series(files).str.replace("./", "", regex=False)

pred_df = pd.DataFrame({"image_id": image_ids, "label": pred_label})

sample_sub = pd.read_csv(SAMPLE_SUB_PATH)
pred_df = sample_sub[["image_id"]].merge(pred_df, on="image_id", how="left")

pred_df["label"] = pred_df["label"].fillna("normal")

pred_df.head(), pred_df.shape



## === cell 9
submission_path = "submission.csv"
pred_df.to_csv(submission_path, index=False)
print("Wrote:", submission_path)
print(pred_df.columns.tolist())
print(pred_df.isna().sum())
print(pred_df.head())
