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

0.17948

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved 0.17948) has done: 'I fix the environment-crashing import issue in the first cell by removing/deferring optional imports that trigger protobuf/TensorFlow incompatibilities, while keeping the core workflow unchanged. Then I fix the tf.data augmentation bug by applying random crop per-image (not per-batch) so the dataset pipeline can be built and used for training/inference. I also make the label mapping consistent with the directory-inferred class order (instead of a hardcoded map) so predicted indices translate to correct class names and submission accuracy improves legitimately. Finally, I ensure the fallback model always runs if no external ensemble models are found, and that a valid `submission.csv` is always written.'

# 9. Code solution

## === cell 0
import os
import random
import numpy as np
import pandas as pd
import tensorflow as tf

hub = None
tfa = None
ss = None

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

tf.config.optimizer.set_jit(False)

DATA_DIR = "../input/paddy-disease-classification"
TRAIN_DIR = os.path.join(DATA_DIR, "train_images")
TEST_DIR = os.path.join(DATA_DIR, "test_images")
SAMPLE_SUB_PATH = os.path.join(DATA_DIR, "sample_submission.csv")

print("TF version:", tf.__version__)
print("Train dir exists:", os.path.isdir(TRAIN_DIR))
print("Test dir exists:", os.path.isdir(TEST_DIR))
print("Sample submission exists:", os.path.isfile(SAMPLE_SUB_PATH))




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
def random_cut_out(images):
    if tfa is None:
        return images
    return tfa.image.random_cutout(images, (32, 32), constant_values=0)


@tf.function
def center_crop_and_random_augmentations_tf(image):
    image = tf.convert_to_tensor(image)
    image = tf.image.random_crop(image, (256, 256, 3))
    image = tf.image.random_brightness(image, 0.2)
    image = tf.image.random_contrast(image, 0.5, 2.0)
    image = tf.image.random_saturation(image, 0.75, 1.25)
    image = tf.image.random_hue(image, 0.1)
    return image


def center_crop_and_random_augmentations_fn(image):
    image = tf.convert_to_tensor(image)
    image = tf.image.random_crop(image, (256, 256, 3))
    image = tf.image.random_brightness(image, 0.2)
    image = tf.image.random_contrast(image, 0.5, 2.0)
    image = tf.image.random_saturation(image, 0.75, 1.25)
    image = tf.image.random_hue(image, 0.1)
    return image.numpy()




## === cell 2
BATCH_SIZE = 16
IMG_SIZE_256 = (256, 256)
IMG_SIZE_300 = (300, 300)
VAL_SPLIT = 0.2

AUTOTUNE = tf.data.AUTOTUNE

train_ds_256 = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    labels="inferred",
    label_mode="categorical",
    image_size=IMG_SIZE_256,
    batch_size=BATCH_SIZE,
    shuffle=True,
    seed=SEED,
    validation_split=VAL_SPLIT,
    subset="training",
)

valid_ds_256 = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    labels="inferred",
    label_mode="categorical",
    image_size=IMG_SIZE_256,
    batch_size=BATCH_SIZE,
    shuffle=False,
    seed=SEED,
    validation_split=VAL_SPLIT,
    subset="validation",
)

train_ds_300 = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    labels="inferred",
    label_mode="categorical",
    image_size=IMG_SIZE_300,
    batch_size=BATCH_SIZE,
    shuffle=True,
    seed=SEED,
    validation_split=VAL_SPLIT,
    subset="training",
)

valid_ds_300 = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    labels="inferred",
    label_mode="categorical",
    image_size=IMG_SIZE_300,
    batch_size=BATCH_SIZE,
    shuffle=False,
    seed=SEED,
    validation_split=VAL_SPLIT,
    subset="validation",
)

num_classes = len(train_ds_256.class_names)

class_names = list(train_ds_256.class_names)
inverse_map = {i: name for i, name in enumerate(class_names)}


def _prep_train_256(x, y):
    x = tf.cast(x, tf.float32) / 255.0
    x = tf.map_fn(
        center_crop_and_random_augmentations_tf, x, fn_output_signature=tf.float32
    )
    return x, y


def _prep_valid(x, y):
    x = tf.cast(x, tf.float32) / 255.0
    return x, y


def _prep_train_300(x, y):
    x = tf.cast(x, tf.float32) / 255.0
    x = tf.map_fn(
        center_crop_and_random_augmentations_tf, x, fn_output_signature=tf.float32
    )
    return x, y


train_datagen = train_ds_256.map(_prep_train_256, num_parallel_calls=AUTOTUNE).prefetch(
    AUTOTUNE
)
valid_datagen = valid_ds_256.map(_prep_valid, num_parallel_calls=AUTOTUNE).prefetch(
    AUTOTUNE
)

train_datagen_300 = train_ds_300.map(
    _prep_train_300, num_parallel_calls=AUTOTUNE
).prefetch(AUTOTUNE)
valid_datagen_300 = valid_ds_300.map(_prep_valid, num_parallel_calls=AUTOTUNE).prefetch(
    AUTOTUNE
)

print("Classes:", class_names)
print("num_classes:", num_classes)




## === cell 3
def _list_test_files(test_dir):
    exts = (".jpg", ".jpeg", ".png", ".bmp")
    files = []
    for name in os.listdir(test_dir):
        if name.lower().endswith(exts):
            files.append(os.path.join(test_dir, name))
    files.sort()
    return files


test_files = _list_test_files(TEST_DIR)
print("Num test files:", len(test_files))
if len(test_files) == 0:
    raise RuntimeError("No test images found in TEST_DIR")


def _decode_jpeg(path):
    img_bytes = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img_bytes, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)  # [0,1]
    return img


def _make_test_ds(target_hw):
    ds = tf.data.Dataset.from_tensor_slices(test_files)
    ds = ds.map(_decode_jpeg, num_parallel_calls=AUTOTUNE)
    ds = ds.map(
        lambda x: tf.image.resize(x, target_hw, method="bilinear", antialias=False),
        num_parallel_calls=AUTOTUNE,
    )
    ds = ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)
    return ds


test_data_256 = _make_test_ds(IMG_SIZE_256)
test_data_300 = _make_test_ds(IMG_SIZE_300)




## === cell 4
def _custom_objects_for_load():
    if hub is not None:
        return {"KerasLayer": hub.KerasLayer}
    return {}


def safe_load_model(path):
    if os.path.exists(path):
        try:
            return tf.keras.models.load_model(
                path, custom_objects=_custom_objects_for_load()
            )
        except Exception as e:
            print(f"Failed to load model at {path}: {repr(e)}")
            return None
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
def _needs_300(model):
    try:
        ish = model.input_shape
        if isinstance(ish, list):
            ish = ish[0]
        return ish[1] == 300 and ish[2] == 300
    except Exception:
        return False


model_train_test_score = []
for model in [m1, m2, m3, m4, m5, m6]:
    if model is None:
        model_train_test_score.append(None)
        continue
    ds = valid_datagen_300 if _needs_300(model) else valid_datagen
    try:
        model_train_test_score.append(model.evaluate(ds, verbose=0))
    except Exception as e:
        print("Evaluation failed for a model; skipping. Error:", repr(e))
        model_train_test_score.append(None)

model_train_test_score




## === cell 6
def predict_model(model, ds256, ds300):
    if model is None:
        return None
    ds = ds300 if _needs_300(model) else ds256
    try:
        return model.predict(ds, verbose=1)
    except Exception as e:
        print("Prediction failed for a model; skipping. Error:", repr(e))
        return None


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
if len(pred_probas) == 0:
    raise RuntimeError("No model predictions available; cannot create submission.")

avg_proba = np.mean(np.stack(pred_probas, axis=0), axis=0)
pred_idx = np.argmax(avg_proba, axis=1)
pred_label = [inverse_map[int(i)] for i in pred_idx]

image_ids = pd.Series([os.path.basename(p) for p in test_files])
pred_df = pd.DataFrame({"image_id": image_ids, "label": pred_label})

sample_sub = pd.read_csv(SAMPLE_SUB_PATH)
pred_df = sample_sub[["image_id"]].merge(pred_df, on="image_id", how="left")
pred_df["label"] = pred_df["label"].fillna(class_names[0] if class_names else "normal")

print(pred_df.head())
print("Submission shape:", pred_df.shape)
print("Missing labels:", pred_df["label"].isna().sum())



## === cell 8
submission_path = "submission.csv"
pred_df.to_csv(submission_path, index=False)
print("Wrote:", submission_path)
print("Columns:", pred_df.columns.tolist())
print(pred_df.head())
