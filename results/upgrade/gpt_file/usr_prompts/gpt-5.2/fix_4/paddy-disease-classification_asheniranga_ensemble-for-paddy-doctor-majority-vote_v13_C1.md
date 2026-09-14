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

# 8. Previous improvement plans

- What this solution (achieved 0.17487) has done: 'I fix the early import/runtime crash by removing incompatible/unused heavy imports (notably `tensorflow_addons`, `tensorflow_hub`, `albumentations`, etc.) and by loading the `.hdf5` models without requiring `hub.KerasLayer` (since these saved models typically don’t need it at inference time). I also correct the test directory path and the hard-coded test size (your code uses `3469`, but the competition test set is 2602), so predictions and filenames align. Finally, I fix the submission-writing step to ensure the output CSV has exactly `image_id,label` columns (your current file writes the index as `Unnamed: 0`, causing the invalid submission error). These changes are execution/format fixes and preserve the ensemble + TTA voting logic.'
- What this solution (achieved 0.17487) has done: 'I fix the early TensorFlow/protobuf crash by forcing the pure-Python protobuf implementation before importing TensorFlow, which is a common Kaggle runtime issue with TF on Py3.10. Then I fix the missing model files problem by loading only models that actually exist in `/kaggle/input` (and error out cleanly if none exist), preserving the same ensemble+TTA voting logic for any models found. I also make the generator fitting and model usage robust so undefined model variables can’t occur, while keeping the same prediction+mode-aggregation semantics. Finally, I ensure the submission is written as a valid `image_id,label` CSV aligned to `sample_submission.csv`.'
- What this solution (achieved 0.17487) has done: 'You’re currently blocked first by a TensorFlow/protobuf incompatibility (the `MessageFactory.GetPrototype` crash), and then by missing external `.hdf5` model files, which prevents any meaningful inference. I fix the TF import crash by pinning the pure-Python protobuf implementation earlier and additionally forcing a safe protobuf version range if present, then provide a robust fallback model that keeps the same “Keras model + TTA + ensemble mode vote” inference semantics when those external models aren’t available. The fallback uses a lightweight TF Hub EfficientNet feature extractor + a small dense head trained on the provided `train.csv`/`train_images` (preserving the same generator-based image pipeline), so you can generate a valid submission and substantially improve accuracy versus the current broken/degenerate run. Finally, I ensure filename alignment and submission formatting stays exactly `image_id,label` with 2602 rows.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import random
import numpy as np
import pandas as pd
import scipy.stats as ss

random.seed(42)
np.random.seed(42)

import tensorflow as tf

tf.random.set_seed(42)

print("TF version:", tf.__version__)



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
print("Using DATA_ROOT:", DATA_ROOT)
print("TEST_DIR:", TEST_DIR)

TRAIN_DIR = os.path.join(DATA_ROOT, "train_images")
assert os.path.isdir(TRAIN_DIR), f"Could not find train_images directory at {TRAIN_DIR}"

TRAIN_CSV = os.path.join(DATA_ROOT, "train.csv")
SAMPLE_SUB = os.path.join(DATA_ROOT, "sample_submission.csv")
assert os.path.isfile(TRAIN_CSV), f"Missing train.csv at {TRAIN_CSV}"
assert os.path.isfile(SAMPLE_SUB), f"Missing sample_submission.csv at {SAMPLE_SUB}"



## === cell 2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import load_img, img_to_array

input_size = 256  # kept for compatibility with original code


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




## === cell 3
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



## === cell 4
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

if len(to_gen_fit) == 0:
    train_df_tmp = pd.read_csv(TRAIN_CSV)
    for _, r in train_df_tmp.head(20).iterrows():
        p = os.path.join(TRAIN_DIR, r["label"], r["image_id"])
        if os.path.exists(p):
            to_gen_fit.append(img_to_array(load_img(p), dtype="uint8"))
        if len(to_gen_fit) >= 10:
            break

if len(to_gen_fit) == 0:
    raise RuntimeError(
        "No images found to fit generator_2 statistics; cannot continue."
    )

generator_2.fit(np.asarray(to_gen_fit))
print("generator_2 fitted on", len(to_gen_fit), "images")



## === cell 5
test_parent = os.path.dirname(TEST_DIR)
test_subfolder = os.path.basename(TEST_DIR)

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
print("n_test:", n_test)




## === cell 6
def safe_load_model(path):
    if not os.path.exists(path):
        return None
    try:
        return tf.keras.models.load_model(path, compile=False)
    except Exception as e:
        print(f"Failed to load model at {path}: {e}")
        return None


CANDIDATE_MODEL_PATHS = [
    ("m1", "../input/paddy-doc-ensemble-models/ensemble_estimators/resnet_150.hdf5"),
    ("m2", "../input/paddy-doc-ensemble-models/ensemble_estimators/effnet_v2_m.hdf5"),
    ("m3", "../input/k/kasunpramodya/paddy-doctor-training/model_effnet_s.hdf5"),
    ("m4", "../input/notebooka9ca40495e/model_effnet_s.hdf5"),
    ("m5", "../input/paddy-doctor-training/model_effnet_b4.hdf5"),
]

loaded_models = {}
for name, path in CANDIDATE_MODEL_PATHS:
    model = safe_load_model(path)
    if model is not None:
        loaded_models[name] = model
        print(f"Loaded {name} from {path}")

m1 = loaded_models.get("m1")
m2 = loaded_models.get("m2")
m3 = loaded_models.get("m3")
m4 = loaded_models.get("m4")
m5 = loaded_models.get("m5")

if len(loaded_models) == 0:
    print(
        "No external .hdf5 models found; training a fallback model from train_images."
    )

    import tensorflow_hub as hub

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

    train_df = pd.read_csv(TRAIN_CSV).copy()
    train_df["filepath"] = train_df.apply(
        lambda r: os.path.join(TRAIN_DIR, r["label"], r["image_id"]), axis=1
    )
    train_df = train_df[train_df["filepath"].apply(os.path.exists)].reset_index(
        drop=True
    )
    assert len(train_df) > 0, "No training images found on disk."

    train_gen = ImageDataGenerator(
        rescale=1.0 / 255.0,
        validation_split=0.1,
        horizontal_flip=True,
        vertical_flip=True,
        preprocessing_function=center_crop_and_random_augmentations_fn,
    )

    train_flow = train_gen.flow_from_dataframe(
        train_df,
        x_col="filepath",
        y_col="label",
        target_size=(256, 256),
        class_mode="categorical",
        batch_size=32,
        shuffle=True,
        subset="training",
        seed=42,
    )
    val_flow = train_gen.flow_from_dataframe(
        train_df,
        x_col="filepath",
        y_col="label",
        target_size=(256, 256),
        class_mode="categorical",
        batch_size=32,
        shuffle=False,
        subset="validation",
        seed=42,
    )

    feature_url = "https://tfhub.dev/tensorflow/efficientnet/b0/feature-vector/1"
    backbone = hub.KerasLayer(feature_url, trainable=False, input_shape=(256, 256, 3))

    inputs = tf.keras.Input(shape=(256, 256, 3))
    x = backbone(inputs)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(10, activation="softmax")(x)
    fallback_model = tf.keras.Model(inputs, outputs)

    fallback_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    fallback_model.fit(
        train_flow,
        validation_data=val_flow,
        epochs=6,
        verbose=1,
    )

    m1 = fallback_model




## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/1345451581.py in <cell line: 0>()
     98 
     99     inputs = tf.keras.Input(shape=(256, 256, 3))
--> 100     x = backbone(inputs)
    101     x = tf.keras.layers.Dropout(0.2)(x)
    102     outputs = tf.keras.layers.Dense(10, activation="softmax")(x)

/usr/local/lib/python3.11/dist-packages/tf_keras/src/utils/traceback_utils.py in error_handler(*args, **kwargs)
     68             # To get the full stack trace, call:
     69             # `tf.debugging.disable_traceback_filtering()`
---> 70             raise e.with_traceback(filtered_tb) from None
     71         finally:
     72             del filtered_tb

/usr/local/lib/python3.11/dist-packages/tensorflow_hub/keras_layer.py in call(self, inputs, training)
    248         # Behave like BatchNormalization. (Dropout is different, b/181839368.)
    249         training = False
--> 250       result = smart_cond.smart_cond(training,
    251                                      lambda: f(training=True),
    252                                      lambda: f(training=False))

/usr/local/lib/python3.11/dist-packages/tensorflow_hub/keras_layer.py in <lambda>()
    250       result = smart_cond.smart_cond(training,
    251                                      lambda: f(training=True),
--> 252                                      lambda: f(training=False))
    253 
    254     # Unwrap dicts returned by signatures.

/usr/local/lib/python3.11/dist-packages/tensorflow/core/function/polymorphism/function_type.py in canonicalize_to_monomorphic(args, kwargs, default_values, capture_types, polymorphic_type)
    581     else:
    582       parameters.append(
--> 583           _make_validated_mono_param(name, arg, poly_parameter.kind,
    584                                      type_context,
    585                                      poly_parameter.type_constraint))

/usr/local/lib/python3.11/dist-packages/tensorflow/core/function/polymorphism/function_type.py in _make_validated_mono_param(name, value, kind, type_context, poly_type)
    520 ) -> Parameter:
    521   """Generates and validates a parameter for Monomorphic FunctionType."""
--> 522   mono_type = trace_type.from_value(value, type_context)
    523 
    524   if poly_type and not mono_type.is_subtype_of(poly_type):

/usr/local/lib/python3.11/dist-packages/tensorflow/core/function/trace_type/trace_type_builder.py in from_value(value, context)
    183 
    184   if util.is_np_ndarray(value):
--> 185     ndarray = value.__array__()
    186     return default_types.TENSOR(ndarray.shape, ndarray.dtype)
    187 

/usr/local/lib/python3.11/dist-packages/keras/src/backend/common/keras_tensor.py in __array__(self)
    106 
    107     def __array__(self):
--> 108         raise ValueError(
    109             "A KerasTensor is symbolic: it's a placeholder for a shape "
    110             "an a dtype. It doesn't have any actual numerical value. "

ValueError: Exception encountered when calling layer 'keras_layer' (type KerasLayer).

A KerasTensor is symbolic: it's a placeholder for a shape an a dtype. It doesn't have any actual numerical value. You cannot convert it to a NumPy array.

Call arguments received by layer 'keras_layer' (type KerasLayer):
  • inputs=<KerasTensor shape=(None, 256, 256, 3), dtype=float32, sparse=False, name=keras_tensor>
  • training=None

## === cell 7
def predict_tta(model, generator, n_classes=10, tta=5):
    enc = np.zeros((generator.n, n_classes), dtype=np.float32)
    for _ in range(tta):
        enc_ = model.predict(generator, verbose=0)
        enc += enc_.astype(np.float32)
    enc /= float(tta)
    return enc


test_encodings = []

for model in [m1, m3, m4]:
    if model is None:
        continue
    test_encodings.append(predict_tta(model, test_data_256, n_classes=10, tta=5))

if m2 is not None:
    test_encodings.append(predict_tta(m2, test_data_300, n_classes=10, tta=5))

if m5 is not None:
    test_encodings.append(predict_tta(m5, test_data_384, n_classes=10, tta=5))

if len(test_encodings) == 0:
    raise RuntimeError(
        "No models available for inference; cannot generate predictions."
    )

print("Ensemble members used:", len(test_encodings))



## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
RuntimeError                              Traceback (most recent call last)
/tmp/ipykernel_11/4009881639.py in <cell line: 0>()
     22 
     23 if len(test_encodings) == 0:
---> 24     raise RuntimeError(
     25         "No models available for inference; cannot generate predictions."
     26     )

RuntimeError: No models available for inference; cannot generate predictions.

## === cell 8
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

files = test_data_256.filenames
files = [f.replace("\\", "/") for f in files]
image_ids = [f.split("/", 1)[-1] for f in files]

s_full = pd.DataFrame({"image_id": image_ids})
for i, pres in enumerate(predictions):
    s_full[f"m{i+1}"] = pres

s_full = s_full.set_index("image_id")
s_full_mode = ss.mode(s_full, axis=1, keepdims=False)
final = pd.DataFrame({"image_id": s_full.index.values, "label": s_full_mode.mode})



## === cell 9
sample = pd.read_csv(SAMPLE_SUB)

final = sample[["image_id"]].merge(final, on="image_id", how="left")
final["label"] = final["label"].fillna("normal")

out_path = "model_submission_v21.csv"
final.to_csv(out_path, index=False)

print(final.head())
print("Saved submission to", out_path, "with shape:", final.shape)
print("Submission columns:", list(final.columns))
assert list(final.columns) == ["image_id", "label"]
assert out_path.endswith(".csv")
assert final.shape[0] == sample.shape[0]
