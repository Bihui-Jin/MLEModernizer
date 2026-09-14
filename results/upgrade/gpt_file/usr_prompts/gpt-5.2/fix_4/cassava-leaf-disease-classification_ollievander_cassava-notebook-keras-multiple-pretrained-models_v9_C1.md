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
Classify each cassava image into four disease categories or a fifth category indicating a healthy leaf.

## Metric
Categorization accuracy.

## Submission Format
```
image_id,label
1000471002.jpg,4
1000840542.jpg,4
etc.
```

## Dataset
**[train/test]_images** the image files.

**train.csv**

- `image_id` the image file name.

- `label` the ID code for the disease.

**sample_submission.csv** A properly formatted sample submission, given the disclosed test set content.

- `image_id` the image file name.

- `label` the predicted ID code for the disease.

**[train/test]_tfrecords** the image files in tfrecord format.

**label_num_to_disease_map.json** The mapping between each disease code and the real disease name.

# 2. Python version

3.9

# 3. Installed packages

No external packages required in the script and installed.

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (124 lines)
            label_num_to_disease_map.json (1 lines)
            sample_submission.csv (2677 lines)
            sample_submission.csv.zip (13.4 kB)
            test.zip (160 Bytes)
            test_images.zip (319.5 MB)
            test_tfrecords.zip (451.9 MB)
            train.csv (18722 lines)
            train.csv.zip (100.0 kB)
            train.zip (162 Bytes)
            train_images.zip (2.2 GB)
            train_tfrecords.zip (3.2 GB)
            cassava-leaf-disease-classification/
                description.md (124 lines)
                label_num_to_disease_map.json (1 lines)
                ... and 10 other files
                cassava-leaf-disease-classification/
                test_images/
                    2574872277.jpg (183.5 kB)
                    1449210447.jpg (100.8 kB)
                    ... and 2674 other files
                    test_images/
                test_tfrecords/
                    ld_test00-1338.tfrec (225.9 MB)
                    ld_test01-1338.tfrec (226.2 MB)
                train_images/
                    478676678.jpg (90.6 kB)
                    2315755156.jpg (59.5 kB)
                    ... and 18719 other files
                    train_images/
                train_tfrecords/
                    ld_train00-1338.tfrec (227.2 MB)
                    ld_train01-1338.tfrec (227.0 MB)
                    ... and 12 other files
            test_images/
                2574872277.jpg (183.5 kB)
                1449210447.jpg (100.8 kB)
                ... and 2674 other files
                test_images/
            test_tfrecords/
                ld_test00-1338.tfrec (225.9 MB)
                ld_test01-1338.tfrec (226.2 MB)
            train_images/
                478676678.jpg (90.6 kB)
                2315755156.jpg (59.5 kB)
                ... and 18719 other files
                train_images/
            train_tfrecords/
                ld_train00-1338.tfrec (227.2 MB)
                ld_train01-1338.tfrec (227.0 MB)
                ... and 12 other files
        input/
            description.md (124 lines)
            label_num_to_disease_map.json (1 lines)
            sample_submission.csv (2677 lines)
            sample_submission.csv.zip (13.4 kB)
            test.zip (160 Bytes)
            test_images.zip (319.5 MB)
            test_tfrecords.zip (451.9 MB)
            train.csv (18722 lines)
            train.csv.zip (100.0 kB)
            train.zip (162 Bytes)
            train_images.zip (2.2 GB)
            train_tfrecords.zip (3.2 GB)
            cassava-leaf-disease-classification/
                description.md (124 lines)
                label_num_to_disease_map.json (1 lines)
                ... and 10 other files
                cassava-leaf-disease-classification/
                test_images/
                    2574872277.jpg (183.5 kB)
                    1449210447.jpg (100.8 kB)
                    ... and 2674 other files
                    test_images/
                test_tfrecords/
                    ld_test00-1338.tfrec (225.9 MB)
                    ld_test01-1338.tfrec (226.2 MB)
                train_images/
                    478676678.jpg (90.6 kB)
                    2315755156.jpg (59.5 kB)
                    ... and 18719 other files
                    train_images/
                train_tfrecords/
                    ld_train00-1338.tfrec (227.2 MB)
                    ld_train01-1338.tfrec (227.0 MB)
                    ... and 12 other files
            test_images/
                2574872277.jpg (183.5 kB)
                1449210447.jpg (100.8 kB)
                ... and 2674 other files
                test_images/
                    2574872277.jpg (183.5 kB)
                    1449210447.jpg (100.8 kB)
                    ... and 2674 other files
                    test_images/
            test_tfrecords/
                ld_test00-1338.tfrec (225.9 MB)
                ld_test01-1338.tfrec (226.2 MB)
            train_images/
                478676678.jpg (90.6 kB)
                2315755156.jpg (59.5 kB)
                ... and 18719 other files
                train_images/
                    478676678.jpg (90.6 kB)
                    2315755156.jpg (59.5 kB)
                    ... and 18719 other files
                    train_images/
            train_tfrecords/
                ld_train00-1338.tfrec (227.2 MB)
                ld_train01-1338.tfrec (227.0 MB)
                ... and 12 other files
        working/
            cassava-leaf-disease-classification/
                description.md (124 lines)
                label_num_to_disease_map.json (1 lines)
                ... and 10 other files
                cassava-leaf-disease-classification/
                test_images/
                    2574872277.jpg (183.5 kB)
                    1449210447.jpg (100.8 kB)
                    ... and 2674 other files
                    test_images/
                test_tfrecords/
                    ld_test00-1338.tfrec (225.9 MB)
                    ld_test01-1338.tfrec (226.2 MB)
                train_images/
                    478676678.jpg (90.6 kB)
                    2315755156.jpg (59.5 kB)
                    ... and 18719 other files
                    train_images/
                train_tfrecords/
                    ld_train00-1338.tfrec (227.2 MB)
                    ld_train01-1338.tfrec (227.0 MB)
                    ... and 12 other files
```

-> data/cassava-leaf-disease-classification/label_num_to_disease_map.json has auto-generated json schema:
{
  "$schema": "http://json-schema.org/schema#",
  "type": "object",
  "properties": {
    "0": {
      "type": "string"
    },
    "1": {
      "type": "string"
    },
    "2": {
      "type": "string"
    },
    "3": {
      "type": "string"
    },
    "4": {
      "type": "string"
    }
  },
  "required": [
    "0",
    "1",
    "2",
    "3",
    "4"
  ]
}

-> data/cassava-leaf-disease-classification/sample_submission.csv has 2676 rows and 2 columns.
The columns are: image_id, label

-> data/cassava-leaf-disease-classification/train.csv has 18721 rows and 2 columns.
The columns are: image_id, label

-> data/label_num_to_disease_map.json has auto-generated json schema:
{
  "$schema": "http://json-schema.org/schema#",
  "type": "object",
  "properties": {
    "0": {
      "type": "string"
    },
    "1": {
      "type": "string"
    },
    "2": {
      "type": "string"
    },
    "3": {
      "type": "string"
    },
    "4": {
      "type": "string"
    }
  },
  "required": [
    "0",
    "1",
    "2",
    "3",
    "4"
  ]
}

-> data/sample_submission.csv has 2676 rows and 2 columns.
The columns are: image_id, label

-> data/train.csv has 18721 rows and 2 columns.
The columns are: image_id, label

-> input/cassava-leaf-disease-classification/label_num_to_disease_map.json has auto-generated json schema:
{
  "$schema": "http://json-schema.org/schema#",
  "type": "object",
  "properties": {
    "0": {
      "type": "string"
    },
    "1": {
      "type": "string"
    },
    "2": {
      "type": "string"
    },
    "3": {
      "type": "string"
    },
    "4": {
      "type": "string"
    }
  },
  "required": [
    "0",
    "1",
    "2",
    "3",
    "4"
  ]
}

-> (stopped after 10 files for performance)

# 5. Target score

0.8750377757630704

# 6. Current score

0.05568

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.32773) has done: 'I fix the pipeline so it runs end-to-end and writes a valid `submission.csv`. The main blockers are (1) loading a Keras `.h5` that fails due to protobuf/TensorFlow/Keras incompatibilities in the environment, and (2) imports that assume unavailable Albumentations symbols, causing the notebook to stop before `pd`, `ImageDataGenerator`, etc. are defined. To keep core logic intact while ensuring a submission is produced, I (a) switch to `tf.keras`-native loading with a safe fallback model if the external weight file can’t be loaded, (b) remove the broken Albumentations imports and keep augmentation minimal/optional, and (c) ensure inference resizes and normalizes images to the expected input shape and that submission rows align with `sample_submission.csv`.'
- What this solution (achieved 0.11697) has done: 'I fix the TensorFlow/protobuf crash causing `MessageFactory.GetPrototype` by forcing TensorFlow to use the pure-Python protobuf implementation before importing `tensorflow`, which resolves this common Kaggle environment incompatibility. I also make the fallback model construction correct for transfer learning by replacing the current “add Dense on top of 1000-way logits” bug with a proper `include_top=False` backbone + pooling + Dense(5), which should substantially improve accuracy versus the current broken head. Finally, I keep the rest of the pipeline intact but ensure preprocessing matches InceptionResNetV2 and that submission ordering matches `sample_submission.csv`, writing a valid `submission.csv`.'
- What this solution (achieved 0.05568) has done: 'I fix the TensorFlow/protobuf crash by forcing a compatible pure-Python protobuf runtime *and* downgrading the protobuf API usage inside the process before importing TensorFlow. Then I make training actually happen (your current script builds generators but never calls `model.fit`), keeping the same model core (either loaded `.h5` or the same InceptionResNetV2 fallback head) so predictions aren’t random. Finally, I ensure preprocessing is consistent between training and inference by using the same InceptionResNetV2 `preprocess_input` in the `ImageDataGenerator`, and keep the submission ordering aligned to `sample_submission.csv` while writing a valid `submission.csv`.'

# 9. Code solution

## === cell 0
import os

INPUT_DIR = "../input/cassava-leaf-disease-classification/"
OUTPUT_DIR = "./"
os.makedirs(OUTPUT_DIR, exist_ok=True)

TRAIN_PATH = os.path.join(INPUT_DIR, "train_images")
TEST_PATH = os.path.join(INPUT_DIR, "test_images")

print("INPUT_DIR exists:", os.path.exists(INPUT_DIR))
print("TRAIN_PATH exists:", os.path.exists(TRAIN_PATH))
print("TEST_PATH exists:", os.path.exists(TEST_PATH))



## === cell 1
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION"] = "2"

try:
    from google.protobuf import internal as _pb_internal  # noqa: F401
    from google.protobuf.internal import api_implementation as _api_impl

    try:
        _api_impl._implementation_type = "python"
    except Exception:
        pass
    print("protobuf implementation:", _api_impl.Type())
except Exception as e:
    print("WARNING: could not inspect/force protobuf implementation:", repr(e))

import tensorflow as tf

MODEL_PATH = "../input/mdpa56/initialweightInceptionResnet4.h5"


def _build_fallback_model(num_classes: int = 5):
    base = tf.keras.applications.InceptionResNetV2(
        include_top=False, weights="imagenet", input_shape=(299, 299, 3)
    )
    x = tf.keras.layers.GlobalAveragePooling2D(name="gap")(base.output)
    x = tf.keras.layers.Dense(num_classes, activation="softmax", name="pred")(x)
    model = tf.keras.Model(inputs=base.input, outputs=x)
    return model


model = None
load_error = None

if os.path.exists(MODEL_PATH):
    try:
        model = tf.keras.models.load_model(MODEL_PATH, compile=False)
        print("Loaded model from:", MODEL_PATH)
    except Exception as e:
        load_error = e
        model = None

if model is None:
    print("WARNING: Could not load provided .h5 model. Using fallback model instead.")
    if load_error is not None:
        print("Load error:", repr(load_error))
    model = _build_fallback_model(num_classes=5)

try:
    input_shape = model.input_shape
    if isinstance(input_shape, list):
        input_shape = input_shape[0]
    H = input_shape[1] or 299
    W = input_shape[2] or 299
except Exception:
    H, W = 299, 299

print("Model input size:", (H, W))



## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 2
import numpy as np
import pandas as pd
from PIL import Image

from sklearn.model_selection import train_test_split
import json

print("Imports OK")



## === cell 3
train = pd.read_csv(os.path.join(INPUT_DIR, "train.csv"))
with open(os.path.join(INPUT_DIR, "label_num_to_disease_map.json")) as f:
    classes = json.load(f)

train["class"] = train["label"].apply(lambda x: classes[str(x)])
print(train.head())



## === cell 4
train["path"] = train["image_id"].apply(
    lambda x: os.path.join(INPUT_DIR, "train_images", str(x))
)
train_df, val_df = train_test_split(
    train, test_size=0.05, random_state=100, stratify=train["label"].values
)
print("Train/val sizes:", len(train_df), len(val_df))



## === cell 5
from tensorflow.keras.preprocessing.image import ImageDataGenerator

batch_size = 12


def transform(image):
    image = image.astype(np.float32)
    return tf.keras.applications.inception_resnet_v2.preprocess_input(image)


datagen = ImageDataGenerator(preprocessing_function=transform).flow_from_dataframe(
    batch_size=batch_size,
    dataframe=train_df,
    directory=os.path.join(INPUT_DIR, "train_images"),
    shuffle=True,
    x_col="image_id",
    y_col="label",
    target_size=(H, W),
    class_mode="categorical",
)

val_datagen = ImageDataGenerator(preprocessing_function=transform).flow_from_dataframe(
    batch_size=batch_size,
    dataframe=val_df,
    directory=os.path.join(INPUT_DIR, "train_images"),
    shuffle=False,
    x_col="image_id",
    y_col="label",
    target_size=(H, W),
    class_mode="categorical",
)



## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4031458976.py in <cell line: 0>()
     10 
     11 
---> 12 datagen = ImageDataGenerator(preprocessing_function=transform).flow_from_dataframe(
     13     batch_size=batch_size,
     14     dataframe=train_df,

/usr/local/lib/python3.11/dist-packages/keras/src/legacy/preprocessing/image.py in flow_from_dataframe(self, dataframe, directory, x_col, y_col, weight_col, target_size, color_mode, classes, class_mode, batch_size, shuffle, seed, save_to_dir, save_prefix, save_format, subset, interpolation, validate_filenames, **kwargs)
   1206             )
   1207 
-> 1208         return DataFrameIterator(
   1209             dataframe,
   1210             directory,

/usr/local/lib/python3.11/dist-packages/keras/src/legacy/preprocessing/image.py in __init__(self, dataframe, directory, image_data_generator, x_col, y_col, weight_col, target_size, color_mode, classes, class_mode, batch_size, shuffle, seed, data_format, save_to_dir, save_prefix, save_format, subset, interpolation, keep_aspect_ratio, dtype, validate_filenames)
    749         self.dtype = dtype
    750         # check that inputs match the required class_mode
--> 751         self._check_params(df, x_col, y_col, weight_col, classes)
    752         if (
    753             validate_filenames

/usr/local/lib/python3.11/dist-packages/keras/src/legacy/preprocessing/image.py in _check_params(self, df, x_col, y_col, weight_col, classes)
    839             types = (str, list, tuple)
    840             if not all(df[y_col].apply(lambda x: isinstance(x, types))):
--> 841                 raise TypeError(
    842                     'If class_mode="{}", y_col="{}" column '
    843                     "values must be type string, list or tuple.".format(

TypeError: If class_mode="categorical", y_col="label" column values must be type string, list or tuple.

## === cell 6
os.makedirs("./checkpoints", exist_ok=True)
print("checkpoints dir ready")



## === cell 7
tf.keras.utils.set_random_seed(123)

if model.optimizer is None:
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
else:
    try:
        model.compile(
            optimizer=model.optimizer,
            loss=model.loss if model.loss is not None else "categorical_crossentropy",
            metrics=["accuracy"],
        )
    except Exception:
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )

ckpt_path = "./checkpoints/best.weights.h5"
callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        ckpt_path,
        monitor="val_accuracy",
        save_best_only=True,
        save_weights_only=True,
        verbose=1,
    )
]

EPOCHS = 1

history = model.fit(
    datagen,
    validation_data=val_datagen,
    epochs=EPOCHS,
    callbacks=callbacks,
    verbose=1,
)

if os.path.exists(ckpt_path):
    try:
        model.load_weights(ckpt_path)
        print("Loaded best weights from:", ckpt_path)
    except Exception as e:
        print("WARNING: could not load checkpoint weights:", repr(e))



## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/3694934130.py in <cell line: 0>()
      4 
      5 # Compile if needed (loaded .h5 might already be compiled; compile=False was used).
----> 6 if model.optimizer is None:
      7     model.compile(
      8         optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),

AttributeError: 'Functional' object has no attribute 'optimizer'

## === cell 8
sample_sub_path = os.path.join(INPUT_DIR, "sample_submission.csv")
sample_sub = pd.read_csv(sample_sub_path)
test_image_ids = sample_sub["image_id"].tolist()

TEST_DIR = os.path.join(INPUT_DIR, "test_images")
print("Num test images in sample:", len(test_image_ids))
print("Example:", test_image_ids[:3])




## === cell 9
def preprocess_pil(img: Image.Image):
    img = img.convert("RGB").resize((W, H))
    arr = np.asarray(img).astype(np.float32)
    arr = tf.keras.applications.inception_resnet_v2.preprocess_input(arr)
    return arr


predictions = []
bs = 32

for i in range(0, len(test_image_ids), bs):
    batch_ids = test_image_ids[i : i + bs]
    batch_arr = []
    for image_id in batch_ids:
        img_path = os.path.join(TEST_DIR, image_id)
        with Image.open(img_path) as img:
            batch_arr.append(preprocess_pil(img))
    batch_arr = np.stack(batch_arr, axis=0)
    probs = model.predict(batch_arr, verbose=0)
    preds = probs.argmax(axis=1).astype(int).tolist()
    predictions.extend(preds)

print("Predictions generated:", len(predictions))



## === cell 10
submission = pd.DataFrame({"image_id": test_image_ids, "label": predictions})
out_path = os.path.join(OUTPUT_DIR, "submission.csv")
submission.to_csv(out_path, index=False)
print("Wrote:", out_path)
print(submission.head())
