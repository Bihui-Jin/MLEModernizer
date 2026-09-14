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

0.7819582955575702

# 6. Current score

0.61099

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.1065) has done: 'The changes focus on eliminating unnecessary file‑system scans and redundant list handling during inference. We restrict the glob to the known test‑image directory, avoiding costly recursive searches that can enumerate thousands of unrelated files. Prediction results are kept as a NumPy array directly (no intermediate Python list or averaging of a single model), and we increase the batch size to reduce the number of generator steps. These adjustments keep the model architecture and inference logic unchanged while cutting I/O and Python‑level overhead, ensuring the script completes well within the 600‑second limit.'
- What this solution (achieved 0.1506) has done: 'The fix adds a protobuf compatibility setting before importing TensorFlow and corrects the model weight path to the Kaggle `/kaggle/input` directory so the pretrained model can be loaded. These minimal changes resolve the import error and enable the pretrained network to generate accurate predictions, moving the score toward the target while keeping the original architecture and workflow intact.'
- What this solution (achieved 0.61099) has done: 'Implemented a feature‑pre‑computation step so the heavy EfficientNet backbone is executed only once per image. The script now:
1. Builds the EfficientNetB3 backbone (frozen) and the full model as before.
2. If no pretrained weights are found, it extracts backbone features for the whole training and validation sets once.
3. Trains only the top dense layer on these cached features, which is orders of magnitude faster than repeatedly passing images through the backbone.
4. Copies the learned dense weights back into the original model so inference on the test set remains unchanged.

These changes keep the exact architecture and loss/optimizer logic while dramatically reducing runtime, ensuring the whole pipeline finishes well within the 600‑second limit.'
- What this solution (achieved 0.61099) has done: 'Implemented two key fixes: (1) correctly retrieve the EfficientNetB3 backbone from the built model (layer index 1) to avoid the empty‑inputs error during feature‑extractor construction; (2) modestly increase the fine‑tuning epochs from 3 to 5 to improve validation accuracy and move the score closer to the target. No other logic changes were made.'
- What this solution (achieved 0.11024) has done: 'The fix corrects the `Input` layer shape by passing a tuple instead of an integer, preventing the ValueError during model construction. It also modestly increases the top‑model training epochs from 5 to 8 to improve validation accuracy and move the Kaggle score closer to the target while keeping the core architecture unchanged.'
- What this solution (achieved 0.61099) has done: 'The fix replaces the incompatible weight‑transfer with a proper reconstruction of the full model using the trained top‑layer, then briefly fine‑tunes the whole network so the predictions use the learned weights. This resolves the ValueError and improves validation accuracy, moving the score toward the target while keeping the original architecture unchanged.'
- What this solution (achieved 0.61099) has done: 'We boost validation accuracy by (1) training the top dense layer a bit longer (8 → 10 epochs), (2) fine‑tuning the full EfficientNet model for one extra epoch (2 → 3 epochs), and (3) adding lightweight random flips to the training images during full‑model training. These tweaks keep the original architecture and loss unchanged while nudging the score toward the target.'

# 9. Code solution

## === cell 0
import os
import tempfile  # added for disk‑based cache

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
try:
    import google.protobuf.message_factory as mf

    if not hasattr(mf.MessageFactory, "GetPrototype"):

        def GetPrototype(self, descriptor):
            return self.GetMessageClass(descriptor)

        mf.MessageFactory.GetPrototype = GetPrototype
except Exception:
    pass

import glob
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from sklearn.model_selection import train_test_split

SEED = 42
DEBUG = False
tf.random.set_seed(SEED)

tf.config.threading.set_intra_op_parallelism_threads(8)
tf.config.threading.set_inter_op_parallelism_threads(8)

gpus = tf.config.list_physical_devices("GPU")
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except Exception as e:
        print("Could not set GPU memory growth:", e)




## === cell 1
def build_model(input_shape=(512, 512, 3), num_classes=5):
    base = EfficientNetB3(
        weights="imagenet", include_top=False, input_shape=input_shape
    )
    base.trainable = False  # <-- speed‑up training, keeps pretrained weights
    x = GlobalAveragePooling2D()(base.output)
    output = Dense(num_classes, activation="softmax")(x)
    model = Model(inputs=base.input, outputs=output)
    model.compile(
        optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    )
    return model




## === cell 2
weight_path = os.path.join(
    "/kaggle/input", "model-ensembling-with-k-fold", "model_v0.39.h5"
)
pretrained_loaded = False  # flag to know if we should skip training
if os.path.exists(weight_path):
    try:
        my_model = load_model(weight_path)
        pretrained_loaded = True
    except Exception as e:
        print("Failed to load pretrained model, building a fresh one:", e)
        my_model = build_model()
else:
    print("Pretrained weight file not found, building a fresh model.")
    my_model = build_model()




## === cell 3
train_csv_path = os.path.join(
    "/kaggle/input", "cassava-leaf-disease-classification", "train.csv"
)
df_train_csv = pd.read_csv(train_csv_path)

train_images_dir = os.path.abspath(
    os.path.join(
        "/kaggle/input",
        "cassava-leaf-disease-classification",
        "train_images",
    )
)
df_train = df_train_csv.copy()
df_train["path"] = df_train["image_id"].apply(
    lambda x: os.path.join(train_images_dir, x)
)

train_df, val_df = train_test_split(
    df_train,
    test_size=0.1,
    random_state=SEED,
    stratify=df_train["label"],
)

TRAIN_BATCH_SIZE = 256
VAL_BATCH_SIZE = 256
AUTOTUNE = tf.data.AUTOTUNE


def _parse_image(filename, label=None):
    img = tf.io.read_file(filename)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [512, 512])
    img = img / 255.0  # rescale
    if label is None:
        return img
    else:
        return img, label


train_paths = train_df["path"].values
train_labels = train_df["label"].values.astype(np.int32)

val_paths = val_df["path"].values
val_labels = val_df["label"].values.astype(np.int32)

if not pretrained_loaded:
    backbone = my_model.layers[1] if len(my_model.layers) > 1 else my_model.layers[0]

    feature_extractor = Model(
        inputs=backbone.input,
        outputs=GlobalAveragePooling2D()(backbone.output),
    )
    feature_extractor.trainable = False

    def _image_dataset(paths, batch_size):
        ds = tf.data.Dataset.from_tensor_slices(paths)
        ds = ds.map(lambda x: _parse_image(x), num_parallel_calls=AUTOTUNE)
        ds = ds.batch(batch_size).prefetch(AUTOTUNE)
        return ds

    train_feat_ds = _image_dataset(train_paths, TRAIN_BATCH_SIZE)
    val_feat_ds = _image_dataset(val_paths, VAL_BATCH_SIZE)

    train_features = feature_extractor.predict(train_feat_ds, verbose=0)
    val_features = feature_extractor.predict(val_feat_ds, verbose=0)

    top_input = tf.keras.Input(shape=(train_features.shape[1],))
    top_output = Dense(5, activation="softmax")(top_input)
    top_model = Model(inputs=top_input, outputs=top_output)
    top_model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    train_feat_dataset = tf.data.Dataset.from_tensor_slices(
        (train_features, train_labels)
    )
    train_feat_dataset = train_feat_dataset.shuffle(
        buffer_size=min(10000, len(train_features)), seed=SEED
    ).batch(TRAIN_BATCH_SIZE)

    val_feat_dataset = tf.data.Dataset.from_tensor_slices((val_features, val_labels))
    val_feat_dataset = val_feat_dataset.batch(VAL_BATCH_SIZE)

    top_model.fit(
        train_feat_dataset,
        validation_data=val_feat_dataset,
        epochs=10,  # increased from 8
        verbose=2,
    )

    my_model = Model(inputs=my_model.input, outputs=top_model(feature_extractor.output))
    backbone.trainable = True
    my_model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    train_dataset_full = tf.data.Dataset.from_tensor_slices((train_paths, train_labels))
    train_dataset_full = train_dataset_full.map(
        _parse_image, num_parallel_calls=AUTOTUNE
    )

    def _augment(img, lbl):
        img = tf.image.random_flip_left_right(img)
        img = tf.image.random_flip_up_down(img)
        return img, lbl

    train_dataset_full = train_dataset_full.map(_augment, num_parallel_calls=AUTOTUNE)

    train_dataset_full = (
        train_dataset_full.shuffle(
            buffer_size=min(10000, len(train_paths)),
            seed=SEED,
            reshuffle_each_iteration=True,
        )
        .batch(TRAIN_BATCH_SIZE)
        .prefetch(AUTOTUNE)
    )

    val_dataset_full = tf.data.Dataset.from_tensor_slices((val_paths, val_labels))
    val_dataset_full = val_dataset_full.map(_parse_image, num_parallel_calls=AUTOTUNE)
    val_dataset_full = val_dataset_full.batch(VAL_BATCH_SIZE).prefetch(AUTOTUNE)

    my_model.fit(
        train_dataset_full,
        validation_data=val_dataset_full,
        epochs=3,  # increased from 2
        verbose=2,
    )

else:
    train_dataset = tf.data.Dataset.from_tensor_slices((train_paths, train_labels))
    train_dataset = train_dataset.map(_parse_image, num_parallel_calls=AUTOTUNE)
    train_dataset = train_dataset.cache()  # memory cache, faster than disk
    train_dataset = train_dataset.shuffle(
        buffer_size=min(10000, len(train_paths)),
        seed=SEED,
        reshuffle_each_iteration=True,
    )
    train_dataset = train_dataset.batch(TRAIN_BATCH_SIZE).prefetch(AUTOTUNE)

    val_dataset = tf.data.Dataset.from_tensor_slices((val_paths, val_labels))
    val_dataset = val_dataset.map(_parse_image, num_parallel_calls=AUTOTUNE)
    val_dataset = val_dataset.cache()
    val_dataset = val_dataset.batch(VAL_BATCH_SIZE).prefetch(AUTOTUNE)

    my_model.fit(train_dataset, validation_data=val_dataset, epochs=3, verbose=2)




## === cell 4
test_dir = os.path.abspath(
    os.path.join(
        "/kaggle/input",
        "cassava-leaf-disease-classification",
        "test_images",
    )
)
if not os.path.isdir(test_dir):
    raise FileNotFoundError(f"Test directory not found: {test_dir}")

test_images = sorted(glob.glob(os.path.join(test_dir, "*.jpg")))
if not test_images:
    raise FileNotFoundError("No test images found in the specified directory.")

df_test = pd.DataFrame(test_images, columns=["path"])




## === cell 5
def make_test_gen(batch_size=256):
    test_paths = df_test["path"].values
    test_dataset = tf.data.Dataset.from_tensor_slices(test_paths)
    test_dataset = test_dataset.map(
        lambda x: _parse_image(x), num_parallel_calls=AUTOTUNE
    )
    test_dataset = test_dataset.batch(batch_size).prefetch(AUTOTUNE)
    return test_dataset




## === cell 6
test_gen = make_test_gen(batch_size=256)
pred_test = my_model.predict(test_gen, verbose=1)
pred_test_labels = np.argmax(pred_test, axis=-1)

final_submission = df_test.copy()
final_submission["image_id"] = final_submission["path"].str.split(os.sep).str[-1]
final_submission["label"] = pred_test_labels.astype(int)

final_csv = final_submission[["image_id", "label"]]
final_csv.to_csv("submission.csv", index=False)
print("Submission file 'submission.csv' created with {} rows.".format(len(final_csv)))




## === cell 7
final_csv.head()
