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

0.1403747355696585

# 6. Current score

0.07175

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved 0.07175) has done: 'The changes replace the slow Python‑based ImageDataGenerator with a fully TensorFlow tf.data pipeline that preloads, preprocesses, caches, augments, batches and prefetches images. This removes the Python‑level per‑image augmentation loop, speeds up I/O and GPU/CPU utilization, and keeps the model architecture, training loop, loss and metrics unchanged, preserving result accuracy while fitting comfortably inside the 600‑second limit.'

# 9. Code solution

## === cell 0
import os, random, warnings
import numpy as np, pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, Flatten, Dense, Dropout
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import mixed_precision
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")


def seed_everything(seed=21):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["TF_DETERMINISTIC_OPS"] = "1"


seed_everything(21)

mixed_precision.set_global_policy("mixed_float16")




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
work_dir = "../input/cassava-leaf-disease-classification/"
train_path = os.path.join(work_dir, "train_images")
test_path = os.path.join(work_dir, "test_images")

train_df = pd.read_csv(os.path.join(work_dir, "train.csv"))

label_map_path = os.path.join(work_dir, "label_num_to_disease_map.json")
real_labels = pd.read_json(label_map_path, typ="series").to_dict()
real_labels = {int(k): v for k, v in real_labels.items()}
train_df["class_name"] = train_df["label"].map(real_labels)




## === cell 2
IMG_SIZE = 320
BATCH_SIZE = 128
IMG_SHAPE = (IMG_SIZE, IMG_SIZE, 3)
N_CLASS = 5

train_split, val_split = train_test_split(
    train_df, test_size=0.05, random_state=42, stratify=train_df["class_name"]
)

class_names = sorted(train_df["class_name"].unique())
class_to_idx = {name: i for i, name in enumerate(class_names)}


def _load_and_preprocess(path, label):
    """Read image file, decode, resize, and apply EfficientNet preprocessing."""
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
    image = tf.keras.applications.efficientnet.preprocess_input(image)
    return image, tf.one_hot(label, N_CLASS)


def _augment(image, label):
    """Apply augmentations comparable to the original ImageDataGenerator."""
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    angle = tf.random.uniform([], -40, 40) * tf.constant(np.pi / 180)
    image = tfa.image.rotate(image, angle, fill_mode="nearest")
    scales = tf.random.uniform([], 0.8, 1.2)
    new_height = tf.cast(scales * IMG_SIZE, tf.int32)
    new_width = tf.cast(scales * IMG_SIZE, tf.int32)
    image = tf.image.resize(image, [new_height, new_width])
    image = tf.image.resize_with_crop_or_pad(image, IMG_SIZE, IMG_SIZE)
    return image, label


def _build_dataset(df, augment=False):
    img_paths = tf.convert_to_tensor(
        df["image_id"].apply(lambda x: os.path.join(train_path, x)).values
    )
    labels = tf.convert_to_tensor(
        df["class_name"].map(class_to_idx).values, dtype=tf.int32
    )
    ds = tf.data.Dataset.from_tensor_slices((img_paths, labels))
    ds = ds.map(_load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    if augment:
        import tensorflow_addons as tfa  # only needed for rotate

        ds = ds.map(_augment, num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.shuffle(1024, seed=42)
    ds = ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return ds


train_ds = _build_dataset(train_split, augment=True)
val_ds = _build_dataset(val_split, augment=False)




## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
ModuleNotFoundError                       Traceback (most recent call last)
/tmp/ipykernel_55/1966099051.py in <cell line: 0>()
     63 
     64 
---> 65 train_ds = _build_dataset(train_split, augment=True)
     66 val_ds = _build_dataset(val_split, augment=False)
     67 

/tmp/ipykernel_55/1966099051.py in _build_dataset(df, augment)
     55     ds = ds.map(_load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
     56     if augment:
---> 57         import tensorflow_addons as tfa  # only needed for rotate
     58 
     59         ds = ds.map(_augment, num_parallel_calls=tf.data.AUTOTUNE)

ModuleNotFoundError: No module named 'tensorflow_addons'

## === cell 3
def create_model():
    model = Sequential()
    model.add(
        EfficientNetB3(
            input_shape=IMG_SHAPE,
            include_top=False,
            weights="imagenet",
        )
    )
    model.add(GlobalAveragePooling2D())
    model.add(Flatten())
    model.add(
        Dense(
            256,
            activation="relu",
            bias_regularizer=tf.keras.regularizers.L1L2(l1=0.01, l2=0.001),
        )
    )
    model.add(Dropout(0.5))
    model.add(Dense(N_CLASS, activation="softmax"))
    return model


leaf_model = create_model()
leaf_model.compile(
    optimizer=Adam(learning_rate=1e-4),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)
leaf_model.summary()




## === cell 4
EPOCHS = 5
leaf_model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    verbose=2,
)




## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/965420260.py in <cell line: 0>()
      1 EPOCHS = 5
      2 leaf_model.fit(
----> 3     train_ds,
      4     validation_data=val_ds,
      5     epochs=EPOCHS,

NameError: name 'train_ds' is not defined

## === cell 5
val_loss, val_acc = leaf_model.evaluate(val_ds, verbose=0)
print(f"Validation accuracy: {val_acc:.4f}")




## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1824420746.py in <cell line: 0>()
----> 1 val_loss, val_acc = leaf_model.evaluate(val_ds, verbose=0)
      2 print(f"Validation accuracy: {val_acc:.4f}")
      3 
      4 

NameError: name 'val_ds' is not defined

## === cell 6
sample_sub = pd.read_csv(os.path.join(work_dir, "sample_submission.csv"))
test_df = sample_sub[["image_id"]].copy()

test_paths = tf.convert_to_tensor(
    test_df["image_id"].apply(lambda x: os.path.join(test_path, x)).values
)
test_ds = tf.data.Dataset.from_tensor_slices(test_paths)
test_ds = test_ds.map(
    lambda p: tf.keras.applications.efficientnet.preprocess_input(
        tf.image.resize(
            tf.image.decode_jpeg(tf.io.read_file(p), channels=3), (IMG_SIZE, IMG_SIZE)
        )
    ),
    num_parallel_calls=tf.data.AUTOTUNE,
)
test_ds = test_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

preds_prob = leaf_model.predict(test_ds, verbose=1)
preds = np.argmax(preds_prob, axis=1)

submission = pd.DataFrame({"image_id": test_df["image_id"], "label": preds})
submission.to_csv("submission.csv", index=False)
print("Submission file saved as submission.csv")
