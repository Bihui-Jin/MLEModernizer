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

3.12

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

0.8319734058627984

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os, glob, json, random, shutil, datetime

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

SEED = 123
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

WORK_DIR = "../input/cassava-leaf-disease-classification/"
OUTPUT_DIR = "./"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("TensorFlow:", tf.__version__)
print("WORK_DIR exists:", os.path.exists(WORK_DIR))

print(
    "Skipping tf.config.experimental.enable_op_determinism() and thread config for stability."
)



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
df = pd.read_csv(os.path.join(WORK_DIR, "train.csv"))

target_folder = "dataset"
train_images_path = os.path.join(WORK_DIR, "train_images")
os.makedirs(target_folder, exist_ok=True)


def _safe_link(src, dst):
    if os.path.exists(dst):
        return True
    try:
        os.symlink(src, dst)
        return True
    except Exception:
        try:
            os.link(src, dst)
            return True
        except Exception:
            try:
                shutil.copy2(src, dst)
                return True
            except Exception:
                return False


need_populate = False
for label in df["label"].unique():
    label_folder = os.path.join(target_folder, str(int(label)))
    if (not os.path.exists(label_folder)) or (len(os.listdir(label_folder)) == 0):
        need_populate = True
        break

if not need_populate:
    sample_rows = df.sample(n=min(200, len(df)), random_state=SEED)
    for _, r in sample_rows.iterrows():
        p = os.path.join(target_folder, str(int(r["label"])), r["image_id"])
        if not os.path.exists(p):
            need_populate = True
            break

if need_populate:
    if os.path.exists(target_folder):
        shutil.rmtree(target_folder)
    os.makedirs(target_folder, exist_ok=True)

    missing_src = 0
    linked = 0
    for label, sub in df.groupby("label", sort=True):
        label_folder = os.path.join(target_folder, str(int(label)))
        os.makedirs(label_folder, exist_ok=True)
        img_ids = sub["image_id"].to_numpy()
        for image_name in img_ids:
            src_path = os.path.join(train_images_path, image_name)
            if not os.path.exists(src_path):
                missing_src += 1
                continue
            dst_path = os.path.join(label_folder, image_name)
            if _safe_link(src_path, dst_path):
                linked += 1

    print(
        f"Populated dataset. Linked/copied: {linked:,}. Missing source images skipped: {missing_src:,}."
    )
else:
    print("dataset folder already populated and looks valid; skipping population.")

print("Dataset root:", os.path.abspath(target_folder))
print("Train images path exists:", os.path.exists(train_images_path))
print(
    "Example label dirs:",
    sorted(
        [
            d
            for d in os.listdir(target_folder)
            if os.path.isdir(os.path.join(target_folder, d))
        ]
    )[:10],
)




## === cell 2
def img_train(
    directory, batch_size=32, image_size=(224, 224), validation_split=0.1, seed=123
):
    train_dataset = tf.keras.utils.image_dataset_from_directory(
        directory,
        labels="inferred",
        label_mode="categorical",
        class_names=None,
        color_mode="rgb",
        batch_size=batch_size,
        image_size=image_size,
        shuffle=True,
        seed=seed,
        validation_split=validation_split,
        subset="training",
        interpolation="bilinear",
        crop_to_aspect_ratio=False,
    )
    if validation_split is not None:
        valid_dataset = tf.keras.utils.image_dataset_from_directory(
            directory,
            labels="inferred",
            label_mode="categorical",
            class_names=None,
            color_mode="rgb",
            batch_size=batch_size,
            image_size=image_size,
            shuffle=True,
            seed=seed,
            validation_split=validation_split,
            subset="validation",
            interpolation="bilinear",
            crop_to_aspect_ratio=False,
        )
        return train_dataset, valid_dataset
    else:
        return train_dataset


dataset = "/kaggle/working/dataset"
train_dataset, valid_dataset = img_train(
    dataset, batch_size=16, image_size=(224, 224), validation_split=0.1, seed=SEED
)

AUTOTUNE = tf.data.AUTOTUNE
options = tf.data.Options()
options.experimental_deterministic = True

train_dataset = train_dataset.with_options(options).cache().prefetch(AUTOTUNE)
valid_dataset = valid_dataset.with_options(options).cache().prefetch(AUTOTUNE)

print("Train batches:", tf.data.experimental.cardinality(train_dataset).numpy())
print("Valid batches:", tf.data.experimental.cardinality(valid_dataset).numpy())




## === cell 3
def data_augmentation():
    return tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal", seed=SEED),
            tf.keras.layers.RandomTranslation(0.2, 0.2, fill_mode="reflect", seed=SEED),
            tf.keras.layers.RandomRotation(0.2, fill_mode="reflect", seed=SEED),
            tf.keras.layers.RandomZoom(0.2, seed=SEED),
            tf.keras.layers.RandomContrast(0.2, seed=SEED),
        ]
    )


data_aug = data_augmentation()

augmented_dataset = train_dataset.repeat(2).map(
    lambda x, y: (data_aug(x, training=True), y), num_parallel_calls=AUTOTUNE
)
train_dataset = (
    train_dataset.concatenate(augmented_dataset)
    .with_options(options)
    .prefetch(AUTOTUNE)
)

print(
    "augmentation train dataset size (batches):",
    tf.data.experimental.cardinality(train_dataset).numpy(),
)




## === cell 4
class SigmoidFocalCrossEntropy(tf.keras.losses.Loss):
    def __init__(self, alpha=0.25, gamma=2.0, from_logits=False, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha
        self.gamma = gamma
        self.from_logits = from_logits

    def call(self, y_true, y_pred):
        if self.from_logits:
            y_pred = tf.sigmoid(y_pred)
        y_pred = tf.clip_by_value(
            y_pred, tf.keras.backend.epsilon(), 1 - tf.keras.backend.epsilon()
        )
        cross_entropy = -y_true * tf.math.log(y_pred) - (1 - y_true) * tf.math.log(
            1 - y_pred
        )
        weight = self.alpha * y_true + (1 - self.alpha) * (1 - y_true)
        focal_loss = weight * ((1 - y_pred) ** self.gamma) * cross_entropy
        return tf.reduce_sum(focal_loss, axis=-1)




## === cell 5
class _HP:
    def Float(self, name, min_value=None, max_value=None, sampling=None):
        if name == "learning_rate":
            return float(np.sqrt(min_value * max_value))
        return float((min_value + max_value) / 2.0)

    def Int(self, name, min_value=None, max_value=None, step=1):
        v = min_value + ((max_value - min_value) // (2 * step)) * step
        return int(v)


class _SimpleTuner:
    def __init__(self, hypermodel, **kwargs):
        self.hypermodel = hypermodel
        self._best_hp = _HP()

    def search(self, *args, **kwargs):
        return

    def get_best_hyperparameters(self, num_trials=1):
        return [self._best_hp]




## === cell 6
def build_model(hp):
    learning_rate = hp.Float(
        "learning_rate", min_value=1e-4, max_value=1e-2, sampling="log"
    )
    dropout_rate = hp.Float("dropout_rate", min_value=0.0, max_value=0.5)
    dense_units = hp.Int("dense_units", min_value=128, max_value=512, step=32)

    try:
        base = tf.keras.applications.efficientnet_v2.EfficientNetV2S(
            weights="imagenet", include_top=False, input_shape=(224, 224, 3)
        )
    except Exception as e:
        print(
            "Warning: could not load imagenet weights, falling back to None. Error:",
            repr(e),
        )
        base = tf.keras.applications.efficientnet_v2.EfficientNetV2S(
            weights=None, include_top=False, input_shape=(224, 224, 3)
        )

    x = tf.keras.layers.GlobalAveragePooling2D()(base.output)
    x = tf.keras.layers.Dense(dense_units, activation="relu")(x)
    x = tf.keras.layers.Dropout(dropout_rate)(x)
    outputs = tf.keras.layers.Dense(5, activation="softmax")(x)
    model = tf.keras.Model(inputs=base.input, outputs=outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=SigmoidFocalCrossEntropy(alpha=0.25, gamma=2, from_logits=False),
        metrics=[tf.keras.metrics.CategoricalAccuracy(name="accuracy")],
    )
    return model


tuner = _SimpleTuner(build_model)

early_stop = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    min_delta=0.001,
    patience=3,
    mode="min",
    verbose=1,
    restore_best_weights=True,
)

tuner.search(
    train_dataset, validation_data=valid_dataset, epochs=2, callbacks=[early_stop]
)

best_hps = tuner.get_best_hyperparameters()[0]
print("Using hyperparameters (deterministic heuristic).")




## === cell 7
def Model_Training(tuner, train_dataset, valid_dataset, epochs=10):
    best_hps = tuner.get_best_hyperparameters()[0]
    model = (
        tuner.hypermodel(best_hps)
        if callable(getattr(tuner, "hypermodel", None))
        else tuner.hypermodel.build(best_hps)
    )

    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        min_delta=0.001,
        patience=5,
        mode="min",
        verbose=1,
        restore_best_weights=True,
    )

    history = model.fit(
        train_dataset,
        epochs=epochs,
        validation_data=valid_dataset,
        callbacks=[early_stop],
        verbose=1,
    )

    plt.plot(history.history.get("accuracy", []))
    plt.plot(history.history.get("val_accuracy", []))
    plt.title("Model accuracy")
    plt.ylabel("Accuracy")
    plt.xlabel("Epoch")
    plt.legend(["Train", "Valid"], loc="upper left")
    plt.show()
    plt.clf()

    plt.plot(history.history.get("loss", []))
    plt.plot(history.history.get("val_loss", []))
    plt.title("Model loss")
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.legend(["Train", "Valid"], loc="upper left")
    plt.show()
    plt.clf()

    return model


model = Model_Training(tuner, train_dataset, valid_dataset, epochs=10)



## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NotFoundError                             Traceback (most recent call last)
/tmp/ipykernel_11/1233967963.py in <cell line: 0>()
     45 
     46 
---> 47 model = Model_Training(tuner, train_dataset, valid_dataset, epochs=10)
     48 

/tmp/ipykernel_11/1233967963.py in Model_Training(tuner, train_dataset, valid_dataset, epochs)
     16     )
     17 
---> 18     history = model.fit(
     19         train_dataset,
     20         epochs=epochs,

/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py in error_handler(*args, **kwargs)
    120             # To get the full stack trace, call:
    121             # `keras.config.disable_traceback_filtering()`
--> 122             raise e.with_traceback(filtered_tb) from None
    123         finally:
    124             del filtered_tb

/usr/local/lib/python3.11/dist-packages/tensorflow/python/eager/execute.py in quick_execute(op_name, num_outputs, inputs, attrs, ctx, name)
     57       e.message += " name: " + name
     58     raise core._status_to_exception(e) from None
---> 59   except TypeError as e:
     60     keras_symbolic_tensors = [x for x in inputs if _is_keras_symbolic_tensor(x)]
     61     if keras_symbolic_tensors:

NotFoundError: Graph execution error:

Detected at node ReadFile defined at (most recent call last):
<stack traces unavailable>
/kaggle/working/dataset/1/2326585522.jpg; No such file or directory
	 [[{{node ReadFile}}]]
	 [[IteratorGetNext]] [Op:__inference_multi_step_on_iterator_107580]

## === cell 8
sample_sub_path = os.path.join(WORK_DIR, "sample_submission.csv")
sample_sub = pd.read_csv(sample_sub_path)

test_dir = os.path.join(WORK_DIR, "test_images")
test_files = sample_sub["image_id"].tolist()


def _load_and_preprocess_path(image_name):
    image_path = tf.strings.join([test_dir, "/", image_name])
    img_bytes = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img_bytes, channels=3)
    img = tf.image.resize(img, [224, 224], method=tf.image.ResizeMethod.BILINEAR)
    img = tf.keras.applications.efficientnet_v2.preprocess_input(
        tf.cast(img, tf.float32)
    )
    return img


test_ds = tf.data.Dataset.from_tensor_slices(tf.constant(test_files))
test_ds = test_ds.map(_load_and_preprocess_path, num_parallel_calls=AUTOTUNE)
test_ds = test_ds.batch(64).prefetch(AUTOTUNE).with_options(options)

probs = model.predict(test_ds, verbose=0)
preds = np.argmax(probs, axis=-1).astype(int)

submission = pd.DataFrame({"image_id": test_files, "label": preds})
submission.to_csv("submission.csv", index=False)

print("Wrote submission.csv with shape:", submission.shape)
print(submission.head())

folder = "/kaggle/working/dataset"
try:
    shutil.rmtree(folder)
    print("Folder Deleted")
except OSError as e:
    print(f"Error: {e}")

## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2706304246.py in <cell line: 0>()
     21 test_ds = test_ds.batch(64).prefetch(AUTOTUNE).with_options(options)
     22 
---> 23 probs = model.predict(test_ds, verbose=0)
     24 preds = np.argmax(probs, axis=-1).astype(int)
     25 

NameError: name 'model' is not defined
