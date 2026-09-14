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

0.8428528256270776

# 6. Current score

0.15022

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.16143) has done: 'The changes remove the problematic `tensorflow_hub` import, replace the unavailable EfficientNet package with TensorFlow’s built‑in EfficientNetB4, add a safe fallback if the custom weight file can’t be loaded, and adjust the image size to match the model’s expected input. These fixes eliminate the import and module errors, ensure `my_model` is defined, and finally write a proper `submission.csv` with the required columns.'
- What this solution (achieved 0.05531) has done: 'I set the protobuf implementation environment variable before importing TensorFlow to avoid the import error, then build the EfficientNet‑B4 model, fine‑tune its top layer on the provided training data for a few epochs, and finally generate predictions for the test set and write a correctly formatted `submission.csv`. This fixes the runtime crash and improves the accuracy toward the target score while keeping the original model architecture unchanged.'
- What this solution (achieved 0.57175) has done: 'The script was missing all required imports, the `SEED` constant, and attempted to use TensorFlow before importing it, which caused the runtime errors. I added the necessary imports (`tensorflow`, `os`, `pandas`, `numpy`, `glob`, and Keras utilities), defined a default `SEED`, and reordered the initialization so TensorFlow is imported before configuring its threading. These minimal changes fix the NameErrors and allow the model to be built, trained, and used for predictions, resulting in a correctly formatted `submission.csv` file.'
- What this solution (achieved 0.31876) has done: 'I add a flag that detects whether the pre‑trained weights file is present. If it is, the model is already fine‑tuned and we can safely skip the two training phases, which removes the most time‑consuming part of the script. When training is needed (weights file missing) I enable multi‑process data loading in `model.fit` (`workers=4, use_multiprocessing=True`) to speed up image preprocessing without changing the model architecture or training schedule. All other logic, paths, and output generation remain unchanged.'
- What this solution (achieved 0.15022) has done: 'I reduced the overhead of Python‑multiprocessing by limiting the data pipelines and `fit` calls to a single worker and disabling multiprocessing. This avoids the costly process‑creation and inter‑process data transfer while preserving the exact model, augmentations, training schedule, and prediction logic, so the results remain unchanged but the overall runtime drops well under the 600‑second limit.'

# 9. Code solution

## === cell 0
import os
import math

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import glob
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator

tf.config.optimizer.set_jit(True)

tf.config.threading.set_inter_op_parallelism_threads(4)
tf.config.threading.set_intra_op_parallelism_threads(4)

SEED = 42
tf.random.set_seed(SEED)
np.random.seed(SEED)

PRETRAINED_WEIGHT_PATH = "../input/model-ensembling-with-k-fold/fineTuned_v0.59.h5"
PRETRAINED_AVAILABLE = os.path.exists(PRETRAINED_WEIGHT_PATH)


def get_model():
    """
    Load a fine‑tuned model if the weight file exists,
    otherwise build a fresh MobileNetV2 backbone with a
    softmax classifier for 5 classes.
    """
    try:
        if PRETRAINED_AVAILABLE:
            return load_model(PRETRAINED_WEIGHT_PATH)
        else:
            raise FileNotFoundError
    except Exception:
        base = tf.keras.applications.MobileNetV2(
            include_top=False, weights="imagenet", input_shape=(224, 224, 3)
        )
        x = GlobalAveragePooling2D()(base.output)
        output = Dense(5, activation="softmax")(x)  # 5 classes
        model = Model(inputs=base.input, outputs=output)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )
        return model


my_model = get_model()




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
train_csv_path = "../input/cassava-leaf-disease-classification/train.csv"
train_images_dir = "../input/cassava-leaf-disease-classification/train_images"

train_df = pd.read_csv(train_csv_path)
train_df["label"] = train_df["label"].astype(str)
train_df["path"] = train_df["image_id"].apply(
    lambda x: os.path.join(train_images_dir, x)
)

BATCH_SIZE = 128
NUM_WORKERS = 1  # use a single worker to avoid multiprocessing overhead

train_idg = ImageDataGenerator(
    rescale=1.0 / 255.0,
    horizontal_flip=True,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
)

train_gen = train_idg.flow_from_dataframe(
    dataframe=train_df,
    x_col="path",
    y_col="label",
    target_size=(224, 224),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=True,
    seed=SEED,
    workers=NUM_WORKERS,
    use_multiprocessing=False,  # disable multiprocessing for speed
)




## === cell 2
steps_per_epoch = math.ceil(len(train_df) / BATCH_SIZE)

if not PRETRAINED_AVAILABLE:
    my_model.layers[0].trainable = False
    my_model.fit(
        train_gen,
        epochs=5,
        steps_per_epoch=steps_per_epoch,
        verbose=1,
        workers=NUM_WORKERS,
        use_multiprocessing=False,
    )

    my_model.layers[0].trainable = True
    my_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    my_model.fit(
        train_gen,
        epochs=15,
        steps_per_epoch=steps_per_epoch,
        verbose=1,
        workers=NUM_WORKERS,
        use_multiprocessing=False,
    )
else:
    pass




## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/665727614.py in <cell line: 0>()
      3 if not PRETRAINED_AVAILABLE:
      4     my_model.layers[0].trainable = False
----> 5     my_model.fit(
      6         train_gen,
      7         epochs=5,

/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py in error_handler(*args, **kwargs)
    120             # To get the full stack trace, call:
    121             # `keras.config.disable_traceback_filtering()`
--> 122             raise e.with_traceback(filtered_tb) from None
    123         finally:
    124             del filtered_tb

/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py in error_handler(*args, **kwargs)
    117             return fn(*args, **kwargs)
    118         except Exception as e:
--> 119             filtered_tb = _process_traceback_frames(e.__traceback__)
    120             # To get the full stack trace, call:
    121             # `keras.config.disable_traceback_filtering()`

TypeError: TensorFlowTrainer.fit() got an unexpected keyword argument 'workers'

## === cell 3
test_images = glob.glob(
    "../input/cassava-leaf-disease-classification/test_images/*.jpg"
)
df_test = pd.DataFrame(test_images, columns=["path"])


def make_test_gen(batch_size=128):
    test_idg = ImageDataGenerator(rescale=1.0 / 255.0)
    test_gen = test_idg.flow_from_dataframe(
        dataframe=df_test,
        x_col="path",
        y_col=None,
        batch_size=batch_size,
        seed=SEED,
        shuffle=False,
        class_mode=None,
        target_size=(224, 224),
        interpolation="nearest",
        workers=NUM_WORKERS,
        use_multiprocessing=False,
    )
    return test_gen


test_gen = make_test_gen(batch_size=128)

test_steps = math.ceil(len(df_test) / 128)
pred_test = my_model.predict(test_gen, steps=test_steps, verbose=1)
pred_test_labels = np.argmax(pred_test, axis=-1)




## === cell 4
final_submission = df_test.copy()
final_submission["image_id"] = final_submission["path"].str.split("/").str[-1]
final_submission["label"] = pred_test_labels
final_csv = final_submission[["image_id", "label"]]

final_csv.to_csv("submission.csv", index=False)
print(final_csv.head())
