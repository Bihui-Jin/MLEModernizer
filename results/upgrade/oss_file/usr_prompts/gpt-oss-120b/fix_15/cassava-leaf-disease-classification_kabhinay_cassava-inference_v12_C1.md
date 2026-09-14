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

0.876095497129042

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.26794) has done: 'The changes increase the data pipeline throughput: larger batch sizes cut the number of training steps, and enabling multiple worker processes lets TensorFlow load and augment images in parallel. The test prediction also uses a batch size > 1, reducing the number of forward passes. These adjustments keep the same model architecture, loss, epochs, and overall training logic, so the final predictions remain equivalent while the total runtime drops below the 600‑second limit.'
- What this solution (achieved 0.34753) has done: 'We increase data loading parallelism and batch size (reducing the number of steps per epoch) while keeping the same model architecture and training schedule. Adding `workers` and `use_multiprocessing` to `model.fit` lets Keras read and augment images concurrently, and a larger batch size halves the number of batches per epoch, both cutting runtime without altering the learning process. The test‑time generator batch size is also raised to speed up prediction.'
- What this solution (achieved 0.05531) has done: 'We speed up the pipeline by parallelising image loading and preprocessing during training (using multiple workers and multiprocessing) and by configuring TensorFlow thread pools, which cuts the I/O‑bound overhead without altering the model architecture or training schedule. All other logic—including the model, epochs, and data splits—remains unchanged, so the predictions stay identical.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

try:
    import tensorflow as tf
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    from tensorflow.keras import layers, models, optimizers
    from tensorflow.keras.applications import MobileNetV2

    tf_available = True
except Exception as e:
    print("TensorFlow import failed:", e)
    tf_available = False

np.random.seed(42)
if tf_available:
    tf.random.set_seed(42)
    tf.config.threading.set_inter_op_parallelism_threads(os.cpu_count())
    tf.config.threading.set_intra_op_parallelism_threads(os.cpu_count())




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
BASE_PATH = "/kaggle/input/cassava-leaf-disease-classification"
TRAIN_CSV = os.path.join(BASE_PATH, "train.csv")
TRAIN_DIR = os.path.join(BASE_PATH, "train_images")
TEST_DIR = os.path.join(BASE_PATH, "test_images")
SUBMISSION_PATH = "submission.csv"

train_df = pd.read_csv(TRAIN_CSV)
train_df["label"] = train_df["label"].astype(str)

if tf_available:
    BATCH_SIZE = 512

    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        validation_split=0.1,
        horizontal_flip=True,
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
    )

    workers = os.cpu_count()
    train_generator = train_datagen.flow_from_dataframe(
        dataframe=train_df,
        directory=TRAIN_DIR,
        x_col="image_id",
        y_col="label",
        target_size=(224, 224),
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        subset="training",
        shuffle=True,
        seed=42,
        workers=workers,
        use_multiprocessing=True,
    )

    val_generator = train_datagen.flow_from_dataframe(
        dataframe=train_df,
        directory=TRAIN_DIR,
        x_col="image_id",
        y_col="label",
        target_size=(224, 224),
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        subset="validation",
        shuffle=False,
        seed=42,
        workers=workers,
        use_multiprocessing=True,
    )

    base_model = MobileNetV2(
        input_shape=(224, 224, 3), include_top=False, weights=None, pooling="avg"
    )
    base_model.trainable = False

    model = models.Sequential(
        [
            base_model,
            layers.Dense(128, activation="relu"),
            layers.Dropout(0.3),
            layers.Dense(5, activation="softmax"),
        ]
    )

    model.compile(
        optimizer=optimizers.Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    model.fit(
        train_generator,
        epochs=15,
        validation_data=val_generator,
        verbose=2,
        workers=workers,
        use_multiprocessing=True,
    )

    base_model.trainable = True
    model.compile(
        optimizer=optimizers.Adam(learning_rate=1e-4),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.fit(
        train_generator,
        epochs=5,
        validation_data=val_generator,
        verbose=2,
        workers=workers,
        use_multiprocessing=True,
    )
else:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import LabelEncoder
    from PIL import Image

    IMG_SIZE = (64, 64)  # small size for fast training

    def load_images(dir_path, file_list, size=IMG_SIZE):
        imgs = []
        for f in file_list:
            path = os.path.join(dir_path, f)
            img = Image.open(path).convert("RGB").resize(size)
            imgs.append(np.array(img, dtype=np.float32) / 255.0)
        return np.stack(imgs)

    train_fnames = train_df["image_id"].tolist()
    X_train = load_images(TRAIN_DIR, train_fnames)
    X_train = X_train.reshape(len(train_fnames), -1)

    le = LabelEncoder()
    y_train = le.fit_transform(train_df["label"])

    clf = LogisticRegression(
        multi_class="multinomial", max_iter=200, n_jobs=os.cpu_count()
    )
    clf.fit(X_train, y_train)

    class SklearnModel:
        def __init__(self, clf, le):
            self.clf = clf
            self.le = le

        def predict(self, generator, verbose=0):
            probs = []
            for batch in generator:
                batch = batch.astype(np.float32) / 255.0
                batch = batch.reshape(batch.shape[0], -1)
                probs.append(self.clf.predict_proba(batch))
            return np.vstack(probs)

    model = SklearnModel(clf, le)




## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/591171074.py in <cell line: 0>()
     74 
     75     # Pass workers to fit for consistent multiprocessing during training
---> 76     model.fit(
     77         train_generator,
     78         epochs=15,

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

## === cell 2
if tf_available:
    test_datagen = ImageDataGenerator(rescale=1.0 / 255)

    test_files = os.listdir(TEST_DIR)
    test_df = pd.DataFrame({"image_id": test_files})

    test_generator = test_datagen.flow_from_dataframe(
        dataframe=test_df,
        directory=TEST_DIR,
        x_col="image_id",
        y_col=None,
        target_size=(224, 224),
        batch_size=256,
        class_mode=None,
        shuffle=False,
        workers=os.cpu_count(),
        use_multiprocessing=True,
    )

    pred_probs = model.predict(test_generator, verbose=1)
    pred_indices = np.argmax(pred_probs, axis=1)
    idx_to_label = {v: k for k, v in train_generator.class_indices.items()}
    pred_labels = [idx_to_label[idx] for idx in pred_indices]
else:
    test_files = os.listdir(TEST_DIR)
    X_test = load_images(TEST_DIR, test_files)
    X_test = X_test.reshape(len(test_files), -1)

    pred_probs = model.clf.predict_proba(X_test)
    pred_indices = np.argmax(pred_probs, axis=1)
    pred_labels = model.le.inverse_transform(pred_indices).astype(str).tolist()




## === cell 3
submission = pd.DataFrame(
    {
        "image_id": test_files,
        "label": pred_labels,
    }
)

submission.to_csv(SUBMISSION_PATH, index=False)
print(f"Submission written to {SUBMISSION_PATH}")

## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/3683680164.py in <cell line: 0>()
----> 1 submission = pd.DataFrame(
      2     {
      3         "image_id": test_files,
      4         "label": pred_labels,
      5     }

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in __init__(self, data, index, columns, dtype, copy)
    776         elif isinstance(data, dict):
    777             # GH#38939 de facto copy defaults to False only in non-dict cases
--> 778             mgr = dict_to_mgr(data, index, columns, dtype=dtype, copy=copy, typ=manager)
    779         elif isinstance(data, ma.MaskedArray):
    780             from numpy.ma import mrecords

/usr/local/lib/python3.11/dist-packages/pandas/core/internals/construction.py in dict_to_mgr(data, index, columns, dtype, typ, copy)
    501             arrays = [x.copy() if hasattr(x, "dtype") else x for x in arrays]
    502 
--> 503     return arrays_to_mgr(arrays, columns, index, dtype=dtype, typ=typ, consolidate=copy)
    504 
    505 

/usr/local/lib/python3.11/dist-packages/pandas/core/internals/construction.py in arrays_to_mgr(arrays, columns, index, dtype, verify_integrity, typ, consolidate)
    112         # figure out the index, if necessary
    113         if index is None:
--> 114             index = _extract_index(arrays)
    115         else:
    116             index = ensure_index(index)

/usr/local/lib/python3.11/dist-packages/pandas/core/internals/construction.py in _extract_index(data)
    675         lengths = list(set(raw_lengths))
    676         if len(lengths) > 1:
--> 677             raise ValueError("All arrays must be of the same length")
    678 
    679         if have_dicts:

ValueError: All arrays must be of the same length
