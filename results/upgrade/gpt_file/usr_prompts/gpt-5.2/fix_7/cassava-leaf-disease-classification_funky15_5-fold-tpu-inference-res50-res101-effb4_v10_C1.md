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

0.2391961317618615

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os, glob, math, re
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from PIL import Image

print("Tensorflow version " + tf.__version__)
print("Keras version " + keras.__version__)

SEED = 42
tf.random.set_seed(SEED)
np.random.seed(SEED)

try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass
try:
    tf.config.optimizer.set_jit(True)
except Exception:
    pass

try:
    tf.config.threading.set_intra_op_parallelism_threads(0)
    tf.config.threading.set_inter_op_parallelism_threads(0)
except Exception:
    pass




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
IMAGE_SIZE = 512
BATCH_SIZE = 64
NUM_CLASSES = 5

DATA_ROOT = "/kaggle/input/cassava-leaf-disease-classification"
TRAIN_CSV = os.path.join(DATA_ROOT, "train.csv")
SAMPLE_SUB = os.path.join(DATA_ROOT, "sample_submission.csv")
TRAIN_IMG_DIR = os.path.join(DATA_ROOT, "train_images")
TEST_IMG_DIR = os.path.join(DATA_ROOT, "test_images")

print("Train CSV exists:", os.path.exists(TRAIN_CSV))
print("Sample submission exists:", os.path.exists(SAMPLE_SUB))
print("Train images dir exists:", os.path.isdir(TRAIN_IMG_DIR))
print("Test images dir exists:", os.path.isdir(TEST_IMG_DIR))




## === cell 2
FOLD_MODEL_PATHS = [
    "../input/tpus-with-5-fold/resnet50_0.h5",
    "../input/tpus-with-5-fold/resnet50_1.h5",
    "../input/tpus-with-5-fold/resnet50_2.h5",
    "../input/tpus-with-5-fold/resnet50_3.h5",
    "../input/tpus-with-5-fold/resnet50_4.h5",
]


def try_load_models(paths):
    models = []
    for p in paths:
        if os.path.exists(p):
            try:
                models.append(keras.models.load_model(p, compile=False))
                print(f"Loaded model: {p}")
            except Exception as e:
                print(f"Failed to load {p}: {e}")
        else:
            print(f"Model not found (will skip): {p}")
    return models


loaded_models = try_load_models(FOLD_MODEL_PATHS)




## === cell 3
def build_resnet50_classifier(image_size=512, num_classes=5):
    inputs = keras.Input(shape=(image_size, image_size, 3))
    x = keras.applications.resnet50.preprocess_input(inputs)
    base = keras.applications.ResNet50(
        include_top=False, weights="imagenet", input_tensor=x, pooling="avg"
    )
    base.trainable = False  # fast + stable within time limit
    x = base.output
    outputs = keras.layers.Dense(num_classes, activation="softmax")(x)
    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def make_train_val_datasets(
    train_df, img_dir, image_size=512, batch_size=64, seed=42, val_frac=0.1
):
    from sklearn.model_selection import train_test_split

    trn_df, val_df = train_test_split(
        train_df,
        test_size=val_frac,
        random_state=seed,
        stratify=train_df["label"],
    )

    def load_and_resize(image_id, label):
        img_path = tf.strings.join([img_dir, "/", image_id])
        img_bytes = tf.io.read_file(img_path)
        img = tf.io.decode_and_crop_jpeg(
            img_bytes, crop_window=[0, 0, 10000000, 10000000], channels=3
        )
        img = tf.image.resize(img, [image_size, image_size], method="bilinear")
        img = tf.cast(img, tf.float32)  # preprocessing is in model via preprocess_input
        img.set_shape([image_size, image_size, 3])
        return img, label

    def df_to_ds(df, training):
        ds = tf.data.Dataset.from_tensor_slices(
            (df["image_id"].values, df["label"].values)
        )
        if training:
            ds = ds.shuffle(
                min(len(df), 8192), seed=seed, reshuffle_each_iteration=True
            )
        options = tf.data.Options()
        options.experimental_deterministic = True
        options.experimental_optimization.apply_default_optimizations = True
        ds = ds.with_options(options)
        ds = ds.map(load_and_resize, num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        return ds

    return df_to_ds(trn_df, True), df_to_ds(val_df, False)




## === cell 4
if len(loaded_models) == 0:
    train_df = pd.read_csv(TRAIN_CSV)
    assert set(train_df.columns) >= {"image_id", "label"}
    train_df["label"] = train_df["label"].astype(int)

    train_ds, val_ds = make_train_val_datasets(
        train_df,
        TRAIN_IMG_DIR,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        seed=SEED,
        val_frac=0.1,
    )

    fallback_model = build_resnet50_classifier(
        image_size=IMAGE_SIZE, num_classes=NUM_CLASSES
    )

    EPOCHS = 3
    history = fallback_model.fit(
        train_ds, validation_data=val_ds, epochs=EPOCHS, verbose=1
    )
    model_list = [fallback_model]
else:
    model_list = loaded_models

print("Number of models used for prediction:", len(model_list))




## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
InvalidArgumentError                      Traceback (most recent call last)
/tmp/ipykernel_11/2922266898.py in <cell line: 0>()
     18 
     19     EPOCHS = 3
---> 20     history = fallback_model.fit(
     21         train_ds, validation_data=val_ds, epochs=EPOCHS, verbose=1
     22     )

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

InvalidArgumentError: Graph execution error:

Detected at node DecodeAndCropJpeg defined at (most recent call last):
<stack traces unavailable>
jpeg::Uncompress failed. Invalid JPEG data or crop window.
	 [[{{node DecodeAndCropJpeg}}]]
	 [[IteratorGetNext]] [Op:__inference_multi_step_on_iterator_11543]

## === cell 5
test_dir = TEST_IMG_DIR

sample_df = pd.read_csv(SAMPLE_SUB)
test_image_ids = sample_df["image_id"].astype(str).tolist()
test_image_paths = [os.path.join(test_dir, iid) for iid in test_image_ids]


def get_preds_model_list(image_paths, image_ids, model_obj_list, batch_size=BATCH_SIZE):
    if len(image_paths) == 0:
        return pd.DataFrame({"image_id": [], "label": []})

    file_ds = tf.data.Dataset.from_tensor_slices(image_paths)

    @tf.function
    def load_resize(path):
        img_bytes = tf.io.read_file(path)
        img = tf.io.decode_and_crop_jpeg(
            img_bytes, crop_window=[0, 0, 10000000, 10000000], channels=3
        )
        img = tf.image.resize(img, [IMAGE_SIZE, IMAGE_SIZE], method="bilinear")
        img = tf.cast(img, tf.float32)  # model does preprocess_input internally
        img.set_shape([IMAGE_SIZE, IMAGE_SIZE, 3])
        return img

    options = tf.data.Options()
    options.experimental_deterministic = True
    options.experimental_optimization.apply_default_optimizations = True

    cache_path = os.path.join("/kaggle/working", f"test_cache_{IMAGE_SIZE}.tfdata")

    ds = (
        file_ds.with_options(options)
        .map(load_resize, num_parallel_calls=tf.data.AUTOTUNE)
        .cache(cache_path)
        .batch(batch_size, drop_remainder=False)
        .prefetch(tf.data.AUTOTUNE)
    )

    n = len(image_paths)
    n_models = len(model_obj_list)
    steps = (
        n + batch_size - 1
    ) // batch_size  # Speed: avoid extra cardinality scans in predict()

    avg_probs = np.zeros((n, NUM_CLASSES), dtype=np.float32)

    for mod in model_obj_list:
        probs = mod.predict(ds, verbose=0, steps=steps)
        avg_probs += probs.astype(np.float32, copy=False)

    avg_probs /= float(n_models)
    preds = avg_probs.argmax(axis=1).astype(np.int64)

    return pd.DataFrame({"image_id": image_ids, "label": preds})


def get_preds(image_paths, image_ids, model_obj, batch_size=BATCH_SIZE):
    return get_preds_model_list(
        image_paths, image_ids, [model_obj], batch_size=batch_size
    )


predict_df = get_preds_model_list(test_image_paths, test_image_ids, model_list)




## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4029963954.py in <cell line: 0>()
     66 
     67 
---> 68 predict_df = get_preds_model_list(test_image_paths, test_image_ids, model_list)
     69 
     70 

NameError: name 'model_list' is not defined

## === cell 6
predict_df = sample_df[["image_id"]].merge(predict_df, on="image_id", how="left")
predict_df["label"] = predict_df["label"].fillna(0).astype(int)

assert list(predict_df.columns) == ["image_id", "label"]
assert len(predict_df) == len(sample_df)

predict_df.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", predict_df.shape)
print(predict_df.head())




## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/725952529.py in <cell line: 0>()
----> 1 predict_df = sample_df[["image_id"]].merge(predict_df, on="image_id", how="left")
      2 predict_df["label"] = predict_df["label"].fillna(0).astype(int)
      3 
      4 assert list(predict_df.columns) == ["image_id", "label"]
      5 assert len(predict_df) == len(sample_df)

NameError: name 'predict_df' is not defined

## === cell 7
try:
    display(predict_df.head(10))
except NameError:
    print(predict_df.head(10))

## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/548262635.py in <cell line: 0>()
      1 try:
----> 2     display(predict_df.head(10))
      3 except NameError:

NameError: name 'predict_df' is not defined

During handling of the above exception, another exception occurred:

NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/548262635.py in <cell line: 0>()
      2     display(predict_df.head(10))
      3 except NameError:
----> 4     print(predict_df.head(10))

NameError: name 'predict_df' is not defined
