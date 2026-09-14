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
Create a classifier to predict whether an image contains a cactus.

## Metric
Area under the ROC curve.

## Submission Format
For each ID in the test set, you must predict a probability for the `has_cactus` variable. The file should contain a header and have the following format:

```
id,has_cactus
000940378805c44108d287872b2f04ce.jpg,0.5
0017242f54ececa4512b4d7937d1e21e.jpg,0.5
001ee6d8564003107853118ab87df407.jpg,0.5
etc.
```

## Dataset
This dataset contains a large number of 32 x 32 thumbnail images containing aerial photos of a cactus. The file name of an image corresponds to its `id`.

- **train/** - the training set images
- **test/** - the test set images (you must predict the labels of these)
- **train.csv** - the training set labels, indicates whether the image has a cactus (`has_cactus = 1`)
- **sample_submission.csv** - a sample submission file in the correct format

# 2. Python version

3.11

# 3. Installed packages

geopandas==0.14.4
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
numpy==1.26.4
opencv-python==4.12.0.88
opencv-python-headless==4.12.0.88
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
protobuf==6.33.0
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
sklearn-pandas==2.2.0
tensorflow==2.18.0
tensorflow-cloud==0.1.5
tensorflow-datasets==4.9.9
tensorflow_decision_forests==1.11.0
tensorflow-hub==0.16.1
tensorflow-io==0.37.1
tensorflow-io-gcs-filesystem==0.37.1
tensorflow-metadata==1.17.2
tensorflow-probability==0.25.0
tensorflow-text==2.18.1

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (56 lines)
            sample_submission.csv (3326 lines)
            sample_submission.csv.zip (67.3 kB)
            test.zip (3.5 MB)
            train.csv (14176 lines)
            train.csv.zip (285.6 kB)
            train.zip (15.0 MB)
            aerial-cactus-identification/
                description.md (56 lines)
                sample_submission.csv (3326 lines)
                ... and 5 other files
                aerial-cactus-identification/
                test/
                    76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                    f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                    ... and 3323 other files
                    test/
                train/
                    775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                    65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                    ... and 14173 other files
                    train/
            test/
                76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                ... and 3323 other files
                test/
            train/
                775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                ... and 14173 other files
                train/
        input/
            description.md (56 lines)
            sample_submission.csv (3326 lines)
            sample_submission.csv.zip (67.3 kB)
            test.zip (3.5 MB)
            train.csv (14176 lines)
            train.csv.zip (285.6 kB)
            train.zip (15.0 MB)
            aerial-cactus-identification/
                description.md (56 lines)
                sample_submission.csv (3326 lines)
                ... and 5 other files
                aerial-cactus-identification/
                test/
                    76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                    f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                    ... and 3323 other files
                    test/
                train/
                    775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                    65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                    ... and 14173 other files
                    train/
            test/
                76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                ... and 3323 other files
                test/
                    76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                    f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                    ... and 3323 other files
                    test/
            train/
                775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                ... and 14173 other files
                train/
                    775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                    65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                    ... and 14173 other files
                    train/
        working/
            aerial-cactus-identification/
                description.md (56 lines)
                sample_submission.csv (3326 lines)
                ... and 5 other files
                aerial-cactus-identification/
                test/
                    76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                    f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                    ... and 3323 other files
                    test/
                train/
                    775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                    65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                    ... and 14173 other files
                    train/
```

-> data/aerial-cactus-identification/sample_submission.csv has 3325 rows and 2 columns.
The columns are: id, has_cactus

-> data/aerial-cactus-identification/train.csv has 14175 rows and 2 columns.
The columns are: id, has_cactus

-> data/sample_submission.csv has 3325 rows and 2 columns.
The columns are: id, has_cactus

-> data/train.csv has 14175 rows and 2 columns.
The columns are: id, has_cactus

-> input/aerial-cactus-identification/sample_submission.csv has 3325 rows and 2 columns.
The columns are: id, has_cactus

-> input/aerial-cactus-identification/train.csv has 14175 rows and 2 columns.
The columns are: id, has_cactus

-> (stopped after 10 files for performance)

# 5. Target score

0.4956

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import sys, subprocess, os, warnings


def _ensure_protobuf_compat():
    try:
        import google.protobuf  # noqa: F401
        import protobuf  # type: ignore  # noqa: F401
    except Exception:
        pass

    try:
        import google.protobuf

        ver = getattr(google.protobuf, "__version__", "")
    except Exception:
        ver = ""

    if ver and ver.split(".")[0].isdigit() and int(ver.split(".")[0]) >= 5:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-q", "protobuf==4.25.3"]
        )
        for m in list(sys.modules.keys()):
            if m.startswith("google.protobuf") or m == "protobuf":
                sys.modules.pop(m, None)


_ensure_protobuf_compat()

import numpy as np
import pandas as pd
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt

tf.keras.utils.set_random_seed(2020)

print("TensorFlow:", tf.__version__)



## === cell 1
BASE = "/kaggle/input/aerial-cactus-identification"
TRAIN_CSV = os.path.join(BASE, "train.csv")
SAMPLE_SUB = os.path.join(BASE, "sample_submission.csv")
TRAIN_DIR = os.path.join(BASE, "train")
TEST_DIR = os.path.join(BASE, "test")

assert os.path.exists(TRAIN_CSV), f"Missing: {TRAIN_CSV}"
assert os.path.isdir(TRAIN_DIR), f"Missing: {TRAIN_DIR}"
assert os.path.isdir(TEST_DIR), f"Missing: {TEST_DIR}"

df = pd.read_csv(TRAIN_CSV)
df.sample(5)



## === cell 2
df["has_cactus"] = df["has_cactus"].astype("str")



## === cell 3
df.has_cactus.value_counts()



## === cell 4
print("Train images:", len(os.listdir(TRAIN_DIR)))
print("Test images:", len(os.listdir(TEST_DIR)))



## === cell 5
example_id = df.iloc[0]["id"]
image = tf.keras.preprocessing.image.load_img(os.path.join(TRAIN_DIR, example_id))
image = tf.keras.preprocessing.image.img_to_array(image)
print(image.shape)
plt.imshow(image.astype("int"))
plt.axis("off")



## === cell 6
idg = tf.keras.preprocessing.image.ImageDataGenerator(
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    brightness_range=(0.8, 1.2),
    horizontal_flip=True,
    validation_split=0.1,
)



## === cell 7
batch_size = 32



## === cell 8
train_idg = idg.flow_from_dataframe(
    df,
    TRAIN_DIR,
    x_col="id",
    y_col="has_cactus",
    target_size=(32, 32),
    batch_size=batch_size,
    seed=2020,
    subset="training",
)



## === cell 9
val_idg = idg.flow_from_dataframe(
    df,
    TRAIN_DIR,
    x_col="id",
    y_col="has_cactus",
    target_size=(32, 32),
    batch_size=batch_size,
    seed=2020,
    subset="validation",
    shuffle=False,
)



## === cell 10
input_layer = tf.keras.layers.Input((32, 32, 3), name="Input_Layer")
preprocess = tf.keras.layers.Lambda(
    tf.keras.applications.vgg16.preprocess_input,
    output_shape=(32, 32, 3),
    name="VGG16_Preprocess",
)(input_layer)

vgg_model = tf.keras.applications.vgg16.VGG16(
    include_top=False, input_shape=(32, 32, 3)
)
vgg_model.trainable = False
vgg = vgg_model(preprocess)

flat = tf.keras.layers.Flatten(name="Flatten")(vgg)
hidden = tf.keras.layers.Dense(512, activation="relu", name="Hidden")(flat)
output = tf.keras.layers.Dense(2, activation="softmax", name="Output_Layer")(hidden)



## === cell 11
model = tf.keras.models.Model(inputs=input_layer, outputs=output)



## === cell 12
model.summary()



## === cell 13
try:
    tf.keras.utils.plot_model(model, show_shapes=True, show_layer_names=True)
except Exception as e:
    print("plot_model skipped:", repr(e))



## === cell 14
model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss=tf.keras.losses.categorical_crossentropy,
    metrics=["acc"],
)



## === cell 15
ckpt_path = "check.keras"
tf_callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        ckpt_path, save_best_only=True, monitor="val_loss", mode="min"
    )
]



## === cell 16
steps_per_epoch = int(np.ceil(train_idg.samples / batch_size))
val_steps = int(np.ceil(val_idg.samples / batch_size))

history = model.fit(
    train_idg,
    steps_per_epoch=steps_per_epoch,
    epochs=10,
    validation_data=val_idg,
    validation_steps=val_steps,
    callbacks=tf_callbacks,
    verbose=2,
)



## === cell 17
plt.figure(figsize=(12, 5))
plt.suptitle("")
plt.subplot(121)
plt.plot(history.history.get("acc", []), label="acc")
plt.plot(history.history.get("val_acc", []), label="val_acc")
plt.legend()
plt.subplot(122)
plt.plot(history.history.get("loss", []), label="loss")
plt.plot(history.history.get("val_loss", []), label="val_loss")
plt.legend()
plt.show()



## === cell 18
final_model = tf.keras.models.load_model(ckpt_path)



## --- ERROR in cell 18, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/usr/local/lib/python3.11/dist-packages/keras/src/saving/serialization_lib.py in deserialize_keras_object(config, custom_objects, safe_mode, **kwargs)
    717         try:
--> 718             instance = cls.from_config(inner_config)
    719         except TypeError as e:

/usr/local/lib/python3.11/dist-packages/keras/src/layers/core/lambda_layer.py in from_config(cls, config, custom_objects, safe_mode)
    198         else:
--> 199             config["function"] = serialization_lib.deserialize_keras_object(
    200                 fn_config, custom_objects=custom_objects

/usr/local/lib/python3.11/dist-packages/keras/src/saving/serialization_lib.py in deserialize_keras_object(config, custom_objects, safe_mode, **kwargs)
    677         fn_name = inner_config
--> 678         return _retrieve_class_or_fn(
    679             fn_name,

/usr/local/lib/python3.11/dist-packages/keras/src/saving/serialization_lib.py in _retrieve_class_or_fn(name, registered_name, module, obj_type, full_config, custom_objects)
    802 
--> 803     raise TypeError(
    804         f"Could not locate {obj_type} '{name}'. "

TypeError: Could not locate function 'preprocess_input'. Make sure custom classes are decorated with `@keras.saving.register_keras_serializable()`. Full object config: {'module': 'builtins', 'class_name': 'function', 'config': 'preprocess_input', 'registered_name': 'function'}

During handling of the above exception, another exception occurred:

TypeError                                 Traceback (most recent call last)
/usr/local/lib/python3.11/dist-packages/keras/src/saving/serialization_lib.py in deserialize_keras_object(config, custom_objects, safe_mode, **kwargs)
    717         try:
--> 718             instance = cls.from_config(inner_config)
    719         except TypeError as e:

/usr/local/lib/python3.11/dist-packages/keras/src/models/model.py in from_config(cls, config, custom_objects)
    581 
--> 582             return functional_from_config(
    583                 cls, config, custom_objects=custom_objects

/usr/local/lib/python3.11/dist-packages/keras/src/models/functional.py in functional_from_config(cls, config, custom_objects)
    550     for layer_data in functional_config["layers"]:
--> 551         process_layer(layer_data)
    552 

/usr/local/lib/python3.11/dist-packages/keras/src/models/functional.py in process_layer(layer_data)
    522         else:
--> 523             layer = serialization_lib.deserialize_keras_object(
    524                 layer_data, custom_objects=custom_objects

/usr/local/lib/python3.11/dist-packages/keras/src/saving/serialization_lib.py in deserialize_keras_object(config, custom_objects, safe_mode, **kwargs)
    719         except TypeError as e:
--> 720             raise TypeError(
    721                 f"{cls} could not be deserialized properly. Please"

TypeError: <class 'keras.src.layers.core.lambda_layer.Lambda'> could not be deserialized properly. Please ensure that components that are Python object instances (layers, models, etc.) returned by `get_config()` are explicitly deserialized in the model's `from_config()` method.

config={'module': 'keras.layers', 'class_name': 'Lambda', 'config': {'name': 'VGG16_Preprocess', 'trainable': True, 'dtype': {'module': 'keras', 'class_name': 'DTypePolicy', 'config': {'name': 'float32'}, 'registered_name': None}, 'function': {'module': 'builtins', 'class_name': 'function', 'config': 'preprocess_input', 'registered_name': 'function'}, 'output_shape': [32, 32, 3], 'arguments': {}}, 'registered_name': None, 'build_config': {'input_shape': [None, 32, 32, 3]}, 'name': 'VGG16_Preprocess', 'inbound_nodes': [{'args': [{'class_name': '__keras_tensor__', 'config': {'shape': [None, 32, 32, 3], 'dtype': 'float32', 'keras_history': ['Input_Layer', 0, 0]}}], 'kwargs': {'mask': None}}]}.

Exception encountered: Could not locate function 'preprocess_input'. Make sure custom classes are decorated with `@keras.saving.register_keras_serializable()`. Full object config: {'module': 'builtins', 'class_name': 'function', 'config': 'preprocess_input', 'registered_name': 'function'}

During handling of the above exception, another exception occurred:

TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3014697235.py in <cell line: 0>()
      1 # Fix: load best model from .keras checkpoint.
----> 2 final_model = tf.keras.models.load_model(ckpt_path)
      3 

/usr/local/lib/python3.11/dist-packages/keras/src/saving/saving_api.py in load_model(filepath, custom_objects, compile, safe_mode)
    187 
    188     if is_keras_zip or is_keras_dir or is_hf:
--> 189         return saving_lib.load_model(
    190             filepath,
    191             custom_objects=custom_objects,

/usr/local/lib/python3.11/dist-packages/keras/src/saving/saving_lib.py in load_model(filepath, custom_objects, compile, safe_mode)
    365             )
    366         with open(filepath, "rb") as f:
--> 367             return _load_model_from_fileobj(
    368                 f, custom_objects, compile, safe_mode
    369             )

/usr/local/lib/python3.11/dist-packages/keras/src/saving/saving_lib.py in _load_model_from_fileobj(fileobj, custom_objects, compile, safe_mode)
    442             config_json = f.read()
    443 
--> 444         model = _model_from_config(
    445             config_json, custom_objects, compile, safe_mode
    446         )

/usr/local/lib/python3.11/dist-packages/keras/src/saving/saving_lib.py in _model_from_config(config_json, custom_objects, compile, safe_mode)
    431     # Construct the model from the configuration file in the archive.
    432     with ObjectSharingScope():
--> 433         model = deserialize_keras_object(
    434             config_dict, custom_objects, safe_mode=safe_mode
    435         )

/usr/local/lib/python3.11/dist-packages/keras/src/saving/serialization_lib.py in deserialize_keras_object(config, custom_objects, safe_mode, **kwargs)
    718             instance = cls.from_config(inner_config)
    719         except TypeError as e:
--> 720             raise TypeError(
    721                 f"{cls} could not be deserialized properly. Please"
    722                 " ensure that components that are Python object"

TypeError: <class 'keras.src.models.functional.Functional'> could not be deserialized properly. Please ensure that components that are Python object instances (layers, models, etc.) returned by `get_config()` are explicitly deserialized in the model's `from_config()` method.

config={'module': 'keras.src.models.functional', 'class_name': 'Functional', 'config': {}, 'registered_name': 'Functional', 'build_config': {'input_shape': None}, 'compile_config': {'optimizer': {'module': 'keras.optimizers', 'class_name': 'Adam', 'config': {'name': 'adam', 'learning_rate': 0.0010000000474974513, 'weight_decay': None, 'clipnorm': None, 'global_clipnorm': None, 'clipvalue': None, 'use_ema': False, 'ema_momentum': 0.99, 'ema_overwrite_frequency': None, 'loss_scale_factor': None, 'gradient_accumulation_steps': None, 'beta_1': 0.9, 'beta_2': 0.999, 'epsilon': 1e-07, 'amsgrad': False}, 'registered_name': None}, 'loss': {'module': 'builtins', 'class_name': 'function', 'config': 'categorical_crossentropy', 'registered_name': 'function'}, 'loss_weights': None, 'metrics': ['acc'], 'weighted_metrics': None, 'run_eagerly': False, 'steps_per_execution': 1, 'jit_compile': True}}.

Exception encountered: <class 'keras.src.layers.core.lambda_layer.Lambda'> could not be deserialized properly. Please ensure that components that are Python object instances (layers, models, etc.) returned by `get_config()` are explicitly deserialized in the model's `from_config()` method.

config={'module': 'keras.layers', 'class_name': 'Lambda', 'config': {'name': 'VGG16_Preprocess', 'trainable': True, 'dtype': {'module': 'keras', 'class_name': 'DTypePolicy', 'config': {'name': 'float32'}, 'registered_name': None}, 'function': {'module': 'builtins', 'class_name': 'function', 'config': 'preprocess_input', 'registered_name': 'function'}, 'output_shape': [32, 32, 3], 'arguments': {}}, 'registered_name': None, 'build_config': {'input_shape': [None, 32, 32, 3]}, 'name': 'VGG16_Preprocess', 'inbound_nodes': [{'args': [{'class_name': '__keras_tensor__', 'config': {'shape': [None, 32, 32, 3], 'dtype': 'float32', 'keras_history': ['Input_Layer', 0, 0]}}], 'kwargs': {'mask': None}}]}.

Exception encountered: Could not locate function 'preprocess_input'. Make sure custom classes are decorated with `@keras.saving.register_keras_serializable()`. Full object config: {'module': 'builtins', 'class_name': 'function', 'config': 'preprocess_input', 'registered_name': 'function'}

## === cell 19
val_pred = final_model.predict(val_idg, verbose=0)
val_pred_class = np.argmax(val_pred, axis=1)
y_true = val_idg.labels

from sklearn.metrics import classification_report, confusion_matrix

print(classification_report(y_true, val_pred_class))
print(confusion_matrix(y_true, val_pred_class))



## --- ERROR in cell 19, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3779627634.py in <cell line: 0>()
      1 # Validation predictions (optional diagnostics; not required for submission).
----> 2 val_pred = final_model.predict(val_idg, verbose=0)
      3 val_pred_class = np.argmax(val_pred, axis=1)
      4 y_true = val_idg.labels
      5 

NameError: name 'final_model' is not defined

## === cell 20
test = pd.DataFrame(sorted(os.listdir(TEST_DIR)), columns=["id"])
test.sample(5)



## === cell 21
test_idg = idg.flow_from_dataframe(
    test,
    TEST_DIR,
    x_col="id",
    y_col=None,
    batch_size=1,
    class_mode=None,
    target_size=(32, 32),
    shuffle=False,
)



## === cell 22
result = final_model.predict(test_idg, verbose=0)



## --- ERROR in cell 22, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3564323730.py in <cell line: 0>()
----> 1 result = final_model.predict(test_idg, verbose=0)
      2 

NameError: name 'final_model' is not defined

## === cell 23
test_prob = result[:, 1].astype(np.float32)
test["has_cactus"] = test_prob



## --- ERROR in cell 23, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1820465144.py in <cell line: 0>()
      1 # Probability for class 1 (has_cactus=1) from softmax output.
----> 2 test_prob = result[:, 1].astype(np.float32)
      3 test["has_cactus"] = test_prob
      4 

NameError: name 'result' is not defined

## === cell 24
sub = test[["id", "has_cactus"]].copy()
sub.to_csv("submission.csv", index=False)
print(sub.head())
print("Wrote submission.csv with shape:", sub.shape)



## --- ERROR in cell 24, traceback:
---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_11/2163521424.py in <cell line: 0>()
      1 # Ensure correct submission format and write CSV.
----> 2 sub = test[["id", "has_cactus"]].copy()
      3 sub.to_csv("submission.csv", index=False)
      4 print(sub.head())
      5 print("Wrote submission.csv with shape:", sub.shape)

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in __getitem__(self, key)
   4106             if is_iterator(key):
   4107                 key = list(key)
-> 4108             indexer = self.columns._get_indexer_strict(key, "columns")[1]
   4109 
   4110         # take() does not accept boolean indexers

/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in _get_indexer_strict(self, key, axis_name)
   6198             keyarr, indexer, new_indexer = self._reindex_non_unique(keyarr)
   6199 
-> 6200         self._raise_if_missing(keyarr, indexer, axis_name)
   6201 
   6202         keyarr = self.take(indexer)

/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in _raise_if_missing(self, key, indexer, axis_name)
   6250 
   6251             not_found = list(ensure_index(key)[missing_mask.nonzero()[0]].unique())
-> 6252             raise KeyError(f"{not_found} not in index")
   6253 
   6254     @overload

KeyError: "['has_cactus'] not in index"

## === cell 25
df_check = pd.read_csv("submission.csv")
print(df_check.head())
print(df_check.columns.tolist(), df_check.shape)

## --- ERROR in cell 25, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/4011213201.py in <cell line: 0>()
      1 # Quick readback check.
----> 2 df_check = pd.read_csv("submission.csv")
      3 print(df_check.head())
      4 print(df_check.columns.tolist(), df_check.shape)

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in read_csv(filepath_or_buffer, sep, delimiter, header, names, index_col, usecols, dtype, engine, converters, true_values, false_values, skipinitialspace, skiprows, skipfooter, nrows, na_values, keep_default_na, na_filter, verbose, skip_blank_lines, parse_dates, infer_datetime_format, keep_date_col, date_parser, date_format, dayfirst, cache_dates, iterator, chunksize, compression, thousands, decimal, lineterminator, quotechar, quoting, doublequote, escapechar, comment, encoding, encoding_errors, dialect, on_bad_lines, delim_whitespace, low_memory, memory_map, float_precision, storage_options, dtype_backend)
   1024     kwds.update(kwds_defaults)
   1025 
-> 1026     return _read(filepath_or_buffer, kwds)
   1027 
   1028 

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in _read(filepath_or_buffer, kwds)
    618 
    619     # Create the parser.
--> 620     parser = TextFileReader(filepath_or_buffer, **kwds)
    621 
    622     if chunksize or iterator:

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in __init__(self, f, engine, **kwds)
   1618 
   1619         self.handles: IOHandles | None = None
-> 1620         self._engine = self._make_engine(f, self.engine)
   1621 
   1622     def close(self) -> None:

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in _make_engine(self, f, engine)
   1878                 if "b" not in mode:
   1879                     mode += "b"
-> 1880             self.handles = get_handle(
   1881                 f,
   1882                 mode,

/usr/local/lib/python3.11/dist-packages/pandas/io/common.py in get_handle(path_or_buf, mode, encoding, compression, memory_map, is_text, errors, storage_options)
    871         if ioargs.encoding and "b" not in ioargs.mode:
    872             # Encoding
--> 873             handle = open(
    874                 handle,
    875                 ioargs.mode,

FileNotFoundError: [Errno 2] No such file or directory: 'submission.csv'
