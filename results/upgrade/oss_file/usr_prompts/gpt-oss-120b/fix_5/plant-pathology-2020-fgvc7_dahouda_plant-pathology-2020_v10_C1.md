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
Detect apple diseases from images.

## Metric
Mean column-wise ROC AUC.

## Submission Format
For each image_id in the test set, you must predict a probability for each target variable. The file should contain a header and have the following format:

```
image_id,
test_0,0.25,0.25,0.25,0.25
test_1,0.25,0.25,0.25,0.25
test_2,0.25,0.25,0.25,0.25
etc.
```

## Dataset
Given a photo of an apple leaf, can you accurately assess its health? This competition will challenge you to distinguish between leaves which are healthy, those which are infected with apple rust, those that have apple scab, and those with more than one disease.

**train.csv**

- `image_id`: the foreign key
- combinations: one of the target labels
- healthy: one of the target labels
- rust: one of the target labels
- scab: one of the target labels

**images**

A folder containing the train and test images, in jpg format.

**test.csv**

- `image_id`: the foreign key

**sample_submission.csv**

- `image_id`: the foreign key
- combinations: one of the target labels
- healthy: one of the target labels
- rust: one of the target labels
- scab: one of the target labels

# 2. Python version

3.9

# 3. Installed packages

geopandas==0.14.4
keras==3.8.0
keras-core==0.1.7
keras-cv==0.9.0
keras-hub==0.18.1
keras-nlp==0.18.1
keras-tuner==1.4.7
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
numpy==1.26.4
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
tf_keras==2.18.0

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (94 lines)
            images.zip (397.8 MB)
            sample_submission.csv (184 lines)
            sample_submission.csv.zip (682 Bytes)
            test.csv (184 lines)
            test.csv.zip (542 Bytes)
            train.csv (1639 lines)
            train.csv.zip (4.6 kB)
            images/
                Train_370.jpg (133.2 kB)
                Test_59.jpg (220.5 kB)
                ... and 1819 other files
            plant-pathology-2020-fgvc7/
                description.md (94 lines)
                images.zip (397.8 MB)
                ... and 6 other files
                images/
                    Train_370.jpg (133.2 kB)
                    Test_59.jpg (220.5 kB)
                    ... and 1819 other files
                plant-pathology-2020-fgvc7/
        input/
            description.md (94 lines)
            images.zip (397.8 MB)
            sample_submission.csv (184 lines)
            sample_submission.csv.zip (682 Bytes)
            test.csv (184 lines)
            test.csv.zip (542 Bytes)
            train.csv (1639 lines)
            train.csv.zip (4.6 kB)
            images/
                Train_370.jpg (133.2 kB)
                Test_59.jpg (220.5 kB)
                ... and 1819 other files
            plant-pathology-2020-fgvc7/
                description.md (94 lines)
                images.zip (397.8 MB)
                ... and 6 other files
                images/
                    Train_370.jpg (133.2 kB)
                    Test_59.jpg (220.5 kB)
                    ... and 1819 other files
                plant-pathology-2020-fgvc7/
        working/
            plant-pathology-2020-fgvc7/
                description.md (94 lines)
                images.zip (397.8 MB)
                ... and 6 other files
                images/
                    Train_370.jpg (133.2 kB)
                    Test_59.jpg (220.5 kB)
                    ... and 1819 other files
                plant-pathology-2020-fgvc7/
```

-> data/plant-pathology-2020-fgvc7/sample_submission.csv has 183 rows and 5 columns.
The columns are: image_id, healthy, multiple_diseases, rust, scab

-> data/plant-pathology-2020-fgvc7/test.csv has 183 rows and 1 columns.
The columns are: image_id

-> data/plant-pathology-2020-fgvc7/train.csv has 1638 rows and 5 columns.
The columns are: image_id, healthy, multiple_diseases, rust, scab

-> data/sample_submission.csv has 183 rows and 5 columns.
The columns are: image_id, healthy, multiple_diseases, rust, scab

-> data/test.csv has 183 rows and 1 columns.
The columns are: image_id

-> data/train.csv has 1638 rows and 5 columns.
The columns are: image_id, healthy, multiple_diseases, rust, scab

-> (stopped after 10 files for performance)

# 5. Target score

0.96962

# 6. Current score

0.50262

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved 0.50262) has done: 'The changes freeze the EfficientNetB7 backbone (so only the lightweight classification head is trained) and add an EarlyStopping callback to cut off training once validation loss stops improving, both dramatically cutting compute time while keeping the same architecture and loss. The batch size is increased to process more images per step, further reducing the number of steps per epoch. These adjustments preserve the original model definition, data pipeline, and evaluation logic, ensuring identical predictions apart from negligible floating‑point variations.'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import tensorflow as tf
import tensorflow.keras.layers as L
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications import EfficientNetB7
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import (
    ReduceLROnPlateau,
    EarlyStopping,
    ModelCheckpoint,
    LearningRateScheduler,
)

from tensorflow.keras import mixed_precision

mixed_precision.set_global_policy("mixed_float16")



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
strategy = tf.distribute.get_strategy()
print("REPLICAS: ", strategy.num_replicas_in_sync)

EPOCHS = 50
BATCH_SIZE = 16 * strategy.num_replicas_in_sync
AUTO = tf.data.experimental.AUTOTUNE



## === cell 2
BASE_IMG_PATH = "/kaggle/input/plant-pathology-2020-fgvc7/images"


def format_path(st):
    return f"{BASE_IMG_PATH}/{st}.jpg"




## === cell 3
train = pd.read_csv("/kaggle/input/plant-pathology-2020-fgvc7/train.csv")
test = pd.read_csv("/kaggle/input/plant-pathology-2020-fgvc7/test.csv")
sub = pd.read_csv("/kaggle/input/plant-pathology-2020-fgvc7/sample_submission.csv")

all_paths = train.image_id.apply(format_path).values
test_paths = test.image_id.apply(format_path).values

label_cols = ["healthy", "multiple_diseases", "rust", "scab"]
all_labels = train[label_cols].values

train_paths, valid_paths, train_labels, valid_labels = train_test_split(
    all_paths, all_labels, test_size=0.06, random_state=2020
)



## === cell 4
f, ax = plt.subplots(3, 6, figsize=(18, 7))
ax = ax.flatten()
for i in range(18):
    img = plt.imread(f"/kaggle/input/plant-pathology-2020-fgvc7/images/Train_{i}.jpg")
    ax[i].set_title(
        train[train["image_id"] == f"Train_{i}"]
        .melt()[train[train["image_id"] == f"Train_{i}"].melt().value == 1]["variable"]
        .values[0]
    )
    ax[i].imshow(img)
plt.show()



## === cell 5
img_size = 224  # smaller size to fit within memory limits


def decode_image(filename, label=None, image_size=(img_size, img_size)):
    bits = tf.io.read_file(filename)
    image = tf.image.decode_jpeg(bits, channels=3)
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.image.resize(image, image_size)
    if label is None:
        return image
    else:
        return image, label


def data_augment(image, label=None, seed=2020):
    image = tf.image.random_flip_left_right(image, seed=seed)
    image = tf.image.random_flip_up_down(image, seed=seed)
    if label is None:
        return image
    else:
        return image, label




## === cell 6
train_dataset = (
    tf.data.Dataset.from_tensor_slices((train_paths, train_labels))
    .map(decode_image, num_parallel_calls=AUTO)
    .cache()  # cache decoded images in memory
    .map(data_augment, num_parallel_calls=AUTO)
    .shuffle(512)
    .batch(BATCH_SIZE)
    .prefetch(AUTO)
)

valid_dataset = (
    tf.data.Dataset.from_tensor_slices((valid_paths, valid_labels))
    .map(decode_image, num_parallel_calls=AUTO)
    .batch(BATCH_SIZE)
    .cache()
    .prefetch(AUTO)
)

test_dataset = (
    tf.data.Dataset.from_tensor_slices(test_paths)
    .map(decode_image, num_parallel_calls=AUTO)
    .batch(BATCH_SIZE)
)



## === cell 7
LR_START = 1e-5
LR_MAX = 1e-4 * strategy.num_replicas_in_sync
LR_MIN = 1e-5
LR_RAMPUP_EPOCHS = 5
LR_SUSTAIN_EPOCHS = 0
LR_EXP_DECAY = 0.8


def lrfn(epoch):
    if epoch < LR_RAMPUP_EPOCHS:
        lr = (LR_MAX - LR_START) / LR_RAMPUP_EPOCHS * epoch + LR_START
    elif epoch < LR_RAMPUP_EPOCHS + LR_SUSTAIN_EPOCHS:
        lr = LR_MAX
    else:
        lr = (LR_MAX - LR_MIN) * LR_EXP_DECAY ** (
            epoch - LR_RAMPUP_EPOCHS - LR_SUSTAIN_EPOCHS
        ) + LR_MIN
    return lr


lr_callback = LearningRateScheduler(lrfn, verbose=True)

rng = list(range(EPOCHS))
y = [lrfn(x) for x in rng]
plt.plot(rng, y)
plt.title("Learning Rate Schedule")
plt.xlabel("Epoch")
plt.ylabel("LR")
plt.show()




## === cell 8
def get_model():
    base_model = EfficientNetB7(
        weights="imagenet",
        include_top=False,
        pooling="avg",
        input_shape=(img_size, img_size, 3),
    )
    base_model.trainable = False
    x = base_model.output
    predictions = Dense(len(label_cols), activation="sigmoid")(x)
    return Model(inputs=base_model.input, outputs=predictions)


with strategy.scope():
    model = get_model()
    model.compile(
        optimizer="nadam", loss="binary_crossentropy", metrics=["binary_accuracy"]
    )
model.summary()



## === cell 9
history = model.fit(
    train_dataset,
    steps_per_epoch=train_labels.shape[0] // BATCH_SIZE,
    validation_data=valid_dataset,
    epochs=EPOCHS,
    callbacks=[
        lr_callback,
        EarlyStopping(
            monitor="val_loss", patience=5, restore_best_weights=True, verbose=1
        ),
        ModelCheckpoint(
            filepath="pretrained_EfficientNetB7.h5",
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),
    ],
)



## === cell 10
plt.plot(history.history["binary_accuracy"], label="Train Acc")
plt.plot(history.history["val_binary_accuracy"], label="Val Acc")
plt.xlabel("epoch")
plt.ylabel("accuracy")
plt.legend()
plt.show()



## === cell 11
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Val Loss")
plt.xlabel("epoch")
plt.ylabel("loss")
plt.legend()
plt.show()



## === cell 12
model = tf.keras.models.load_model("pretrained_EfficientNetB7.h5")



## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_56/2947927801.py in <cell line: 0>()
----> 1 model = tf.keras.models.load_model("pretrained_EfficientNetB7.h5")
      2 

/usr/local/lib/python3.11/dist-packages/keras/src/saving/saving_api.py in load_model(filepath, custom_objects, compile, safe_mode)
    194         )
    195     if str(filepath).endswith((".h5", ".hdf5")):
--> 196         return legacy_h5_format.load_model_from_hdf5(
    197             filepath, custom_objects=custom_objects, compile=compile
    198         )

/usr/local/lib/python3.11/dist-packages/keras/src/legacy/saving/legacy_h5_format.py in load_model_from_hdf5(filepath, custom_objects, compile)
    131 
    132         with saving_options.keras_option_scope(use_legacy_config=True):
--> 133             model = saving_utils.model_from_config(
    134                 model_config, custom_objects=custom_objects
    135             )

/usr/local/lib/python3.11/dist-packages/keras/src/legacy/saving/saving_utils.py in model_from_config(config, custom_objects)
     83     config = _find_replace_nested_dict(config, "keras.", "keras.")
     84 
---> 85     return serialization.deserialize_keras_object(
     86         config,
     87         module_objects=MODULE_OBJECTS.ALL_OBJECTS,

/usr/local/lib/python3.11/dist-packages/keras/src/legacy/saving/serialization.py in deserialize_keras_object(identifier, module_objects, custom_objects, printable_module_name)
    493 
    494             if "custom_objects" in arg_spec.args:
--> 495                 deserialized_obj = cls.from_config(
    496                     cls_config,
    497                     custom_objects={

/usr/local/lib/python3.11/dist-packages/keras/src/models/model.py in from_config(cls, config, custom_objects)
    580             from keras.src.models.functional import functional_from_config
    581 
--> 582             return functional_from_config(
    583                 cls, config, custom_objects=custom_objects
    584             )

/usr/local/lib/python3.11/dist-packages/keras/src/models/functional.py in functional_from_config(cls, config, custom_objects)
    549     # First, we create all layers and enqueue nodes to be processed
    550     for layer_data in functional_config["layers"]:
--> 551         process_layer(layer_data)
    552 
    553     # Then we process nodes in order of layer depth.

/usr/local/lib/python3.11/dist-packages/keras/src/models/functional.py in process_layer(layer_data)
    517             # Legacy format deserialization (no "module" key)
    518             # used for H5 and SavedModel formats
--> 519             layer = saving_utils.model_from_config(
    520                 layer_data, custom_objects=custom_objects
    521             )

/usr/local/lib/python3.11/dist-packages/keras/src/legacy/saving/saving_utils.py in model_from_config(config, custom_objects)
     83     config = _find_replace_nested_dict(config, "keras.", "keras.")
     84 
---> 85     return serialization.deserialize_keras_object(
     86         config,
     87         module_objects=MODULE_OBJECTS.ALL_OBJECTS,

/usr/local/lib/python3.11/dist-packages/keras/src/legacy/saving/serialization.py in deserialize_keras_object(identifier, module_objects, custom_objects, printable_module_name)
    471         # In this case we are dealing with a Keras config dictionary.
    472         config = identifier
--> 473         (cls, cls_config) = class_and_config_for_serialized_keras_object(
    474             config, module_objects, custom_objects, printable_module_name
    475         )

/usr/local/lib/python3.11/dist-packages/keras/src/legacy/saving/serialization.py in class_and_config_for_serialized_keras_object(config, module_objects, custom_objects, printable_module_name)
    352     )
    353     if cls is None:
--> 354         raise ValueError(
    355             f"Unknown {printable_module_name}: '{class_name}'. "
    356             "Please ensure you are using a `keras.utils.custom_object_scope` "

ValueError: Unknown layer: 'Cast'. Please ensure you are using a `keras.utils.custom_object_scope` and that this object is included in the scope. See https://www.tensorflow.org/guide/keras/save_and_serialize#registering_the_custom_object for details.

## === cell 13
probs = model.predict(test_dataset, verbose=1)
sub.loc[:, label_cols] = probs
sub.to_csv("submission.csv", index=False)
print("Submission saved to submission.csv")
sub.head()
