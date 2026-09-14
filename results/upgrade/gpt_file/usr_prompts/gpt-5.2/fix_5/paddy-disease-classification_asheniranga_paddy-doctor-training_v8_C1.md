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

albumentations==2.0.8
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
seaborn==0.12.2
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

0.96543

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved 0.06764) has done: 'I fix the runtime blockers that prevent the notebook from importing/defining key objects and from writing a valid submission CSV. Specifically: remove the incompatible `tensorflow_addons` dependency (it triggers the protobuf `MessageFactory` error) and replace its cutout augmentation with an equivalent pure-TensorFlow implementation; update Keras checkpoint/saving filenames to the required `.keras`/`.weights.h5` extensions for TF/Keras 2.18; and make `Compose`/`ImageDataGenerator`/`plt` reliably available by using their current import locations. I also skip `plot_model()` if Graphviz/dot isn’t available (non-critical) and ensure test `image_id` formatting matches the sample submission (no directory prefixes). The core model (Xception + GAP + Dense softmax) and training loop remain the same, and the script finish by writing `model_submission_v5.csv`.'

# 9. Code solution

## === cell 0
import os
import random
import numpy as np
import pandas as pd

import tensorflow as tf

import cv2
import albumentations as A
from albumentations import Compose

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator

SEED = 42
tf.keras.utils.set_random_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

tf.config.optimizer.set_jit(True)

print("TF:", tf.__version__)




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
train_meta_data = "../input/paddy-disease-classification/train.csv"
train_data_dir = "../input/paddy-disease-classification/train_images"
epochs = 100
lr = 1e-4
valid_split = 0.2
input_size = 224
batch_size = 32
classes = 10
initializer = tf.keras.initializers.HeUniform()
optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
loss = tf.keras.losses.categorical_crossentropy




## === cell 2
early_stop = tf.keras.callbacks.EarlyStopping(
    patience=15, monitor="val_loss", restore_best_weights=True, verbose=1
)

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    patience=5, monitor="val_loss", factor=0.75, verbose=1
)

checkpoint = tf.keras.callbacks.ModelCheckpoint(
    filepath="best_chp.keras", monitor="val_loss", verbose=1, save_best_only=True
)




## === cell 3
get_transform = Compose(
    [
        A.CoarseDropout(
            max_holes=16,
            min_holes=8,
            max_height=16,
            max_width=16,
            min_height=8,
            min_width=8,
            p=0.2,
        )
    ]
)




## === cell 4
meta = pd.read_csv(train_meta_data)
class_names = sorted(meta["label"].unique().tolist())
print("Detected classes:", class_names, "count:", len(class_names))

generator = ImageDataGenerator(
    rescale=1 / 255,
    rotation_range=10,
    shear_range=0.25,
    zoom_range=0.1,
    horizontal_flip=True,
    vertical_flip=True,
    validation_split=valid_split,
)

train_datagen = generator.flow_from_directory(
    train_data_dir + "/",
    target_size=(input_size, input_size),
    batch_size=batch_size,
    subset="training",
    seed=SEED,
    classes=class_names,
    class_mode="categorical",
    shuffle=True,
)

valid_datagen = generator.flow_from_directory(
    train_data_dir + "/",
    target_size=(input_size, input_size),
    batch_size=batch_size,
    subset="validation",
    seed=SEED,
    classes=class_names,
    class_mode="categorical",
    shuffle=False,
)

train_ds = train_datagen
valid_ds = valid_datagen

print("train_datagen.num_classes:", train_datagen.num_classes)
print("valid_datagen.num_classes:", valid_datagen.num_classes)




## === cell 5
xb, yb = next(iter(train_datagen))
xv, yv = next(iter(valid_datagen))
print("Sanity check batch shapes:", xb.shape, yb.shape, xv.shape, yv.shape)




## === cell 6
pass




## === cell 7
pass




## === cell 8
meta.head()




## === cell 9
pass




## === cell 10
pass




## === cell 11
back_bone = tf.keras.applications.Xception(
    weights="imagenet", input_shape=(input_size, input_size, 3), include_top=False
)
back_bone.summary()




## === cell 12
pass




## === cell 13
back_bone.trainable = False

input_layer = Input(shape=(input_size, input_size, 3))
x = back_bone(input_layer, training=False)
x = GlobalAveragePooling2D()(x)
output_layer = Dense(train_datagen.num_classes, activation="softmax")(x)

model = Model(input_layer, output_layer)

model.compile(optimizer=optimizer, loss=loss, metrics=["accuracy"], jit_compile=True)




## === cell 14
model.summary()




## === cell 15
history = model.fit(
    train_ds,
    validation_data=valid_ds,
    epochs=epochs,
    callbacks=[early_stop, reduce_lr, checkpoint],
    verbose=1,
    workers=min(8, (os.cpu_count() or 2)),
    use_multiprocessing=True,
    max_queue_size=32,
)




## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3085213168.py in <cell line: 0>()
      1 # Speed-only change: increase data loader parallelism for DirectoryIterator inside fit.
      2 # This does not alter training semantics or batches.
----> 3 history = model.fit(
      4     train_ds,
      5     validation_data=valid_ds,

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

## === cell 16
model.evaluate(
    valid_ds, verbose=1, workers=min(8, (os.cpu_count() or 2)), use_multiprocessing=True
)




## --- ERROR in cell 16, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/2997630180.py in <cell line: 0>()
----> 1 model.evaluate(
      2     valid_ds, verbose=1, workers=min(8, (os.cpu_count() or 2)), use_multiprocessing=True
      3 )
      4 
      5 

/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py in error_handler(*args, **kwargs)
    120             # To get the full stack trace, call:
    121             # `keras.config.disable_traceback_filtering()`
--> 122             raise e.with_traceback(filtered_tb) from None
    123         finally:
    124             del filtered_tb

/usr/local/lib/python3.11/dist-packages/keras/src/backend/tensorflow/trainer.py in evaluate(self, x, y, batch_size, verbose, sample_weight, steps, callbacks, return_dict, **kwargs)
    442         use_cached_eval_dataset = kwargs.pop("_use_cached_eval_dataset", False)
    443         if kwargs:
--> 444             raise ValueError(f"Arguments not recognized: {kwargs}")
    445 
    446         if use_cached_eval_dataset:

ValueError: Arguments not recognized: {'workers': 8, 'use_multiprocessing': True}

## === cell 17
pass




## === cell 18
pass




## === cell 19
temp_hist = pd.DataFrame(history.history)
temp_hist.to_csv("model_xception_history.csv", index=False)
temp_hist.head()




## --- ERROR in cell 19, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2543656814.py in <cell line: 0>()
----> 1 temp_hist = pd.DataFrame(history.history)
      2 temp_hist.to_csv("model_xception_history.csv", index=False)
      3 temp_hist.head()
      4 
      5 

NameError: name 'history' is not defined

## === cell 20
model.save("model_xception.keras")




## === cell 21
model.save_weights("model_xception_weights.weights.h5")




## === cell 22
test_loc = "../input/paddy-disease-classification/test_images"

test_data = ImageDataGenerator(rescale=1.0 / 255).flow_from_directory(
    directory=test_loc,
    target_size=(input_size, input_size),
    batch_size=batch_size,
    classes=["."],
    shuffle=False,
)

test_ds = test_data




## === cell 23
train_datagen.class_indices




## === cell 24
pred_probs = model.predict(
    test_ds,
    verbose=1,
    workers=min(8, (os.cpu_count() or 2)),
    use_multiprocessing=True,
    max_queue_size=32,
)
predict_max = np.argmax(pred_probs, axis=1)




## --- ERROR in cell 24, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/898859233.py in <cell line: 0>()
----> 1 pred_probs = model.predict(
      2     test_ds,
      3     verbose=1,
      4     workers=min(8, (os.cpu_count() or 2)),
      5     use_multiprocessing=True,

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

TypeError: TensorFlowTrainer.predict() got an unexpected keyword argument 'workers'

## === cell 25
inverse_map = {v: k for k, v in train_datagen.class_indices.items()}
predictions = [inverse_map[k] for k in predict_max]




## --- ERROR in cell 25, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2727795600.py in <cell line: 0>()
      1 inverse_map = {v: k for k, v in train_datagen.class_indices.items()}
----> 2 predictions = [inverse_map[k] for k in predict_max]
      3 
      4 

NameError: name 'predict_max' is not defined

## === cell 26
files = test_data.filenames
sub = pd.DataFrame({"image_id": files, "label": predictions})

sub["image_id"] = sub["image_id"].str.replace("./", "", regex=False)
sub["image_id"] = sub["image_id"].str.replace(".\\", "", regex=False)
sub["image_id"] = sub["image_id"].str.replace("/", "", regex=False)

sample = pd.read_csv("../input/paddy-disease-classification/sample_submission.csv")
sub = sample[["image_id"]].merge(sub, on="image_id", how="left")

if sub["label"].isna().any():
    most_common_label = meta["label"].mode().iloc[0]
    sub["label"] = sub["label"].fillna(most_common_label)

sub.to_csv("model_submission_v5.csv", index=False)
sub.head()




## --- ERROR in cell 26, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1089095821.py in <cell line: 0>()
      1 files = test_data.filenames
----> 2 sub = pd.DataFrame({"image_id": files, "label": predictions})
      3 
      4 sub["image_id"] = sub["image_id"].str.replace("./", "", regex=False)
      5 sub["image_id"] = sub["image_id"].str.replace(".\\", "", regex=False)

NameError: name 'predictions' is not defined

## === cell 27
sub.label.value_counts()

## --- ERROR in cell 27, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3442297246.py in <cell line: 0>()
----> 1 sub.label.value_counts()

NameError: name 'sub' is not defined
