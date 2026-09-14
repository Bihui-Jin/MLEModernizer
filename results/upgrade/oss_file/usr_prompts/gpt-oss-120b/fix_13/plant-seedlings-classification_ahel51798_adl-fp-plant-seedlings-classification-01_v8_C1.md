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
Classify plant seedlings into their respective species.

## Metric
Micro-averaged F1-score.

## Submission Format
For each `file` in the test set, you must predict a probability for the `species` variable. The file should contain a header and have the following format:

```
file,species
0021e90e4.png,Maize
003d61042.png,Sugar beet
007b3da8b.png,Common wheat
etc.
```

## Dataset
The list of species is as follows:

```
Black-grass
Charlock
Cleavers
Common Chickweed
Common wheat
Fat Hen
Loose Silky-bent
Maize
Scentless Mayweed
Shepherds Purse
Small-flowered Cranesbill
Sugar beet
```

- **train.csv** - the training set, with plant species organized by folder
- **test.csv** - the test set, you need to predict the species of each image
- **sample_submission.csv** - a sample submission file in the correct format

# 2. Python version

3.12

# 3. Installed packages

geopandas==0.14.4
google-api-python-client==2.177.0
ipython==7.34.0
ipython-genutils==0.2.0
ipython_pygments_lexers==1.1.1
ipython-sql==0.5.0
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
pillow==11.3.0
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
            description.md (84 lines)
            sample_submission.csv (667 lines)
            sample_submission.csv.zip (4.6 kB)
            test.zip (259.0 MB)
            train.zip (1.5 GB)
            plant-seedlings-classification/
                description.md (84 lines)
                sample_submission.csv (667 lines)
                ... and 3 other files
                plant-seedlings-classification/
                test/
                    5db43df54.png (177.5 kB)
                    09d34fe5b.png (156.0 kB)
                    ... and 664 other files
                    test/
                train/
                    Black-grass/
                        2ed589264.png (44.6 kB)
                        840a7ed59.png (708.1 kB)
                        ... and 219 other files
                    Charlock/
                        ee4a02bf9.png (229.3 kB)
                        e795c53c9.png (354.4 kB)
                        ... and 322 other files
                    ... and 11 other folders
            test/
                5db43df54.png (177.5 kB)
                09d34fe5b.png (156.0 kB)
                ... and 664 other files
                test/
            train/
                Black-grass/
                    2ed589264.png (44.6 kB)
                    840a7ed59.png (708.1 kB)
                    ... and 219 other files
                Charlock/
                    ee4a02bf9.png (229.3 kB)
                    e795c53c9.png (354.4 kB)
                    ... and 322 other files
                ... and 11 other folders
        input/
            description.md (84 lines)
            sample_submission.csv (667 lines)
            sample_submission.csv.zip (4.6 kB)
            test.zip (259.0 MB)
            train.zip (1.5 GB)
            plant-seedlings-classification/
                description.md (84 lines)
                sample_submission.csv (667 lines)
                ... and 3 other files
                plant-seedlings-classification/
                test/
                    5db43df54.png (177.5 kB)
                    09d34fe5b.png (156.0 kB)
                    ... and 664 other files
                    test/
                train/
                    Black-grass/
                        2ed589264.png (44.6 kB)
                        840a7ed59.png (708.1 kB)
                        ... and 219 other files
                    Charlock/
                        ee4a02bf9.png (229.3 kB)
                        e795c53c9.png (354.4 kB)
                        ... and 322 other files
                    ... and 11 other folders
            test/
                5db43df54.png (177.5 kB)
                09d34fe5b.png (156.0 kB)
                ... and 664 other files
                test/
                    5db43df54.png (177.5 kB)
                    09d34fe5b.png (156.0 kB)
                    ... and 664 other files
                    test/
            train/
                Black-grass/
                    2ed589264.png (44.6 kB)
                    840a7ed59.png (708.1 kB)
                    ... and 219 other files
                Charlock/
                    ee4a02bf9.png (229.3 kB)
                    e795c53c9.png (354.4 kB)
                    ... and 322 other files
                ... and 11 other folders
        working/
            plant-seedlings-classification/
                description.md (84 lines)
                sample_submission.csv (667 lines)
                ... and 3 other files
                plant-seedlings-classification/
                test/
                    5db43df54.png (177.5 kB)
                    09d34fe5b.png (156.0 kB)
                    ... and 664 other files
                    test/
                train/
                    Black-grass/
                        2ed589264.png (44.6 kB)
                        840a7ed59.png (708.1 kB)
                        ... and 219 other files
                    Charlock/
                        ee4a02bf9.png (229.3 kB)
                        e795c53c9.png (354.4 kB)
                        ... and 322 other files
                    ... and 11 other folders
```

-> data/plant-seedlings-classification/sample_submission.csv has 666 rows and 2 columns.
The columns are: file, species

-> data/sample_submission.csv has 666 rows and 2 columns.
The columns are: file, species

-> input/plant-seedlings-classification/sample_submission.csv has 666 rows and 2 columns.
The columns are: file, species

-> input/sample_submission.csv has 666 rows and 2 columns.
The columns are: file, species

-> working/plant-seedlings-classification/sample_submission.csv has 666 rows and 2 columns.
The columns are: file, species

# 5. Target score

0.98488

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved 0.14414) has done: 'We increase the batch size (reducing the number of gradient steps per epoch) and enable multiprocessing workers for the ImageDataGenerator, which cuts the per‑epoch overhead without changing the model, loss, or augmentation logic. The rest of the pipeline—including the InceptionV3 architecture, learning‑rate schedule, callbacks, and prediction steps—remains exactly the same, so results stay unchanged while training finishes well before the 600‑second limit.'

# 9. Code solution

## === cell 0
import os, random, pickle, gc, matplotlib.pyplot as plt
import numpy as np, pandas as pd
from datetime import datetime, timedelta

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

np.random.seed(42)
random.seed(42)
os.environ["PYTHONHASHSEED"] = "42"

import tensorflow as tf

tf.config.optimizer.set_jit(True)
tf.keras.mixed_precision.set_global_policy("mixed_float16")
tf.random.set_seed(42)

from tensorflow.keras.callbacks import (
    ReduceLROnPlateau,
    EarlyStopping,
    LearningRateScheduler,
    ModelCheckpoint,
)

print("TensorFlow version:", tf.__version__)



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
strategy = tf.distribute.get_strategy()
print("REPLICAS:", strategy.num_replicas_in_sync)

start_time = datetime.now()
print("Start time:", start_time)
end_training_by_tdelta = timedelta(seconds=8400)
this_run_file_prefix = start_time.strftime("%Y%m%d_%H%M_")
print("Run prefix:", this_run_file_prefix)



## === cell 2
IMAGE_SIZE = [299, 299]  # ResNet/Inception default
height, width = IMAGE_SIZE
BATCH_SIZE = 128 * strategy.num_replicas_in_sync
CLASSES = [
    "Black-grass",
    "Charlock",
    "Cleavers",
    "Common Chickweed",
    "Common wheat",
    "Fat Hen",
    "Loose Silky-bent",
    "Maize",
    "Scentless Mayweed",
    "Shepherds Purse",
    "Small-flowered Cranesbill",
    "Sugar beet",
]
num_classes = len(CLASSES)

train_dir = "/kaggle/input/plant-seedlings-classification/train"
test_dir = "/kaggle/input/plant-seedlings-classification/test"


def get_datasets(train_dir, test_dir, img_size, batch_size, seed=42):
    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        labels="inferred",
        label_mode="categorical",
        class_names=CLASSES,
        color_mode="rgb",
        batch_size=batch_size,
        image_size=img_size,
        shuffle=True,
        seed=seed,
        validation_split=0.1,
        subset="training",
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        labels="inferred",
        label_mode="categorical",
        class_names=CLASSES,
        color_mode="rgb",
        batch_size=batch_size,
        image_size=img_size,
        shuffle=False,
        seed=seed,
        validation_split=0.1,
        subset="validation",
    )
    test_ds = tf.keras.utils.image_dataset_from_directory(
        test_dir,
        labels=None,
        label_mode=None,
        class_names=["test"],
        color_mode="rgb",
        batch_size=1,
        image_size=img_size,
        shuffle=False,
    )
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.prefetch(AUTOTUNE)
    val_ds = val_ds.prefetch(AUTOTUNE)
    test_ds = test_ds.prefetch(AUTOTUNE)
    return train_ds, val_ds, test_ds


train_ds, val_ds, test_ds = get_datasets(train_dir, test_dir, IMAGE_SIZE, BATCH_SIZE)

print(
    f"Train batches: {tf.data.experimental.cardinality(train_ds).numpy()}, "
    f"Val batches: {tf.data.experimental.cardinality(val_ds).numpy()}, "
    f"Test samples: {tf.data.experimental.cardinality(test_ds).numpy()}"
)




## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_54/2204415257.py in <cell line: 0>()
     74 
     75 
---> 76 train_ds, val_ds, test_ds = get_datasets(train_dir, test_dir, IMAGE_SIZE, BATCH_SIZE)
     77 
     78 print(

/tmp/ipykernel_54/2204415257.py in get_datasets(train_dir, test_dir, img_size, batch_size, seed)
     56     )
     57     # Test dataset (no labels)
---> 58     test_ds = tf.keras.utils.image_dataset_from_directory(
     59         test_dir,
     60         labels=None,

/usr/local/lib/python3.11/dist-packages/keras/src/utils/image_dataset_utils.py in image_dataset_from_directory(directory, labels, label_mode, class_names, color_mode, batch_size, image_size, shuffle, seed, validation_split, subset, interpolation, follow_links, crop_to_aspect_ratio, pad_to_aspect_ratio, data_format, verbose)
    230     if seed is None:
    231         seed = np.random.randint(1e6)
--> 232     image_paths, labels, class_names = dataset_utils.index_directory(
    233         directory,
    234         labels,

/usr/local/lib/python3.11/dist-packages/keras/src/utils/dataset_utils.py in index_directory(directory, labels, formats, class_names, shuffle, seed, follow_links, verbose)
    548         if class_names is not None:
    549             if labels is None:
--> 550                 raise ValueError(
    551                     "When `labels=None` (no labels), argument `class_names` "
    552                     "cannot be specified."

ValueError: When `labels=None` (no labels), argument `class_names` cannot be specified.

## === cell 3
def create_InceptionV3_model():
    augmentation = tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip(mode="horizontal_and_vertical"),
            tf.keras.layers.RandomRotation(factor=1.0),  # 0‑360°
            tf.keras.layers.RandomZoom(height_factor=0.5, width_factor=0.5),
            tf.keras.layers.RandomTranslation(height_factor=0.3, width_factor=0.3),
            tf.keras.layers.RandomShear(angle=0.3),
        ]
    )
    base = tf.keras.applications.InceptionV3(
        weights="imagenet", include_top=False, input_shape=[*IMAGE_SIZE, 3]
    )
    base.trainable = False
    model = tf.keras.Sequential(
        [
            augmentation,
            tf.keras.applications.inception_v3.preprocess_input,
            base,
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dense(num_classes, activation="softmax"),
        ]
    )
    model.compile(
        optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
    )
    return model


def write_history(j, filename):
    history_dict = [0] * no_of_models
    for i in range(j + 1):
        if historys[i] != 0:
            history_dict[i] = historys[i].history
    with open(filename + ".pkl", "ab") as f:
        pickle.dump(history_dict, f)


def plot_history(hist):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(hist["accuracy"], label="Acc")
    plt.plot(hist["val_accuracy"], label="Val Acc")
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(hist["loss"], label="Loss")
    plt.plot(hist["val_loss"], label="Val Loss")
    plt.legend()
    plt.show()




## === cell 4
def lrfn(epoch):
    if epoch < LR_RAMPUP_EPOCHS:
        lr = LR_START + epoch * (LR_MAX - LR_START) / LR_RAMPUP_EPOCHS
    elif epoch < LR_RAMPUP_EPOCHS + LR_SUSTAIN_EPOCHS:
        lr = LR_MAX
    else:
        lr = LR_MIN + (LR_MAX - LR_MIN) * LR_EXP_DECAY ** (
            epoch - LR_RAMPUP_EPOCHS - LR_SUSTAIN_EPOCHS
        )
    return lr


LR_START = 1e-5
LR_MAX = 5e-5 * strategy.num_replicas_in_sync
LR_MIN = LR_START
LR_RAMPUP_EPOCHS = 5
LR_SUSTAIN_EPOCHS = 0
LR_EXP_DECAY = 0.80
lr_callback = LearningRateScheduler(lrfn, verbose=True)

rng = list(range(30))
plt.plot(rng, [lrfn(x) for x in rng])
plt.title("Learning Rate schedule")
plt.show()



## === cell 5
no_of_models = 1
EPOCHS = 20  # unchanged from original
historys = [0] * no_of_models

checkpoint_path = "/kaggle/working/best_model.keras"
early_stop = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(
    monitor="val_accuracy", patience=2, factor=0.5, min_lr=1e-5, verbose=1
)
model_ckpt = ModelCheckpoint(
    filepath=checkpoint_path,
    save_best_only=True,
    monitor="val_loss",
    mode="min",
    verbose=1,
)

with strategy.scope():
    models = [create_InceptionV3_model() for _ in range(no_of_models)]

models[0].summary()

for j in range(no_of_models):
    start = datetime.now()
    if (start - start_time) > end_training_by_tdelta:
        print("Time limit reached, stopping training.")
        break
    historys[j] = models[j].fit(
        train_ds,
        epochs=EPOCHS,
        validation_data=val_ds,
        callbacks=[reduce_lr, early_stop, lr_callback, model_ckpt],
        verbose=2,
    )
    write_history(j, this_run_file_prefix + f"model_{j}")
    models[j].save(this_run_file_prefix + f"model_{j}.keras")
    gc.collect()

print("Training finished at", datetime.now())



## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_54/4065367209.py in <cell line: 0>()
     17 
     18 with strategy.scope():
---> 19     models = [create_InceptionV3_model() for _ in range(no_of_models)]
     20 
     21 models[0].summary()

/tmp/ipykernel_54/4065367209.py in <listcomp>(.0)
     17 
     18 with strategy.scope():
---> 19     models = [create_InceptionV3_model() for _ in range(no_of_models)]
     20 
     21 models[0].summary()

/tmp/ipykernel_54/3565227036.py in create_InceptionV3_model()
      7             tf.keras.layers.RandomZoom(height_factor=0.5, width_factor=0.5),
      8             tf.keras.layers.RandomTranslation(height_factor=0.3, width_factor=0.3),
----> 9             tf.keras.layers.RandomShear(angle=0.3),
     10         ]
     11     )

/usr/local/lib/python3.11/dist-packages/keras/src/layers/preprocessing/image_preprocessing/random_shear.py in __init__(self, x_factor, y_factor, interpolation, fill_mode, fill_value, data_format, seed, **kwargs)
     81         **kwargs,
     82     ):
---> 83         super().__init__(data_format=data_format, **kwargs)
     84         self.x_factor = self._set_factor_with_name(x_factor, "x_factor")
     85         self.y_factor = self._set_factor_with_name(y_factor, "y_factor")

/usr/local/lib/python3.11/dist-packages/keras/src/layers/preprocessing/image_preprocessing/base_image_preprocessing_layer.py in __init__(self, factor, bounding_box_format, data_format, **kwargs)
     15         self, factor=None, bounding_box_format=None, data_format=None, **kwargs
     16     ):
---> 17         super().__init__(**kwargs)
     18         self.bounding_box_format = bounding_box_format
     19         self.data_format = backend_config.standardize_data_format(data_format)

/usr/local/lib/python3.11/dist-packages/keras/src/layers/preprocessing/tf_data_layer.py in __init__(self, **kwargs)
     17 
     18     def __init__(self, **kwargs):
---> 19         super().__init__(**kwargs)
     20         self.backend = backend_utils.DynamicBackend()
     21         self._allow_non_tensor_positional_args = True

/usr/local/lib/python3.11/dist-packages/keras/src/layers/layer.py in __init__(self, activity_regularizer, trainable, dtype, autocast, name, **kwargs)
    285             self._input_shape_arg = input_shape_arg
    286         if kwargs:
--> 287             raise ValueError(
    288                 "Unrecognized keyword arguments "
    289                 f"passed to {self.__class__.__name__}: {kwargs}"

ValueError: Unrecognized keyword arguments passed to RandomShear: {'angle': 0.3}

## === cell 6
if os.path.exists(checkpoint_path):
    best_model = tf.keras.models.load_model(checkpoint_path)
    print("Loaded best model from checkpoint.")
else:
    best_model = models[0]
    print("Checkpoint not found; using the last trained model.")

if isinstance(historys[0], tf.keras.callbacks.History):
    history_to_analyze = historys[0].history
    plot_history(history_to_analyze)
else:
    print("No training history available to plot.")



## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_54/1849393659.py in <cell line: 0>()
      3     print("Loaded best model from checkpoint.")
      4 else:
----> 5     best_model = models[0]
      6     print("Checkpoint not found; using the last trained model.")
      7 

NameError: name 'models' is not defined

## === cell 7
preds = best_model.predict(
    test_ds,
    verbose=0,
)

class_list = [CLASSES[p.argmax()] for p in preds]


def plot_test_images_with_predictions(dataset, predictions, num_images=25):
    num_images = min(num_images, tf.data.experimental.cardinality(dataset).numpy())
    plt.figure(figsize=(15, 15))
    cols = 5
    rows = (num_images + cols - 1) // cols
    for i, (img_batch, fname) in enumerate(
        zip(dataset.unbatch().take(num_images), predictions)
    ):
        img = img_batch.numpy().astype("uint8")
        plt.subplot(rows, cols, i + 1)
        plt.imshow(img)
        plt.title(f"Pred: {predictions[i]}")
        plt.axis("off")
    plt.tight_layout()
    plt.show()


filenames = test_ds.file_paths
plot_test_images_with_predictions(test_ds, class_list, num_images=25)



## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_54/1771109744.py in <cell line: 0>()
      1 # Predict on the test dataset; test_ds yields (image, ) tuples, so we take the first element.
----> 2 preds = best_model.predict(
      3     test_ds,
      4     verbose=0,
      5 )

NameError: name 'best_model' is not defined

## === cell 8
submission = pd.DataFrame(
    {"file": [os.path.basename(f) for f in filenames], "species": class_list}
)
submission.to_csv("submission.csv", index=False)
print("Submission saved to submission.csv")

## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_54/3086770090.py in <cell line: 0>()
      1 submission = pd.DataFrame(
----> 2     {"file": [os.path.basename(f) for f in filenames], "species": class_list}
      3 )
      4 submission.to_csv("submission.csv", index=False)
      5 print("Submission saved to submission.csv")

NameError: name 'filenames' is not defined
