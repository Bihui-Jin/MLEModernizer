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

N/A

# 9. Code solution

## === cell 0
import os, random, pickle, gc, matplotlib.pyplot as plt
import numpy as np, pandas as pd
from datetime import datetime, timedelta

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
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
def visualize_class_images(base_directory, rows=2, cols=6):
    class_dirs = [
        d
        for d in os.listdir(base_directory)
        if os.path.isdir(os.path.join(base_directory, d))
    ]
    imgs = []
    for cls in class_dirs:
        cls_path = os.path.join(base_directory, cls)
        files = [
            f for f in os.listdir(cls_path) if os.path.isfile(os.path.join(cls_path, f))
        ]
        if not files:
            continue  # skip empty folders
        img_path = os.path.join(cls_path, random.choice(files))
        imgs.append((img_path, cls))
    random.shuffle(imgs)
    imgs = imgs[: rows * cols]
    plt.figure(figsize=(15, 6))
    for i, (p, name) in enumerate(imgs):
        img = plt.imread(p)
        plt.subplot(rows, cols, i + 1)
        plt.imshow(img)
        plt.title(name)
        plt.axis("off")
    plt.tight_layout()
    plt.show()


train_dir = "/kaggle/input/plant-seedlings-classification/train"
visualize_class_images(train_dir)




## === cell 3
def define_generators(train_directory, validation_directory, test_directory):
    workers = 4
    train_datagen = ImageDataGenerator(
        rotation_range=360,
        width_shift_range=0.3,
        height_shift_range=0.3,
        shear_range=0.3,
        zoom_range=0.5,
        vertical_flip=True,
        horizontal_flip=True,
    )
    train_gen = train_datagen.flow_from_directory(
        train_directory,
        target_size=(height, width),
        batch_size=BATCH_SIZE,
        color_mode="rgb",
        class_mode="categorical",
        classes=CLASSES,  # enforce correct class list
        shuffle=True,
        seed=42,
        workers=workers,
        use_multiprocessing=True,
        max_queue_size=10,
    )

    val_datagen = ImageDataGenerator()
    val_gen = val_datagen.flow_from_directory(
        validation_directory,
        target_size=(height, width),
        batch_size=BATCH_SIZE,
        color_mode="rgb",
        class_mode="categorical",
        classes=CLASSES,  # enforce correct class list
        shuffle=False,
        seed=42,
        workers=workers,
        use_multiprocessing=True,
        max_queue_size=10,
    )

    test_datagen = ImageDataGenerator()
    test_gen = test_datagen.flow_from_directory(
        directory=os.path.dirname(test_directory),
        classes=["test"],
        target_size=(height, width),
        batch_size=1,
        color_mode="rgb",
        shuffle=False,
        class_mode=None,
    )
    return train_gen, val_gen, test_gen


def visualize_generator_samples(generator, rows=2, cols=6):
    images, labels = next(generator)
    class_labels = list(generator.class_indices.keys())
    plt.figure(figsize=(15, 6))
    for i in range(rows * cols):
        plt.subplot(rows, cols, i + 1)
        plt.imshow(images[i].astype("uint8"))
        plt.title(class_labels[np.argmax(labels[i])])
        plt.axis("off")
    plt.tight_layout()
    plt.show()




## === cell 4
IMAGE_SIZE = [299, 299]  # ResNet/Inception default
height, width = IMAGE_SIZE
BATCH_SIZE = 32 * strategy.num_replicas_in_sync
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

train_directory = "/kaggle/input/plant-seedlings-classification/train"
validation_directory = train_directory
test_directory = "/kaggle/input/plant-seedlings-classification/test"

train_gen, val_gen, test_gen = define_generators(
    train_directory, validation_directory, test_directory
)

print(
    f"Train steps: {len(train_gen)}  Val steps: {len(val_gen)}  Test samples: {test_gen.samples}"
)

visualize_generator_samples(train_gen)




## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3269567048.py in <cell line: 0>()
     23 test_directory = "/kaggle/input/plant-seedlings-classification/test"
     24 
---> 25 train_gen, val_gen, test_gen = define_generators(
     26     train_directory, validation_directory, test_directory
     27 )

/tmp/ipykernel_55/2102691981.py in define_generators(train_directory, validation_directory, test_directory)
     11         horizontal_flip=True,
     12     )
---> 13     train_gen = train_datagen.flow_from_directory(
     14         train_directory,
     15         target_size=(height, width),

TypeError: ImageDataGenerator.flow_from_directory() got an unexpected keyword argument 'workers'

## === cell 5
def create_InceptionV3_model():
    base = tf.keras.applications.InceptionV3(
        weights="imagenet", include_top=False, input_shape=[*IMAGE_SIZE, 3]
    )
    base.trainable = True
    model = tf.keras.Sequential(
        [
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




## === cell 6
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



## === cell 7
no_of_models = 1
EPOCHS = 20  # shortened to keep runtime reasonable
historys = [0] * no_of_models

checkpoint_path = "/kaggle/working/best_model.keras"  # must end with .keras
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
        train_gen,
        epochs=EPOCHS,
        steps_per_epoch=train_gen.samples // BATCH_SIZE,
        validation_data=val_gen,
        validation_steps=val_gen.samples // BATCH_SIZE,
        callbacks=[reduce_lr, early_stop, lr_callback, model_ckpt],
        workers=4,
        use_multiprocessing=True,
    )
    write_history(j, this_run_file_prefix + f"model_{j}")
    models[j].save(this_run_file_prefix + f"model_{j}.keras")
    gc.collect()

print("Training finished at", datetime.now())



## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1445952580.py in <cell line: 0>()
     27         break
     28     historys[j] = models[j].fit(
---> 29         train_gen,
     30         epochs=EPOCHS,
     31         steps_per_epoch=train_gen.samples // BATCH_SIZE,

NameError: name 'train_gen' is not defined

## === cell 8
best_model = tf.keras.models.load_model(checkpoint_path)
history_to_analyze = historys[0].history
plot_history(history_to_analyze)



## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_55/774484073.py in <cell line: 0>()
----> 1 best_model = tf.keras.models.load_model(checkpoint_path)
      2 history_to_analyze = historys[0].history
      3 plot_history(history_to_analyze)
      4 

/usr/local/lib/python3.11/dist-packages/keras/src/saving/saving_api.py in load_model(filepath, custom_objects, compile, safe_mode)
    198         )
    199     elif str(filepath).endswith(".keras"):
--> 200         raise ValueError(
    201             f"File not found: filepath={filepath}. "
    202             "Please ensure the file is an accessible `.keras` "

ValueError: File not found: filepath=/kaggle/working/best_model.keras. Please ensure the file is an accessible `.keras` zip file.

## === cell 9
preds = best_model.predict(test_gen, steps=test_gen.samples, verbose=0)
class_list = [CLASSES[p.argmax()] for p in preds]


def plot_test_images_with_predictions(generator, predictions, num_images=25):
    num_images = min(num_images, generator.samples)
    plt.figure(figsize=(15, 15))
    cols = 5
    rows = (num_images + cols - 1) // cols
    for i in range(num_images):
        img_path = os.path.join(generator.directory, generator.filenames[i])
        img = plt.imread(img_path)
        plt.subplot(rows, cols, i + 1)
        plt.imshow(img)
        plt.title(f"Pred: {predictions[i]}")
        plt.axis("off")
    plt.tight_layout()
    plt.show()


plot_test_images_with_predictions(test_gen, class_list, num_images=25)



## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/190055530.py in <cell line: 0>()
----> 1 preds = best_model.predict(test_gen, steps=test_gen.samples, verbose=0)
      2 class_list = [CLASSES[p.argmax()] for p in preds]
      3 
      4 
      5 def plot_test_images_with_predictions(generator, predictions, num_images=25):

NameError: name 'best_model' is not defined

## === cell 10
submission = pd.DataFrame(
    {"file": [os.path.basename(f) for f in test_gen.filenames], "species": class_list}
)
submission.to_csv("submission.csv", index=False)
print("Submission saved to submission.csv")

## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/272353991.py in <cell line: 0>()
      1 submission = pd.DataFrame(
----> 2     {"file": [os.path.basename(f) for f in test_gen.filenames], "species": class_list}
      3 )
      4 submission.to_csv("submission.csv", index=False)
      5 print("Submission saved to submission.csv")

NameError: name 'test_gen' is not defined
