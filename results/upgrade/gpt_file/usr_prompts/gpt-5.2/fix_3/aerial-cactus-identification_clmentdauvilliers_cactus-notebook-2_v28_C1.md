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
scikit-image==0.25.2
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

0.8118

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd

for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames[:5]:
        print(os.path.join(dirname, filename))



## === cell 1
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from zipfile import ZipFile

print("TF version:", tf.__version__)



## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 2
path = "/kaggle/input/aerial-cactus-identification/"
files_dataframe = pd.read_csv(path + "train.csv", dtype={"id": str, "has_cactus": str})
files_dataframe.head()



## === cell 3
training_files = ("train/" + files_dataframe["id"]).tolist()
print("Training sample:")
print(training_files[:2])

os.makedirs("./training", exist_ok=True)
os.makedirs("./test", exist_ok=True)

with ZipFile(path + "train.zip", "r") as zipper:
    zipper.extractall("./training")

with ZipFile(path + "test.zip", "r") as zipper:
    zipper.extractall("./test")

print("Train dir exists:", os.path.isdir("./training/train"))
print("Test dir exists:", os.path.isdir("./test/test"))

if os.path.isdir("./training/train"):
    print(
        "Num train images:",
        len([f for f in os.listdir("./training/train") if f.lower().endswith(".jpg")]),
    )
if os.path.isdir("./test/test"):
    print(
        "Num test images:",
        len([f for f in os.listdir("./test/test") if f.lower().endswith(".jpg")]),
    )



## === cell 4
class_reparts = files_dataframe["has_cactus"].value_counts()
ax = class_reparts.plot.bar()



## === cell 5
total_samples = files_dataframe["has_cactus"].size
print("Total number of samples: ", total_samples)
has_cactus_weight = total_samples / (2 * class_reparts["1"])
no_cactus_weight = total_samples / (2 * class_reparts["0"])
class_weights = {0: float(no_cactus_weight), 1: float(has_cactus_weight)}
print("Class weights: ", class_weights)



## === cell 6
import matplotlib.pyplot as plt
from matplotlib.image import imread

plt.figure(figsize=(36, 12))
for i, k in enumerate(np.random.randint(0, len(files_dataframe), size=(20,))):
    plt.subplot(4, 5, i + 1)
    plt.imshow(imread("./training/train/" + files_dataframe["id"].iloc[k]))
    plt.title("Label :" + str(files_dataframe["has_cactus"].iloc[k]))
plt.tight_layout()



## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/3309526934.py in <cell line: 0>()
      6 for i, k in enumerate(np.random.randint(0, len(files_dataframe), size=(20,))):
      7     plt.subplot(4, 5, i + 1)
----> 8     plt.imshow(imread("./training/train/" + files_dataframe["id"].iloc[k]))
      9     plt.title("Label :" + str(files_dataframe["has_cactus"].iloc[k]))
     10 plt.tight_layout()

/usr/local/lib/python3.11/dist-packages/matplotlib/image.py in imread(fname, format)
   1561             "``np.array(PIL.Image.open(urllib.request.urlopen(url)))``."
   1562             )
-> 1563     with img_open(fname) as image:
   1564         return (_pil_png_to_float_array(image)
   1565                 if isinstance(image, PIL.PngImagePlugin.PngImageFile) else

/usr/local/lib/python3.11/dist-packages/PIL/Image.py in open(fp, mode, formats)
   3511     if is_path(fp):
   3512         filename = os.fspath(fp)
-> 3513         fp = builtins.open(filename, "rb")
   3514         exclusive_fp = True
   3515     else:

FileNotFoundError: [Errno 2] No such file or directory: './training/train/c24194b034dfa7be314182c94d45f027.jpg'

## === cell 7
import skimage.exposure as exposure


def preprocess(img):
    p2, p98 = np.percentile(img, (3, 97))
    img_rescale = exposure.rescale_intensity(img, in_range=(p2, p98))
    return img_rescale


plt.figure(figsize=(12, 6))
plt.subplot(121)
img = imread("./training/train/" + files_dataframe["id"].iloc[0])
plt.imshow(img)
plt.title("Before histogram equalization")

plt.subplot(122)
img2 = preprocess(img)
plt.imshow(img2)
plt.title("After histogram equalization")
plt.tight_layout()



## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/437621629.py in <cell line: 0>()
     10 plt.figure(figsize=(12, 6))
     11 plt.subplot(121)
---> 12 img = imread("./training/train/" + files_dataframe["id"].iloc[0])
     13 plt.imshow(img)
     14 plt.title("Before histogram equalization")

/usr/local/lib/python3.11/dist-packages/matplotlib/image.py in imread(fname, format)
   1561             "``np.array(PIL.Image.open(urllib.request.urlopen(url)))``."
   1562             )
-> 1563     with img_open(fname) as image:
   1564         return (_pil_png_to_float_array(image)
   1565                 if isinstance(image, PIL.PngImagePlugin.PngImageFile) else

/usr/local/lib/python3.11/dist-packages/PIL/Image.py in open(fp, mode, formats)
   3511     if is_path(fp):
   3512         filename = os.fspath(fp)
-> 3513         fp = builtins.open(filename, "rb")
   3514         exclusive_fp = True
   3515     else:

FileNotFoundError: [Errno 2] No such file or directory: './training/train/2de8f189f1dce439766637e75df0ee27.jpg'

## === cell 8
generator = ImageDataGenerator(
    samplewise_center=True,
    samplewise_std_normalization=True,
    validation_split=0.1,
    preprocessing_function=preprocess,
)

noPreprocessGenerator = ImageDataGenerator(
    samplewise_center=True,
    samplewise_std_normalization=True,
    validation_split=0.1,
    vertical_flip=True,
    horizontal_flip=True,
    rotation_range=45,
)

noAugmentationGenerator = ImageDataGenerator(
    samplewise_center=True,
    samplewise_std_normalization=True,
    preprocessing_function=preprocess,
)



## === cell 9
training_generator = generator.flow_from_dataframe(
    dataframe=files_dataframe,
    directory="./training/train/",
    x_col="id",
    y_col="has_cactus",
    class_mode="categorical",
    target_size=(32, 32),
    batch_size=32,
    subset="training",
    shuffle=True,
    seed=42,
)

validation_generator = generator.flow_from_dataframe(
    dataframe=files_dataframe,
    directory="./training/train/",
    x_col="id",
    y_col="has_cactus",
    class_mode="categorical",
    target_size=(32, 32),
    batch_size=32,
    subset="validation",
    shuffle=False,
    seed=42,
)

noproc_training_generator = noPreprocessGenerator.flow_from_dataframe(
    dataframe=files_dataframe,
    directory="./training/train/",
    x_col="id",
    y_col="has_cactus",
    class_mode="categorical",
    target_size=(32, 32),
    batch_size=32,
    subset="training",
    shuffle=True,
    seed=42,
)

noproc_validation_generator = noPreprocessGenerator.flow_from_dataframe(
    dataframe=files_dataframe,
    directory="./training/train/",
    x_col="id",
    y_col="has_cactus",
    class_mode="categorical",
    target_size=(32, 32),
    batch_size=32,
    subset="validation",
    shuffle=False,
    seed=42,
)

print("training_generator.n:", training_generator.n)
print("validation_generator.n:", validation_generator.n)



## === cell 10
plt.figure(figsize=(36, 12))
imgs, labels = next(validation_generator)
plotindx = 1
for img, label in zip(imgs[:35], labels[:35]):
    plt.subplot(7, 5, plotindx)
    plt.imshow((img - img.min()) / (img.max() - img.min() + 1e-8))
    labl = int(np.argmax(label))
    plt.title("Label :" + str(labl))
    plotindx += 1
plt.tight_layout()



## === cell 11
from tensorflow.keras import layers, models

model = models.Sequential()
model.add(layers.Flatten(input_shape=(32, 32, 3)))
model.add(layers.Dense(256, activation="relu"))
model.add(layers.Dense(256, activation="relu"))
model.add(layers.Dropout(0.1))
model.add(layers.Dense(128, activation="relu"))
model.add(layers.Dropout(0.1))
model.add(layers.Dense(128, activation="relu"))
model.add(layers.Dropout(0.1))
model.add(layers.Dense(64, activation="relu"))
model.add(layers.Dense(2, activation="softmax"))

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
model.summary()



## === cell 12
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint

reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=5, min_lr=0.001)

first_model_save = ModelCheckpoint(
    "/tmp/checkpoint_phase1.keras",
    monitor="val_loss",
    mode="min",
    save_best_only=True,
)
second_model_save = ModelCheckpoint(
    "/tmp/checkpoint_phase2.keras",
    monitor="val_loss",
    mode="min",
    save_best_only=True,
)



## === cell 13
steps_per_epoch = max(1, training_generator.n // training_generator.batch_size)

history = model.fit(
    training_generator,
    validation_data=validation_generator,
    steps_per_epoch=steps_per_epoch,
    verbose=1,
    epochs=1,
    class_weight=class_weights,
    callbacks=[first_model_save],
)

print("Checkpoint phase1 exists:", os.path.exists("/tmp/checkpoint_phase1.keras"))



## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/2208193087.py in <cell line: 0>()
      2 steps_per_epoch = max(1, training_generator.n // training_generator.batch_size)
      3 
----> 4 history = model.fit(
      5     training_generator,
      6     validation_data=validation_generator,

/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py in error_handler(*args, **kwargs)
    120             # To get the full stack trace, call:
    121             # `keras.config.disable_traceback_filtering()`
--> 122             raise e.with_traceback(filtered_tb) from None
    123         finally:
    124             del filtered_tb

/usr/local/lib/python3.11/dist-packages/keras/src/trainers/data_adapters/py_dataset_adapter.py in get_tf_dataset(self)
    293             ]
    294             if len(batches) == 0:
--> 295                 raise ValueError("The PyDataset has length 0")
    296             self._output_signature = data_adapter_utils.get_tensor_spec(batches)
    297 

ValueError: The PyDataset has length 0

## === cell 14
generator_p2 = ImageDataGenerator(
    samplewise_center=True,
    samplewise_std_normalization=True,
    validation_split=0.1,
    preprocessing_function=preprocess,
)

training_generator_p2 = generator_p2.flow_from_dataframe(
    dataframe=files_dataframe,
    directory="./training/train/",
    x_col="id",
    y_col="has_cactus",
    class_mode="categorical",
    target_size=(32, 32),
    batch_size=32,
    subset="training",
    shuffle=True,
    seed=42,
)

validation_generator_p2 = generator_p2.flow_from_dataframe(
    dataframe=files_dataframe,
    directory="./training/train/",
    x_col="id",
    y_col="has_cactus",
    class_mode="categorical",
    target_size=(32, 32),
    batch_size=32,
    subset="validation",
    shuffle=False,
    seed=42,
)



## === cell 15
from keras.models import load_model

if os.path.exists("/tmp/checkpoint_phase1.keras"):
    model = load_model("/tmp/checkpoint_phase1.keras")
else:
    print(
        "Warning: /tmp/checkpoint_phase1.keras not found; continuing with current model."
    )



## === cell 16
steps_per_epoch_p2 = max(1, training_generator_p2.n // training_generator_p2.batch_size)

history = model.fit(
    training_generator_p2,
    validation_data=validation_generator_p2,
    steps_per_epoch=steps_per_epoch_p2,
    verbose=1,
    epochs=16,
    class_weight=class_weights,
    callbacks=[second_model_save, reduce_lr],
)

print("Checkpoint phase2 exists:", os.path.exists("/tmp/checkpoint_phase2.keras"))



## --- ERROR in cell 16, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/2583122507.py in <cell line: 0>()
      1 steps_per_epoch_p2 = max(1, training_generator_p2.n // training_generator_p2.batch_size)
      2 
----> 3 history = model.fit(
      4     training_generator_p2,
      5     validation_data=validation_generator_p2,

/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py in error_handler(*args, **kwargs)
    120             # To get the full stack trace, call:
    121             # `keras.config.disable_traceback_filtering()`
--> 122             raise e.with_traceback(filtered_tb) from None
    123         finally:
    124             del filtered_tb

/usr/local/lib/python3.11/dist-packages/keras/src/trainers/data_adapters/py_dataset_adapter.py in get_tf_dataset(self)
    293             ]
    294             if len(batches) == 0:
--> 295                 raise ValueError("The PyDataset has length 0")
    296             self._output_signature = data_adapter_utils.get_tensor_spec(batches)
    297 

ValueError: The PyDataset has length 0

## === cell 17
if os.path.exists("/tmp/checkpoint_phase2.keras"):
    model = load_model("/tmp/checkpoint_phase2.keras")
else:
    print("Warning: /tmp/checkpoint_phase2.keras not found; using current model.")



## === cell 18
sample_sub = pd.read_csv(path + "sample_submission.csv", dtype={"id": str})
test_df = sample_sub[["id"]].copy()

test_generator = noAugmentationGenerator.flow_from_dataframe(
    dataframe=test_df,
    directory="./test/test/",
    x_col="id",
    y_col=None,
    class_mode=None,
    target_size=(32, 32),
    batch_size=32,
    shuffle=False,
)

print("test_generator.n:", test_generator.n)



## === cell 19
probas = model.predict(test_generator, verbose=0)

class_indices = training_generator.class_indices  # e.g., {'0': 0, '1': 1}
pos_index = class_indices.get("1", 1)
pos_proba = probas[:, pos_index].astype(np.float32)

print(
    "Pred proba stats:",
    float(pos_proba.min()),
    float(pos_proba.max()),
    float(pos_proba.mean()),
)



## --- ERROR in cell 19, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/4180238182.py in <cell line: 0>()
      1 # Predict probabilities for class "1" (has_cactus)
----> 2 probas = model.predict(test_generator, verbose=0)
      3 
      4 class_indices = training_generator.class_indices  # e.g., {'0': 0, '1': 1}
      5 pos_index = class_indices.get("1", 1)

/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py in error_handler(*args, **kwargs)
    120             # To get the full stack trace, call:
    121             # `keras.config.disable_traceback_filtering()`
--> 122             raise e.with_traceback(filtered_tb) from None
    123         finally:
    124             del filtered_tb

/usr/local/lib/python3.11/dist-packages/keras/src/trainers/data_adapters/py_dataset_adapter.py in get_tf_dataset(self)
    293             ]
    294             if len(batches) == 0:
--> 295                 raise ValueError("The PyDataset has length 0")
    296             self._output_signature = data_adapter_utils.get_tensor_spec(batches)
    297 

ValueError: The PyDataset has length 0

## === cell 20
import shutil

os.makedirs("./training/train", exist_ok=True)
for dirname, _, filenames in os.walk("./test/test/"):
    for filename in filenames:
        if filename.lower().endswith(".jpg"):
            src = os.path.join(dirname, filename)
            dst = os.path.join("./training/train", filename)
            if not os.path.exists(dst):
                shutil.copy(src, dst)



## === cell 21
images = []
for dirname, _, filenames in os.walk("./training/train"):
    for filename in filenames:
        if filename.lower().endswith(".jpg"):
            images.append(os.path.join(dirname, filename))
print(len(images))
print(images[:5])



## === cell 22
firstPredictions = tf.argmax(probas, axis=1).numpy().astype(int)

test_files_data = pd.DataFrame(
    {
        "id": [
            file for file in os.listdir("./test/test") if file.lower().endswith(".jpg")
        ],
        "has_cactus": firstPredictions.astype(str),
    }
)

final_training_dataframe = pd.concat(
    (files_dataframe[["id", "has_cactus"]], test_files_data), ignore_index=True
)
print(final_training_dataframe.head())



## --- ERROR in cell 22, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2260422409.py in <cell line: 0>()
      1 # Pseudo-label dataframe (kept for compatibility with original notebook flow)
----> 2 firstPredictions = tf.argmax(probas, axis=1).numpy().astype(int)
      3 
      4 test_files_data = pd.DataFrame(
      5     {

NameError: name 'probas' is not defined

## === cell 23
generator_final = ImageDataGenerator(
    samplewise_center=True,
    samplewise_std_normalization=True,
    vertical_flip=True,
    horizontal_flip=True,
    rotation_range=30,
    validation_split=0.1,
    shear_range=10,
    preprocessing_function=preprocess,
)

final_training_generator = generator_final.flow_from_dataframe(
    dataframe=final_training_dataframe,
    directory="./training/train/",
    x_col="id",
    y_col="has_cactus",
    class_mode="categorical",
    target_size=(32, 32),
    batch_size=32,
    subset="training",
    shuffle=True,
    seed=42,
)
final_training_valid = generator_final.flow_from_dataframe(
    dataframe=final_training_dataframe,
    directory="./training/train/",
    x_col="id",
    y_col="has_cactus",
    class_mode="categorical",
    target_size=(32, 32),
    batch_size=32,
    subset="validation",
    shuffle=False,
    seed=42,
)



## --- ERROR in cell 23, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1026609856.py in <cell line: 0>()
     11 
     12 final_training_generator = generator_final.flow_from_dataframe(
---> 13     dataframe=final_training_dataframe,
     14     directory="./training/train/",
     15     x_col="id",

NameError: name 'final_training_dataframe' is not defined

## === cell 24
final_steps_per_epoch = max(
    1, final_training_generator.n // final_training_generator.batch_size
)

history = model.fit(
    final_training_generator,
    validation_data=final_training_valid,
    steps_per_epoch=final_steps_per_epoch,
    verbose=1,
    epochs=1,
)



## --- ERROR in cell 24, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1389908241.py in <cell line: 0>()
      1 # Keep original final fit call, but ensure steps_per_epoch is valid.
      2 final_steps_per_epoch = max(
----> 3     1, final_training_generator.n // final_training_generator.batch_size
      4 )
      5 

NameError: name 'final_training_generator' is not defined

## === cell 25
from keras.models import load_model

if os.path.exists("/tmp/checkpoint_phase2.keras"):
    model = load_model("/tmp/checkpoint_phase2.keras")
else:
    print("Warning: /tmp/checkpoint_phase2.keras not found; using current model.")



## === cell 26
final_probas = model.predict(test_generator, verbose=0)
pos_proba_final = final_probas[:, pos_index].astype(np.float32)

output = pd.DataFrame({"id": test_df["id"].values, "has_cactus": pos_proba_final})
output.to_csv("submission.csv", index=False)
print(output.head())
print("Wrote submission.csv with shape:", output.shape)



## --- ERROR in cell 26, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/3634861566.py in <cell line: 0>()
----> 1 final_probas = model.predict(test_generator, verbose=0)
      2 pos_proba_final = final_probas[:, pos_index].astype(np.float32)
      3 
      4 output = pd.DataFrame({"id": test_df["id"].values, "has_cactus": pos_proba_final})
      5 output.to_csv("submission.csv", index=False)

/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py in error_handler(*args, **kwargs)
    120             # To get the full stack trace, call:
    121             # `keras.config.disable_traceback_filtering()`
--> 122             raise e.with_traceback(filtered_tb) from None
    123         finally:
    124             del filtered_tb

/usr/local/lib/python3.11/dist-packages/keras/src/trainers/data_adapters/py_dataset_adapter.py in get_tf_dataset(self)
    293             ]
    294             if len(batches) == 0:
--> 295                 raise ValueError("The PyDataset has length 0")
    296             self._output_signature = data_adapter_utils.get_tensor_spec(batches)
    297 

ValueError: The PyDataset has length 0

## === cell 27
try:
    shutil.rmtree("test")
except OSError:
    print("Test files already erased")
try:
    shutil.rmtree("training")
except OSError:
    print("Training files already erased")
