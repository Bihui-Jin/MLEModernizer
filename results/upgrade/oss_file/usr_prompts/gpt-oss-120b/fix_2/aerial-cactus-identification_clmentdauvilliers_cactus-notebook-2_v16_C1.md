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

0.951

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
import os
from zipfile import ZipFile
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from matplotlib.image import imread



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
base_path = "/kaggle/input/aerial-cactus-identification/"

files_dataframe = pd.read_csv(os.path.join(base_path, "train.csv"), dtype=str)
files_dataframe.head()



## === cell 2
train_dir = os.path.join(base_path, "train")
test_dir = os.path.join(base_path, "test")




## === cell 3
class_reparts = files_dataframe["has_cactus"].value_counts()
ax = class_reparts.plot.bar()
plt.title("Class distribution")



## === cell 4
total_samples = files_dataframe["has_cactus"].size
has_cactus_weight = total_samples / (2 * class_reparts["1"])
no_cactus_weight = total_samples / (2 * class_reparts["0"])
class_weights = {0: no_cactus_weight, 1: has_cactus_weight}
print("Class weights:", class_weights)



## === cell 5
training_files = "train/" + files_dataframe["id"]
plt.figure(figsize=(36, 12))
for i, k in enumerate(np.random.randint(0, len(files_dataframe), size=20)):
    plt.subplot(4, 5, i + 1)
    img_path = os.path.join(train_dir, training_files.iloc[k])
    plt.imshow(imread(img_path))
    plt.title("Label: " + str(files_dataframe["has_cactus"].iloc[k]))
plt.tight_layout()



## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_55/803658893.py in <cell line: 0>()
      5     plt.subplot(4, 5, i + 1)
      6     img_path = os.path.join(train_dir, training_files.iloc[k])
----> 7     plt.imshow(imread(img_path))
      8     plt.title("Label: " + str(files_dataframe["has_cactus"].iloc[k]))
      9 plt.tight_layout()

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

FileNotFoundError: [Errno 2] No such file or directory: '/kaggle/input/aerial-cactus-identification/train/train/a030c34806ccb9bc96df7ade80159e53.jpg'

## === cell 6
import skimage.exposure as exposure


def preprocess(img):
    p2, p98 = np.percentile(img, (2, 98))
    return exposure.rescale_intensity(img, in_range=(p2, p98))


plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
img = imread(os.path.join(train_dir, training_files.iloc[0]))
plt.imshow(img)
plt.title("Before preprocessing")
plt.subplot(1, 2, 2)
plt.imshow(preprocess(img))
plt.title("After preprocessing")
plt.tight_layout()



## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_55/167212135.py in <cell line: 0>()
     10 plt.figure(figsize=(12, 5))
     11 plt.subplot(1, 2, 1)
---> 12 img = imread(os.path.join(train_dir, training_files.iloc[0]))
     13 plt.imshow(img)
     14 plt.title("Before preprocessing")

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

FileNotFoundError: [Errno 2] No such file or directory: '/kaggle/input/aerial-cactus-identification/train/train/2de8f189f1dce439766637e75df0ee27.jpg'

## === cell 7
generator = ImageDataGenerator(
    samplewise_center=True,
    samplewise_std_normalization=True,
    vertical_flip=True,
    horizontal_flip=True,
    rotation_range=45,
    shear_range=10,
    validation_split=0.1,
    preprocessing_function=preprocess,
)

noAugmentationGenerator = ImageDataGenerator(
    samplewise_center=True,
    samplewise_std_normalization=True,
    preprocessing_function=preprocess,
)



## === cell 8
training_generator = generator.flow_from_dataframe(
    dataframe=files_dataframe,
    directory=train_dir,
    x_col="id",
    y_col="has_cactus",
    class_mode="categorical",
    target_size=(32, 32),
    batch_size=32,
    subset="training",
    shuffle=True,
)

validation_generator = generator.flow_from_dataframe(
    dataframe=files_dataframe,
    directory=train_dir,
    x_col="id",
    y_col="has_cactus",
    class_mode="categorical",
    target_size=(32, 32),
    batch_size=32,
    subset="validation",
    shuffle=True,
)



## === cell 9
imgs, labels = next(validation_generator)
plt.figure(figsize=(12, 8))
for idx, (img, label) in enumerate(zip(imgs[:12], labels[:12])):
    plt.subplot(3, 4, idx + 1)
    plt.imshow(img)
    plt.title(f"Label: {np.argmax(label)}")
    plt.axis("off")
plt.tight_layout()



## === cell 10
from tensorflow.keras import datasets, layers, models



## === cell 11
model = models.Sequential()
model.add(layers.Conv2D(32, (5, 5), activation="relu", input_shape=(32, 32, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Dropout(0.2))

model.add(layers.Conv2D(64, (3, 3), activation="relu"))
model.add(layers.Conv2D(64, (3, 3), activation="relu"))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Dropout(0.2))

model.add(layers.Conv2D(128, (3, 3), activation="relu"))
model.add(layers.Conv2D(128, (3, 3), activation="relu"))
model.add(layers.Dropout(0.2))

model.add(layers.Flatten())
model.add(layers.Dense(256, activation="relu"))
model.add(layers.Dense(256, activation="relu"))
model.add(layers.Dropout(0.1))
model.add(layers.Dense(128, activation="relu"))
model.add(layers.Dense(2, activation="softmax"))

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
model.summary()



## === cell 12
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint

reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=5, min_lr=0.001)

save_best_model = ModelCheckpoint(
    "/tmp/checkpoint.keras",
    monitor="val_accuracy",
    mode="max",
    save_best_only=True,
    save_weights_only=False,
)



## === cell 13
history = model.fit(
    training_generator,
    validation_data=validation_generator,
    steps_per_epoch=training_generator.n // training_generator.batch_size,
    epochs=8,
    class_weight=class_weights,
    callbacks=[reduce_lr, save_best_model],
    verbose=1,
)



## === cell 14
test_filenames = sorted(os.listdir(test_dir))
test_df = pd.DataFrame({"filename": test_filenames})

test_generator = noAugmentationGenerator.flow_from_dataframe(
    dataframe=test_df,
    directory=test_dir,
    x_col="filename",
    y_col=None,
    class_mode=None,
    target_size=(32, 32),
    batch_size=32,
    shuffle=False,
)



## === cell 15
from tensorflow.keras.models import load_model

model = load_model("/tmp/checkpoint.keras")



## === cell 16
pred_probs = model.predict(test_generator, verbose=0)[:, 1]

submission = pd.DataFrame({"id": test_filenames, "has_cactus": pred_probs})
submission.to_csv("submission.csv", index=False)
print("Submission saved to submission.csv with shape:", submission.shape)



## --- ERROR in cell 16, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_55/1871168223.py in <cell line: 0>()
      2 pred_probs = model.predict(test_generator, verbose=0)[:, 1]
      3 
----> 4 submission = pd.DataFrame({"id": test_filenames, "has_cactus": pred_probs})
      5 submission.to_csv("submission.csv", index=False)
      6 print("Submission saved to submission.csv with shape:", submission.shape)

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

## === cell 17
import shutil

for folder in ["training", "test"]:
    try:
        shutil.rmtree(folder)
    except OSError:
        pass
