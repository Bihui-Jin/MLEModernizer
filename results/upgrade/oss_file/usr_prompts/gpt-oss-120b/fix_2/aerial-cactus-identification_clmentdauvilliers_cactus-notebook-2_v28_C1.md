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
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import os
from zipfile import ZipFile
from tensorflow.keras.preprocessing.image import ImageDataGenerator

for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames:
        print(os.path.join(dirname, filename))




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
path = "/kaggle/input/aerial-cactus-identification/"
files_dataframe = pd.read_csv(path + "train.csv", dtype=str)
files_dataframe.head()




## === cell 2
os.makedirs("./training", exist_ok=True)
os.makedirs("./test", exist_ok=True)

with ZipFile(path + "train.zip", "r") as zipper:
    zipper.extractall("./training/")  # creates ./training/train/ inside the zip

with ZipFile(path + "test.zip", "r") as zipper:
    zipper.extractall("./test/")  # creates ./test/test/ inside the zip




## === cell 3
class_reparts = files_dataframe["has_cactus"].value_counts()




## === cell 4
total_samples = files_dataframe["has_cactus"].size
has_cactus_weight = total_samples / (2 * class_reparts["1"])
no_cactus_weight = total_samples / (2 * class_reparts["0"])
class_weights = {0: no_cactus_weight, 1: has_cactus_weight}
print("Class weights:", class_weights)




## === cell 5
try:
    import matplotlib.pyplot as plt
    from matplotlib.image import imread

    plt.figure(figsize=(12, 12))
    for i, k in enumerate(np.random.randint(0, len(files_dataframe), size=9)):
        plt.subplot(3, 3, i + 1)
        img_path = "./training/train/" + files_dataframe.iloc[k]["id"]
        img = imread(img_path)
        plt.imshow(img)
        plt.title("Label: " + str(files_dataframe["has_cactus"].iloc[k]))
    plt.show()
except Exception as e:
    print("Visualization skipped:", e)




## === cell 6
import skimage.exposure as exposure


def preprocess(img):
    p2, p98 = np.percentile(img, (3, 97))
    return exposure.rescale_intensity(img, in_range=(p2, p98))




## === cell 7
generator = ImageDataGenerator(
    samplewise_center=True,
    samplewise_std_normalization=True,
    preprocessing_function=preprocess,
    validation_split=0.1,
    horizontal_flip=True,
    vertical_flip=True,
    rotation_range=45,
)

noAugmentationGenerator = ImageDataGenerator(
    samplewise_center=True,
    samplewise_std_normalization=True,
    preprocessing_function=preprocess,
)




## === cell 8
train_gen = generator.flow_from_dataframe(
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

val_gen = generator.flow_from_dataframe(
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




## === cell 9
import tensorflow as tf
from tensorflow.keras import layers, models

model = models.Sequential(
    [
        layers.Flatten(input_shape=(32, 32, 3)),
        layers.Dense(256, activation="relu"),
        layers.Dense(256, activation="relu"),
        layers.Dropout(0.1),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.1),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.1),
        layers.Dense(64, activation="relu"),
        layers.Dense(2, activation="softmax"),  # two classes
    ]
)

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
model.summary()




## === cell 10
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint

reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=3, min_lr=1e-5)

checkpoint_path = "/tmp/best_model.keras"
model_checkpoint = ModelCheckpoint(
    checkpoint_path,
    monitor="val_loss",
    mode="min",
    save_best_only=True,
    save_weights_only=False,
    verbose=1,
)




## === cell 11
history = model.fit(
    train_gen,
    validation_data=val_gen,
    steps_per_epoch=train_gen.n // train_gen.batch_size,
    validation_steps=val_gen.n // val_gen.batch_size,
    epochs=5,
    class_weight=class_weights,
    callbacks=[reduce_lr, model_checkpoint],
    verbose=1,
)




## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_55/3786750693.py in <cell line: 0>()
      1 # Train for a few epochs (increase if time permits)
----> 2 history = model.fit(
      3     train_gen,
      4     validation_data=val_gen,
      5     steps_per_epoch=train_gen.n // train_gen.batch_size,

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

## === cell 12
model = tf.keras.models.load_model(checkpoint_path)




## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_55/661559714.py in <cell line: 0>()
      1 # Load the best checkpoint (optional – model is already trained)
----> 2 model = tf.keras.models.load_model(checkpoint_path)
      3 
      4 

/usr/local/lib/python3.11/dist-packages/keras/src/saving/saving_api.py in load_model(filepath, custom_objects, compile, safe_mode)
    198         )
    199     elif str(filepath).endswith(".keras"):
--> 200         raise ValueError(
    201             f"File not found: filepath={filepath}. "
    202             "Please ensure the file is an accessible `.keras` "

ValueError: File not found: filepath=/tmp/best_model.keras. Please ensure the file is an accessible `.keras` zip file.

## === cell 13
test_gen = noAugmentationGenerator.flow_from_directory(
    directory="./test/",
    target_size=(32, 32),
    batch_size=32,
    class_mode=None,
    shuffle=False,
)




## === cell 14
probas = model.predict(test_gen, verbose=0)
has_cactus_prob = probas[:, 1]  # probability of class 1




## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_55/2647234367.py in <cell line: 0>()
      1 # Predict probabilities for the positive class (index 1)
----> 2 probas = model.predict(test_gen, verbose=0)
      3 has_cactus_prob = probas[:, 1]  # probability of class 1
      4 
      5 

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

## === cell 15
filenames = [os.path.basename(path) for path in test_gen.filenames]
submission = pd.DataFrame({"id": filenames, "has_cactus": has_cactus_prob})
submission.to_csv("submission.csv", index=False)
print("Submission saved to submission.csv, shape:", submission.shape)




## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/131289994.py in <cell line: 0>()
      1 # Build submission dataframe – filenames are returned relative to the directory
      2 filenames = [os.path.basename(path) for path in test_gen.filenames]
----> 3 submission = pd.DataFrame({"id": filenames, "has_cactus": has_cactus_prob})
      4 submission.to_csv("submission.csv", index=False)
      5 print("Submission saved to submission.csv, shape:", submission.shape)

NameError: name 'has_cactus_prob' is not defined

## === cell 16
import shutil

shutil.rmtree("./training", ignore_errors=True)
shutil.rmtree("./test", ignore_errors=True)
