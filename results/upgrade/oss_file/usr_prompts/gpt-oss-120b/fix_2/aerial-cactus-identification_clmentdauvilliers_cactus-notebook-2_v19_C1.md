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

0.9438

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
import matplotlib.pyplot as plt
from zipfile import ZipFile
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import skimage.exposure as exposure
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
base_path = "/kaggle/input/aerial-cactus-identification/"

train_df = pd.read_csv(os.path.join(base_path, "train.csv"), dtype=str)

total = len(train_df)
counts = train_df["has_cactus"].value_counts()
class_weights = {
    0: total / (2 * counts.get("0", 1)),
    1: total / (2 * counts.get("1", 1)),
}
print("Class weights:", class_weights)



## === cell 2
os.makedirs("./training", exist_ok=True)
os.makedirs("./test", exist_ok=True)

with ZipFile(os.path.join(base_path, "train.zip"), "r") as z:
    z.extractall("./training/")

with ZipFile(os.path.join(base_path, "test.zip"), "r") as z:
    z.extractall("./test/")




## === cell 3
def preprocess(img):
    """Simple contrast stretching."""
    p2, p98 = np.percentile(img, (3, 97))
    return exposure.rescale_intensity(img, in_range=(p2, p98))




## === cell 4
train_gen = ImageDataGenerator(
    samplewise_center=True,
    samplewise_std_normalization=True,
    vertical_flip=True,
    horizontal_flip=True,
    rotation_range=45,
    shear_range=10,
    preprocessing_function=preprocess,
    validation_split=0.25,
)

val_gen = ImageDataGenerator(
    samplewise_center=True,
    samplewise_std_normalization=True,
    preprocessing_function=preprocess,
    validation_split=0.25,
)



## === cell 5
training_generator = train_gen.flow_from_dataframe(
    dataframe=train_df,
    directory="./training/train/",
    x_col="id",
    y_col="has_cactus",
    class_mode="categorical",
    target_size=(32, 32),
    batch_size=32,
    subset="training",
    shuffle=True,
)

validation_generator = val_gen.flow_from_dataframe(
    dataframe=train_df,
    directory="./training/train/",
    x_col="id",
    y_col="has_cactus",
    class_mode="categorical",
    target_size=(32, 32),
    batch_size=32,
    subset="validation",
    shuffle=False,
)



## === cell 6
model = models.Sequential(
    [
        layers.Conv2D(32, (5, 5), activation="relu", input_shape=(32, 32, 3)),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.2),
        layers.Conv2D(64, (3, 3), activation="relu"),
        layers.Conv2D(64, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.2),
        layers.Conv2D(128, (3, 3), activation="relu"),
        layers.Conv2D(128, (3, 3), activation="relu"),
        layers.Dropout(0.2),
        layers.Flatten(),
        layers.Dense(256, activation="relu"),
        layers.Dense(256, activation="relu"),
        layers.Dropout(0.1),
        layers.Dense(128, activation="relu"),
        layers.Dense(2, activation="softmax"),
    ]
)

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

model.summary()



## === cell 7
reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=5, min_lr=0.001)

checkpoint_path = "/tmp/best_model.keras"
save_best_model = ModelCheckpoint(
    filepath=checkpoint_path,
    monitor="val_accuracy",
    mode="max",
    save_best_only=True,
    save_weights_only=False,
)



## === cell 8
history = model.fit(
    training_generator,
    validation_data=validation_generator,
    epochs=10,
    callbacks=[reduce_lr, save_best_model],
    verbose=1,
)



## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_55/3101149864.py in <cell line: 0>()
----> 1 history = model.fit(
      2     training_generator,
      3     validation_data=validation_generator,
      4     epochs=10,
      5     callbacks=[reduce_lr, save_best_model],

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

## === cell 9
test_gen = ImageDataGenerator(
    samplewise_center=True,
    samplewise_std_normalization=True,
    preprocessing_function=preprocess,
).flow_from_directory(
    directory="./test/",
    target_size=(32, 32),
    batch_size=32,
    class_mode=None,
    shuffle=False,
)



## === cell 10
best_model = tf.keras.models.load_model(checkpoint_path)

pred_probs = best_model.predict(test_gen, verbose=0)[:, 1]

ids = [os.path.basename(p) for p in test_gen.filenames]
submission = pd.DataFrame({"id": ids, "has_cactus": pred_probs})
submission.to_csv("submission.csv", index=False)
print("Submission saved to submission.csv")



## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_55/536168084.py in <cell line: 0>()
      1 # load best model (optional – ensures we use the checkpoint)
----> 2 best_model = tf.keras.models.load_model(checkpoint_path)
      3 
      4 # predict probabilities for the cactus class (index 1)
      5 pred_probs = best_model.predict(test_gen, verbose=0)[:, 1]

/usr/local/lib/python3.11/dist-packages/keras/src/saving/saving_api.py in load_model(filepath, custom_objects, compile, safe_mode)
    198         )
    199     elif str(filepath).endswith(".keras"):
--> 200         raise ValueError(
    201             f"File not found: filepath={filepath}. "
    202             "Please ensure the file is an accessible `.keras` "

ValueError: File not found: filepath=/tmp/best_model.keras. Please ensure the file is an accessible `.keras` zip file.

## === cell 11
import shutil

for folder in ["training", "test"]:
    try:
        shutil.rmtree(folder)
    except OSError:
        pass
