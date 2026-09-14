# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


# 1. Kaggle task description

## Task
Create a classifier to predict the severity of diabetic retinopathy.

## Metric
Quadratic weighted kappa, which measures the agreement between two ratings. This metric typically varies from 0 (random agreement between raters) to 1 (complete agreement between raters). In the event that there is less agreement between the raters than expected by chance, this metric may go below 0. The quadratic weighted kappa is calculated between the scores assigned by the human rater and the predicted scores.

Images have five possible ratings, 0,1,2,3,4.  Each image is characterized by a tuple *(e*,*e)*, which corresponds to its scores by *Rater A* (human) and *Rater B* (predicted).  The quadratic weighted kappa is calculated as follows. First, an N x N histogram matrix *O* is constructed, such that *O* corresponds to the number of images that received a rating *i* by *A* and a rating *j* by *B*. An *N-by-N* matrix of weights, *w*, is calculated based on the difference between raters' scores:

An *N-by-N* histogram matrix of expected ratings, *E*, is calculated, assuming that there is no correlation between rating scores.  This is calculated as the outer product between each rater's histogram vector of ratings, normalized such that *E* and *O* have the same sum.

## Submission Format
```
id_code,diagnosis
0005cfc8afb6,0
003f0afdcd15,0
etc.
```

## Dataset
You are provided with a large set of retina images taken using [fundus photography](https://en.wikipedia.org/wiki/Fundus_photography) under a variety of imaging conditions.

Labels are on a scale of 0 to 4:

> 0 - No DR
> 1 - Mild
> 2 - Moderate
> 3 - Severe
> 4 - Proliferative DR

Images may contain artifacts, be out of focus, underexposed, or overexposed. The images were gathered from multiple clinics using a variety of cameras over an extended period of time, which will introduce further variation.

- **train.csv** - the training labels
- **test.csv** - the test set (you must predict the `diagnosis` value for these variables)
- **sample_submission.csv** - a sample submission file in the correct format
- **train.zip** - the training set images
- **test.zip** - the public test set images

# 2. Python version

3.7

# 3. Installed packages

geopandas==0.14.4
keras==3.8.0
keras-core==0.1.7
keras-cv==0.9.0
keras-hub==0.18.1
keras-nlp==0.18.1
keras-tuner==1.4.7
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
protobuf==6.33.0
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
            description.md (118 lines)
            sample_submission.csv (368 lines)
            sample_submission.csv.zip (3.2 kB)
            test.csv (368 lines)
            test.csv.zip (2.9 kB)
            test.zip (160 Bytes)
            test_images.zip (902.9 MB)
            train.csv (3296 lines)
            train.csv.zip (27.5 kB)
            train.zip (162 Bytes)
            train_images.zip (7.7 GB)
            aptos2019-blindness-detection/
                description.md (118 lines)
                sample_submission.csv (368 lines)
                ... and 9 other files
                aptos2019-blindness-detection/
                test_images/
                    218c822a3dd9.png (5.7 MB)
                    0e82bcacc475.png (5.2 MB)
                    ... and 365 other files
                    test_images/
                train_images/
                    184a185e7447.png (337.5 kB)
                    c4aef0d88d1b.png (876.6 kB)
                    ... and 3293 other files
                    train_images/
            test_images/
                218c822a3dd9.png (5.7 MB)
                0e82bcacc475.png (5.2 MB)
                ... and 365 other files
                test_images/
            train_images/
                184a185e7447.png (337.5 kB)
                c4aef0d88d1b.png (876.6 kB)
                ... and 3293 other files
                train_images/
        input/
            description.md (118 lines)
            sample_submission.csv (368 lines)
            sample_submission.csv.zip (3.2 kB)
            test.csv (368 lines)
            test.csv.zip (2.9 kB)
            test.zip (160 Bytes)
            test_images.zip (902.9 MB)
            train.csv (3296 lines)
            train.csv.zip (27.5 kB)
            train.zip (162 Bytes)
            train_images.zip (7.7 GB)
            aptos2019-blindness-detection/
                description.md (118 lines)
                sample_submission.csv (368 lines)
                ... and 9 other files
                aptos2019-blindness-detection/
                test_images/
                    218c822a3dd9.png (5.7 MB)
                    0e82bcacc475.png (5.2 MB)
                    ... and 365 other files
                    test_images/
                train_images/
                    184a185e7447.png (337.5 kB)
                    c4aef0d88d1b.png (876.6 kB)
                    ... and 3293 other files
                    train_images/
            test_images/
                218c822a3dd9.png (5.7 MB)
                0e82bcacc475.png (5.2 MB)
                ... and 365 other files
                test_images/
                    218c822a3dd9.png (5.7 MB)
                    0e82bcacc475.png (5.2 MB)
                    ... and 365 other files
                    test_images/
            train_images/
                184a185e7447.png (337.5 kB)
                c4aef0d88d1b.png (876.6 kB)
                ... and 3293 other files
                train_images/
                    184a185e7447.png (337.5 kB)
                    c4aef0d88d1b.png (876.6 kB)
                    ... and 3293 other files
                    train_images/
        working/
            aptos2019-blindness-detection/
                description.md (118 lines)
                sample_submission.csv (368 lines)
                ... and 9 other files
                aptos2019-blindness-detection/
                test_images/
                    218c822a3dd9.png (5.7 MB)
                    0e82bcacc475.png (5.2 MB)
                    ... and 365 other files
                    test_images/
                train_images/
                    184a185e7447.png (337.5 kB)
                    c4aef0d88d1b.png (876.6 kB)
                    ... and 3293 other files
                    train_images/
```

-> data/aptos2019-blindness-detection/sample_submission.csv has 367 rows and 2 columns.
The columns are: id_code, diagnosis

-> data/aptos2019-blindness-detection/test.csv has 367 rows and 1 columns.
The columns are: id_code

-> data/aptos2019-blindness-detection/train.csv has 3295 rows and 2 columns.
The columns are: id_code, diagnosis

-> data/sample_submission.csv has 367 rows and 2 columns.
The columns are: id_code, diagnosis

-> data/test.csv has 367 rows and 1 columns.
The columns are: id_code

-> data/train.csv has 3295 rows and 2 columns.
The columns are: id_code, diagnosis

-> input/aptos2019-blindness-detection/sample_submission.csv has 367 rows and 2 columns.
The columns are: id_code, diagnosis

-> (stopped after 10 files for performance)

# 5. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import os

print(os.listdir("../input"))

train_df = pd.read_csv("../input/train.csv")
print("Shape of train data: {0}".format(train_df.shape))
test_df = pd.read_csv("../input/test.csv")
print("Shape of test data: {0}".format(test_df.shape))

diagnosis_df = pd.DataFrame(
    {
        "diagnosis": [0, 1, 2, 3, 4],
        "diagnosis_label": ["No DR", "Mild", "Moderate", "Severe", "Proliferative DR"],
    }
)

train_df = train_df.merge(diagnosis_df, how="left", on="diagnosis")

train_image_files = [
    os.path.join(dp, f)
    for dp, dn, fn in os.walk(os.path.expanduser("../input/train_images"))
    for f in fn
]
train_images_df = pd.DataFrame(
    {
        "files": train_image_files,
        "id_code": [
            os.path.splitext(os.path.basename(file))[0] for file in train_image_files
        ],
    }
)
train_df = train_df.merge(train_images_df, how="left", on="id_code")
del train_images_df
print("Shape of train data after merge: {0}".format(train_df.shape))

test_image_files = [
    os.path.join(dp, f)
    for dp, dn, fn in os.walk(os.path.expanduser("../input/test_images"))
    for f in fn
]
test_images_df = pd.DataFrame(
    {
        "files": test_image_files,
        "id_code": [
            os.path.splitext(os.path.basename(file))[0] for file in test_image_files
        ],
    }
)

test_df = test_df.merge(test_images_df, how="left", on="id_code")
del test_images_df
print("Shape of test data after merge: {0}".format(test_df.shape))




## === cell 1
train_df.head()




## === cell 2
test_df.head()




## === cell 3
IMG_SIZE = 150
N_CLASSES = train_df.diagnosis.nunique()
CLASSES = list(map(str, range(N_CLASSES)))
BATCH_SIZE = 32
EPOCHS = 5  # increased epochs for better performance

import tensorflow as tf

print(tf.__version__)

from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_df["diagnosis"] = train_df["diagnosis"].astype(str)

train_data_gen = ImageDataGenerator(rescale=1.0 / 255, validation_split=0.3)

train_data = train_data_gen.flow_from_dataframe(
    dataframe=train_df,
    x_col="files",
    y_col="diagnosis",
    batch_size=BATCH_SIZE,
    shuffle=True,
    classes=CLASSES,
    class_mode="sparse",
    target_size=(IMG_SIZE, IMG_SIZE),
    subset="training",
)

validation_data = train_data_gen.flow_from_dataframe(
    dataframe=train_df,
    x_col="files",
    y_col="diagnosis",
    batch_size=BATCH_SIZE,
    shuffle=True,
    classes=CLASSES,
    class_mode="sparse",
    target_size=(IMG_SIZE, IMG_SIZE),
    subset="validation",
)

test_data_gen = ImageDataGenerator(rescale=1.0 / 255)
test_data = test_data_gen.flow_from_dataframe(
    dataframe=test_df,
    x_col="files",
    batch_size=1,
    shuffle=False,
    class_mode=None,
    target_size=(IMG_SIZE, IMG_SIZE),
)




## === cell 4
model = tf.keras.models.Sequential(
    [
        tf.keras.layers.Conv2D(
            64, (3, 3), activation="relu", input_shape=(IMG_SIZE, IMG_SIZE, 3)
        ),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(1024, activation="relu"),
        tf.keras.layers.Dense(512, activation="relu"),
        tf.keras.layers.Dense(256, activation="relu"),
        tf.keras.layers.Dense(N_CLASSES, activation="softmax"),
    ]
)

opt = tf.keras.optimizers.Adam(learning_rate=0.001, epsilon=1e-6)
model.compile(
    optimizer=opt, loss="sparse_categorical_crossentropy", metrics=["accuracy"]
)
model.summary()




## === cell 5
model.fit(
    train_data,
    epochs=EPOCHS,
    validation_data=validation_data,
)




## === cell 6
predictions = model.predict(test_data, steps=len(test_data), verbose=0)




## === cell 7
pred_classes = np.argmax(predictions, axis=1)

results = pd.DataFrame(
    {"id_code": test_df["id_code"].values, "diagnosis": pred_classes}
)

results.head()




## === cell 8
results.to_csv("submission.csv", index=False)
print("Submission saved to submission.csv")
