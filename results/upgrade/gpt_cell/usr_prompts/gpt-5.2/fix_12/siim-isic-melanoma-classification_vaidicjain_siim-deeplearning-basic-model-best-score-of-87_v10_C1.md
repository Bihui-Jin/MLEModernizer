# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


# 1. Kaggle task description

## Task
Predict whether a lesion is malignant (0 denotes **benign**, and 1 indicates **malignant**).

## Metric
Area under the ROC curve.

## Submission Format
For each `image_name` in the test set, you must predict the probability (`target`) that the sample is **malignant**. The file should contain a header and have the following format:

```
image_name,target
ISIC_0052060,0.7
ISIC_0052349,0.9
ISIC_0058510,0.8
ISIC_0073313,0.5
ISIC_0073502,0.5
etc.
```

## Dataset 
The images are provided in DICOM format.

Images are also provided in JPEG and TFRecord format (in the `jpeg` and `tfrecords` directories, respectively). Images in TFRecord format have been resized to a uniform 1024x1024.

Metadata is also provided outside of the DICOM format, in CSV files. See the `Columns` section for a description.

### Files
- **train.csv** - the training set
- **test.csv** - the test set
- **sample_submission.csv** - a sample submission file in the correct format

### Columns
- `image_name` - unique identifier, points to filename of related DICOM image
- `patient_id` - unique patient identifier
- `sex` - the sex of the patient (when unknown, will be blank)
- `age_approx` - approximate patient age at time of imaging
- `anatom_site_general_challenge` - location of imaged site
- `diagnosis` - detailed diagnosis information (train only)
- `benign_malignant` - indicator of malignancy of imaged lesion
- `target` - binarized version of the target variable

# 2. Python version

3.8

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
            description.md (176 lines)
            jpeg.zip (24.7 GB)
            sample_submission.csv (4143 lines)
            sample_submission.csv.zip (16.4 kB)
            test.csv (4143 lines)
            test.csv.zip (42.5 kB)
            test.zip (6.4 GB)
            tfrecords.zip (9.3 GB)
            train.csv (28985 lines)
            train.csv.zip (299.7 kB)
            train.zip (46.0 GB)
            jpeg/
                test/
                    ISIC_1440063.jpg (1.1 MB)
                    ISIC_0815802.jpg (853.1 kB)
                    ... and 4140 other files
                train/
                    ISIC_1845271.jpg (1.0 MB)
                    ISIC_1970027.jpg (138.4 kB)
                    ... and 28982 other files
            siim-isic-melanoma-classification/
                description.md (176 lines)
                jpeg.zip (24.7 GB)
                ... and 9 other files
                jpeg/
                    test/
                        ISIC_1440063.jpg (1.1 MB)
                        ISIC_0815802.jpg (853.1 kB)
                        ... and 4140 other files
                    train/
                        ISIC_1845271.jpg (1.0 MB)
                        ISIC_1970027.jpg (138.4 kB)
                        ... and 28982 other files
                siim-isic-melanoma-classification/
                test/
                    ISIC_0052212.dcm (1.5 MB)
                    ISIC_0076545.dcm (4.0 MB)
                    ... and 4140 other files
                    test/
                tfrecords/
                    test00-2071.tfrec (579.6 MB)
                    test01-2071.tfrec (583.5 MB)
                    ... and 14 other files
                train/
                    ISIC_0015719.dcm (2.4 MB)
                    ISIC_0068279.dcm (1.3 MB)
                    ... and 28982 other files
                    train/
            test/
                ISIC_0052212.dcm (1.5 MB)
                ISIC_0076545.dcm (4.0 MB)
                ... and 4140 other files
                test/
            tfrecords/
                test00-2071.tfrec (579.6 MB)
                test01-2071.tfrec (583.5 MB)
                ... and 14 other files
            train/
                ISIC_0015719.dcm (2.4 MB)
                ISIC_0068279.dcm (1.3 MB)
                ... and 28982 other files
                train/
        input/
            description.md (176 lines)
            jpeg.zip (24.7 GB)
            sample_submission.csv (4143 lines)
            sample_submission.csv.zip (16.4 kB)
            test.csv (4143 lines)
            test.csv.zip (42.5 kB)
            test.zip (6.4 GB)
            tfrecords.zip (9.3 GB)
            train.csv (28985 lines)
            train.csv.zip (299.7 kB)
            train.zip (46.0 GB)
            jpeg/
                test/
                    ISIC_1440063.jpg (1.1 MB)
                    ISIC_0815802.jpg (853.1 kB)
                    ... and 4140 other files
                train/
                    ISIC_1845271.jpg (1.0 MB)
                    ISIC_1970027.jpg (138.4 kB)
                    ... and 28982 other files
            siim-isic-melanoma-classification/
                description.md (176 lines)
                jpeg.zip (24.7 GB)
                ... and 9 other files
                jpeg/
                    test/
                        ISIC_1440063.jpg (1.1 MB)
                        ISIC_0815802.jpg (853.1 kB)
                        ... and 4140 other files
                    train/
                        ISIC_1845271.jpg (1.0 MB)
                        ISIC_1970027.jpg (138.4 kB)
                        ... and 28982 other files
                siim-isic-melanoma-classification/
                test/
                    ISIC_0052212.dcm (1.5 MB)
                    ISIC_0076545.dcm (4.0 MB)
                    ... and 4140 other files
                    test/
                tfrecords/
                    test00-2071.tfrec (579.6 MB)
                    test01-2071.tfrec (583.5 MB)
                    ... and 14 other files
                train/
                    ISIC_0015719.dcm (2.4 MB)
                    ISIC_0068279.dcm (1.3 MB)
                    ... and 28982 other files
                    train/
            test/
                ISIC_0052212.dcm (1.5 MB)
                ISIC_0076545.dcm (4.0 MB)
                ... and 4140 other files
                test/
                    ISIC_0052212.dcm (1.5 MB)
                    ISIC_0076545.dcm (4.0 MB)
                    ... and 4140 other files
                    test/
            tfrecords/
                test00-2071.tfrec (579.6 MB)
                test01-2071.tfrec (583.5 MB)
                ... and 14 other files
            train/
                ISIC_0015719.dcm (2.4 MB)
                ISIC_0068279.dcm (1.3 MB)
                ... and 28982 other files
                train/
                    ISIC_0015719.dcm (2.4 MB)
                    ISIC_0068279.dcm (1.3 MB)
                    ... and 28982 other files
                    train/
        working/
            siim-isic-melanoma-classification/
                description.md (176 lines)
                jpeg.zip (24.7 GB)
                ... and 9 other files
                jpeg/
                    test/
                        ISIC_1440063.jpg (1.1 MB)
                        ISIC_0815802.jpg (853.1 kB)
                        ... and 4140 other files
                    train/
                        ISIC_1845271.jpg (1.0 MB)
                        ISIC_1970027.jpg (138.4 kB)
                        ... and 28982 other files
                siim-isic-melanoma-classification/
                test/
                    ISIC_0052212.dcm (1.5 MB)
                    ISIC_0076545.dcm (4.0 MB)
                    ... and 4140 other files
                    test/
                tfrecords/
                    test00-2071.tfrec (579.6 MB)
                    test01-2071.tfrec (583.5 MB)
                    ... and 14 other files
                train/
                    ISIC_0015719.dcm (2.4 MB)
                    ISIC_0068279.dcm (1.3 MB)
                    ... and 28982 other files
                    train/
```

-> data/sample_submission.csv has 4142 rows and 2 columns.
The columns are: image_name, target

-> data/siim-isic-melanoma-classification/sample_submission.csv has 4142 rows and 2 columns.
The columns are: image_name, target

-> data/siim-isic-melanoma-classification/test.csv has 4142 rows and 5 columns.
The columns are: image_name, patient_id, sex, age_approx, anatom_site_general_challenge

-> data/siim-isic-melanoma-classification/train.csv has 28984 rows and 8 columns.
The columns are: image_name, patient_id, sex, age_approx, anatom_site_general_challenge, diagnosis, benign_malignant, target

-> data/test.csv has 4142 rows and 5 columns.
The columns are: image_name, patient_id, sex, age_approx, anatom_site_general_challenge

-> data/train.csv has 28984 rows and 8 columns.
The columns are: image_name, patient_id, sex, age_approx, anatom_site_general_challenge, diagnosis, benign_malignant, target

-> input/sample_submission.csv has 4142 rows and 2 columns.
The columns are: image_name, target

-> (stopped after 10 files for performance)

# 5. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", None)

try:
    import google.protobuf  # noqa: F401
    from google.protobuf import __version__ as _pb_version
except Exception:
    _pb_version = None


def _pb_major(v):
    try:
        return int(str(v).split(".", 1)[0])
    except Exception:
        return None


if _pb_major(_pb_version) is None or _pb_major(_pb_version) >= 4:
    import sys
    import subprocess

    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-q", "protobuf==3.20.*"]
    )

import pandas as pd
import numpy as np
import tensorflow as tf


class _EFNShim:
    EfficientNetB0 = tf.keras.applications.EfficientNetB0
    EfficientNetB1 = tf.keras.applications.EfficientNetB1
    EfficientNetB2 = tf.keras.applications.EfficientNetB2
    EfficientNetB3 = tf.keras.applications.EfficientNetB3
    EfficientNetB4 = tf.keras.applications.EfficientNetB4
    EfficientNetB5 = tf.keras.applications.EfficientNetB5
    EfficientNetB6 = tf.keras.applications.EfficientNetB6
    EfficientNetB7 = tf.keras.applications.EfficientNetB7


efn = _EFNShim()

from keras.metrics import *
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input

try:
    from kaggle_datasets import KaggleDatasets
except Exception:
    KaggleDatasets = None

from tensorflow.keras.preprocessing.image import ImageDataGenerator


## === cell 1
try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
    print('Running on TPU ', tpu.master())
except ValueError:
    tpu = None

if tpu:
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.experimental.TPUStrategy(tpu)
else:
    strategy = tf.distribute.get_strategy()

print("REPLICAS: ", strategy.num_replicas_in_sync)


## === cell 2
PATH = '/kaggle/input/siim-isic-melanoma-classification'
train_images_dir = PATH+'/jpeg/train/'
test_images_dir = PATH+'/jpeg/test/'
train_csv = PATH+'/train.csv'
test_csv  = PATH+'/test.csv'


## === cell 3
train_df = pd.read_csv(train_csv)
test_df = pd.read_csv(test_csv)


train_df['image_name'] = train_df['image_name'] + '.jpg'
test_df['image_name'] = test_df['image_name'] + '.jpg'

train_df = train_df.sort_values(['patient_id'])
train_df = train_df.reset_index()

train_df = train_df.sort_values(['target'])
train_df = train_df.reset_index()

train_df = train_df.drop(['index', 'patient_id', 'diagnosis', 'benign_malignant'], axis = 1)
test_df = test_df.drop(['patient_id'], axis = 1)

train_df['anatom_site_general_challenge'] = train_df['anatom_site_general_challenge'].replace(np.nan, 'torso')
test_df['anatom_site_general_challenge'] = test_df['anatom_site_general_challenge'].replace(np.nan, 'torso')

train_df = train_df.dropna()

test_df['age_approx'] = test_df['age_approx'] / train_df['age_approx'].mean()
train_df['age_approx'] = train_df['age_approx'] / train_df['age_approx'].mean()

train_df['sex'] = train_df['sex'].replace('female', 0)
test_df['sex'] = test_df['sex'].replace('female', 0)
train_df['sex'] = train_df['sex'].replace('male', 1)
test_df['sex'] = test_df['sex'].replace('male', 1)

train_df['target'] = train_df['target'].replace(0,'0')
train_df['target'] = train_df['target'].replace(1,'1')

train_df['anatom_site_general_challenge'] = pd.Categorical(train_df['anatom_site_general_challenge'])
train_df['anatom_site_general_challenge'] = train_df.anatom_site_general_challenge.cat.codes
test_df['anatom_site_general_challenge'] = pd.Categorical(test_df['anatom_site_general_challenge'])
test_df['anatom_site_general_challenge'] = test_df.anatom_site_general_challenge.cat.codes



print(train_df)
test_df


## === cell 4
val_split = 0.1


## === cell 5
val = train_df[24000:25000]
val = pd.concat([val, train_df[33057:]])
train = train_df[25000:33058]


## === cell 7
target_size=(128, 128)
batch_size=32


## === cell 8
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255.0,
    width_shift_range=0.15,
    height_shift_range=0.15,
    horizontal_flip=True,
    brightness_range=[0.5, 1.5],
)
val_datagen = ImageDataGenerator(rescale=1.0 / 255.0)

train_generator = train_datagen.flow_from_dataframe(
    dataframe=train,
    directory=train_images_dir,
    x_col="image_name",
    y_col="target",
    classes=["0", "1"],
    class_mode="binary",
    color_mode="rgb",
    target_size=target_size,
    batch_size=batch_size,
)
val_generator = val_datagen.flow_from_dataframe(
    dataframe=val,
    directory=train_images_dir,
    x_col="image_name",
    y_col="target",
    classes=["0", "1"],
    class_mode="binary",
    color_mode="rgb",
    target_size=target_size,
    batch_size=batch_size,
)


## === cell 9
with strategy.scope():
    base_model = tf.keras.applications.EfficientNetB0(
        include_top=False,
        input_shape=(128, 128, 3),
        weights="imagenet",
    )

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(16, activation="relu")(x)
    x = Dense(1, activation="sigmoid")(x)

    model = Model(inputs=base_model.input, outputs=x)

    for layer in base_model.layers:
        layer.trainable = False

    METRICS = [
        BinaryAccuracy(name="accuracy"),
        AUC(name="auc"),
    ]
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=[METRICS])


## === cell 10
with strategy.scope():
    base_model = tf.keras.applications.EfficientNetB0(
        include_top=False,
        input_shape=(128, 128, 3),
        weights="imagenet",
    )

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(16, activation="relu")(x)
    x = Dense(1, activation="sigmoid")(x)

    model = Model(inputs=base_model.input, outputs=x)

    for layer in base_model.layers:
        layer.trainable = False

    METRICS = [
        BinaryAccuracy(name="accuracy"),
        AUC(name="auc"),
    ]
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=METRICS)

model.fit(
    x=train_generator, epochs=6, validation_data=val_generator
)  # , class_weight = weights)


## === cell 11
timage_path = test_images_dir + test_df['image_name']
timage_path = tf.convert_to_tensor(timage_path, dtype=tf.string)
tmetadata = test_df[['sex','age_approx','anatom_site_general_challenge']]
tmetadata = np.array(tmetadata)
tmetadata = tf.convert_to_tensor(tmetadata)

np.shape(timage_path)


## === cell 12
def tmap_fn(path,metadata):
    
    image = tf.image.decode_jpeg(tf.io.read_file(path))

    image = tf.image.resize(image, [128, 128])
    
    image = tf.cast(image, tf.float32) / 255.0
    return (image, metadata)

tdataset = tf.data.Dataset.from_tensor_slices((timage_path, tmetadata))
tdataset = tdataset.map(tmap_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)
tdataset = tdataset.batch(10982)
tdataset = tdataset.prefetch(tf.data.experimental.AUTOTUNE)
timages, tmetadatas = tf.compat.v1.data.make_one_shot_iterator(tdataset).get_next()

test_labels = model.predict([timages])
test_df = pd.read_csv(test_csv)

np.shape(test_labels)
test_df = test_df.reset_index()
df_sub = pd.DataFrame()
df_sub['image_name'] = test_df['image_name']
df_sub['target'] = test_labels.astype(np.float32)
df_sub.to_csv('submission.csv', index=False)
