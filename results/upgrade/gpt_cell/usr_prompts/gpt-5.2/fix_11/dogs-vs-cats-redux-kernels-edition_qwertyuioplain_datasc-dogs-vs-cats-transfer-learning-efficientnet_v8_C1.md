# Goal

I want you to improve my Kaggle competition solution to increase the score toward a target. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (evaluation score improvement); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Ensure it runs end-to-end and produces a valid submission file.


# 1. Kaggle task description

## Task
Given a dataset of images of dogs and cats, predict if an image is a dog or a cat.

## Metric
Log loss.

## Submission Format
For each image in the test set, you must submit a probability that image is a dog. The file should have a header and be in the following format:

```
id,label
1,0.5
2,0.5
3,0.5
...
```

## Dataset
The train folder contains 25,000 images of dogs and cats. Each image in this folder has the label as part of the filename. The test folder contains 12,500 images, named according to a numeric id.

# 2. Python version

3.10

# 3. Installed packages

geopandas==0.14.4
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
pytorch-ignite==0.5.3
pytorch-lightning==2.5.5
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
torch==2.6.0+cu124
torchao==0.10.0
torchaudio==2.6.0+cu124
torchdata==0.11.0
torchinfo==1.8.0
torchmetrics==1.8.2
torchsummary==1.5.1
torchtune==0.6.1
torchvision==0.21.0+cu124

# 4. Data file paths

```
/
    kaggle/
        data/
            cat.1714.jpg (7.8 kB)
            cat.10025.jpg (18.4 kB)
            ... and 24998 other files
            description.md (50 lines)
            sample_submission.csv (2501 lines)
            sample_submission.csv.zip (6.0 kB)
            test.zip (56.6 MB)
            train.zip (513.0 MB)
            dogs-vs-cats-redux-kernels-edition/
                cat.1714.jpg (7.8 kB)
                cat.10025.jpg (18.4 kB)
                ... and 24998 other files
                description.md (50 lines)
                sample_submission.csv (2501 lines)
                sample_submission.csv.zip (6.0 kB)
                test.zip (56.6 MB)
                train.zip (513.0 MB)
                dogs-vs-cats-redux-kernels-edition/
                test/
                    test/
                    unknown/
                        900.jpg (42.3 kB)
                        572.jpg (30.6 kB)
                        ... and 2498 other files
                train/
                    cat/
                        cat.4838.jpg (20.2 kB)
                        cat.1314.jpg (21.7 kB)
                        ... and 11240 other files
                    dog/
                        dog.6712.jpg (35.3 kB)
                        dog.7152.jpg (36.1 kB)
                        ... and 11256 other files
                    train/
            test/
                test/
                unknown/
                    900.jpg (42.3 kB)
                    572.jpg (30.6 kB)
                    ... and 2498 other files
            train/
                cat/
                    cat.4838.jpg (20.2 kB)
                    cat.1314.jpg (21.7 kB)
                    ... and 11240 other files
                dog/
                    dog.6712.jpg (35.3 kB)
                    dog.7152.jpg (36.1 kB)
                    ... and 11256 other files
                train/
        input/
            cat.1714.jpg (7.8 kB)
            cat.10025.jpg (18.4 kB)
            ... and 24998 other files
            description.md (50 lines)
            sample_submission.csv (2501 lines)
            sample_submission.csv.zip (6.0 kB)
            test.zip (56.6 MB)
            train.zip (513.0 MB)
            dogs-vs-cats-redux-kernels-edition/
                cat.1714.jpg (7.8 kB)
                cat.10025.jpg (18.4 kB)
                ... and 24998 other files
                description.md (50 lines)
                sample_submission.csv (2501 lines)
                sample_submission.csv.zip (6.0 kB)
                test.zip (56.6 MB)
                train.zip (513.0 MB)
                dogs-vs-cats-redux-kernels-edition/
                test/
                    test/
                    unknown/
                        900.jpg (42.3 kB)
                        572.jpg (30.6 kB)
                        ... and 2498 other files
                train/
                    cat/
                        cat.4838.jpg (20.2 kB)
                        cat.1314.jpg (21.7 kB)
                        ... and 11240 other files
                    dog/
                        dog.6712.jpg (35.3 kB)
                        dog.7152.jpg (36.1 kB)
                        ... and 11256 other files
                    train/
            test/
                test/
                    test/
                    unknown/
                        900.jpg (42.3 kB)
                        572.jpg (30.6 kB)
                        ... and 2498 other files
                unknown/
                    900.jpg (42.3 kB)
                    572.jpg (30.6 kB)
                    ... and 2498 other files
            train/
                cat/
                    cat.4838.jpg (20.2 kB)
                    cat.1314.jpg (21.7 kB)
                    ... and 11240 other files
                dog/
                    dog.6712.jpg (35.3 kB)
                    dog.7152.jpg (36.1 kB)
                    ... and 11256 other files
                train/
                    cat/
                        cat.4838.jpg (20.2 kB)
                        cat.1314.jpg (21.7 kB)
                        ... and 11240 other files
                    dog/
                        dog.6712.jpg (35.3 kB)
                        dog.7152.jpg (36.1 kB)
                        ... and 11256 other files
                    train/
        working/
            dogs-vs-cats-redux-kernels-edition/
                cat.1714.jpg (7.8 kB)
                cat.10025.jpg (18.4 kB)
                ... and 24998 other files
                description.md (50 lines)
                sample_submission.csv (2501 lines)
                sample_submission.csv.zip (6.0 kB)
                test.zip (56.6 MB)
                train.zip (513.0 MB)
                dogs-vs-cats-redux-kernels-edition/
                test/
                    test/
                    unknown/
                        900.jpg (42.3 kB)
                        572.jpg (30.6 kB)
                        ... and 2498 other files
                train/
                    cat/
                        cat.4838.jpg (20.2 kB)
                        cat.1314.jpg (21.7 kB)
                        ... and 11240 other files
                    dog/
                        dog.6712.jpg (35.3 kB)
                        dog.7152.jpg (36.1 kB)
                        ... and 11256 other files
                    train/
```

-> data/dogs-vs-cats-redux-kernels-edition/sample_submission.csv has 2500 rows and 2 columns.
The columns are: id, label

-> data/sample_submission.csv has 2500 rows and 2 columns.
The columns are: id, label

-> input/dogs-vs-cats-redux-kernels-edition/sample_submission.csv has 2500 rows and 2 columns.
The columns are: id, label

-> input/sample_submission.csv has 2500 rows and 2 columns.
The columns are: id, label

-> working/dogs-vs-cats-redux-kernels-edition/sample_submission.csv has 2500 rows and 2 columns.
The columns are: id, label

# 5. Target score

0.08565

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import numpy as np
import os

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "2")

import sys
import subprocess

subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "protobuf<5"])

import glob

import tensorflow as tf

import matplotlib.pyplot as plt

AUTOTUNE = tf.data.experimental.AUTOTUNE

import torch


## === cell 1
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))


## === cell 2
import zipfile
zip_df = zipfile.ZipFile("/kaggle/input/dogs-vs-cats-redux-kernels-edition/train.zip", 'r')
zip_df.extractall("/kaggle/working/")
zip_df.close()
zip_df = zipfile.ZipFile("/kaggle/input/dogs-vs-cats-redux-kernels-edition/test.zip", 'r')
zip_df.extractall("/kaggle/working/")
zip_df.close()


## === cell 3
dataset_path = './dataset'
if not os.path.isdir(dataset_path):
    os.mkdir(dataset_path)

train_path = os.path.join(dataset_path,'train')
val_path = os.path.join(dataset_path,'val')
test_path = os.path.join(dataset_path,'test')
if not os.path.isdir(train_path):
    os.mkdir(train_path)
if not os.path.isdir(val_path):
    os.mkdir(val_path)
if not os.path.isdir(test_path):
    os.mkdir(test_path)


## === cell 4
def get_path(path,name):
    return glob.glob(path+'/*.'+name)
check = lambda x: 1 if x.split('.')[1].split('/')[-1] == 'dog' else 0


## === cell 5
data_list = get_path('./train','jpg')
result = list(map(check,data_list))


## === cell 6
print('dogs:',result.count(1),'cats:',result.count(0))


## === cell 7
dogs_list = [i for i in data_list if check(i)]
cats_list = [i for i in data_list if not check(i)]


## === cell 8
split_ratio = 0.8


## === cell 9
train_data = []
val_data = []
train_label = []
val_label = []

n_pairs = min(len(dogs_list), len(cats_list))

for i in range(n_pairs):
    if i < len(data_list) / 2 * split_ratio:
        train_data.append(dogs_list[i])
        train_data.append(cats_list[i])
    else:
        val_data.append(dogs_list[i])
        val_data.append(cats_list[i])

train_label = list(map(check, train_data))
val_label = list(map(check, val_data))


## === cell 10
class_label = ['dog','cat']


## === cell 11
img_size = 224

def preprocess_image(image):
  image = tf.image.decode_jpeg(image, channels=3)
  image = tf.image.resize(image, [img_size, img_size])

  return image


## === cell 12
def load_and_preprocess_image(path):
  image = tf.io.read_file(path)
  return preprocess_image(image)


## === cell 13
train_paths = tf.constant(train_data, dtype=tf.string)
val_paths = tf.constant(val_data, dtype=tf.string)
train_labels = tf.constant(train_label, dtype=tf.int32)
val_labels = tf.constant(val_label, dtype=tf.int32)

ds_train = tf.data.Dataset.from_tensor_slices((train_paths, train_labels))
ds_val = tf.data.Dataset.from_tensor_slices((val_paths, val_labels))


def load_and_preprocess_from_path_label(path, label):
    path = tf.cast(path, tf.string)
    label = tf.cast(label, tf.int32)
    return load_and_preprocess_image(path), tf.one_hot(label, 2)


ds_train = ds_train.map(load_and_preprocess_from_path_label)
ds_val = ds_val.map(load_and_preprocess_from_path_label)


## === cell 14
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers


## === cell 15
batch_size = 64
dsb_train = ds_train.batch(batch_size=batch_size, drop_remainder=True)
dsb_train = dsb_train.prefetch(tf.data.AUTOTUNE)

dsb_val = ds_val.batch(batch_size=batch_size, drop_remainder=True)


## === cell 16
model = EfficientNetB0(weights="imagenet")


## === cell 17
img_augmentation = Sequential(
    [
        layers.RandomRotation(factor=0.15),
        layers.RandomTranslation(height_factor=0.1, width_factor=0.1),
        layers.RandomFlip(),
        layers.RandomContrast(factor=0.1),
    ],
    name="img_augmentation",
)


## === cell 18
def build_model(num_classes):
    inputs = layers.Input(shape=(img_size, img_size, 3))
    x = img_augmentation(inputs)
    model = EfficientNetB0(include_top=False, input_tensor=x, weights="imagenet")

    model.trainable = False

    x = layers.GlobalAveragePooling2D(name="avg_pool")(model.output)
    x = layers.BatchNormalization()(x)

    top_dropout_rate = 0.2
    x = layers.Dropout(top_dropout_rate, name="top_dropout")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="pred")(x)

    model = tf.keras.Model(inputs, outputs, name="EfficientNet")
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-2)
    model.compile(
        optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"]
    )
    return model


## === cell 19
strategy = tf.distribute.MirroredStrategy()


## === cell 20
steps_per_epoch = max(1, len(train_data) // batch_size)
validation_steps = max(1, len(val_data) // batch_size)

new_model = build_model(num_classes=2)

epochs = 10
hist = new_model.fit(
    dsb_train,
    epochs=epochs,
    steps_per_epoch=steps_per_epoch,
    validation_data=dsb_val,
    validation_steps=validation_steps,
    verbose=2,
)


## === cell 21
def plot_hist(hist):
    plt.plot(hist.history["accuracy"])
    plt.plot(hist.history["val_accuracy"])
    plt.title("model accuracy")
    plt.ylabel("accuracy")
    plt.xlabel("epoch")
    plt.legend(["train", "validation"], loc="upper left")
    plt.show()


## === cell 22
def plot_hist(hist):
    history = hist.history if hasattr(hist, "history") else (hist or {})
    if not isinstance(history, dict):
        history = {}

    acc_candidates = [
        "accuracy",
        "acc",
        "categorical_accuracy",
        "sparse_categorical_accuracy",
    ]
    val_acc_candidates = [
        "val_accuracy",
        "val_acc",
        "val_categorical_accuracy",
        "val_sparse_categorical_accuracy",
    ]

    acc_key = next((k for k in acc_candidates if k in history), None)
    val_acc_key = next((k for k in val_acc_candidates if k in history), None)

    if acc_key is None:
        print(f"No accuracy metric found in history keys: {list(history.keys())}")
        return

    plt.plot(history[acc_key])
    if val_acc_key is not None:
        plt.plot(history[val_acc_key])
        plt.legend(["train", "validation"], loc="upper left")
    else:
        plt.legend(["train"], loc="upper left")

    plt.title("model accuracy")
    plt.ylabel("accuracy")
    plt.xlabel("epoch")
    plt.show()


plot_hist(hist)


## === cell 23
test_list = get_path('./test','jpg')
id_load = lambda x : int(x.split('/')[-1].split('.')[0])
id_list = list(map(id_load,test_list))


## === cell 24
test_paths = tf.constant(test_list, dtype=tf.string)
test_ids = tf.constant(id_list, dtype=tf.int32)

ds_test = tf.data.Dataset.from_tensor_slices((test_paths, test_ids))


def test(image, id):
    return load_and_preprocess_image(image), id


ds_test = ds_test.map(test)
dsb_test = ds_test.batch(batch_size=100, drop_remainder=True)


## === cell 25
submission = {'id':[],'label':[]}
dog_prediction = lambda x:x[1]

for batch in dsb_test:
    results = new_model.predict(batch[0])
    id = batch[1].numpy()
    
    submission['id'].extend(id)
    submission['label'].extend(map(dog_prediction,results))
    


## === cell 26
import pandas as pd

submission_df=pd.DataFrame(submission)
submission_df.to_csv('submission.csv',index=False)
