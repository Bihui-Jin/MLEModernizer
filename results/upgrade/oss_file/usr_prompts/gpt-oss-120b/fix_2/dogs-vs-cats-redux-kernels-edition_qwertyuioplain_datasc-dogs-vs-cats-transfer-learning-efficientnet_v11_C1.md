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

0.07542

# 6. Current score

0.04857

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

- What this solution (achieved 0.04857) has done: 'I fixed the protobuf import issue by removing the unnecessary `torch` import, corrected the file‑path handling to search recursively in the extracted folders, cast dataset elements to string tensors so `tf.io.read_file` receives the proper type, and rebuilt the pipeline so it creates a valid `submission.csv` with all test IDs.'

# 9. Code solution

## === cell 0
import numpy as np
import os
import glob
import tensorflow as tf
import matplotlib.pyplot as plt
import timeit

AUTOTUNE = tf.data.experimental.AUTOTUNE



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
device_name = tf.test.gpu_device_name()
if "GPU" not in device_name:
    print("GPU device not found")
print("Found GPU at: {}".format(device_name))
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices("GPU")))



## === cell 2
import zipfile

zip_df = zipfile.ZipFile(
    "/kaggle/input/dogs-vs-cats-redux-kernels-edition/train.zip", "r"
)
zip_df.extractall("/kaggle/working/")
zip_df.close()
zip_df = zipfile.ZipFile(
    "/kaggle/input/dogs-vs-cats-redux-kernels-edition/test.zip", "r"
)
zip_df.extractall("/kaggle/working/")
zip_df.close()



## === cell 3
BASE_DIR = "/kaggle/working/dogs-vs-cats-redux-kernels-edition"


def get_path(path, ext):
    """Recursively fetch all files with given extension."""
    pattern = os.path.join(path, "**", f"*.{ext}")
    return glob.glob(pattern, recursive=True)


dog_check = lambda x: 1 if x.split("/")[-1].split(".")[0].startswith("dog") else 0



## === cell 4
data_list = get_path(os.path.join(BASE_DIR, "train"), "jpg")
print("Total train images:", len(data_list))
print(
    "dogs:",
    sum(map(dog_check, data_list)),
    "cats:",
    sum(1 - dog_check(p) for p in data_list),
)



## === cell 5
dogs_list = [p for p in data_list if dog_check(p) == 1]
cats_list = [p for p in data_list if dog_check(p) == 0]

train_data, val_data = [], []
split_ratio = 0.8
half = int(len(dogs_list) / 2)  # both lists have the same length

for i in range(half):
    if i < half * split_ratio:
        train_data.append(dogs_list[i])
        train_data.append(cats_list[i])
    else:
        val_data.append(dogs_list[i])
        val_data.append(cats_list[i])

train_label = list(map(dog_check, train_data))
val_label = list(map(dog_check, val_data))



## === cell 6
img_size = 224


def preprocess_image(image):
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [img_size, img_size])
    return image


def load_and_preprocess_image(path):
    image = tf.io.read_file(path)
    return preprocess_image(image)


def load_and_preprocess_from_path_label(path, label):
    path = tf.cast(path, tf.string)  # ensure string tensor
    label = tf.cast(label, tf.int32)
    return load_and_preprocess_image(path), tf.one_hot(label, 2)


ds_train = tf.data.Dataset.from_tensor_slices((train_data, train_label))
ds_val = tf.data.Dataset.from_tensor_slices((val_data, val_label))

ds_train = ds_train.map(
    load_and_preprocess_from_path_label, num_parallel_calls=AUTOTUNE
)
ds_val = ds_val.map(load_and_preprocess_from_path_label, num_parallel_calls=AUTOTUNE)



## === cell 7
batch_size = 64
ds_batch_train = (
    ds_train.shuffle(1000).batch(batch_size, drop_remainder=True).prefetch(AUTOTUNE)
)
ds_batch_val = ds_val.batch(batch_size, drop_remainder=True).prefetch(AUTOTUNE)



## === cell 8
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras import layers, Sequential

img_augmentation = Sequential(
    [
        layers.RandomRotation(factor=0.15),
        layers.RandomTranslation(height_factor=0.1, width_factor=0.1),
        layers.RandomFlip(),
        layers.RandomContrast(factor=0.1),
    ],
    name="img_augmentation",
)


def build_model(num_classes):
    inputs = layers.Input(shape=(img_size, img_size, 3))
    x = img_augmentation(inputs)
    backbone = EfficientNetB0(include_top=False, input_tensor=x, weights="imagenet")
    backbone.trainable = False
    x = layers.GlobalAveragePooling2D(name="avg_pool")(backbone.output)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2, name="top_dropout")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="pred")(x)
    model = tf.keras.Model(inputs, outputs, name="EfficientNet")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-2),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model




## === cell 9
model = build_model(num_classes=2)
epochs = 10
hist = model.fit(ds_batch_train, epochs=epochs, validation_data=ds_batch_val, verbose=2)




## === cell 10
def plot_hist(hist):
    plt.plot(hist.history["accuracy"], label="train")
    plt.plot(hist.history["val_accuracy"], label="val")
    plt.title("Model accuracy")
    plt.ylabel("accuracy")
    plt.xlabel("epoch")
    plt.legend()
    plt.show()


plot_hist(hist)



## === cell 11
test_list = get_path(os.path.join(BASE_DIR, "test"), "jpg")
print("Total test images:", len(test_list))
id_load = lambda x: int(os.path.splitext(os.path.basename(x))[0])
id_list = list(map(id_load, test_list))




## === cell 12
def test_map(image_path, img_id):
    image_path = tf.cast(image_path, tf.string)
    img = load_and_preprocess_image(image_path)
    return img, img_id


ds_test = tf.data.Dataset.from_tensor_slices((test_list, id_list))
ds_test = ds_test.map(test_map, num_parallel_calls=AUTOTUNE)
dsb_test = ds_test.batch(100, drop_remainder=False)



## === cell 13
submission = {"id": [], "label": []}
dog_prediction = lambda probs: probs[1]  # probability of class 1 (dog)

for batch_imgs, batch_ids in dsb_test:
    probs = model.predict(batch_imgs, verbose=0)
    submission["id"].extend(batch_ids.numpy().tolist())
    submission["label"].extend([dog_prediction(p) for p in probs])



## === cell 14
import pandas as pd

submission_df = pd.DataFrame(submission)
submission_df.to_csv("submission.csv", index=False)
