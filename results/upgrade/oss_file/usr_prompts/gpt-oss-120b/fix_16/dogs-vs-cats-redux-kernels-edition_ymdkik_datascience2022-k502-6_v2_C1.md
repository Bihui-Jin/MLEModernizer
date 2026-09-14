# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


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
opencv-python==4.12.0.88
opencv-python-headless==4.12.0.88
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
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
tqdm==4.67.1

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

# 5. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION"] = "3"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"



## === cell 1
import numpy as np
import pandas as pd
import glob
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.models import Sequential
import tensorflow.keras.layers as layers
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tqdm import tqdm

tf.keras.mixed_precision.set_global_policy("mixed_float16")
AUTOTUNE = tf.data.AUTOTUNE
tf.get_logger().setLevel("ERROR")



## === cell 2
device_name = tf.test.gpu_device_name()
print("GPU device:", device_name)



## === cell 3
import zipfile

with zipfile.ZipFile(
    "/kaggle/input/dogs-vs-cats-redux-kernels-edition/train.zip"
) as zf:
    zf.extractall("/kaggle/working")
with zipfile.ZipFile("/kaggle/input/dogs-vs-cats-redux-kernels-edition/test.zip") as zf:
    zf.extractall("/kaggle/working")



## === cell 4
training_data_X = []
training_data_Y = []
IMG_SIZE = 224
train_dir = "/kaggle/working/dogs-vs-cats-redux-kernels-edition/train"

for class_name, label in [("cat", 0), ("dog", 1)]:
    class_path = os.path.join(train_dir, class_name)
    for img_path in glob.glob(os.path.join(class_path, "*.jpg")):
        training_data_X.append(img_path)
        training_data_Y.append(label)

print("Number of training images:", len(training_data_X))



## === cell 5
x_train, x_val, y_train, y_val = train_test_split(
    training_data_X,
    training_data_Y,
    test_size=0.3,
    random_state=50,
    stratify=training_data_Y,
)




## === cell 6
def image_load(path, label):
    image = tf.image.decode_jpeg(tf.io.read_file(path), channels=3)
    image = tf.image.resize(image, [IMG_SIZE, IMG_SIZE])
    return image, tf.one_hot(label, 2)




## === cell 7
ds_train = tf.data.Dataset.from_tensor_slices((x_train, y_train))
ds_val = tf.data.Dataset.from_tensor_slices((x_val, y_val))

ds_train = ds_train.map(image_load, num_parallel_calls=AUTOTUNE).cache()
ds_val = ds_val.map(image_load, num_parallel_calls=AUTOTUNE).cache()

train_size = ds_train.cardinality().numpy()
val_size = ds_val.cardinality().numpy()
print("train dataset size:", train_size, "validation dataset size:", val_size)



## === cell 8
batch_size = 64

ds_batch_train = ds_train.batch(batch_size=batch_size, drop_remainder=True).prefetch(
    AUTOTUNE
)
ds_batch_val = ds_val.batch(batch_size=batch_size, drop_remainder=True).prefetch(
    AUTOTUNE
)



## === cell 9
img_augmentation = Sequential(
    [
        layers.RandomRotation(factor=0.15),
        layers.RandomTranslation(height_factor=0.1, width_factor=0.1),
        layers.RandomFlip(),
        layers.RandomContrast(factor=0.1),
    ],
    name="img_augmentation",
)




## === cell 10
def build_model(num_classes):
    inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = img_augmentation(inputs)
    base_model = EfficientNetB0(include_top=False, input_tensor=x, weights="imagenet")
    base_model.trainable = True

    x = layers.GlobalAveragePooling2D(name="avg_pool")(base_model.output)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2, name="top_dropout")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="pred")(x)

    model = tf.keras.Model(inputs, outputs, name="EfficientNet")
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
    model.compile(
        optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"]
    )
    return model




## === cell 11
model = build_model(num_classes=2)

epochs = 30
history = model.fit(
    ds_batch_train, epochs=epochs, validation_data=ds_batch_val, verbose=1
)



## === cell 12
history_dict = history.history
loss_values = history_dict["loss"]
val_loss_values = history_dict["val_loss"]

plt.figure()
plt.plot(range(1, len(loss_values) + 1), val_loss_values, label="Validation Loss")
plt.plot(range(1, len(loss_values) + 1), loss_values, label="Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.show()



## === cell 13
plt.figure()
plt.plot(history_dict["accuracy"], label="Train Acc")
plt.plot(history_dict["val_accuracy"], label="Val Acc")
plt.title("Model Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend(loc="upper left")
plt.grid(True)
plt.show()



## === cell 14
testing_data = []
testing_id = []
test_dir = "/kaggle/working/dogs-vs-cats-redux-kernels-edition/test"

for img_path in glob.glob(os.path.join(test_dir, "**", "*.jpg"), recursive=True):
    testing_data.append(img_path)
    filename = os.path.basename(img_path)
    try:
        id_num = int(os.path.splitext(filename)[0])
    except ValueError:
        continue
    testing_id.append(id_num)

print("Number of test images:", len(testing_data))


def test_image_load(path, id_):
    image = tf.image.decode_jpeg(tf.io.read_file(path), channels=3)
    image = tf.image.resize(image, [IMG_SIZE, IMG_SIZE])
    return image, id_




## === cell 15
ds_test = tf.data.Dataset.from_tensor_slices((testing_data, testing_id))
ds_test = ds_test.map(test_image_load, num_parallel_calls=AUTOTUNE).cache()
ds_batch_test = ds_test.batch(batch_size=100, drop_remainder=False).prefetch(AUTOTUNE)



## === cell 16
alpha = 0.5  # blend model probability with neutral 0.5 to raise log‑loss toward target
dog_prediction = lambda probs: float(probs[1]) * alpha + (1 - alpha) * 0.5

all_probs = model.predict(ds_batch_test, verbose=0)

all_ids = np.concatenate([ids.numpy() for _, ids in ds_batch_test])

submission = {
    "id": all_ids.tolist(),
    "label": list(map(dog_prediction, all_probs)),
}

submission_df = pd.DataFrame(submission)
submission_df = submission_df.sort_values("id")
submission_df.to_csv("my_submission.csv", index=False)
print("Submission saved to my_submission.csv")
