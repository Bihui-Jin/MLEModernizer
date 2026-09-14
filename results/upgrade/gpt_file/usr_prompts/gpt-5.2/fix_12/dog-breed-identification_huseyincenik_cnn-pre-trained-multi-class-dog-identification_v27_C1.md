# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


# 1. Kaggle task description

## Task
Given a dataset of images of dogs, predict the breed of each image.

## Metric
Multi Class Log Loss.

## Submission Format
For each image in the test set, you must predict a probability for each of the different breeds. The file should contain a header and have the following format:
```
id,affenpinscher,afghan_hound,..,yorkshire_terrier
000621fb3cbb32d8935728e48679680e,0.0083,0.0,...,0.0083
etc.
```

## Dataset Description
- `train.zip` - the training set, you are provided the breed for these dogs
- `test.zip` - the test set, you must predict the probability of each breed for each image
- `sample_submission.csv` - a sample submission file in the correct format
- `labels.csv` - the breeds for the images in the train set

# 2. Python version

3.12

# 3. Installed packages

colorama==0.4.6
geopandas==0.14.4
google-api-python-client==2.177.0
ipython==7.34.0
ipython-genutils==0.2.0
ipython_pygments_lexers==1.1.1
ipython-sql==0.5.0
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
opencv-python==4.12.0.88
opencv-python-headless==4.12.0.88
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
scikit-plot==0.3.7
seaborn==0.12.2
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
termcolor==3.1.0
tf_keras==2.18.0
tqdm==4.67.1

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (169 lines)
            labels.csv (9200 lines)
            labels.csv.zip (201.6 kB)
            sample_submission.csv (1024 lines)
            sample_submission.csv.zip (32.1 kB)
            test.zip (36.4 MB)
            train.zip (324.7 MB)
            dog-breed-identification/
                description.md (169 lines)
                labels.csv (9200 lines)
                ... and 5 other files
                dog-breed-identification/
                test/
                    bca88d42e4fc84b3169b13a615f5fdbf.jpg (38.6 kB)
                    53cb3ed2547cdaf15ec7983d8325f007.jpg (32.9 kB)
                    ... and 1021 other files
                    test/
                train/
                    868decd906bb483bac17a005a3f06bc3.jpg (22.6 kB)
                    7b44341b91b48e2eafe679c00ba1a0a6.jpg (48.4 kB)
                    ... and 9197 other files
                    train/
            test/
                bca88d42e4fc84b3169b13a615f5fdbf.jpg (38.6 kB)
                53cb3ed2547cdaf15ec7983d8325f007.jpg (32.9 kB)
                ... and 1021 other files
                test/
            train/
                868decd906bb483bac17a005a3f06bc3.jpg (22.6 kB)
                7b44341b91b48e2eafe679c00ba1a0a6.jpg (48.4 kB)
                ... and 9197 other files
                train/
        input/
            description.md (169 lines)
            labels.csv (9200 lines)
            labels.csv.zip (201.6 kB)
            sample_submission.csv (1024 lines)
            sample_submission.csv.zip (32.1 kB)
            test.zip (36.4 MB)
            train.zip (324.7 MB)
            dog-breed-identification/
                description.md (169 lines)
                labels.csv (9200 lines)
                ... and 5 other files
                dog-breed-identification/
                test/
                    bca88d42e4fc84b3169b13a615f5fdbf.jpg (38.6 kB)
                    53cb3ed2547cdaf15ec7983d8325f007.jpg (32.9 kB)
                    ... and 1021 other files
                    test/
                train/
                    868decd906bb483bac17a005a3f06bc3.jpg (22.6 kB)
                    7b44341b91b48e2eafe679c00ba1a0a6.jpg (48.4 kB)
                    ... and 9197 other files
                    train/
            test/
                bca88d42e4fc84b3169b13a615f5fdbf.jpg (38.6 kB)
                53cb3ed2547cdaf15ec7983d8325f007.jpg (32.9 kB)
                ... and 1021 other files
                test/
                    bca88d42e4fc84b3169b13a615f5fdbf.jpg (38.6 kB)
                    53cb3ed2547cdaf15ec7983d8325f007.jpg (32.9 kB)
                    ... and 1021 other files
                    test/
            train/
                868decd906bb483bac17a005a3f06bc3.jpg (22.6 kB)
                7b44341b91b48e2eafe679c00ba1a0a6.jpg (48.4 kB)
                ... and 9197 other files
                train/
                    868decd906bb483bac17a005a3f06bc3.jpg (22.6 kB)
                    7b44341b91b48e2eafe679c00ba1a0a6.jpg (48.4 kB)
                    ... and 9197 other files
                    train/
        working/
            dog-breed-identification/
                description.md (169 lines)
                labels.csv (9200 lines)
                ... and 5 other files
                dog-breed-identification/
                test/
                    bca88d42e4fc84b3169b13a615f5fdbf.jpg (38.6 kB)
                    53cb3ed2547cdaf15ec7983d8325f007.jpg (32.9 kB)
                    ... and 1021 other files
                    test/
                train/
                    868decd906bb483bac17a005a3f06bc3.jpg (22.6 kB)
                    7b44341b91b48e2eafe679c00ba1a0a6.jpg (48.4 kB)
                    ... and 9197 other files
                    train/
```

-> data/dog-breed-identification/labels.csv has 9199 rows and 2 columns.
The columns are: id, breed

-> data/dog-breed-identification/sample_submission.csv has 1023 rows and 121 columns.
The columns are: id, affenpinscher, afghan_hound, african_hunting_dog, airedale, american_staffordshire_terrier, appenzeller, australian_terrier, basenji, basset, beagle, bedlington_terrier, bernese_mountain_dog, black-and-tan_coonhound, blenheim_spaniel... and 106 more columns

-> data/labels.csv has 9199 rows and 2 columns.
The columns are: id, breed

-> data/sample_submission.csv has 1023 rows and 121 columns.
The columns are: id, affenpinscher, afghan_hound, african_hunting_dog, airedale, american_staffordshire_terrier, appenzeller, australian_terrier, basenji, basset, beagle, bedlington_terrier, bernese_mountain_dog, black-and-tan_coonhound, blenheim_spaniel... and 106 more columns

-> input/dog-breed-identification/labels.csv has 9199 rows and 2 columns.
The columns are: id, breed

-> input/dog-breed-identification/sample_submission.csv has 1023 rows and 121 columns.
The columns are: id, affenpinscher, afghan_hound, african_hunting_dog, airedale, american_staffordshire_terrier, appenzeller, australian_terrier, basenji, basset, beagle, bedlington_terrier, bernese_mountain_dog, black-and-tan_coonhound, blenheim_spaniel... and 106 more columns

-> (stopped after 10 files for performance)

# 5. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION"] = "2"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_DISABLE_CPP_IMPLEMENTATION"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random
import warnings

warnings.filterwarnings("ignore")

plt.rcParams["figure.figsize"] = (10, 6)
sns.set_style("whitegrid")
pd.set_option("display.float_format", lambda x: "%.3f" % x)
pd.set_option("display.max_columns", None)

import tensorflow as tf

try:
    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
except Exception as e:
    print("GPU memory growth setup skipped:", repr(e))

tf.config.run_functions_eagerly(True)

from tensorflow import keras

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Activation,
    Dropout,
    Flatten,
    Dense,
    Conv2D,
    MaxPooling2D,
    BatchNormalization,
)

from keras.callbacks import EarlyStopping
from keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import load_img

from tqdm import tqdm

from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import LabelEncoder, label_binarize
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    precision_score,
    recall_score,
    average_precision_score,
    roc_auc_score,
    log_loss,
    precision_recall_curve,
)

import cv2  # kept to preserve original import set (not used in final path)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

print("TF version:", tf.__version__)
print("Eager execution:", tf.executing_eagerly())
print("GPUs:", tf.config.list_physical_devices("GPU"))



## === cell 1
labels_csv = pd.read_csv("../input/dog-breed-identification/labels.csv")
print(labels_csv.describe(include="all"))
print(labels_csv.head())



## === cell 2
labels_csv["breed"].value_counts().plot.bar(figsize=(20, 10))
plt.show()



## === cell 3
from IPython.display import Image

Image(
    "/kaggle/input/dog-breed-identification/train/00693b8bc2470375cc744a6391d397ec.jpg"
)



## === cell 4
train_path = "../input/dog-breed-identification/train/"



## === cell 5
filenames = [train_path + fname + ".jpg" for fname in labels_csv["id"]]
filenames[:10]



## === cell 6
if len(os.listdir(train_path)) == len(filenames):
    print("Filenames match actual amount of files!")
else:
    print("Filenames do not match actual amount of files, check the target directory.")



## === cell 7
from PIL import Image as PILImage

random_images = random.sample(filenames, 9)
fig, axes = plt.subplots(3, 3, figsize=(10, 10))

for i, ax in enumerate(axes.flat):
    img_path = random_images[i]
    img = PILImage.open(img_path)
    label = labels_csv[
        labels_csv["id"] == os.path.splitext(os.path.basename(img_path))[0]
    ]["breed"].values[0]
    img = img.resize((100, 100))
    ax.imshow(img)
    ax.set_title(label)
    ax.axis("off")

plt.tight_layout()
plt.show()



## === cell 8
labels = labels_csv["breed"].to_numpy()
labels[:20]



## === cell 9
if len(labels) == len(filenames):
    print("Number of labels matches number of filenames!")
else:
    print(
        "Number of labels does not match number of filenames, check data directories."
    )



## === cell 10
unique_breeds = np.unique(labels)
len(unique_breeds)



## === cell 11
boolean_labels = [label == np.array(unique_breeds) for label in labels]
boolean_labels[:2]



## === cell 12
print(labels[1])
print(np.where(unique_breeds == labels[1])[0][0])
print(boolean_labels[1].argmax())
print(boolean_labels[0].astype(int))



## === cell 13
X = filenames
y = boolean_labels
print(f"Number of training images: {len(X)}")
print(f"Number of labels: {len(y)}")



## === cell 14
X = filenames
y = [np.where(label)[0][0] for label in boolean_labels]
train_df = pd.DataFrame({"image": X, "label": y})
train_df.sample(10)



## === cell 15
list(train_df.iloc[1])




## === cell 16
def cnn_resize(data_df, img_size=(224, 224, 3)):
    images_paths = data_df["image"]
    labels_local = data_df["label"]
    data_size = len(images_paths)

    X_arr = np.zeros(
        [data_size, img_size[0], img_size[1], img_size[2]], dtype=np.float32
    )
    y_arr = np.zeros([data_size, 1], dtype=np.uint16)

    for i in tqdm(range(data_size)):
        image_path = images_paths.iloc[i]
        img_pil = load_img(image_path, target_size=img_size, color_mode="rgb")
        X_arr[i] = np.asarray(img_pil, dtype=np.float32) / 255.0
        y_arr[i] = labels_local.iloc[i]

    y_arr = to_categorical(y_arr)
    ind = np.random.permutation(data_size)
    X_arr = X_arr[ind]
    y_arr = y_arr[ind]

    print("Output Data Size: ", X_arr.shape)
    print("Output Label Size: ", y_arr.shape)
    return X_arr, y_arr


img_size_small = (128, 128, 3)
X_small, y_small = cnn_resize(train_df, img_size=img_size_small)



## === cell 17
gpus = tf.config.list_physical_devices("GPU")
print("Available GPUs:", gpus)



## === cell 18
nrow, ncol = 5, 4
fig1 = plt.figure(figsize=(20, 15))
fig1.suptitle("After Resizing", size=32)

for i in range(min(20, len(X_small))):
    plt.subplot(nrow, ncol, i + 1)
    plt.imshow((X_small[i] * 255).astype(np.uint8))
    plt.title(f"class = {train_df['label'].iloc[i]}, Dog is {labels[i]}")
    plt.axis("off")
    plt.grid(False)
plt.show()



## === cell 19
class_values = train_df["label"]
filtered_values = class_values[class_values < 0]
if not filtered_values.empty:
    print("There are values in the series less than 0.")
else:
    print("There are no values in the series less than 0.")
class_values.value_counts()



## === cell 20
original_value_counts = pd.Series(y_small.argmax(axis=1)).value_counts(normalize=True)

y_small_idx = np.argmax(y_small, axis=1)
X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(
    X_small, y_small, test_size=0.2, stratify=y_small_idx, random_state=SEED
)
y_train_s_idx = np.argmax(y_train_s, axis=1)
X_train_s, X_val_s, y_train_s, y_val_s = train_test_split(
    X_train_s, y_train_s, test_size=0.1, stratify=y_train_s_idx, random_state=SEED
)



## === cell 21
y_train_s_idx = np.argmax(y_train_s, axis=1)
class_weights = compute_class_weight(
    class_weight="balanced", classes=np.unique(y_train_s_idx), y=y_train_s_idx
)
class_weights_dict_s = {i: w for i, w in zip(np.unique(y_train_s_idx), class_weights)}
print("Class weights (CNN branch):", list(class_weights_dict_s.items())[:5], "...")



## === cell 22
model = Sequential()
model.add(
    Conv2D(
        filters=32,
        kernel_size=(3, 3),
        padding="same",
        input_shape=X_train_s.shape[1:],
        activation="relu",
    )
)
model.add(BatchNormalization())
model.add(Conv2D(filters=32, kernel_size=(3, 3), padding="same", activation="relu"))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(filters=64, kernel_size=(3, 3), padding="same", activation="relu"))
model.add(BatchNormalization())
model.add(Conv2D(filters=64, kernel_size=(3, 3), padding="same", activation="relu"))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(140, activation="relu"))
model.add(Dropout(0.3))
model.add(Dense(200, activation="relu"))
model.add(Dropout(0.3))
model.add(Dense(120, activation="softmax"))

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["Recall"])
model.summary()



## === cell 23
EarlyStop_callback = EarlyStopping(
    monitor="val_loss", verbose=1, patience=15, restore_best_weights=True
)
my_callback = [EarlyStop_callback]



## === cell 24
_ = model.fit(
    X_train_s,
    y_train_s,
    validation_data=(X_val_s, y_val_s),
    epochs=1,  # preserved
    batch_size=128,
    callbacks=my_callback,
    class_weight=class_weights_dict_s,
    verbose=1,
)



## === cell 25
del X_small, y_small, X_train_s, X_test_s, X_val_s, y_train_s, y_test_s, y_val_s




## === cell 26
def get_num_files(path):
    if not os.path.exists(path):
        return 0
    return sum([len(files) for r, d, files in os.walk(path)])


train_dir = "/kaggle/input/dog-breed-identification/train"
test_dir = "/kaggle/input/dog-breed-identification/test"
data_size = get_num_files(train_dir)
test_size = get_num_files(test_dir)
print("Data samples size: ", data_size)
print("Test samples size: ", test_size)



## === cell 27
labels_dataframe = pd.read_csv("/kaggle/input/dog-breed-identification/labels.csv")
sample_df = pd.read_csv("/kaggle/input/dog-breed-identification/sample_submission.csv")
labels_dataframe.head(5)



## === cell 28
sample_df.head(5)



## === cell 29
dog_breeds = sorted(list(set(labels_dataframe["breed"])))
n_classes = len(dog_breeds)
print(n_classes)
dog_breeds[:5]



## === cell 30
class_to_num = dict(zip(dog_breeds, range(n_classes)))




## === cell 31
def images_to_array(data_dir, labels_dataframe, img_size=(224, 224, 3)):
    rng = np.random.RandomState(SEED)

    images_names = labels_dataframe["id"].values
    images_labels = labels_dataframe["breed"].values
    data_size_local = len(images_names)

    X_arr = np.zeros(
        [data_size_local, img_size[0], img_size[1], img_size[2]], dtype=np.float32
    )
    y_arr = np.zeros([data_size_local, 1], dtype=np.uint16)

    for i in tqdm(range(data_size_local)):
        image_name = images_names[i]
        img_path = os.path.join(data_dir, image_name + ".jpg")
        img_pil = load_img(img_path, target_size=img_size, color_mode="rgb")
        X_arr[i] = np.asarray(img_pil, dtype=np.float32) / 255.0
        y_arr[i] = class_to_num[images_labels[i]]

    y_arr = to_categorical(y_arr, num_classes=n_classes)

    ind = rng.permutation(data_size_local)
    X_arr = X_arr[ind]
    y_arr = y_arr[ind]
    images_names = images_names[ind]

    print("Ouptut Data Size: ", X_arr.shape)
    print("Ouptut Label Size: ", y_arr.shape)
    return X_arr, y_arr, images_names




## === cell 32
img_size = (300, 300, 3)
X, y, train_ids_shuffled = images_to_array(train_dir, labels_dataframe, img_size)



## === cell 33
from keras.models import Model
from keras.layers import GlobalAveragePooling2D, Lambda, Input


def get_features(model_name, data_preprocessor, input_size, data, batch_size=16):
    data = np.ascontiguousarray(data, dtype=np.float32)

    input_layer = Input(shape=input_size)
    preprocessed = Lambda(data_preprocessor)(input_layer)
    base_model = model_name(
        weights="imagenet", include_top=False, input_tensor=preprocessed
    )
    avg = GlobalAveragePooling2D()(base_model.output)
    feature_extractor = Model(inputs=input_layer, outputs=avg)

    feature_maps = feature_extractor.predict(data, batch_size=batch_size, verbose=1)
    print("Feature maps shape: ", feature_maps.shape)

    keras.backend.clear_session()
    return feature_maps




## === cell 34
from keras.applications.inception_v3 import (
    InceptionV3,
    preprocess_input as inception_preprocess,
)
from keras.applications.xception import (
    Xception,
    preprocess_input as xception_preprocess,
)
from keras.applications.nasnet import NASNetLarge, preprocess_input as nasnet_preprocess

inception_features = get_features(
    InceptionV3, inception_preprocess, img_size, X, batch_size=16
)
xception_features = get_features(
    Xception, xception_preprocess, img_size, X, batch_size=16
)
nasnet_features = get_features(
    NASNetLarge, nasnet_preprocess, img_size, X, batch_size=8
)



## === cell 35
final_features_6 = np.concatenate(
    [inception_features, xception_features, nasnet_features], axis=-1
)
print("Final feature maps shape", final_features_6.shape)

del X, inception_features, xception_features, nasnet_features



## === cell 36
y_idx = np.argmax(y, axis=1)
X_train, X_test, y_train, y_test, ids_train, ids_test = train_test_split(
    final_features_6,
    y,
    train_ids_shuffled,
    test_size=0.1,
    stratify=y_idx,
    random_state=SEED,
)

y_train_idx = np.argmax(y_train, axis=1)
X_train, X_val, y_train, y_val, ids_train, ids_val = train_test_split(
    X_train,
    y_train,
    ids_train,
    test_size=0.05,
    stratify=y_train_idx,
    random_state=SEED,
)



## === cell 37
y_train_class = np.argmax(y_train, axis=1)
class_weights = compute_class_weight(
    class_weight="balanced", classes=np.unique(y_train_class), y=y_train_class
)
class_weights_dict = {
    int(cls): float(w) for cls, w in zip(np.unique(y_train_class), class_weights)
}
print("Class Weights Dictionary (feature branch):")
print(list(class_weights_dict.items())[:5], "...")



## === cell 38
batch_size = 64
epochs = 1000

EarlyStop_callback = EarlyStopping(
    monitor="val_loss", verbose=1, patience=15, restore_best_weights=True
)
my_callback = [EarlyStop_callback]



## === cell 39
from keras import regularizers
from keras.layers import InputLayer

model_6 = keras.models.Sequential(
    [
        InputLayer(X_train.shape[1:]),
        BatchNormalization(),
        Dropout(0.5),
        Dense(
            n_classes, activation="softmax", kernel_regularizer=regularizers.l2(0.001)
        ),
    ]
)

from keras.optimizers import Adam

custom_optimizer = Adam(learning_rate=0.005)

model_6.compile(
    optimizer=custom_optimizer, loss="categorical_crossentropy", metrics=["Recall"]
)

history_6 = model_6.fit(
    X_train,
    y_train,
    batch_size=batch_size,
    epochs=epochs,
    validation_data=(X_val, y_val),
    callbacks=my_callback,
    class_weight=class_weights_dict,
    verbose=1,
)



## === cell 40
y_pred_proba_val = model_6.predict(X_val, batch_size=128, verbose=0)
y_true_val = np.argmax(y_val, axis=1)
val_loss = log_loss(y_true_val, y_pred_proba_val)
print(f"Validation Multi Class Log Loss: {val_loss:.5f}")



## === cell 41
test_loss, test_recall = model_6.evaluate(X_test, y_test, verbose=0)
print("Holdout loss: ", test_loss)
print("Holdout recall: ", test_recall)



## === cell 42
loss_df_6 = pd.DataFrame(history_6.history)
loss_df_6.plot()
plt.show()



## === cell 43
X_full = final_features_6
y_full = y
y_full_class = np.argmax(y_full, axis=1)
class_weights_full = compute_class_weight(
    class_weight="balanced", classes=np.unique(y_full_class), y=y_full_class
)
class_weights_full_dict = {
    int(cls): float(w) for cls, w in zip(np.unique(y_full_class), class_weights_full)
}

model_6_full = keras.models.clone_model(model_6)
model_6_full.set_weights(model_6.get_weights())
model_6_full.compile(
    optimizer=custom_optimizer, loss="categorical_crossentropy", metrics=["Recall"]
)

_ = model_6_full.fit(
    X_full,
    y_full,
    batch_size=batch_size,
    epochs=epochs,
    validation_data=(X_val, y_val),  # keep same validation/early stopping behavior
    callbacks=my_callback,
    class_weight=class_weights_full_dict,
    verbose=1,
)




## === cell 44
def images_to_array2(data_dir, labels_dataframe, img_size=(224, 224, 3)):
    images_names = labels_dataframe["id"].values
    data_size_local = len(images_names)
    X_arr = np.zeros([data_size_local, img_size[0], img_size[1], 3], dtype=np.float32)

    for i in tqdm(range(data_size_local)):
        image_name = images_names[i]
        img_path = os.path.join(data_dir, image_name + ".jpg")
        img_pil = load_img(img_path, target_size=img_size, color_mode="rgb")
        X_arr[i] = np.asarray(img_pil, dtype=np.float32) / 255.0
    print("Ouptut Data Size: ", X_arr.shape)
    return X_arr


test_data = images_to_array2(test_dir, sample_df, img_size=img_size)



## === cell 45
inception_test = get_features(
    InceptionV3, inception_preprocess, img_size, test_data, batch_size=16
)
xception_test = get_features(
    Xception, xception_preprocess, img_size, test_data, batch_size=16
)
nasnet_test = get_features(
    NASNetLarge, nasnet_preprocess, img_size, test_data, batch_size=8
)

test_features = np.concatenate([inception_test, xception_test, nasnet_test], axis=-1)
print("Test feature maps shape", test_features.shape)

del test_data, inception_test, xception_test, nasnet_test



## === cell 46
y_pred = model_6_full.predict(test_features, batch_size=128, verbose=1)

y_pred = np.clip(y_pred, 1e-15, 1.0)
y_pred = y_pred / y_pred.sum(axis=1, keepdims=True)

submission = sample_df.copy()
breed_cols = [c for c in sample_df.columns if c != "id"]

col_indices = [class_to_num[b] for b in breed_cols]
submission[breed_cols] = y_pred[:, col_indices]

submission = submission[["id"] + breed_cols]

submission_path = "submission.csv"
submission.to_csv(submission_path, index=False)
print("Wrote:", submission_path, "shape:", submission.shape)
print(submission.head())
