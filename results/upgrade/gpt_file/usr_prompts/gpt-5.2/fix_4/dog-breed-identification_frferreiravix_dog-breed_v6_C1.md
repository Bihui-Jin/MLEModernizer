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

3.11

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
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
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

# 5. Target score

0.27843

# 6. Current score

0.32073

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 4.85454) has done: 'I fix the environment-breaking import error by avoiding the legacy `tensorflow.keras`/protobuf path mix and using `tf.keras` consistently. I correct the feature extractor construction (it currently returns `None` because the base model is being called incorrectly), which is the direct cause of the `None values not supported` error during `model.fit`. I also fix test image loading to ignore non-JPG entries/subfolders (which caused `IsADirectoryError`) and ensure submission IDs align with the predicted rows by sorting filenames and using `sample_submission.csv` column order. These changes are minimal, keep the same overall approach (InceptionV3 feature extraction + single Dense softmax head), and ensure a valid `submission.csv` is written.'
- What this solution (achieved 4.88761) has done: 'I fix the TensorFlow/Keras import crash caused by the protobuf `MessageFactory.GetPrototype` incompatibility by forcing the pure-Python protobuf implementation before importing TensorFlow (this is score-neutral but unblocks execution). I also fix the `None values not supported` training error by making sure `load_img(...); img_to_array(...)` produces numeric arrays and that the feature extractor returns a real NumPy array (the current code assigns PIL images into a NumPy array, which can yield invalid/None tensors downstream). Finally, I keep the same InceptionV3 feature-extraction + softmax-head approach, but ensure preprocessing is correctly applied and predictions are aligned to `sample_submission.csv` columns and test IDs so the logloss improves drastically toward the target and a valid `submission.csv` is always written.'
- What this solution (achieved 0.32073) has done: 'I fix the two blockers causing your current crash and poor score: (1) the protobuf/TensorFlow import mismatch by using `tf.keras` consistently (dropping the `tf_keras` mix) while keeping the same model/training approach, and (2) the `None values not supported` optimizer error by ensuring feature arrays are `float32` and the Adam epsilon is set explicitly (the current `epsilon=None` can propagate `None` into the update). I also keep your submission formatting logic but add a safety alignment so the prediction columns always match `sample_submission.csv` order and stay properly normalized/clipped for logloss. These are minimal, score-improving correctness fixes without changing the core InceptionV3-features + softmax-head design.'

# 9. Code solution

## === cell 0
import os
import gc
import time

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from tqdm.autonotebook import tqdm

import tensorflow as tf

from tensorflow.keras import Sequential
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.layers import (
    Dense,
    Dropout,
    Lambda,
    Input,
    GlobalAveragePooling2D,
)
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

import matplotlib.pyplot as plt
import seaborn as sns

np.random.seed(42)
tf.random.set_seed(42)

print("TF version:", tf.__version__)



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        break



## === cell 2
labels = pd.read_csv("/kaggle/input/dog-breed-identification/labels.csv")
labels.head()



## === cell 3
classes = sorted(list(set(labels["breed"])))
n_classes = len(classes)
print("Total unique breed {}".format(n_classes))

class_to_num = dict(zip(classes, range(n_classes)))



## === cell 4
input_shape = (331, 331, 3)


def images_to_array(directory, label_dataframe, target_size=input_shape):
    image_labels = label_dataframe["breed"].values
    images = np.zeros(
        [len(label_dataframe), target_size[0], target_size[1], target_size[2]],
        dtype=np.uint8,
    )
    y = np.zeros([len(label_dataframe), 1], dtype=np.int32)

    for ix, image_name in enumerate(tqdm(label_dataframe["id"].values)):
        img_dir = os.path.join(directory, image_name + ".jpg")
        img = load_img(img_dir, target_size=target_size[:2])
        img = img_to_array(img, dtype="uint8")
        images[ix] = img
        del img

        dog_breed = image_labels[ix]
        y[ix] = class_to_num[dog_breed]

    y = to_categorical(y, num_classes=n_classes)
    return images, y




## === cell 5
t = time.time()
X, y = images_to_array("/kaggle/input/dog-breed-identification/train", labels[:])
print("runtime in seconds: {}".format(time.time() - t))
print("X dtype/min/max:", X.dtype, X.min(), X.max())
print("y shape:", y.shape)



## === cell 6
pass



## === cell 7
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=np.argmax(y, axis=1)
)

print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)



## === cell 8
lrr = ReduceLROnPlateau(
    monitor="val_accuracy", factor=0.01, patience=3, min_lr=1e-5, verbose=1
)
EarlyStop = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)



## === cell 9
batch_size = 128
epochs = 20
learn_rate = 0.001

sgd = SGD(learning_rate=learn_rate, momentum=0.9, nesterov=False)

adam = Adam(
    learning_rate=learn_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-7, amsgrad=False
)



## === cell 10
img_size = (331, 331, 3)


def get_features(model_name, model_preprocessor, input_size, data):
    data = np.asarray(data, dtype=np.float32)

    input_layer = Input(shape=input_size)
    preprocessed = Lambda(lambda x: model_preprocessor(tf.cast(x, tf.float32)))(
        input_layer
    )

    base_model = model_name(
        weights="imagenet", include_top=False, input_shape=input_size
    )
    base_model.trainable = False
    base_out = base_model(preprocessed)

    avg = GlobalAveragePooling2D()(base_out)
    feature_extractor = Model(inputs=input_layer, outputs=avg)

    feature_maps = feature_extractor.predict(data, verbose=1, batch_size=batch_size)
    if feature_maps is None:
        raise RuntimeError(
            "Feature extractor returned None; check preprocessing/model wiring."
        )
    feature_maps = np.asarray(feature_maps, dtype=np.float32)
    print("Feature maps shape: ", feature_maps.shape, "dtype:", feature_maps.dtype)
    return feature_maps




## === cell 11
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input

inception_preprocessor = preprocess_input

inception_features = get_features(
    InceptionV3, inception_preprocessor, img_size, X_train
)
print("Inception feature maps shape", inception_features.shape)



## === cell 12
del X, X_train
gc.collect()



## === cell 13
print("Inception feature maps shape", inception_features.shape)



## === cell 14
model = Sequential()
model.add(Dropout(0.7, input_shape=(inception_features.shape[1],)))
model.add(Dense(n_classes, activation="softmax"))

model.compile(optimizer=adam, loss="categorical_crossentropy", metrics=["accuracy"])

history = model.fit(
    inception_features,
    y_train,
    batch_size=batch_size,
    epochs=epochs,
    validation_split=0.2,
    callbacks=[lrr, EarlyStop],
    verbose=1,
)



## === cell 15
acc = history.history.get("accuracy", [])
val_acc = history.history.get("val_accuracy", [])
loss = history.history.get("loss", [])
val_loss = history.history.get("val_loss", [])
ep = range(1, len(loss) + 1)

plt.figure()
plt.plot(ep, acc, label="Train accuracy")
plt.plot(ep, val_acc, label="Val accuracy")
plt.title("Training & validation accuracy")
plt.legend()

plt.figure()
plt.plot(ep, loss, label="Train loss")
plt.plot(ep, val_loss, label="Val loss")
plt.title("Training & validation loss")
plt.legend()
plt.show()



## === cell 16
del inception_features
gc.collect()




## === cell 17
def images_to_array_test(test_path, img_size=(331, 331, 3)):
    fnames = sorted(
        [
            f
            for f in os.listdir(test_path)
            if f.lower().endswith(".jpg") and os.path.isfile(os.path.join(test_path, f))
        ]
    )
    test_filenames = [os.path.join(test_path, f) for f in fnames]

    data_size = len(test_filenames)
    images = np.zeros([data_size, img_size[0], img_size[1], 3], dtype=np.uint8)

    for ix, img_dir in enumerate(tqdm(test_filenames)):
        img = load_img(img_dir, target_size=img_size[:2])
        img = img_to_array(img, dtype="uint8")
        images[ix] = img
        del img

    print("Ouptut Data Size: ", images.shape)
    return images, fnames


test_path = "/kaggle/input/dog-breed-identification/test/"
test_data, test_fnames = images_to_array_test(test_path, img_size)




## === cell 18
def extact_features(data):
    inception_features_local = get_features(
        InceptionV3, inception_preprocessor, img_size, data
    )
    print("Inception feature maps shape", inception_features_local.shape)
    return inception_features_local


test_features = extact_features(test_data)



## === cell 19
del test_data
gc.collect()



## === cell 20
pred = model.predict(test_features, verbose=1, batch_size=batch_size)



## === cell 21
sample_sub = pd.read_csv("/kaggle/input/dog-breed-identification/sample_submission.csv")
sub_classes = list(sample_sub.columns[1:])  # official breed column order

ids = [os.path.splitext(f)[0] for f in test_fnames]

preds_df = pd.DataFrame(pred, columns=classes)
preds_df.insert(0, "id", ids)

preds_df = preds_df.set_index("id").reindex(columns=sub_classes).reset_index()

prob_cols = sub_classes
eps = 1e-15
preds_df[prob_cols] = preds_df[prob_cols].clip(eps, 1.0)
preds_df[prob_cols] = preds_df[prob_cols].div(preds_df[prob_cols].sum(axis=1), axis=0)

preds_df.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", preds_df.shape)
print(preds_df.head())
