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

# 5. Target score

0.64296

# 6. Current score

0.93872

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 0.87383) has done: 'I clean up the imports that caused the initial protobuf error, add missing `os` import, fix undefined variables by using the trained `model_1` for predictions, and replace the later error‑prone cells with harmless placeholders so the notebook runs straight through and writes a correct `submission.csv` file.'
- What this solution (achieved 0.93872) has done: 'We set the protobuf implementation before importing TensorFlow to avoid the `MessageFactory` error, switch all Keras‑applications imports to `tensorflow.keras.applications` (so they share the same backend), correct the stratification argument to be 1‑D, and give the simple classifier a bit more training capacity (increase epochs and early‑stopping patience). These fixes unblock the notebook, ensure a valid `submission.csv`, and should lower the log‑loss toward the target.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

plt.rcParams["figure.figsize"] = (10, 6)
sns.set_style("whitegrid")
pd.set_option("display.float_format", lambda x: "%.3f" % x)
pd.set_option("display.max_columns", None)

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    Input,
    InputLayer,
    Dense,
    Dropout,
    Lambda,
    GlobalAveragePooling2D,
    BatchNormalization,
)
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.utils import to_categorical

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
def get_num_files(path):
    """
    Counts the number of files in a folder.
    """
    if not os.path.exists(path):
        return 0
    return sum([len(files) for r, d, files in os.walk(path)])




## === cell 2
train_dir = "/kaggle/input/dog-breed-identification/train"
test_dir = "/kaggle/input/dog-breed-identification/test"
data_size = get_num_files(train_dir)
test_size = get_num_files(test_dir)
print("Data samples size: ", data_size)
print("Test samples size: ", test_size)




## === cell 3
labels_dataframe = pd.read_csv("/kaggle/input/dog-breed-identification/labels.csv")
sample_df = pd.read_csv("/kaggle/input/dog-breed-identification/sample_submission.csv")
print(labels_dataframe.head())




## === cell 4
print(sample_df.head())




## === cell 5
dog_breeds = sorted(list(set(labels_dataframe["breed"])))
n_classes = len(dog_breeds)
print("Number of breeds:", n_classes)
print(dog_breeds[:5])




## === cell 6
class_to_num = dict(zip(dog_breeds, range(n_classes)))




## === cell 7
def images_to_array(data_dir, labels_dataframe, img_size=(224, 224, 3)):
    """
    Load images, resize, and one‑hot encode labels.
    """
    images_names = labels_dataframe["id"].values
    images_labels = labels_dataframe["breed"].values
    data_len = len(images_names)
    X = np.zeros([data_len, img_size[0], img_size[1], img_size[2]], dtype=np.uint8)
    y = np.zeros([data_len, 1], dtype=np.uint8)
    for i in tqdm(range(data_len)):
        img_path = os.path.join(data_dir, images_names[i] + ".jpg")
        img = load_img(img_path, target_size=img_size)
        X[i] = img
        y[i] = class_to_num[images_labels[i]]
    y = to_categorical(y)
    perm = np.random.permutation(data_len)
    return X[perm], y[perm]




## === cell 8
img_size = (224, 224, 3)
X, y = images_to_array(train_dir, labels_dataframe, img_size)




## === cell 9
def get_features(model_fn, preprocessor, input_size, data):
    """
    Build a feature extractor using a pretrained model and return its output.
    """
    input_tensor = Input(shape=input_size)
    x = Lambda(preprocessor)(input_tensor)
    base = model_fn(weights="imagenet", include_top=False, input_shape=input_size)(x)
    out = GlobalAveragePooling2D()(base)
    extractor = Model(inputs=input_tensor, outputs=out)
    features = extractor.predict(data, batch_size=64, verbose=1)
    print("Feature maps shape:", features.shape)
    return features




## === cell 10
from tensorflow.keras.applications.inception_v3 import (
    InceptionV3,
    preprocess_input as inc_preprocess,
)

inception_features = get_features(InceptionV3, inc_preprocess, img_size, X)




## === cell 11
from tensorflow.keras.applications.xception import (
    Xception,
    preprocess_input as xcep_preprocess,
)

xception_features = get_features(Xception, xcep_preprocess, img_size, X)




## === cell 12
from tensorflow.keras.applications.nasnet import (
    NASNetLarge,
    preprocess_input as nas_preprocess,
)

nasnet_features = get_features(NASNetLarge, nas_preprocess, img_size, X)




## === cell 13
from tensorflow.keras.applications.inception_resnet_v2 import (
    InceptionResNetV2,
    preprocess_input as inc_res_preprocess,
)

inc_resnet_features = get_features(InceptionResNetV2, inc_res_preprocess, img_size, X)




## === cell 14
from tensorflow.keras.applications.vgg16 import (
    VGG16,
    preprocess_input as vgg_preprocess,
)

vgg16_features = get_features(VGG16, vgg_preprocess, img_size, X)




## === cell 15
from tensorflow.keras.applications.resnet50 import (
    ResNet50,
    preprocess_input as resnet_preprocess,
)

resnet50_features = get_features(ResNet50, resnet_preprocess, img_size, X)




## === cell 16
from tensorflow.keras.applications.mobilenet_v2 import (
    MobileNetV2,
    preprocess_input as mobilenet_preprocess,
)

mobilenet_v2_features = get_features(MobileNetV2, mobilenet_preprocess, img_size, X)




## === cell 17
del X  # free memory




## === cell 18
final_features = np.concatenate(
    [
        inception_features,
        xception_features,
        nasnet_features,
        inc_resnet_features,
        vgg16_features,
        resnet50_features,
        mobilenet_v2_features,
    ],
    axis=-1,
)
print("Final feature maps shape", final_features.shape)




## === cell 19
y_labels = np.argmax(y, axis=1)

X_train, X_test, y_train, y_test = train_test_split(
    final_features, y, test_size=0.2, stratify=y_labels, random_state=42
)

X_train, X_val, y_train, y_val = train_test_split(
    X_train,
    y_train,
    test_size=0.1,
    stratify=np.argmax(y_train, axis=1),
    random_state=42,
)




## === cell 20
label_encoder = LabelEncoder()
y_train_enc = label_encoder.fit_transform(np.argmax(y_train, axis=1))
class_weights_arr = compute_class_weight(
    "balanced", classes=np.unique(y_train_enc), y=y_train_enc
)
class_weights = {i: w for i, w in enumerate(class_weights_arr)}
print("Class weights computed.")




## === cell 21
early_stop = EarlyStopping(monitor="val_loss", patience=20, restore_best_weights=True)




## === cell 22
model_1 = Sequential(
    [
        InputLayer(input_shape=final_features.shape[1:]),
        Dropout(0.5),
        Dense(n_classes, activation="softmax"),
    ]
)
model_1.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
history_1 = model_1.fit(
    X_train,
    y_train,
    batch_size=128,
    epochs=120,
    validation_data=(X_val, y_val),
    callbacks=[early_stop],
    class_weight=class_weights,
    verbose=2,
)




## === cell 23
loss, accuracy = model_1.evaluate(X_test, y_test, verbose=0)
print("Test loss:", loss)
print("Test accuracy:", accuracy)




## === cell 24
pass




## === cell 25
pass




## === cell 26
pass




## === cell 27
pass




## === cell 28
test_pred_probs = model_1.predict(X_test, batch_size=128)




## === cell 29
pass




## === cell 30
def images_to_array2(data_dir, labels_dataframe, img_size=(224, 224, 3)):
    """
    Load test images (no labels) and return a NumPy array.
    """
    ids = labels_dataframe["id"].values
    n = len(ids)
    X = np.zeros([n, img_size[0], img_size[1], img_size[2]], dtype=np.uint8)
    for i in tqdm(range(n)):
        img_path = os.path.join(data_dir, ids[i] + ".jpg")
        img = load_img(img_path, target_size=img_size)
        X[i] = img
    print("Test data shape:", X.shape)
    return X




## === cell 31
test_data = images_to_array2(test_dir, sample_df, img_size)




## === cell 32
test_inception = get_features(InceptionV3, inc_preprocess, img_size, test_data)
test_xception = get_features(Xception, xcep_preprocess, img_size, test_data)
test_nasnet = get_features(NASNetLarge, nas_preprocess, img_size, test_data)
test_inc_resnet = get_features(
    InceptionResNetV2, inc_res_preprocess, img_size, test_data
)
test_vgg16 = get_features(VGG16, vgg_preprocess, img_size, test_data)
test_resnet50 = get_features(ResNet50, resnet_preprocess, img_size, test_data)
test_mobilenet = get_features(MobileNetV2, mobilenet_preprocess, img_size, test_data)

test_features = np.concatenate(
    [
        test_inception,
        test_xception,
        test_nasnet,
        test_inc_resnet,
        test_vgg16,
        test_resnet50,
        test_mobilenet,
    ],
    axis=-1,
)
print("Test feature maps shape", test_features.shape)




## === cell 33
del test_data  # free memory




## === cell 34
y_pred = model_1.predict(test_features, batch_size=128)




## === cell 35
for breed in dog_breeds:
    sample_df[breed] = y_pred[:, class_to_num[breed]]
sample_df.to_csv("submission.csv", index=False)
print('Submission file "submission.csv" written.')




## === cell 36
pass




## === cell 37
pass




## === cell 38
pass




## === cell 39
pass




## === cell 40
pass




## === cell 41
pass




## === cell 42
pass




## === cell 43
pass




## === cell 44
pass




## === cell 45
pass




## === cell 46
pass




## === cell 47
pass




## === cell 48
pass
