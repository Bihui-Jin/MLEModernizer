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

0.26052

# 6. Current score

0.84622

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 0.58265) has done: 'I fix the missing imports for the feature‑extraction helper, lower the training epochs so the notebook finishes quickly, and use a simple dense classifier (keeping the same pretrained‑CNN feature backbone) to produce a valid `submission.csv`. These minimal changes keep the original modelling approach while ensuring the pipeline runs end‑to‑end and writes the required file.'
- What this solution (achieved 0.5119) has done: 'I fixed the protobuf import error by switching all Keras‑application imports to the TensorFlow‑Keras equivalents, ensured images are loaded as NumPy arrays, added a small hidden dense layer to the classifier (a modest architectural tweak that can improve log‑loss without changing the overall approach), and extended the training epochs slightly to let the model converge a bit better. These minimal changes resolve the runtime crash and should move the log‑loss toward the target score while keeping the core pipeline intact.'
- What this solution (achieved 0.84622) has done: 'I add a standard‑scaler to normalise the concatenated CNN features before training and also apply the same transformation to the test features. Normalising the feature space usually lets the simple dense head converge to a better solution, reducing the log‑loss and moving the score closer to the target while keeping the original architecture unchanged.'

# 9. Code solution

## === cell 0
import os, random, csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import LabelEncoder, label_binarize, StandardScaler
from sklearn.metrics import (
    log_loss,
    precision_score,
    recall_score,
    average_precision_score,
)
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import (
    Input,
    Lambda,
    Dense,
    Dropout,
    BatchNormalization,
    InputLayer,
)
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.applications.inception_v3 import (
    InceptionV3,
    preprocess_input as inc_preprocess,
)
from tensorflow.keras.applications.xception import (
    Xception,
    preprocess_input as xcep_preprocess,
)
from tensorflow.keras.applications.nasnet import (
    NASNetLarge,
    preprocess_input as nas_preprocess,
)
from tensorflow.keras.applications.inception_resnet_v2 import (
    InceptionResNetV2,
    preprocess_input as inc_resnet_preprocess,
)
import warnings

warnings.filterwarnings("ignore")




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
train_dir = "/kaggle/input/dog-breed-identification/train"
test_dir = "/kaggle/input/dog-breed-identification/test"
labels_path = "/kaggle/input/dog-breed-identification/labels.csv"
sample_sub_path = "/kaggle/input/dog-breed-identification/sample_submission.csv"

labels_df = pd.read_csv(labels_path)
sample_df = pd.read_csv(sample_sub_path)

dog_breeds = sorted(labels_df["breed"].unique())
n_classes = len(dog_breeds)
class_to_num = {breed: idx for idx, breed in enumerate(dog_breeds)}




## === cell 2
def get_features(model_fn, preprocess_fn, img_size, data):
    """
    Extract features using a pretrained Keras model.
    """
    input_layer = Input(shape=img_size)
    preprocessed = Lambda(preprocess_fn)(input_layer)
    base = model_fn(weights="imagenet", include_top=False, input_shape=img_size)(
        preprocessed
    )
    pooled = tf.keras.layers.GlobalAveragePooling2D()(base)
    feature_extractor = Model(inputs=input_layer, outputs=pooled)
    features = feature_extractor.predict(data, batch_size=64, verbose=0)
    return features




## === cell 3
def images_to_array(data_dir, labels_dataframe, img_size=(224, 224, 3)):
    ids = labels_dataframe["id"].values
    breeds = labels_dataframe["breed"].values
    N = len(ids)
    X = np.zeros((N, *img_size), dtype=np.uint8)
    y = np.zeros((N, 1), dtype=np.int32)
    for i, (img_id, breed) in enumerate(zip(ids, breeds)):
        img_path = os.path.join(data_dir, f"{img_id}.jpg")
        img = load_img(img_path, target_size=img_size)
        img = np.array(img, dtype=np.uint8)
        X[i] = img
        y[i] = class_to_num[breed]
    y = to_categorical(y, num_classes=n_classes)
    return X, y




## === cell 4
img_size = (224, 224, 3)
X_raw, y_raw = images_to_array(train_dir, labels_df, img_size)




## === cell 5
inc_feat = get_features(InceptionV3, inc_preprocess, img_size, X_raw)
xcep_feat = get_features(Xception, xcep_preprocess, img_size, X_raw)
nas_feat = get_features(NASNetLarge, nas_preprocess, img_size, X_raw)
inc_resnet_feat = get_features(
    InceptionResNetV2, inc_resnet_preprocess, img_size, X_raw
)

train_features = np.concatenate(
    [inc_feat, xcep_feat, nas_feat, inc_resnet_feat], axis=1
)




## === cell 6
X_train, X_val, y_train, y_val = train_test_split(
    train_features, y_raw, test_size=0.1, stratify=y_raw, random_state=42
)

scaler = StandardScaler()
scaler.fit(X_train)  # fit on training part only
X_train = scaler.transform(X_train)
X_val = scaler.transform(X_val)

y_train_int = np.argmax(y_train, axis=1)
class_weights_vals = compute_class_weight(
    "balanced", classes=np.arange(n_classes), y=y_train_int
)
class_weights = {i: w for i, w in enumerate(class_weights_vals)}




## === cell 7
model = Sequential(
    [
        InputLayer(input_shape=X_train.shape[1:]),
        Dropout(0.5),
        Dense(256, activation="relu"),
        Dropout(0.5),
        Dense(n_classes, activation="softmax"),
    ]
)
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

early_stop = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)




## === cell 8
model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=30,
    batch_size=128,
    callbacks=[early_stop],
    class_weight=class_weights,
    verbose=2,
)




## === cell 9
def images_to_array_test(data_dir, ids, img_size=(224, 224, 3)):
    N = len(ids)
    X = np.zeros((N, *img_size), dtype=np.uint8)
    for i, img_id in enumerate(ids):
        img_path = os.path.join(data_dir, f"{img_id}.jpg")
        img = load_img(img_path, target_size=img_size)
        img = np.array(img, dtype=np.uint8)
        X[i] = img
    return X




## === cell 10
test_ids = sample_df["id"].values
X_test_raw = images_to_array_test(test_dir, test_ids, img_size)

inc_feat_test = get_features(InceptionV3, inc_preprocess, img_size, X_test_raw)
xcep_feat_test = get_features(Xception, xcep_preprocess, img_size, X_test_raw)
nas_feat_test = get_features(NASNetLarge, nas_preprocess, img_size, X_test_raw)
inc_resnet_feat_test = get_features(
    InceptionResNetV2, inc_resnet_preprocess, img_size, X_test_raw
)

test_features = np.concatenate(
    [inc_feat_test, xcep_feat_test, nas_feat_test, inc_resnet_feat_test], axis=1
)

test_features = scaler.transform(test_features)




## === cell 11
test_pred = model.predict(test_features, batch_size=128, verbose=0)




## === cell 12
for breed in dog_breeds:
    idx = class_to_num[breed]
    sample_df[breed] = test_pred[:, idx]

submission_path = "submission.csv"
sample_df.to_csv(submission_path, index=False)
print(f"Submission written to {submission_path}")
