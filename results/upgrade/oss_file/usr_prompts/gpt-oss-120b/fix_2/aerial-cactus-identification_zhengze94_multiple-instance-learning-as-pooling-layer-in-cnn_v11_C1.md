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

3.7

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
pillow==11.3.0
protobuf==6.33.0
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
scipy==1.15.3
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

0.7965

# 6. Current score

0.98565

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved 0.98565) has done: 'I fix the import error caused by the protobuf‑TensorFlow mismatch, guard image loading so that only valid images are used, correct the custom noisyand layer (use input_shape[-1] instead of the outdated .value attribute), adjust the model compilation and history key names, and finally ensure the script writes a proper `sample_submission.csv` with the required columns. These changes remove the runtime crashes, allow the model to train, and produce a valid submission file, moving the solution toward the target AUC score.'

# 9. Code solution

## === cell 0
import os, glob, cv2, numpy as np, pandas as pd
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D,
    BatchNormalization,
    Activation,
    MaxPooling2D,
    Dense,
    Layer,
)
from tensorflow.keras.optimizers import RMSprop
from sklearn.model_selection import train_test_split

print("Input directories:", os.listdir("../input"))




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
def load_imgs(path):
    """
    Load images from a directory, keep only those successfully read.
    """
    imgs = {}
    for f in os.listdir(path):
        fp = os.path.join(path, f)
        img = cv2.imread(fp)
        if img is not None and img.shape == (32, 32, 3):
            imgs[f] = img.astype(np.float32) / 255.0  # normalise here
    return imgs




## === cell 2
train_img_dir = "../input/train/train/"
test_img_dir = "../input/test/test/"

img_train = load_imgs(train_img_dir)
img_test = load_imgs(test_img_dir)




## === cell 3
train_csv = pd.read_csv("../input/train.csv")
X_train = []
Y_train = []

for _, row in train_csv.iterrows():
    img_id = row["id"]
    if img_id in img_train:
        X_train.append(img_train[img_id])
        Y_train.append(int(row["has_cactus"]))

X_train = np.array(X_train)
Y_train = np.array(Y_train)

print("Training data shape:", X_train.shape, "=>", Y_train.shape)




## === cell 4
from scipy.ndimage import gaussian_filter


def img_sharpen(img):
    blurred = gaussian_filter(img, 2)
    blurred2 = gaussian_filter(blurred, 2)
    alpha = 15
    return blurred + alpha * (blurred - blurred2)




## === cell 5
sharp_img_xtrain = [img_sharpen(im) for im in X_train]
sharp_xtrain = np.array(sharp_img_xtrain)




## === cell 6
x_train, x_val, y_train, y_val = train_test_split(
    sharp_xtrain, Y_train, test_size=0.2, random_state=42
)




## === cell 7
class noisyand(Layer):
    def __init__(self, num_classes, a=20, **kwargs):
        super(noisyand, self).__init__(**kwargs)
        self.num_classes = num_classes
        self.a = max(1, a)

    def build(self, input_shape):
        channel_dim = int(input_shape[-1])
        self.b = self.add_weight(
            name="b", shape=(1, channel_dim), initializer="uniform", trainable=True
        )
        super(noisyand, self).build(input_shape)

    def call(self, x):
        mean = tf.reduce_mean(x, axis=[1, 2])  # (batch, channels)
        b = self.b
        a = self.a
        top = tf.nn.sigmoid(a * (mean - b)) - tf.nn.sigmoid(-a * b)
        bot = tf.nn.sigmoid(a * (1 - b)) - tf.nn.sigmoid(-a * b)
        return top / bot

    def compute_output_shape(self, input_shape):
        return input_shape




## === cell 8
def define_model(input_shape=(32, 32, 3), num_classes=1):
    model = Sequential()
    model.add(
        Conv2D(64, (3, 3), padding="same", activation="relu", input_shape=input_shape)
    )
    model.add(Conv2D(64, (3, 3), padding="same", activation="relu"))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(MaxPooling2D())

    model.add(Conv2D(128, (3, 3), activation="relu"))
    model.add(MaxPooling2D())

    model.add(Conv2D(128, (3, 3), activation="relu"))
    model.add(Conv2D(128, (1, 1), activation="relu"))

    model.add(noisyand(num_classes + 1))
    model.add(Dense(num_classes, activation="sigmoid"))
    return model




## === cell 9
model = define_model()
model.compile(
    loss=tf.keras.losses.BinaryCrossentropy(), optimizer=RMSprop(), metrics=["accuracy"]
)
model.summary()




## === cell 10
epochs = 30
history = model.fit(
    x_train,
    y_train,
    batch_size=32,
    epochs=epochs,
    validation_data=(x_val, y_val),
    verbose=1,
)




## === cell 11
submission = pd.read_csv("../input/sample_submission.csv")
preds = np.empty(len(submission), dtype=np.float32)

for idx in tqdm(range(len(submission)), desc="Predicting"):
    img_id = submission.loc[idx, "id"]
    if img_id in img_test:
        img = img_test[img_id]
    else:
        img_path = os.path.join(test_img_dir, img_id)
        img = cv2.imread(img_path).astype(np.float32) / 255.0
    img = img_sharpen(img)
    prob = model.predict(img.reshape((1, 32, 32, 3)), verbose=0)[0][0]
    preds[idx] = prob

submission["has_cactus"] = preds
submission.to_csv("sample_submission.csv", index=False)
print("Submission saved to sample_submission.csv")
