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

0.5583

# 6. Current score

0.86871

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.84542) has done: 'Diagnosis: The crash happens when Keras calls `noisyand.build()`: the code uses `input_shape[-1].value`, which was valid in older TF1/early TF2 when shapes were `tf.Dimension` objects. In TF/Keras 2.18, `input_shape[-1]` is already a plain `int`, so accessing `.value` raises `AttributeError: 'int' object has no attribute 'value'`.  
Patch summary: In cell 13 (where `define_model()` is invoked and triggers `noisyand.build()`), update the `noisyand.build()` method to derive the channel dimension in a version-compatible way (use `int(input_shape[-1])` and fall back to `.value` only if present). This keeps the layer behavior identical while fixing shape extraction.  
Updated cells: Only cell 13 is updated (redefines `noisyand` with a fixed `build`, then builds and summarizes the model) so the notebook can proceed.  
Compatibility notes for cell k+1: `model` remains a `tf.keras.Sequential` model created by the same `define_model()` function and is compiled in cell 14 exactly as before. No interface or variable name changes.  
Assumptions: `input_shape[-1]` is statically known (not `None`) for this model, which is true given the explicit `input_shape=(32,32,3)` in `define_model()`.'
- What this solution (achieved 0.83079) has done: 'Diagnosis: Cell 16 crashes because in modern `tf.keras` (TF 2.18), the `History.history` dict uses the metric name `'accuracy'` and `'val_accuracy'`, not the legacy aliases `'acc'`/`'val_acc'`. Since the model was compiled with `metrics=['accuracy']`, those legacy keys are absent, causing `KeyError: 'acc'`.  
Patch summary: Update cell 16 to read the correct keys, while keeping backward compatibility by falling back to `'acc'`/`'val_acc'` if they exist. This preserves the plotting logic and does not change training, metrics, or model behavior.  
Updated cells: Only cell 16 is modified.  
Compatibility notes for cell k+1: Cell 17 uses loss keys (`'loss'`, `'val_loss'`) which are unchanged, and it does not depend on any variables from cell 16, so no interface issues arise.  
Assumptions: The compiled metric remains exactly `'accuracy'` (as in cell 14), so `'accuracy'`/`'val_accuracy'` are present in `history.history`.'
- What this solution (achieved 0.86871) has done: 'The crash happens because `tf.keras.Sequential` models don’t implement the scikit-learn methods `predict_proba` and `predict_classes`. In this cell we should use `model.predict(...)` to get probabilities, then threshold them to obtain class predictions, while keeping the ROC computation semantics the same. I replace those two calls with Keras-compatible equivalents and ensure the shapes are 1D arrays as expected by `roc_curve`. No other cells or model logic are changed.'

# 9. Code solution

## === cell 0

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


import os
print(os.listdir("../input"))



## === cell 1
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import glob
%matplotlib inline 
import os


## === cell 2
from tqdm import tqdm
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", None)

import google.protobuf  # noqa: F401

from google.protobuf import message_factory as _message_factory

if not hasattr(_message_factory.MessageFactory, "GetPrototype"):

    def _GetPrototype(self, descriptor):
        if hasattr(self, "GetMessageClass"):
            return self.GetMessageClass(descriptor)
        return type(None)

    _message_factory.MessageFactory.GetPrototype = _GetPrototype

import tensorflow.keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Dense,
    Conv2D,
    Flatten,
    Dropout,
    MaxPooling2D,
    Activation,
    BatchNormalization,
    LeakyReLU,
    GlobalAveragePooling2D,
)
from tensorflow.keras.optimizers import Adam, SGD
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
from PIL import Image


## === cell 3
import pandas as pd
import cv2
import os

def load_imgs(path):
    imgs = {}
    for f in os.listdir(path):
        fname = os.path.join(path, f)
        imgs[f] = cv2.imread(fname)
    return imgs

img_train = load_imgs('../input/train/train/')
img_test = load_imgs('../input/test/test/')


## === cell 4
train_csv = pd.read_csv("../input/train.csv")
import numpy as np
import cv2

X_train = []
Y_train = []

for _, row in train_csv.iterrows():
    X_train.append(img_train[row["id"]] / 255)
    Y_train.append(int(row["has_cactus"]))

X_train = np.array(X_train)
Y_train = np.array(Y_train)

target_h, target_w = X_train.shape[1], X_train.shape[2]
x_test_list = []
for f in sorted(img_test.keys()):
    im = img_test.get(f, None)
    if im is None:
        continue
    if im.ndim == 2:  # grayscale -> 3 channels
        im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
    if im.shape[2] == 4:  # BGRA -> BGR
        im = cv2.cvtColor(im, cv2.COLOR_BGRA2BGR)
    if im.shape[0] != target_h or im.shape[1] != target_w:
        im = cv2.resize(im, (target_w, target_h), interpolation=cv2.INTER_AREA)
    x_test_list.append(im.astype(np.float32) / 255.0)

X_test = np.array(x_test_list, dtype=np.float32)

print("Training data shape:", X_train.shape, "=>", Y_train.shape)


## === cell 5
%matplotlib inline
import numpy as np
from matplotlib import pyplot as plt
plt.rcParams["axes.grid"] = False


## === cell 6
fig, axes = plt.subplots(1, 5, figsize=(15, 4))
axes[0].imshow(X_train[0])
axes[0].set_title("Has cactus:" + str(Y_train[0]))
axes[1].imshow(X_train[1])
axes[1].set_title("Has cactus:" + str(Y_train[0]))
axes[2].imshow(X_train[2])
axes[2].set_title("Has cactus:" + str(Y_train[0]))
axes[3].imshow(X_train[1000])
axes[3].set_title("Has cactus:" + str(Y_train[1000]))
axes[4].imshow(X_train[1050])
axes[4].set_title("Has cactus:" + str(Y_train[1050]))


## === cell 7
from scipy.ndimage import gaussian_filter

def img_sharpen(img):
    blurred_f = gaussian_filter(img, 2)

    filter_blurred_f = gaussian_filter(blurred_f, 2)

    alpha = 15
    sharpened = blurred_f + alpha * (blurred_f - filter_blurred_f)
    return sharpened


## === cell 8
sharp_img_xtrain = []

for im in X_train:
    sharp_img_xtrain.append(img_sharpen(im))


## === cell 9
fig, axes = plt.subplots(1, 5, figsize=(15, 4))
axes[0].imshow(sharp_img_xtrain[0])
axes[0].set_title("Has cactus:" + str(Y_train[0]))
axes[1].imshow(sharp_img_xtrain[1])
axes[1].set_title("Has cactus:" + str(Y_train[0]))
axes[2].imshow(sharp_img_xtrain[2])
axes[2].set_title("Has cactus:" + str(Y_train[0]))
axes[3].imshow(sharp_img_xtrain[1000])
axes[3].set_title("Has cactus:" + str(Y_train[1000]))
axes[4].imshow(sharp_img_xtrain[1050])
axes[4].set_title("Has cactus:" + str(Y_train[1050]))


## === cell 10
from sklearn.model_selection import train_test_split
from numpy import array

sharp_xtrain = array(sharp_img_xtrain)
x_train, x_test, y_train, y_test = train_test_split(X_train, Y_train, test_size=0.2)


## === cell 11
import tensorflow as tf

class noisyand(tf.keras.layers.Layer):
    def __init__(self, num_classes, a = 20, **kwargs):
        self.num_classes = num_classes
        self.a = max(1,a)
        super(noisyand,self).__init__(**kwargs)

    def build(self, input_shape):
        self.b = self.add_weight(name = "b",shape = (1,input_shape[-1].value), initializer = "uniform",trainable = True)
        super(noisyand,self).build(input_shape)

    def call(self,x):
        mean = tf.reduce_mean(x, axis = [1,2])
        return (tf.nn.sigmoid(self.a * (mean - self.b)) - tf.nn.sigmoid(-self.a * self.b)) / (tf.nn.sigmoid(self.a * (1 - self.b)) - tf.nn.sigmoid(-self.a * self.b))
    
    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[3]


## === cell 12
def define_model(input_shape= (32,32,3), num_classes=1):
    model = Sequential()
    model.add(Conv2D(64, kernel_size=(3, 3),
                     activation='relu',
                     padding = 'same',
                     input_shape=input_shape))
    
    model.add(Conv2D(64, (3, 3), padding = 'same', activation='relu'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D())
    
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D())

    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(Conv2D(128, (1, 1), activation='relu'))
    
    model.add(noisyand(num_classes+1))
    model.add(Dense(num_classes, activation='sigmoid'))
    
    return model


## === cell 13
import tensorflow as tf


class noisyand(tf.keras.layers.Layer):
    def __init__(self, num_classes, a=20, **kwargs):
        self.num_classes = num_classes
        self.a = max(1, a)
        super(noisyand, self).__init__(**kwargs)

    def build(self, input_shape):
        ch = input_shape[-1]
        if hasattr(ch, "value"):
            ch = ch.value
        ch = int(ch)
        self.b = self.add_weight(
            name="b",
            shape=(1, ch),
            initializer="uniform",
            trainable=True,
        )
        super(noisyand, self).build(input_shape)

    def call(self, x):
        mean = tf.reduce_mean(x, axis=[1, 2])
        return (
            tf.nn.sigmoid(self.a * (mean - self.b)) - tf.nn.sigmoid(-self.a * self.b)
        ) / (tf.nn.sigmoid(self.a * (1 - self.b)) - tf.nn.sigmoid(-self.a * self.b))

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[3]


model = define_model()
model.summary()


## === cell 14
model.compile(loss=tensorflow.keras.losses.binary_crossentropy,
                  optimizer=tensorflow.keras.optimizers.RMSprop(),
                  metrics=['accuracy'])


## === cell 15
epoch=20
history = model.fit(x_train, y_train,
         batch_size=32,
         epochs=epoch,
         verbose=1,
         validation_data=(x_test, y_test))


## === cell 16
acc = history.history.get("accuracy", history.history.get("acc"))
epochs_ = range(0, epoch)
plt.plot(epochs_, acc, label="training accuracy")

acc_val = history.history.get("val_accuracy", history.history.get("val_acc"))
plt.scatter(epochs_, acc_val, label="validation accuracy")
plt.ylim([0.85, 1.0])
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title("Accuracy Plot of Model")

plt.legend()


## === cell 17
acc=history.history['loss']
epochs_=range(0,epoch)
plt.plot(epochs_,acc,label='training loss')

acc_val=history.history['val_loss']
plt.scatter(epochs_,acc_val,label="validation loss")
plt.ylim([0,0.5])
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Loss Plot of Model')

plt.legend()


## === cell 18
submission_set=pd.read_csv('../input/sample_submission.csv')
submission_set.head()


## === cell 19
predictions=np.empty((submission_set.shape[0],))
    
for n in tqdm(range(submission_set.shape[0])):
    data=np.array(Image.open('../input/test/test/'+submission_set.id[n]))
    data=data.astype(np.float32)/255
    data=img_sharpen(data)
    predictions[n]=model.predict(data.reshape((1,32,32,3)))[0]

    
submission_set['has_cactus']=predictions
submission_set.to_csv('sample_submission.csv',index=False)

submission_set.head()


## === cell 20
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import numpy as np

clf = model
y_pred_proba = clf.predict(x_test).reshape(-1)
y_pred = (y_pred_proba >= 0.5).astype(int)

fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(
    fpr, tpr, color="darkorange", lw=1.5, label="ROC curve (area = %0.2f)" % roc_auc
)
plt.plot([0, 1], [0, 1], color="navy", lw=1.5, linestyle="--")
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver operating characteristic example")
plt.legend(loc="lower right")
plt.show()
