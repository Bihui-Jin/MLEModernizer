# Goal

You will receive environment details and a partial notebook export.

# Requirements

- Fix the bug that causes the error in cell k.
- Do NOT adjust any other non-buggy cells.
- You may reference cell k+1 only to preserve variable/interface compatibility.
- Do not complete or extend code logic in cell k, k+1, or later cells.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (bug fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Output must follow your strict format: Diagnosis / Patch summary / Updated cells / Compatibility notes for cell k+1 / Assumptions.


# 1. Python version

3.7

# 2. Installed packages

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
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
sklearn-pandas==2.2.0
tf_keras==2.18.0
tqdm==4.67.1

# 3. Data file paths

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

# 4. Code solution

## === cell 0

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


import os
print(os.listdir("../input"))



## === cell 1
import os

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "2")

try:
    from google.protobuf.message_factory import MessageFactory as _MessageFactory

    if not hasattr(_MessageFactory, "GetPrototype"):
        if hasattr(_MessageFactory, "GetMessageClass"):

            def _GetPrototype(self, descriptor):
                return self.GetMessageClass(descriptor)

        else:

            def _GetPrototype(self, descriptor):
                raise AttributeError(
                    "MessageFactory does not support GetPrototype/GetMessageClass in this protobuf version."
                )

        _MessageFactory.GetPrototype = _GetPrototype
except Exception:
    pass

from matplotlib import pyplot as plt

from tf_keras.preprocessing import image
from tf_keras.preprocessing.image import ImageDataGenerator

import pandas as pd
from tf_keras.layers import (
    Conv2D,
    MaxPooling2D,
    GlobalMaxPool2D,
    Dropout,
    Dense,
    Flatten,
    BatchNormalization,
)
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import numpy as np
from tf_keras.models import Sequential
from tf_keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from sklearn.metrics import roc_auc_score, roc_curve, f1_score
from sklearn.utils import class_weight


## === cell 2
output_dir = '../input/aerial-cactus-identification/model_output/CNN'
seed = 7
np.random.seed(seed)


## === cell 3
train_df = pd.read_csv("../input/train.csv")
train_df.head()


## === cell 4
class_weights = class_weight.compute_class_weight(
    class_weight="balanced",
    classes=np.unique(train_df["has_cactus"]),
    y=train_df["has_cactus"],
)
print(class_weights)


## === cell 5
train_image = []

for id in tqdm(range(len(train_df))):
    img = image.load_img('../input/train/train/'+train_df['id'][id],target_size=(32,32)) 
    img = image.img_to_array(img)
    img = img/255
    train_image.append(img)
X = np.array(train_image)


## === cell 6
X.shape


## === cell 7
plt.imshow(X[1])


## === cell 8
y = np.array(train_df.drop(['id'],axis=1))
y.shape


## === cell 9
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2)


## === cell 10
X_train.shape,X_test.shape,y_train.shape,y_test.shape


## === cell 11
img_gen = ImageDataGenerator(horizontal_flip=True,vertical_flip=True,zoom_range=0.1,rotation_range=40,brightness_range=(0.5,1.0),
                             height_shift_range=0.2,width_shift_range=0.2)

test_datagen = ImageDataGenerator()
validation_generator = test_datagen.flow(X_test, y_test)


## === cell 12
model = Sequential()
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation="relu", input_shape=(32,32,3)))
model.add(BatchNormalization())
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(rate=0.25))
model.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(rate=0.25))
model.add(Conv2D(filters=256, kernel_size=(3, 3), activation="relu"))
model.add(BatchNormalization())
model.add(Conv2D(filters=256, kernel_size=(3, 3), activation="relu"))
model.add(BatchNormalization())
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))
model.summary()


## === cell 13
callbacks = [
ModelCheckpoint(filepath="weights.best.hdf5",monitor='val_acc',save_best_only=True, mode='max'),
EarlyStopping(monitor="val_loss",mode='auto',patience=20,restore_best_weights=True),
ReduceLROnPlateau(monitor='val_loss',mode='auto',patience=3,min_lr=0.0001)
]


## === cell 14
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=80, validation_data=(X_test, y_test), batch_size=32,shuffle=True,callbacks=callbacks)


## === cell 15
pred = {}
def predictions(imagepath,imagename):
    img = image.load_img(imagepath,target_size=(32,32,3))
    img = image.img_to_array(img)
    proba = model.predict(img.reshape(1,32,32,3))   
    pred.update( {imagename : (int(proba[0][0]))} )  


## === cell 16
import os

weights_path = "weights.best.hdf5"
if os.path.exists(weights_path):
    model.load_weights(weights_path)
else:
    print(
        f"Warning: '{weights_path}' not found. Using current model weights from the last training run."
    )


## === cell 17
y_hat = model.predict_proba(X_test)
get_auc = roc_auc_score(y_test,y_hat)*100.0
print(get_auc)


## --- ERROR in cell 17, traceback:
[0;31m---------------------------------------------------------------------------[0m
[0;31mAttributeError[0m                            Traceback (most recent call last)
[0;32m/tmp/ipykernel_11/1701864911.py[0m in [0;36m<cell line: 0>[0;34m()[0m
[0;32m----> 1[0;31m [0my_hat[0m [0;34m=[0m [0mmodel[0m[0;34m.[0m[0mpredict_proba[0m[0;34m([0m[0mX_test[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m      2[0m [0mget_auc[0m [0;34m=[0m [0mroc_auc_score[0m[0;34m([0m[0my_test[0m[0;34m,[0m[0my_hat[0m[0;34m)[0m[0;34m*[0m[0;36m100.0[0m[0;34m[0m[0;34m[0m[0m
[1;32m      3[0m [0mprint[0m[0;34m([0m[0mget_auc[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m

[0;31mAttributeError[0m: 'Sequential' object has no attribute 'predict_proba'

## === cell 18
files = os.listdir("../input/test/test")
for file in tqdm(files):
    predictions("../input/test/test/"+file,file)
