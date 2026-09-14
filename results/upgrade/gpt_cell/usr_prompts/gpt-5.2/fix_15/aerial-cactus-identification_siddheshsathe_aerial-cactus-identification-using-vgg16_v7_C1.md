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
"""
Your current script likely doesn’t yield a Kaggle score because it never writes a `submission.csv` file, even though it builds a `df` of predictions. I’ll keep your model/training/inference logic intact and only add the minimal steps needed to reliably create a valid submission: read `sample_submission.csv` to enforce correct row order/IDs, merge predictions onto it (avoiding missing/extra IDs), and write `submission.csv`. This also prevents accidental ID misalignment from `glob()` ordering, which can otherwise hurt AUC. No changes are made to the architecture, loss, optimizer, epochs, or your prediction “shrink_to_half” calibration.
"""


## === cell 1
pass



## === cell 2
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
print(os.listdir("../input/"))



## === cell 3
!mkdir has_cactus has_no_cactus 



## === cell 4
df = pd.read_csv('../input/train.csv')



## === cell 5
import shutil
images_having_cactus = []
images_having_no_cactus = []

for i in df[df['has_cactus'] == 1]['id']:
    p = os.path.join('../input/train/train/', i)
    images_having_cactus.append(p)

for i in df[df['has_cactus'] == 0]['id']:
    p = os.path.join('../input/train/train/', i)
    images_having_no_cactus.append(p)

for i in images_having_cactus:
    shutil.copy(i, './has_cactus/')
for i in images_having_no_cactus:
    shutil.copy(i, './has_no_cactus/')



## === cell 6
print('Has Cactus: {}'.format(df[df['has_cactus'] == 1]['id'].count()))
print('Has No Cactus: {}'.format(df[df['has_cactus'] == 0]['id'].count()))



## === cell 7
def augument_data(
    directory,              # Directory where augumentation is needed. Same dir will have sample images
    number_of_images_to_add # Image count to add 
):
    print('Images to add: {}'.format(number_of_images_to_add))
    import cv2
    from glob import glob
    l = glob(directory + '/*.jpg')
    for image in l:
        if number_of_images_to_add == 0:
            break
        img = cv2.imread(image)
        h_img = cv2.flip(img, 0)
        v_img = cv2.flip(img, 1)
        cv2.imwrite(directory + '/h_img_{}.jpg'.format(number_of_images_to_add), h_img)
        number_of_images_to_add -= 1
        cv2.imwrite(directory + '/v_img_{}.jpg'.format(number_of_images_to_add), v_img)
        number_of_images_to_add -= 1



## === cell 8
augument_data('./has_no_cactus/', df[df['has_cactus'] == 1]['id'].count() - df[df['has_cactus'] == 0]['id'].count())



## === cell 9
!mkdir -p curated_data/train_data curated_data/validation_data/has_cactus
!mkdir -p curated_data/validation_data/has_no_cactus
!mv has_cactus has_no_cactus curated_data/train_data



## === cell 10
from glob import glob
import shutil
l = glob('curated_data/train_data/has_cactus/*.jpg')
for i in range(300):
    shutil.move(l[i], 'curated_data/validation_data/has_cactus')

l = glob('curated_data/train_data/has_no_cactus/*.jpg')
for i in range(300):
    shutil.move(l[i], 'curated_data/validation_data/has_no_cactus')



## === cell 11
import os

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "2")

try:
    from google.protobuf import message_factory as _message_factory
    from google.protobuf import symbol_database as _symbol_database

    if not hasattr(_message_factory.MessageFactory, "GetPrototype"):
        _sym_db = _symbol_database.Default()

        def _GetPrototype(self, descriptor):
            return _sym_db.GetPrototype(descriptor)

        _message_factory.MessageFactory.GetPrototype = _GetPrototype
except Exception:
    pass

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Input, Flatten, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.applications.vgg16 import VGG16



## === cell 12
datagen = ImageDataGenerator(
    featurewise_std_normalization=True,
    samplewise_std_normalization=True,
    horizontal_flip=True,
    vertical_flip=True
)



## === cell 13
train_data = datagen.flow_from_directory(
    'curated_data/train_data/',
    class_mode='categorical'
)

validation_data = datagen.flow_from_directory(
    'curated_data/validation_data/',
    class_mode='categorical'
)



## === cell 14
vgg16_model = VGG16(
    include_top=False,
    weights='imagenet',
    input_shape=(256, 256, 3)
)



## === cell 15
vgg16_model.summary()



## === cell 16
for layer in vgg16_model.layers[1:19]:
    layer.trainable = False



## === cell 17
x = vgg16_model.output
x = Flatten()(x)
x = Dense(1024)(x)
x = Dropout(0.5)(x)
x = Dense(1024, activation="relu")(x)
predictions = Dense(2, activation="softmax")(x)

model = Model(inputs=vgg16_model.input, outputs=predictions)

model.compile(
    loss="binary_crossentropy",
    optimizer=SGD(learning_rate=0.0001, momentum=0.9),
    metrics=["accuracy"],
)



## === cell 18
model.fit(train_data, epochs=4, validation_data=validation_data)



## === cell 19
import cv2
from glob import glob

test_images = glob("../input/test/test/*.jpg")
df = pd.DataFrame(columns=["id", "has_cactus"])

shrink_to_half = 0.0005  # was 0.02

for img in test_images:
    i = cv2.imread(img)
    i = cv2.resize(i, (256, 256))
    pred = model.predict(i.reshape(1, 256, 256, 3), verbose=0)
    p_raw = float(pred[0][0])
    p = 0.5 + shrink_to_half * (p_raw - 0.5)
    tempDf = pd.DataFrame({"id": [img.split("/")[-1]], "has_cactus": [p]})
    df = pd.concat([df, tempDf], ignore_index=True)

df["has_cactus"] = pd.to_numeric(df["has_cactus"], errors="coerce")



## === cell 20
sample_path = "../input/sample_submission.csv"
if not os.path.exists(sample_path):
    sample_path = "../input/aerial-cactus-identification/sample_submission.csv"

sample_sub = pd.read_csv(sample_path)

sub = sample_sub[["id"]].merge(df, on="id", how="left")
sub["has_cactus"] = sub["has_cactus"].fillna(0.5).astype(float)

sub.to_csv("submission.csv", index=False)
print(sub.head())
print("Wrote submission.csv with shape:", sub.shape)

!rm -rf curated_data/
```

## --- ERROR in cell 20, traceback:
[0;36m  File [0;32m"/tmp/ipykernel_11/3738870782.py"[0;36m, line [0;32m18[0m
[0;31m    ```[0m
[0m    ^[0m
[0;31mSyntaxError[0m[0;31m:[0m invalid syntax
