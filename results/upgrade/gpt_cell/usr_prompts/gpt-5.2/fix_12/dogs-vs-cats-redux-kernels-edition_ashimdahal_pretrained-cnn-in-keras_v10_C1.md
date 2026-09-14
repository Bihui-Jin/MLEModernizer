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

3.8

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

0.21328

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", None)

import sys
import subprocess

try:
    import google.protobuf  # noqa: F401
    from google.protobuf import __version__ as _pb_ver

    _major = int(_pb_ver.split(".")[0])
except Exception:
    _major = None

if _major is None or _major >= 6:
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-q", "protobuf>=3.20.3,<6"]
    )

import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf

import cv2

import zipfile
import matplotlib.pyplot as plt


## === cell 1
TEST_DIR = '../input/dogs-vs-cats-redux-kernels-edition/test.zip'
TRAIN_DIR = '../input/dogs-vs-cats-redux-kernels-edition/train.zip'


## === cell 2
with zipfile.ZipFile(TRAIN_DIR,'r') as trainfile:
    trainfile.extractall()
with zipfile.ZipFile(TEST_DIR,'r') as trainfile:
    trainfile.extractall()


## === cell 3
!ls


## === cell 4
testdir ='test/'
traindir = 'train/'


## === cell 5
def _dir_has_jpgs(d):
    return os.path.isdir(d) and any(f.lower().endswith(".jpg") for f in os.listdir(d))


testdir = None
test_candidates = [
    "test/",
    os.path.join("test", "test", "unknown") + os.sep,
    os.path.join("test", "test") + os.sep,
    os.path.join("dogs-vs-cats-redux-kernels-edition", "test", "test", "unknown")
    + os.sep,
    os.path.join("dogs-vs-cats-redux-kernels-edition", "test", "test") + os.sep,
]
for cand in test_candidates:
    if _dir_has_jpgs(cand):
        testdir = cand
        break

if testdir is None:
    found_unknown = []
    found_any_jpg = []
    for root, dirs, files in os.walk("."):
        root_norm = os.path.normpath(root)
        if os.path.basename(root_norm).lower() == "unknown" and _dir_has_jpgs(
            root_norm
        ):
            parts = root_norm.replace("\\", "/").lower().split("/")
            if len(parts) >= 3 and parts[-3:] == ["test", "test", "unknown"]:
                found_unknown.append(root_norm + os.sep)
        if files and any(f.lower().endswith(".jpg") for f in files):
            found_any_jpg.append(root_norm + os.sep)

    if found_unknown:
        testdir = sorted(found_unknown)[0]
    else:
        test_like = [
            p for p in found_any_jpg if "/test" in p.replace("\\", "/").lower()
        ]
        if test_like:
            testdir = sorted(test_like)[0]
        else:
            raise FileNotFoundError(
                "Could not find extracted test directory under current workspace"
            )

all_images = None
traindir = None

if os.path.isdir("train/") and any(
    f.lower().endswith(".jpg") for f in os.listdir("train/")
):
    traindir = "train/"
    all_images = [traindir + i for i in os.listdir(traindir)]
else:
    train_root_candidates = [
        "train",
        os.path.join("train", "train"),
        os.path.join("dogs-vs-cats-redux-kernels-edition", "train"),
        os.path.join("dogs-vs-cats-redux-kernels-edition", "train", "train"),
    ]

    found_cat_dir = None
    found_dog_dir = None
    for root in train_root_candidates:
        cat_dir = os.path.join(root, "cat")
        dog_dir = os.path.join(root, "dog")
        if _dir_has_jpgs(cat_dir) and _dir_has_jpgs(dog_dir):
            found_cat_dir, found_dog_dir = cat_dir, dog_dir
            break

    if found_cat_dir is None or found_dog_dir is None:
        for root, dirs, files in os.walk("."):
            base = os.path.basename(os.path.normpath(root)).lower()
            if base in ("cat", "dog") and _dir_has_jpgs(root):
                if base == "cat" and found_cat_dir is None:
                    found_cat_dir = root
                elif base == "dog" and found_dog_dir is None:
                    found_dog_dir = root
            if found_cat_dir is not None and found_dog_dir is not None:
                break

    if found_cat_dir is not None and found_dog_dir is not None:
        all_images = [
            os.path.join(found_cat_dir, i) for i in os.listdir(found_cat_dir)
        ] + [os.path.join(found_dog_dir, i) for i in os.listdir(found_dog_dir)]
    else:
        raise FileNotFoundError("Could not find extracted train images under 'train/'")

test_images = [testdir + i for i in os.listdir(testdir)]

limit = int(0.8 * len(all_images))

train_images = all_images[0:limit]
validation_images = all_images[limit:]


## === cell 6
img = cv2.imread(train_images[1])
plt.imshow(img)


## === cell 7
rows, columns = 160,160


## === cell 8
def getallimages(path):
    actualdata = np.ndarray((len(path),rows,columns,3),dtype=np.uint8)
    for index , file in enumerate(path):
        img = cv2.imread(file)
        img= cv2.resize(img, (rows, columns), interpolation=cv2.INTER_CUBIC)
        actualdata[index] = img
    return actualdata
train = getallimages(train_images)
test = getallimages(test_images)


## === cell 9
validation = getallimages(validation_images)


## === cell 10
test.shape


## === cell 11
label = [1 if 'dog' in i else 0 for i in train_images]
validation_label = [1 if 'dog' in i else 0 for i in validation_images]

validation_label[:10]


## === cell 12
image_shape = (rows,rows,3)


## === cell 13
type(train)


## === cell 14
base_model = tf.keras.applications.ResNet101(
    weights = 'imagenet', include_top=False, input_shape=image_shape)


## === cell 15
base_model.trainable=False


## === cell 16
base_model.summary()


## === cell 17
model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
   
    tf.keras.layers.Dense(1,activation='sigmoid')
    
])


## === cell 18
model.summary()


## === cell 20
base_learning_rate = 0.001
model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=base_learning_rate),
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])


## === cell 21
epochs = 5
validation_steps=20



## === cell 22
model.fit(x=np.array(train),y=np.array(label),validation_data=(np.array(validation),np.array(validation_label)) ,batch_size=32,epochs=epochs,shuffle=True)


## === cell 23
prediction = model.predict(test, verbose=1)


## === cell 24
plt.xlabel(prediction[4][0])
plt.imshow(test[4])


## === cell 26
 
test_id = [i.split('/')[1][:-4] for i in test_images]

predictions_df = pd.DataFrame({'id': test_id, 'label': prediction[:,0]})
predictions_df
predictions_df.to_csv("submission.csv", index=False,header=True)
