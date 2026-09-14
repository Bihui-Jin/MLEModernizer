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
google-api-python-client==2.177.0
ipython==7.34.0
ipython-genutils==0.2.0
ipython_pygments_lexers==1.1.1
ipython-sql==0.5.0
joblib==1.5.2
keras==3.8.0
keras-core==0.1.7
keras-cv==0.9.0
keras-hub==0.18.1
keras-nlp==0.18.1
keras-tuner==1.4.7
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
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
from IPython.display import display

image_size = (32, 32)


## === cell 1
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "2")

from PIL import Image
from os import listdir
from os.path import join
from pandas import read_csv


def pil_load_img(path, target_size=None):
    img = Image.open(path).convert("RGB")
    if target_size is not None:
        img = img.resize((target_size[1], target_size[0]), resample=Image.BILINEAR)
    return img


train_labels = read_csv("../input/train.csv")
for image_name in listdir("../input/train/train")[:10]:
    image = pil_load_img(
        join("../input/train/train", image_name), target_size=image_size
    )
    display(train_labels[train_labels["id"] == image_name]["has_cactus"].item())
    display(image)


## === cell 2
from google.protobuf import message_factory as _message_factory

if not hasattr(_message_factory.MessageFactory, "GetPrototype"):
    if hasattr(_message_factory.MessageFactory, "GetMessageClass"):
        _message_factory.MessageFactory.GetPrototype = (
            _message_factory.MessageFactory.GetMessageClass
        )
    else:
        def _GetPrototype(self, descriptor):
            return self.GetMessages([descriptor])[descriptor.full_name]

        _message_factory.MessageFactory.GetPrototype = _GetPrototype

from tf_keras.applications.vgg19 import VGG19, preprocess_input
from tf_keras.preprocessing.image import img_to_array, load_img
from tf_keras.utils import to_categorical
from numpy import array
from os import listdir
from os.path import join
from pandas import read_csv
from tqdm import tqdm_notebook


def extract_features(label_path, set_path):
    images = []
    labels = []

    model = VGG19(include_top=False, input_shape=(image_size[0], image_size[1], 3))

    train_labels = read_csv(label_path)
    for image_name in tqdm_notebook(listdir(set_path)):
        image = load_img(join(set_path, image_name), target_size=image_size)
        images.append(img_to_array(image))
        label = train_labels[train_labels["id"] == image_name]["has_cactus"].item()
        labels.append(label)

    training_images = preprocess_input(array(images))
    training_labels = array(labels)

    features = model.predict(training_images)

    return features, training_labels


## === cell 3
from joblib import dump
from os import listdir

features, training_labels = extract_features('../input/train.csv', '../input/train/train')

dump(features, 'features.dat')
dump(training_labels, 'labels.dat')

display(listdir('.'))


## --- ERROR in cell 3, traceback:
[0;31m---------------------------------------------------------------------------[0m
[0;31mIsADirectoryError[0m                         Traceback (most recent call last)
[0;32m/tmp/ipykernel_11/1611314368.py[0m in [0;36m<cell line: 0>[0;34m()[0m
[1;32m      2[0m [0;32mfrom[0m [0mos[0m [0;32mimport[0m [0mlistdir[0m[0;34m[0m[0;34m[0m[0m
[1;32m      3[0m [0;34m[0m[0m
[0;32m----> 4[0;31m [0mfeatures[0m[0;34m,[0m [0mtraining_labels[0m [0;34m=[0m [0mextract_features[0m[0;34m([0m[0;34m'../input/train.csv'[0m[0;34m,[0m [0;34m'../input/train/train'[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m      5[0m [0;34m[0m[0m
[1;32m      6[0m [0;31m# Save the features and labels to files[0m[0;34m[0m[0;34m[0m[0m

[0;32m/tmp/ipykernel_11/753402888.py[0m in [0;36mextract_features[0;34m(label_path, set_path)[0m
[1;32m     33[0m     [0mtrain_labels[0m [0;34m=[0m [0mread_csv[0m[0;34m([0m[0mlabel_path[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m     34[0m     [0;32mfor[0m [0mimage_name[0m [0;32min[0m [0mtqdm_notebook[0m[0;34m([0m[0mlistdir[0m[0;34m([0m[0mset_path[0m[0;34m)[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m---> 35[0;31m         [0mimage[0m [0;34m=[0m [0mload_img[0m[0;34m([0m[0mjoin[0m[0;34m([0m[0mset_path[0m[0;34m,[0m [0mimage_name[0m[0;34m)[0m[0;34m,[0m [0mtarget_size[0m[0;34m=[0m[0mimage_size[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     36[0m         [0mimages[0m[0;34m.[0m[0mappend[0m[0;34m([0m[0mimg_to_array[0m[0;34m([0m[0mimage[0m[0;34m)[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m     37[0m         [0mlabel[0m [0;34m=[0m [0mtrain_labels[0m[0;34m[[0m[0mtrain_labels[0m[0;34m[[0m[0;34m"id"[0m[0;34m][0m [0;34m==[0m [0mimage_name[0m[0;34m][0m[0;34m[[0m[0;34m"has_cactus"[0m[0;34m][0m[0;34m.[0m[0mitem[0m[0;34m([0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/tf_keras/src/utils/image_utils.py[0m in [0;36mload_img[0;34m(path, grayscale, color_mode, target_size, interpolation, keep_aspect_ratio)[0m
[1;32m    420[0m         [0;32mif[0m [0misinstance[0m[0;34m([0m[0mpath[0m[0;34m,[0m [0mpathlib[0m[0;34m.[0m[0mPath[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m    421[0m             [0mpath[0m [0;34m=[0m [0mstr[0m[0;34m([0m[0mpath[0m[0;34m.[0m[0mresolve[0m[0;34m([0m[0;34m)[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 422[0;31m         [0;32mwith[0m [0mopen[0m[0;34m([0m[0mpath[0m[0;34m,[0m [0;34m"rb"[0m[0;34m)[0m [0;32mas[0m [0mf[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    423[0m             [0mimg[0m [0;34m=[0m [0mpil_image[0m[0;34m.[0m[0mopen[0m[0;34m([0m[0mio[0m[0;34m.[0m[0mBytesIO[0m[0;34m([0m[0mf[0m[0;34m.[0m[0mread[0m[0;34m([0m[0;34m)[0m[0;34m)[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m    424[0m     [0;32melse[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m

[0;31mIsADirectoryError[0m: [Errno 21] Is a directory: '../input/train/train/train'

## === cell 4
from joblib import load
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten 
from pathlib import Path

x_train = load('features.dat')
y_train = load('labels.dat')

model = Sequential()
model.add(Flatten(input_shape=x_train.shape[1:]))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))
model.summary()

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=50, shuffle=True, validation_split=0.2)

Path('model_structure.json').write_text(model.to_json())
model.save_weights('model_weights.h5')

display(listdir('.'))
