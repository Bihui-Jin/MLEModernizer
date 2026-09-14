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

3.12

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

12.75741

# 6. Current score

17.90649

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plan

- What this solution (achieved 17.90649) has done: 'Diagnosis: Cell 18 crashes because `test_dir` is hard-coded to `/kaggle/working/test/test`, but after extraction the actual images are under `/kaggle/working/test/test/unknown` (or similar nested variants). The existing candidate search doesn’t include the common nested path `/kaggle/working/test/test/test/unknown`, so `test_dir` remains pointing to a non-existent directory and `os.listdir(test_dir)` raises `FileNotFoundError`.

Patch summary: In cell 18 only, expand and harden the test directory resolution by checking a small set of likely extracted locations and, if needed, walking under `/kaggle/working/test` to find a directory that contains image files. This keeps the rest of the prediction/submission logic unchanged and ensures `load_test_data()` is called with an existing directory.

Updated cells: (cell 18 only)
Compatibility notes for cell k+1: No interface/variable changes; `X_test`, `test_filenames`, `predictions`, and `output_df` are produced exactly as before, just with a valid `test_dir`.
Assumptions: Test images are present somewhere under `/kaggle/working/test` after unzip, typically in an `unknown` folder; image extensions are standard (.jpg/.jpeg/.png/.bmp).'

# 9. Code solution

## === cell 0

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))



## === cell 1
import zipfile

with zipfile.ZipFile('/kaggle/input/dogs-vs-cats-redux-kernels-edition/train.zip','r') as zip_ref:
    zip_ref.extractall('/kaggle/working/train')

with zipfile.ZipFile('/kaggle/input/dogs-vs-cats-redux-kernels-edition/test.zip','r') as zip_ref:
    zip_ref.extractall('/kaggle/working/test')


## === cell 2
import numpy as np  # numpy（Pythonでの機械学習の計算をより速く、効率的に行えるようにする拡張モジュール）
import pandas as pd
import os
import shutil
import cv2
import matplotlib.pyplot as plt

from google.protobuf import message_factory as _message_factory

if not hasattr(_message_factory.MessageFactory, "GetPrototype"):
    if hasattr(_message_factory.MessageFactory, "GetMessageClass"):
        _message_factory.MessageFactory.GetPrototype = (
            _message_factory.MessageFactory.GetMessageClass
        )
    elif hasattr(_message_factory, "GetMessageClass"):

        def _get_prototype(self, descriptor):
            return _message_factory.GetMessageClass(descriptor)

        _message_factory.MessageFactory.GetPrototype = _get_prototype

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "2")

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
import csv


## === cell 3
train_dir = '/kaggle/working/train/train'


## === cell 4
test_dir = '/kaggle/working/test/test'


## === cell 5
IMG_SIZE = 64


## === cell 6
def load_data(data_dir, sample_size=1000):
    images = [] # 写真を入れるリスト
    labels = [] # その写真のラベル(犬か猫かを判断する)を入れる用のリスト
    files = os.listdir(data_dir)[:sample_size] # サイズを指定
    for file in files: 
        img_path = os.path.join(data_dir, file) # ディレクトリのパスとファイル名（写真名）をくっつけて（join）、ファイル（写真）のパスを作る。それをimg_pathに入れる
        img = cv2.imread(img_path) 
        if img is not None: 
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE)) # 写真のサイズを調整
            images.append(img) # リストの中に一枚ずつ格納
            label = 1 if 'dog' in file else 0 # 写真が犬なら 1, 猫なら 0とラベル付して
            labels.append(label) # ラベルをリストに格納
        else: #　写真がきちんと読み込めていない場合
            print(f'error {img_path}')  # デバッグ文
    return np.array(images) / 255.0, np.array(labels) # 戻り値


## === cell 7
def load_data(data_dir, sample_size=1000):
    img_exts = (".jpg", ".jpeg", ".png", ".bmp")

    def _is_image_file(fn: str) -> bool:
        return fn.lower().endswith(img_exts)

    def _has_class_subdirs_with_images(p: str) -> bool:
        cat_dir = os.path.join(p, "cat")
        dog_dir = os.path.join(p, "dog")
        if not (os.path.isdir(cat_dir) and os.path.isdir(dog_dir)):
            return False
        try:
            cat_files = os.listdir(cat_dir)
            dog_files = os.listdir(dog_dir)
        except OSError:
            return False
        return any(_is_image_file(f) for f in cat_files) or any(
            _is_image_file(f) for f in dog_files
        )

    parent_dir = os.path.dirname(os.path.normpath(data_dir))

    candidates = [
        data_dir,  # include the provided dir itself for resolution checks
        parent_dir,  # <-- added to recover from an extra nested path level
        os.path.join(parent_dir, "train"),
        os.path.join(parent_dir, "test"),
        os.path.join(parent_dir, "unknown"),
        os.path.join(data_dir, "train"),
        os.path.join(data_dir, "test"),
        os.path.join(data_dir, "unknown"),
        os.path.join(data_dir, "cat"),
        os.path.join(data_dir, "dog"),
        os.path.join(data_dir, "train", "train"),
        os.path.join(data_dir, "test", "test"),
        os.path.join(data_dir, "test", "unknown"),
        os.path.join(data_dir, "train", "cat"),
        os.path.join(data_dir, "train", "dog"),
        os.path.join(data_dir, "test", "cat"),
        os.path.join(data_dir, "test", "dog"),
        os.path.join(data_dir, "test", "test", "unknown"),
        os.path.join(data_dir, "train", "train", "cat"),
        os.path.join(data_dir, "train", "train", "dog"),
    ]

    resolved_dir = None
    for p in candidates:
        if os.path.isdir(p):
            if _has_class_subdirs_with_images(p):
                resolved_dir = p
                break
            try:
                files_here = os.listdir(p)
            except OSError:
                continue
            if any(_is_image_file(f) for f in files_here):
                resolved_dir = p
                break

    if resolved_dir is None and os.path.isdir(data_dir):
        for root, dirs, files in os.walk(data_dir):
            if any(_is_image_file(f) for f in files):
                resolved_dir = root
                break
            if "cat" in dirs and "dog" in dirs:
                cat_dir = os.path.join(root, "cat")
                dog_dir = os.path.join(root, "dog")
                try:
                    cat_files = os.listdir(cat_dir)
                    dog_files = os.listdir(dog_dir)
                except OSError:
                    cat_files, dog_files = [], []
                if any(_is_image_file(f) for f in cat_files) or any(
                    _is_image_file(f) for f in dog_files
                ):
                    resolved_dir = root
                    break
            if "unknown" in dirs:
                unk_dir = os.path.join(root, "unknown")
                try:
                    unk_files = os.listdir(unk_dir)
                except OSError:
                    unk_files = []
                if any(_is_image_file(f) for f in unk_files):
                    resolved_dir = unk_dir
                    break

    if resolved_dir is None and os.path.isdir(parent_dir):
        for root, dirs, files in os.walk(parent_dir):
            if any(_is_image_file(f) for f in files):
                resolved_dir = root
                break

    if resolved_dir is None:
        raise FileNotFoundError(f"No valid image directory found under: {data_dir}")

    images = []  # 写真を入れるリスト
    labels = []  # その写真のラベル(犬か猫かを判断する)を入れる用のリスト

    cat_subdir = os.path.join(resolved_dir, "cat")
    dog_subdir = os.path.join(resolved_dir, "dog")

    if os.path.isdir(cat_subdir) and os.path.isdir(dog_subdir):
        cat_files = sorted([f for f in os.listdir(cat_subdir) if _is_image_file(f)])
        dog_files = sorted([f for f in os.listdir(dog_subdir) if _is_image_file(f)])

        combined = [(os.path.join(cat_subdir, f), 0) for f in cat_files] + [
            (os.path.join(dog_subdir, f), 1) for f in dog_files
        ]
        combined = combined[:sample_size]

        for img_path, label in combined:
            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                images.append(img)
                labels.append(label)
            else:
                print(f"error {img_path}")
    else:
        files = sorted([f for f in os.listdir(resolved_dir) if _is_image_file(f)])[
            :sample_size
        ]
        for file in files:
            img_path = os.path.join(resolved_dir, file)
            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))  # 写真のサイズを調整
                images.append(img)  # リストの中に一枚ずつ格納
                label = 1 if "dog" in file else 0
                labels.append(label)  # ラベルをリストに格納
            else:  # 写真がきちんと読み込めていない場合
                print(f"error {img_path}")  # デバッグ文

    return np.array(images) / 255.0, np.array(labels)  # 戻り値


X_train, y_train = load_data(train_dir, sample_size=1000)
X_test, y_test = load_data(test_dir, sample_size=500)


## === cell 9
datagen = ImageDataGenerator(
    rotation_range=20, # 画像の回転(最大20度まで)
    width_shift_range=0.2, # 水平方向に移動
    height_shift_range=0.2, # 垂直方向に移動
    shear_range=0.2, # 画像のせん断
    zoom_range=0.2, # ズーム
    horizontal_flip=True, # 水平方向に反転
    fill_mode='nearest' # 空白の処理
)


## === cell 10
def create_model(neuron):
    Dense = keras.layers.Dense
    Conv2D = keras.layers.Conv2D
    MaxPooling2D = keras.layers.MaxPooling2D
    Flatten = keras.layers.Flatten
    Dropout = keras.layers.Dropout
    
    model = keras.models.Sequential()
    
    
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)))
    model.add(MaxPooling2D((2, 2)))
    
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    
    model.add(Flatten())
    model.add(Dense(neuron, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    return model


## === cell 11
def save_model(model, filename):
    model.save(filename)


## === cell 12
def load_existing_model(filename):
    return load_model(filename)


## === cell 13
def fit_epoch(neuron, batch, epochs, initial_epoch=0, model_filename='model.h5'):
    if initial_epoch == 0: # モデルを作るのが初めての場合
        model = create_model(neuron)
    else: # すでにモデルを作っている場合
        model = load_existing_model(model_filename) #モデルの読み込み
        
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    hist = model.fit(datagen.flow(X_train, y_train, batch_size=batch),
                     steps_per_epoch=len(X_train) // batch,
                     validation_data=(X_test, y_test),
                     epochs=epochs,
                     initial_epoch=initial_epoch,
                     verbose=1)
    
    score = model.evaluate(X_test, y_test, verbose=1)
    print('正解率=', score[1], 'loss=', score[0])
    
    save_model(model, model_filename)
    
    
    
    plt.plot(hist.history['accuracy'])
    plt.plot(hist.history['val_accuracy'])
    plt.title('Accuracy')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

    plt.plot(hist.history['loss'])
    plt.plot(hist.history['val_loss'])
    plt.title('Loss')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()


## === cell 14
neuron = 512  # ニューロンの数
batch = 8  # バッチサイズ
model_filename = 'model.h5'  # モデルのファイル名


## === cell 15
fit_epoch(neuron=neuron, batch=batch, epochs=10, initial_epoch=0, model_filename=model_filename)


## === cell 16
fit_epoch(neuron=neuron, batch=batch, epochs=20, initial_epoch=10, model_filename=model_filename)


## === cell 17
fit_epoch(neuron=neuron, batch=batch, epochs=30, initial_epoch=20, model_filename=model_filename)


## === cell 18
import numpy as np
import pandas as pd
import os
import cv2
import tensorflow.keras as keras
from tensorflow.keras.models import load_model

IMG_SIZE = 64  # 画像のサイズ
test_dir = "/kaggle/working/test/test"  # テストデータのディレクトリ
model_filename = "model.h5"  # 学習したモデルのファイル名
output_csv = "/kaggle/working/submission.csv"  # 提出するCSVファイルのパス

_img_exts = (".jpg", ".jpeg", ".png", ".bmp")


def _dir_has_images(p: str) -> bool:
    if not os.path.isdir(p):
        return False
    try:
        return any(fn.lower().endswith(_img_exts) for fn in os.listdir(p))
    except OSError:
        return False


_candidates = [
    test_dir,
    os.path.join(test_dir, "unknown"),
    "/kaggle/working/test/unknown",
    "/kaggle/working/test/test/unknown",
    "/kaggle/working/test/test/test",
    "/kaggle/working/test/test/test/unknown",
]

_resolved = None
for _p in _candidates:
    if _dir_has_images(_p):
        _resolved = _p
        break

if _resolved is None and os.path.isdir("/kaggle/working/test"):
    for _root, _dirs, _files in os.walk("/kaggle/working/test"):
        if any(f.lower().endswith(_img_exts) for f in _files):
            _resolved = _root
            break

if _resolved is None:
    raise FileNotFoundError(
        f"Could not locate test images under /kaggle/working/test (last tried base: {test_dir})"
    )

test_dir = _resolved


def load_test_data(data_dir):
    images = []
    filenames = os.listdir(data_dir)
    for file in filenames:
        img_path = os.path.join(data_dir, file)
        img = cv2.imread(img_path)
        if img is not None:
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            images.append(img)
        else:
            print(f"Error reading {img_path}")
    return np.array(images) / 255.0, filenames


X_test, test_filenames = load_test_data(test_dir)

model = load_model(model_filename)

predictions = model.predict(X_test)

predictions = (predictions > 0.5).astype(int).flatten()

output_df = pd.DataFrame(
    {
        "id": [
            os.path.splitext(f)[0] for f in test_filenames
        ],  # ファイル名から拡張子を除去してIDにする
        "label": predictions,
    }
)
output_df.to_csv(output_csv, index=False)

print(f"CSVファイル {output_csv} を作成しました")
