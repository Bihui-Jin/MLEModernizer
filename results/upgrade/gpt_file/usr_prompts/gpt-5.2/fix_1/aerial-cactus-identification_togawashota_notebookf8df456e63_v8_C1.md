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

3.13

# 3. Installed packages

No external packages required in the script and installed.

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

0.8359651666666666

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 1

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        



## === cell 3
import zipfile

extract_dir = '/kaggle/working'

with zipfile.ZipFile('/kaggle/input/aerial-cactus-identification/train.zip', 'r') as zip_ref:
    zip_ref.extractall(os.path.join(extract_dir, 'train'))

with zipfile.ZipFile('/kaggle/input/aerial-cactus-identification/test.zip', 'r') as zip_ref:
    zip_ref.extractall(os.path.join(extract_dir, 'test'))    


## === cell 4
for dirname, _, _ in os.walk('/kaggle/working'):
    print(dirname)

## === cell 6
!pip install -U efficientnet

## === cell 8
train_dir = '/kaggle/working/train/train'

test_dir = '/kaggle/working/test'

train_df = pd.read_csv('/kaggle/input/aerial-cactus-identification/train.csv')

train_df.head(20)

## === cell 10
def count_files(directory):
    return len([f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))])

test_dir2 = '/kaggle/working/test/test'

train_count = count_files(train_dir)
test_count = count_files(test_dir2)

print(f'Train images: {train_count}')
print(f'Test images: {test_count}')

## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/1469657478.py in <cell line: 0>()
      7 
      8 # トレーニングとテストディレクトリのファイル数を取得
----> 9 train_count = count_files(train_dir)
     10 test_count = count_files(test_dir2)
     11 

/tmp/ipykernel_11/1469657478.py in count_files(directory)
      1 # ディレクトリ内のファイル数を数える関数
      2 def count_files(directory):
----> 3     return len([f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))])
      4 
      5 # テスト画像が保存されているディレクトリのパスを変数 test_dir2 に代入

FileNotFoundError: [Errno 2] No such file or directory: '/kaggle/working/train/train'

## === cell 12
class_ratio = train_df['has_cactus'].value_counts(normalize=True) * 100
print(class_ratio)

## === cell 13
import matplotlib.pyplot as plt

counts = train_df['has_cactus'].value_counts()

labels = ['Has Cactus (1)', 'No Cactus (0)']
colors = ['lightgreen', 'lightcoral']

plt.figure(figsize=(6, 6))
plt.pie(counts, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors)
plt.title('Distribution of Cactus Presence (has_cactus)')
plt.axis('equal')  # 円を真円にする
plt.show()

## === cell 15
from sklearn.utils.class_weight import compute_class_weight

class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_df['has_cactus']),
    y=train_df['has_cactus']
)
class_weights_dict = dict(enumerate(class_weights))
print(class_weights_dict)


## === cell 17
import cv2
cactus = []
cactus.append(cv2.imread(train_dir + '/' + train_df['id'][0]))
cactus.append(cv2.imread(train_dir + '/' + train_df['id'][1]))
cactus.append(cv2.imread(train_dir + '/' + train_df['id'][2]))
cactus.append(cv2.imread(train_dir + '/' + train_df['id'][8]))
cactus.append(cv2.imread(train_dir + '/' + train_df['id'][9]))
cactus.append(cv2.imread(train_dir + '/' + train_df['id'][12]))
cactus.append(cv2.imread(train_dir + '/' + train_df['id'][6]))
cactus.append(cv2.imread(train_dir + '/' + train_df['id'][7]))
cactus.append(cv2.imread(train_dir + '/' + train_df['id'][11]))
cactus.append(cv2.imread(train_dir + '/' + train_df['id'][14]))
cactus.append(cv2.imread(train_dir + '/' + train_df['id'][16]))
cactus.append(cv2.imread(train_dir + '/' + train_df['id'][17]))


labels = ['cactus','cactus','cactus','cactus','cactus','cactus',
          'no cactus','no cactus',' no cactus','no cactus','no cactus',' no cactus']

import matplotlib.pyplot as plt

plt.figure(figsize=[10,10])
for x in range(0,12):
    plt.subplot(4, 3,x+1)
    plt.imshow(cactus[x])
    plt.title(labels[x])
    x += 1

plt.tight_layout()
plt.show()

## --- ERROR in cell 17, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/169737849.py in <cell line: 0>()
     25 for x in range(0,12):
     26     plt.subplot(4, 3,x+1)
---> 27     plt.imshow(cactus[x])
     28     plt.title(labels[x])
     29     x += 1

/usr/local/lib/python3.11/dist-packages/matplotlib/pyplot.py in imshow(X, cmap, norm, aspect, interpolation, alpha, vmin, vmax, origin, extent, interpolation_stage, filternorm, filterrad, resample, url, data, **kwargs)
   2693         interpolation_stage=None, filternorm=True, filterrad=4.0,
   2694         resample=None, url=None, data=None, **kwargs):
-> 2695     __ret = gca().imshow(
   2696         X, cmap=cmap, norm=norm, aspect=aspect,
   2697         interpolation=interpolation, alpha=alpha, vmin=vmin,

/usr/local/lib/python3.11/dist-packages/matplotlib/__init__.py in inner(ax, data, *args, **kwargs)
   1444     def inner(ax, *args, data=None, **kwargs):
   1445         if data is None:
-> 1446             return func(ax, *map(sanitize_sequence, args), **kwargs)
   1447 
   1448         bound = new_sig.bind(ax, *args, **kwargs)

/usr/local/lib/python3.11/dist-packages/matplotlib/axes/_axes.py in imshow(self, X, cmap, norm, aspect, interpolation, alpha, vmin, vmax, origin, extent, interpolation_stage, filternorm, filterrad, resample, url, **kwargs)
   5661                               **kwargs)
   5662 
-> 5663         im.set_data(X)
   5664         im.set_alpha(alpha)
   5665         if im.get_clip_path() is None:

/usr/local/lib/python3.11/dist-packages/matplotlib/image.py in set_data(self, A)
    699         if (self._A.dtype != np.uint8 and
    700                 not np.can_cast(self._A.dtype, float, "same_kind")):
--> 701             raise TypeError("Image data of dtype {} cannot be converted to "
    702                             "float".format(self._A.dtype))
    703 

TypeError: Image data of dtype object cannot be converted to float

## === cell 18
import cv2
cactus = []
cactus.append(cv2.imread(train_dir + '/' + train_df['id'][0]))
cactus.append(cv2.imread(train_dir + '/' + train_df['id'][1]))
cactus.append(cv2.imread(train_dir + '/' + train_df['id'][2]))
cactus.append(cv2.imread(train_dir + '/' + train_df['id'][3]))
cactus.append(cv2.imread(train_dir + '/' + train_df['id'][4]))
cactus.append(cv2.imread(train_dir + '/' + train_df['id'][5]))
cactus.append(cv2.imread(train_dir + '/' + train_df['id'][6]))
cactus.append(cv2.imread(train_dir + '/' + train_df['id'][7]))
cactus.append(cv2.imread(train_dir + '/' + train_df['id'][8]))
cactus.append(cv2.imread(train_dir + '/' + train_df['id'][9]))
cactus.append(cv2.imread(train_dir + '/' + train_df['id'][10]))
cactus.append(cv2.imread(train_dir + '/' + train_df['id'][11]))

import matplotlib.pyplot as plt

plt.figure(figsize=(10, 10))
for x in range(0,12):
    plt.subplot(4, 3, x + 1)
    plt.imshow(cactus[x])
    x += 1

plt.tight_layout()
plt.show()

## --- ERROR in cell 18, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4096204402.py in <cell line: 0>()
     20 for x in range(0,12):
     21     plt.subplot(4, 3, x + 1)
---> 22     plt.imshow(cactus[x])
     23     x += 1
     24 

/usr/local/lib/python3.11/dist-packages/matplotlib/pyplot.py in imshow(X, cmap, norm, aspect, interpolation, alpha, vmin, vmax, origin, extent, interpolation_stage, filternorm, filterrad, resample, url, data, **kwargs)
   2693         interpolation_stage=None, filternorm=True, filterrad=4.0,
   2694         resample=None, url=None, data=None, **kwargs):
-> 2695     __ret = gca().imshow(
   2696         X, cmap=cmap, norm=norm, aspect=aspect,
   2697         interpolation=interpolation, alpha=alpha, vmin=vmin,

/usr/local/lib/python3.11/dist-packages/matplotlib/__init__.py in inner(ax, data, *args, **kwargs)
   1444     def inner(ax, *args, data=None, **kwargs):
   1445         if data is None:
-> 1446             return func(ax, *map(sanitize_sequence, args), **kwargs)
   1447 
   1448         bound = new_sig.bind(ax, *args, **kwargs)

/usr/local/lib/python3.11/dist-packages/matplotlib/axes/_axes.py in imshow(self, X, cmap, norm, aspect, interpolation, alpha, vmin, vmax, origin, extent, interpolation_stage, filternorm, filterrad, resample, url, **kwargs)
   5661                               **kwargs)
   5662 
-> 5663         im.set_data(X)
   5664         im.set_alpha(alpha)
   5665         if im.get_clip_path() is None:

/usr/local/lib/python3.11/dist-packages/matplotlib/image.py in set_data(self, A)
    699         if (self._A.dtype != np.uint8 and
    700                 not np.can_cast(self._A.dtype, float, "same_kind")):
--> 701             raise TypeError("Image data of dtype {} cannot be converted to "
    702                             "float".format(self._A.dtype))
    703 

TypeError: Image data of dtype object cannot be converted to float

## === cell 20
from keras import applications
from tensorflow.keras.applications import EfficientNetB3
from keras import callbacks
from keras.models import Sequential

## --- ERROR in cell 20, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 22
train_df['has_cactus'] = train_df['has_cactus'].astype('str')

## === cell 24
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import random

def custom_preprocessing(image):
    k = random.randint(0, 3)
    image = np.rot90(image, k)
    
    if random.random() > 0.5:
        image = np.fliplr(image)

    if random.random() > 0.5:
        image = np.flipud(image)

    factor = random.uniform(0.8, 1.2)
    image = np.clip(image * factor, 0, 255).astype(np.uint32) / 255.0 # 画像値を0〜255から0〜1に正規化
        
    return image
    
train_datagen = ImageDataGenerator(
    validation_split=0.10,      # データの10%を検証用に分ける
    
    preprocessing_function=custom_preprocessing,  # カスタム処理を適用
)

train_generator = train_datagen.flow_from_dataframe(
    dataframe = train_df,
    directory = train_dir,
    x_col="id",
    y_col="has_cactus",
    target_size=(32,32),
    subset="training",
    batch_size=128,      # バッチサイズ：64→128に変更
    shuffle=True,
    class_mode="binary"
)

val_generator = train_datagen.flow_from_dataframe(
    dataframe = train_df,
    directory = train_dir,
    x_col="id",
    y_col="has_cactus",
    target_size=(32,32),
    subset="validation",
    batch_size=64,      # バッチサイズ：256→64に変更
    shuffle=True,
    class_mode="binary"
)

## === cell 27
import matplotlib.pyplot as plt

cactus = []
for i in range(12):
    img = cv2.imread(train_dir + '/' + train_df['id'][i])
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # OpenCVはBGRなのでRGBに変換
    cactus.append(img)

cactus_augmented = [custom_preprocessing(img) for img in cactus]

plt.figure(figsize=(10, 10))
for i in range(12):
    plt.subplot(4, 3, i + 1)
    plt.imshow(cactus_augmented[i])
    plt.title(f"Image {i+1}")
    plt.axis('off')

plt.tight_layout()
plt.show()


## --- ERROR in cell 27, traceback:
---------------------------------------------------------------------------
error                                     Traceback (most recent call last)
/tmp/ipykernel_11/4159764922.py in <cell line: 0>()
      5 for i in range(12):
      6     img = cv2.imread(train_dir + '/' + train_df['id'][i])
----> 7     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # OpenCVはBGRなのでRGBに変換
      8     cactus.append(img)
      9 

error: OpenCV(4.12.0) /io/opencv/modules/imgproc/src/color.cpp:199: error: (-215:Assertion failed) !_src.empty() in function 'cvtColor'


## === cell 28
test_datagen = ImageDataGenerator(
    rescale=1/255        # 画像値を0〜255から0〜1に正規化
)

test_generator = test_datagen.flow_from_directory(
    directory = test_dir,
    target_size=(32,32),
    batch_size=1,
    shuffle=False,
    class_mode=None     # ラベル（クラス）がない前提で読み込み（予測専用）
)

## === cell 31
from keras.layers import Dense
from keras.optimizers import Adam

efficient_net = EfficientNetB3(
    weights='imagenet',      # ImageNet で学習された EfficientNetB3 の重みを 再利用
    input_shape=(32,32,3),
    include_top=False,       # 上位層（分類層）は使わない
    pooling='max'            # 出力を1ベクトルにする
)

model = Sequential()        # KerasのSequentialモデルを初期化
model.add(efficient_net)    #  EfficientNetB3モデルを追加
model.add(Dense(units = 120, activation='relu'))   # 全結合層（Dense層）を追加
model.add(Dense(units = 120, activation = 'relu')) # もう一層全結合層（Dense層）を追加
model.add(Dense(units = 1, activation='sigmoid'))  # 最終出力層を追加（出力は0～1）
model.summary()  # モデル全体の構成・パラメータ数を表示

## === cell 33
model.compile(
    optimizer=Adam(learning_rate=0.0001),  # 最適化手法をAdamに設定、学習率は0.0001
    loss='binary_crossentropy',            # 損失関数をバイナリクロスエントロピーに設定
    metrics=['accuracy']                   # 学習中や評価時に「正解率」を計測するよう指定
)

## === cell 35
history = model.fit(
    train_generator,
    epochs = 50,
    steps_per_epoch = 123,   # 246→123に変更。通常、訓練画像枚数 / バッチサイズ でok
    validation_data = val_generator,
    validation_steps = 27,    # 27→27で変更なし。通常は 検証画像枚数 / バッチサイズ でok
    class_weight=class_weights_dict  # クラスウェイトの適用
)

## --- ERROR in cell 35, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/1592235620.py in <cell line: 0>()
----> 1 history = model.fit(
      2     train_generator,
      3     epochs = 50,
      4     # 1エポック内で何回学習ステップを行うか（何バッチ処理するか）を指定
      5     steps_per_epoch = 123,   # 246→123に変更。通常、訓練画像枚数 / バッチサイズ でok

/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py in error_handler(*args, **kwargs)
    120             # To get the full stack trace, call:
    121             # `keras.config.disable_traceback_filtering()`
--> 122             raise e.with_traceback(filtered_tb) from None
    123         finally:
    124             del filtered_tb

/usr/local/lib/python3.11/dist-packages/keras/src/trainers/data_adapters/py_dataset_adapter.py in get_tf_dataset(self)
    293             ]
    294             if len(batches) == 0:
--> 295                 raise ValueError("The PyDataset has length 0")
    296             self._output_signature = data_adapter_utils.get_tensor_spec(batches)
    297 

ValueError: The PyDataset has length 0

## === cell 37
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1,len(acc) + 1)

plt.plot(epochs,acc,'bo',label = 'Training Accuracy')
plt.plot(epochs,val_acc,'b',label = 'Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()
plt.figure()

plt.plot(epochs,loss,'bo',label = 'Training loss')
plt.plot(epochs,val_loss,'b',label = 'Validation Loss')
plt.title('Training and Validation Loss')
plt.legend()

plt.show()

## --- ERROR in cell 37, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3440241191.py in <cell line: 0>()
----> 1 acc = history.history['accuracy']
      2 val_acc = history.history['val_accuracy']
      3 loss = history.history['loss']
      4 val_loss = history.history['val_loss']
      5 

NameError: name 'history' is not defined

## === cell 39
preds = model.predict(
    test_generator,
    steps=len(test_generator.filenames),
    verbose=1    # 進捗バーも表示される（追記）
)

## --- ERROR in cell 39, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/1152484999.py in <cell line: 0>()
      1 # preds = model.predict_generator
----> 2 preds = model.predict(
      3     test_generator,
      4     steps=len(test_generator.filenames),
      5     verbose=1    # 進捗バーも表示される（追記）

/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py in error_handler(*args, **kwargs)
    120             # To get the full stack trace, call:
    121             # `keras.config.disable_traceback_filtering()`
--> 122             raise e.with_traceback(filtered_tb) from None
    123         finally:
    124             del filtered_tb

/usr/local/lib/python3.11/dist-packages/keras/src/trainers/data_adapters/py_dataset_adapter.py in get_tf_dataset(self)
    293             ]
    294             if len(batches) == 0:
--> 295                 raise ValueError("The PyDataset has length 0")
    296             self._output_signature = data_adapter_utils.get_tensor_spec(batches)
    297 

ValueError: The PyDataset has length 0

## === cell 40
image_ids = [name.split('/')[-1] for name in test_generator.filenames]
predictions = preds.flatten()
data = {'id': image_ids, 'has_cactus':predictions} 
submission = pd.DataFrame(data)
print(submission.head(20))







## --- ERROR in cell 40, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2429617665.py in <cell line: 0>()
      1 image_ids = [name.split('/')[-1] for name in test_generator.filenames]
----> 2 predictions = preds.flatten()
      3 data = {'id': image_ids, 'has_cactus':predictions}
      4 submission = pd.DataFrame(data)
      5 print(submission.head(20))

NameError: name 'preds' is not defined

## === cell 42
submission.to_csv("/kaggle/working/submission.csv", index=False)

## --- ERROR in cell 42, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2768606786.py in <cell line: 0>()
----> 1 submission.to_csv("/kaggle/working/submission.csv", index=False)

NameError: name 'submission' is not defined

## === cell 43
print(os.listdir("/kaggle/working"))
