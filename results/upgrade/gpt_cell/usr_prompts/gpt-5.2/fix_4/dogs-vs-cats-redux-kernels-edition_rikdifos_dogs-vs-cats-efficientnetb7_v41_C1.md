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
protobuf==6.33.0
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
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
tf_keras==2.18.0

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

0.19583

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
!pip install -q efficientnet


## === cell 1

import sys, subprocess

try:
    import google.protobuf
    from packaging import version

    _pb_ver = version.parse(google.protobuf.__version__)
    if _pb_ver.major >= 5:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-q", "protobuf==4.25.3"]
        )
        import os

        os._exit(0)
except Exception:
    pass

try:
    get_ipython().run_line_magic("matplotlib", "inline")
    get_ipython().run_line_magic("config", "InlineBackend.figure_format = 'svg'")
except Exception:
    pass

import warnings

warnings.filterwarnings("ignore")
import os, cv2, re, random, time, zipfile, gc
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import log_loss, accuracy_score
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator, load_img
from keras import layers, models, optimizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import RMSprop, Adam

import tensorflow as tf
from tensorflow.keras.applications import efficientnet as efn


## === cell 2
PATH = '/kaggle/input/dogs-vs-cats-redux-kernels-edition/'
train_image_path = os.path.join(PATH, 'train.zip')
test_image_path = os.path.join(PATH, 'test.zip')

with zipfile.ZipFile(train_image_path,"r") as z:
    z.extractall("./data") # target dir
    z.close()
    
with zipfile.ZipFile(test_image_path,"r") as z:
    z.extractall("./data")
    z.close()


## === cell 3
start = time.time() 

TRAIN_DIR = './data/train/'
TEST_DIR = './data/test/'

train_images = [TRAIN_DIR+i for i in os.listdir(TRAIN_DIR)] # use this for full dataset
test_images = [TEST_DIR+i for i in os.listdir(TEST_DIR)]


## === cell 4
def txt_dig(text):
    '''Input string, if it is a number, output the number, 
       if not, output the original string'''
    return int(text) if text.isdigit() else text

def natural_keys(text):
    '''Enter a string, separate the number from the text, 
       and convert the number string to int'''
    return [ txt_dig(c) for c in re.split('(\d+)', text) ]


## === cell 5
train_images.sort(key=natural_keys) # 依据编号进行重新排序
test_images.sort(key=natural_keys)

train_images = train_images[0:7500] +  train_images[17500:25000]  #抽样
random.seed(558)
random.shuffle(train_images)


## === cell 6
IMG_WIDTH = 128
IMG_HEIGHT = 128
x = []
for img in train_images:
    x.append(cv2.resize(cv2.imread(img), 
                        (IMG_WIDTH, IMG_HEIGHT), 
                        interpolation=cv2.INTER_CUBIC))
    
test = []
for img in test_images:
    test.append(cv2.resize(cv2.imread(img), 
                        (IMG_WIDTH, IMG_HEIGHT), 
                        interpolation=cv2.INTER_CUBIC))
    
print('The shape of train data is {}'.format(np.array(x).shape))
print('The shape of test data is {}'.format(np.array(test).shape))

plt.rcParams['figure.facecolor'] = 'white'
y = []
for i in train_images:
    if 'dog' in i:
        y.append(1)
    elif 'cat' in i:
        y.append(0)
len(y)

x = np.array(x)
y = np.array(y)
test = np.array(test)
sns.countplot(y)


## === cell 7
random.seed(558)
plt.subplots(facecolor='white',figsize=(10,20))
sample = random.choice(train_images)
image = load_img(sample)
plt.subplot(131)
plt.imshow(image)

sample = random.choice(train_images)
image = load_img(sample)
plt.subplot(132)
plt.imshow(image)

sample = random.choice(train_images)
image = load_img(sample)
plt.subplot(133)
plt.imshow(image)


## === cell 8
plt.subplots(facecolor='white',figsize=(10,20))
plt.subplot(131)
plt.imshow(cv2.cvtColor(x[1024,:,:,:], cv2.COLOR_BGR2RGB))
plt.subplot(132)
plt.imshow(cv2.cvtColor(x[546,:,:,:], cv2.COLOR_BGR2RGB))
plt.subplot(133)
plt.imshow(cv2.cvtColor(x[742,:,:,:], cv2.COLOR_BGR2RGB))


## === cell 9
x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=2020)


## === cell 10
model = models.Sequential()


efnModel = efn.EfficientNetB7(weights = 'imagenet', 
                       input_shape = (IMG_WIDTH, IMG_HEIGHT,3), 
                       include_top = False)
model.add(efnModel)
model.add(layers.GlobalAveragePooling2D())
model.add(layers.Dense(1, activation='sigmoid'))

opt1 = RMSprop(lr=1e-5, decay=1e-6)
opt2 = Adam(lr=2e-4) 

model.compile(loss='binary_crossentropy',
              optimizer = opt2, 
              metrics = ['accuracy'])

model.summary()


## === cell 11
datagen = ImageDataGenerator(
            rescale=1. / 255,            # 将数据放缩到0-1范围内
            rotation_range=40,           # 图像随机旋转的角度范围
            width_shift_range=0.2,       # 图像在水平方向上平移的范围
            height_shift_range=0.2,      # 图像在垂直方向上平移的范围
            shear_range=0.2,             # 随机错切变换的角度
            zoom_range=0.2,              # 图像随机缩放的范围
            horizontal_flip=True,        # 随机将一半图像水平翻转
            fill_mode='nearest')         # 填充新创建像素的方法

val_datagen = ImageDataGenerator(rescale=1./255)


## === cell 12
def plot_gened(train_images,seed=320):
    '''plot pictures after processing
    '''
    df = pd.DataFrame({'filename': train_images})
    np.random.seed(seed)
    vis_df = df.sample(n=1).reset_index(drop=True)
    vis_df['category'] = '0'
    vis_gen = ImageDataGenerator(
            rescale=1. / 255,             # Scale data to 0-1 range
            rotation_range=40,            # The angle range of the image randomly rotated
            width_shift_range=0.2,        # The range of image translation in the horizontal direction
            height_shift_range=0.2,       # The range of image translation in the vertical direction
            shear_range=0.2,              # Random staggered transformation angle
            zoom_range=0.2,               # Random image zoom range
            horizontal_flip=True,         # Randomly flip half of the image horizontally
            fill_mode='nearest')          # How to fill in newly created pixels

    vis_gen0 = vis_gen.flow_from_dataframe(vis_df,
                                       x_col='filename',
                                       y_col='category',
                                       target_size=(IMG_WIDTH, IMG_HEIGHT),
                                       batch_size = 16)
    plt.rcParams['figure.facecolor'] = 'white'
    plt.figure(figsize=(8, 8))
    for i in range(0, 9):
        plt.subplot(3, 3, i+1)
        for X_batch, Y_batch in vis_gen0:
            image = X_batch[0]
            plt.imshow(image)
            break
    plt.tight_layout()
    plt.show()
    
plot_gened(train_images)    


## === cell 13
BATCH_SIZE = 16
datagen = datagen.flow(x_train, y_train, batch_size=BATCH_SIZE)
val_datagen = val_datagen.flow(x_val, y_val, batch_size=BATCH_SIZE)

earlystop1 = EarlyStopping(patience=5)
earlystop2 = ReduceLROnPlateau(monitor = 'val_accuracy', min_lr = 0.001, 
                               patience = 5, mode = 'min', 
                               verbose = 1)

history = model.fit(datagen, 
                    steps_per_epoch=45,
                    epochs=20,
                    validation_data=val_datagen,
                    callbacks=[earlystop1, earlystop2],
                    validation_steps=25)


## === cell 14
plt.rcParams['figure.facecolor'] = 'white'
model_loss = pd.DataFrame(history.history)
model_loss.head()
model_loss[['accuracy','val_accuracy']].plot(ylim=[0,1]);
model_loss[['loss','val_loss']].plot(ylim=[0,1]);


## === cell 15
x_val = x_val.astype('float32') / 255
val_preds = model.predict(x_val)
print(val_preds.ravel().dtype)
val_preds_class = np.where(val_preds.ravel() > 0.5, 1, 0) 

print('Out of Fold Accuracy is {:.5}'.format(accuracy_score(y_val, val_preds_class)))
print('Out of Fold log loss is {:.5}'.format(log_loss(y_val, val_preds.ravel() \
                                                      .astype('float64'))))


## === cell 16
test = test.astype('float32') / 255
test_pred = model.predict(test)
submission = pd.DataFrame({'id': range(1, len(test_images) + 1), 'label': test_pred.ravel()})
submission.to_csv('submission.csv', index = False)
print('This program costs {:.2f} seconds'.format(time.time()-start))
submission


## === cell 17
!rm -rf /kaggle/working/data/
