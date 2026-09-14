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
Predict engagement with a pet's profile based on the photograph for that profile.

## Metric
Root mean squared error.

## Submission Format
For each `Id` in the test set, you must predict a probability for the target variable, `Pawpularity`. The file should contain a header and have the following format:

```
Id, Pawpularity
0008dbfb52aa1dc6ee51ee02adf13537, 99.24
0014a7b528f1682f0cf3b73a991c17a0, 61.71
0019c1388dfcd30ac8b112fb4250c251, 6.23
00307b779c82716b240a24f028b0031b, 9.43
00320c6dd5b4223c62a9670110d47911, 70.89
etc.
```

## Dataset
- **train/** - Folder containing training set photos of the form **{id}.jpg**, where **{id}** is a unique Pet Profile ID.
- **train.csv** - Metadata (described below) for each photo in the training set as well as the target, the photo's Pawpularity score. The Id column gives the photo's unique Pet Profile ID corresponding the photo's file name.

The train.csv and test.csv files contain metadata for photos in the training set and test set, respectively. Each pet photo is labeled with the value of 1 (Yes) or 0 (No) for each of the following features:

- **Focus** - Pet stands out against uncluttered background, not too close / far.
- **Eyes** - Both eyes are facing front or near-front, with at least 1 eye / pupil decently clear.
- **Face** - Decently clear face, facing front or near-front.
- **Near** - Single pet taking up significant portion of photo (roughly over 50% of photo width or height).
- **Action** - Pet in the middle of an action (e.g., jumping).
- **Accessory** - Accompanying physical or digital accessory / prop (i.e. toy, digital sticker), excluding collar and leash.
- **Group** - More than 1 pet in the photo.
- **Collage** - Digitally-retouched photo (i.e. with digital photo frame, combination of multiple photos).
- **Human** - Human in the photo.
- **Occlusion** - Specific undesirable objects blocking part of the pet (i.e. human, cage or fence). Note that not all blocking objects are considered occlusion.
- **Info** - Custom-added text or labels (i.e. pet name, description).
- **Blur** - Noticeably out of focus or noisy, especially for the pet's eyes and face. For Blur entries, "Eyes" column is always set to 0.

# 2. Python version

3.10

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

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (132 lines)
            sample_submission.csv (993 lines)
            sample_submission.csv.zip (22.7 kB)
            test.csv (993 lines)
            test.csv.zip (22.5 kB)
            test.zip (102.2 MB)
            train.csv (8921 lines)
            train.csv.zip (213.0 kB)
            train.zip (926.9 MB)
            petfinder-pawpularity-score/
                description.md (132 lines)
                sample_submission.csv (993 lines)
                ... and 7 other files
                petfinder-pawpularity-score/
                test/
                    a5c4de4c29e2097889f0d4cbd566625c.jpg (57.3 kB)
                    2e5cda2c4cf0530423e8181b7f8c67bc.jpg (94.9 kB)
                    ... and 990 other files
                    test/
                train/
                    e449bbacd2930d6f6ab4589182a09185.jpg (95.7 kB)
                    cfe66786a9c53db0e6936209291cc67d.jpg (247.5 kB)
                    ... and 8918 other files
                    train/
            test/
                a5c4de4c29e2097889f0d4cbd566625c.jpg (57.3 kB)
                2e5cda2c4cf0530423e8181b7f8c67bc.jpg (94.9 kB)
                ... and 990 other files
                test/
            train/
                e449bbacd2930d6f6ab4589182a09185.jpg (95.7 kB)
                cfe66786a9c53db0e6936209291cc67d.jpg (247.5 kB)
                ... and 8918 other files
                train/
        input/
            description.md (132 lines)
            sample_submission.csv (993 lines)
            sample_submission.csv.zip (22.7 kB)
            test.csv (993 lines)
            test.csv.zip (22.5 kB)
            test.zip (102.2 MB)
            train.csv (8921 lines)
            train.csv.zip (213.0 kB)
            train.zip (926.9 MB)
            petfinder-pawpularity-score/
                description.md (132 lines)
                sample_submission.csv (993 lines)
                ... and 7 other files
                petfinder-pawpularity-score/
                test/
                    a5c4de4c29e2097889f0d4cbd566625c.jpg (57.3 kB)
                    2e5cda2c4cf0530423e8181b7f8c67bc.jpg (94.9 kB)
                    ... and 990 other files
                    test/
                train/
                    e449bbacd2930d6f6ab4589182a09185.jpg (95.7 kB)
                    cfe66786a9c53db0e6936209291cc67d.jpg (247.5 kB)
                    ... and 8918 other files
                    train/
            test/
                a5c4de4c29e2097889f0d4cbd566625c.jpg (57.3 kB)
                2e5cda2c4cf0530423e8181b7f8c67bc.jpg (94.9 kB)
                ... and 990 other files
                test/
                    a5c4de4c29e2097889f0d4cbd566625c.jpg (57.3 kB)
                    2e5cda2c4cf0530423e8181b7f8c67bc.jpg (94.9 kB)
                    ... and 990 other files
                    test/
            train/
                e449bbacd2930d6f6ab4589182a09185.jpg (95.7 kB)
                cfe66786a9c53db0e6936209291cc67d.jpg (247.5 kB)
                ... and 8918 other files
                train/
                    e449bbacd2930d6f6ab4589182a09185.jpg (95.7 kB)
                    cfe66786a9c53db0e6936209291cc67d.jpg (247.5 kB)
                    ... and 8918 other files
                    train/
        working/
            petfinder-pawpularity-score/
                description.md (132 lines)
                sample_submission.csv (993 lines)
                ... and 7 other files
                petfinder-pawpularity-score/
                test/
                    a5c4de4c29e2097889f0d4cbd566625c.jpg (57.3 kB)
                    2e5cda2c4cf0530423e8181b7f8c67bc.jpg (94.9 kB)
                    ... and 990 other files
                    test/
                train/
                    e449bbacd2930d6f6ab4589182a09185.jpg (95.7 kB)
                    cfe66786a9c53db0e6936209291cc67d.jpg (247.5 kB)
                    ... and 8918 other files
                    train/
```

-> data/petfinder-pawpularity-score/sample_submission.csv has 992 rows and 2 columns.
The columns are: Id, Pawpularity

-> data/petfinder-pawpularity-score/test.csv has 992 rows and 13 columns.
The columns are: Id, Subject Focus, Eyes, Face, Near, Action, Accessory, Group, Collage, Human, Occlusion, Info, Blur

-> data/petfinder-pawpularity-score/train.csv has 8920 rows and 14 columns.
The columns are: Id, Subject Focus, Eyes, Face, Near, Action, Accessory, Group, Collage, Human, Occlusion, Info, Blur, Pawpularity

-> data/sample_submission.csv has 992 rows and 2 columns.
The columns are: Id, Pawpularity

-> data/test.csv has 992 rows and 13 columns.
The columns are: Id, Subject Focus, Eyes, Face, Near, Action, Accessory, Group, Collage, Human, Occlusion, Info, Blur

-> data/train.csv has 8920 rows and 14 columns.
The columns are: Id, Subject Focus, Eyes, Face, Near, Action, Accessory, Group, Collage, Human, Occlusion, Info, Blur, Pawpularity

-> (stopped after 10 files for performance)

# 5. Target score

20.56963

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import pandas as pd
import numpy as np
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import MaxPool2D
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import GlobalAveragePooling2D
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

from keras_preprocessing  import image
from keras_preprocessing.image import ImageDataGenerator
from scipy import stats
from sklearn.metrics import mean_squared_error
from scipy import special
from sklearn.preprocessing import StandardScaler


## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
images_dir = r"../input/petfinder-pawpularity-score/train"
images_dir_test = r"../input/petfinder-pawpularity-score/test"


## === cell 2
def image_size(r):
    img = cv2.imread(r)
    return img.shape[0] * img.shape[1]

def image_ratio(r):
    img = cv2.imread(r)
    return img.shape[0] / img.shape[1]


## === cell 3
df_beg = pd.read_csv('../input/petfinder-pawpularity-score/train.csv')
df_beg['file'] = df_beg['Id'] + ".jpg"
df_beg['path'] = images_dir + '/' + df_beg['file']


## === cell 4
df_beg['image_size'] = df_beg.apply(lambda r: image_size(r['path']), axis=1)
df_beg['image_ratio'] = df_beg.apply(lambda r: image_ratio(r['path']), axis=1)

test_df = pd.read_csv('../input/petfinder-pawpularity-score/test.csv')
test_df['file'] = test_df['Id'] + ".jpg"
test_df['path'] = images_dir_test + '/' + test_df['file']
test_df['image_size'] = test_df.apply(lambda r: image_size(r['path']), axis=1)
test_df['image_ratio'] = test_df.apply(lambda r: image_ratio(r['path']), axis=1)




df = df_beg.copy()


## === cell 6
y_bc, lambda_bc_train = stats.boxcox(df['Pawpularity'])
df['Pawpularity_box'] = y_bc


df = shuffle(df)
train_size = 0.91
train_cut = int(len(df) * train_size)
train_df = df[:train_cut].copy()
validate_df = df[train_cut:].copy()


## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1137879887.py in <cell line: 0>()
----> 1 y_bc, lambda_bc_train = stats.boxcox(df['Pawpularity'])
      2 df['Pawpularity_box'] = y_bc
      3 
      4 
      5 df = shuffle(df)

NameError: name 'stats' is not defined

## === cell 7
def to_log(r):
    return np.log(r['Pawpularity'])

def to_sqrt(r):
    return (np.sqrt(r['Pawpularity']))


## === cell 8
train_df['Pawpularity_lg'] = train_df.apply(lambda r: to_log(r), axis=1)
validate_df['Pawpularity_lg'] = validate_df.apply(lambda r: to_log(r), axis=1)

train_df['Pawpularity_sqrt'] = train_df.apply(lambda r: to_sqrt(r), axis=1)
validate_df['Pawpularity_sqrt'] = validate_df.apply(lambda r: to_sqrt(r), axis=1)


## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/558245237.py in <cell line: 0>()
----> 1 train_df['Pawpularity_lg'] = train_df.apply(lambda r: to_log(r), axis=1)
      2 validate_df['Pawpularity_lg'] = validate_df.apply(lambda r: to_log(r), axis=1)
      3 
      4 train_df['Pawpularity_sqrt'] = train_df.apply(lambda r: to_sqrt(r), axis=1)
      5 validate_df['Pawpularity_sqrt'] = validate_df.apply(lambda r: to_sqrt(r), axis=1)

NameError: name 'train_df' is not defined

## === cell 9
train_df.columns


## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1723399785.py in <cell line: 0>()
----> 1 train_df.columns

NameError: name 'train_df' is not defined

## === cell 10

x_train = train_df[['Subject Focus', 'Eyes', 'Face', 'Near', 'Action', 'Accessory',
       'Group', 'Collage', 'Human', 'Occlusion', 'Info', 'Blur', 
                    'image_size', 'image_ratio']].copy()
x_validate = validate_df[['Subject Focus', 'Eyes', 'Face', 'Near', 'Action', 'Accessory',
       'Group', 'Collage', 'Human', 'Occlusion', 'Info', 'Blur', 
                    'image_size', 'image_ratio']].copy()


sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_validate = sc.transform (x_validate)



y_train = train_df['Pawpularity_box']
y_validate = validate_df['Pawpularity_box']


## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3335541879.py in <cell line: 0>()
      1 #tabular data
      2 
----> 3 x_train = train_df[['Subject Focus', 'Eyes', 'Face', 'Near', 'Action', 'Accessory',
      4        'Group', 'Collage', 'Human', 'Occlusion', 'Info', 'Blur',
      5                     'image_size', 'image_ratio']].copy()

NameError: name 'train_df' is not defined

## === cell 12
train_df.hist('Pawpularity')
train_df.hist('Pawpularity_lg')
train_df.hist('Pawpularity_sqrt')
train_df.hist('Pawpularity_box')
train_df.hist('image_ratio')
train_df.hist('image_size')


## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3946133790.py in <cell line: 0>()
----> 1 train_df.hist('Pawpularity')
      2 train_df.hist('Pawpularity_lg')
      3 train_df.hist('Pawpularity_sqrt')
      4 train_df.hist('Pawpularity_box')
      5 train_df.hist('image_ratio')

NameError: name 'train_df' is not defined

## === cell 14
train_df.columns


## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1723399785.py in <cell line: 0>()
----> 1 train_df.columns

NameError: name 'train_df' is not defined

## === cell 15
image_size = (150, 300, 3)
batch_size = 5
classes_no = 1


## === cell 16
class CustomDataGen(tf.keras.utils.Sequence):
    
    def __init__(self, df, X_col, y_col,
                 batch_size,
                 input_size=image_size,
                 shuffle=True):
        
        self.df = df.copy()
        self.X_col = X_col
        self.y_col = y_col
        self.batch_size = batch_size
        self.input_size = input_size
        self.shuffle = shuffle
        
        self.n = len(self.df)
        self.n_name = df[y_col].nunique()

    
    def on_epoch_end(self):
        if self.shuffle:
            self.df = self.df.sample(frac=1).reset_index(drop=True)
    
    def __get_input(self, path, target_size):
    

        image = tf.keras.preprocessing.image.load_img(path)
        image_arr = tf.keras.preprocessing.image.img_to_array(image)

        image_arr = tf.image.resize(image_arr,(target_size[0], target_size[1])).numpy()

        return image_arr/255.
    
    def __get_output(self, label, num_classes):
        return label #label -- regresijai #tf.keras.utils.to_categorical(label, num_classes=num_classes) - classifikacija
    
    def __get_data(self, batches):

        path_batch = batches[self.X_col]

        
        name_batch = batches[self.y_col]


        X_batch = np.asarray([self.__get_input(x, self.input_size) for x in path_batch])

        y0_batch = np.asarray([self.__get_output(y, self.n_name) for y in name_batch])


        return X_batch, y0_batch
    
    def __getitem__(self, index):
        
        batches = self.df[index * self.batch_size:(index + 1) * self.batch_size]
        X, y = self.__get_data(batches)        
        return X, y
    
    def __len__(self):
        return self.n // self.batch_size


## === cell 17
traingen = CustomDataGen(train_df,
                         X_col='path',
                         y_col='Pawpularity_box',
                         batch_size=batch_size, input_size=image_size)

valgen = CustomDataGen(validate_df,
                       X_col='path',
                       y_col='Pawpularity_box',
                       batch_size=batch_size, input_size=image_size)
testgen = CustomDataGen(test_df,
                       X_col='path',
                       y_col='file',
                       batch_size=batch_size, input_size=image_size)


## --- ERROR in cell 17, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2582088872.py in <cell line: 0>()
----> 1 traingen = CustomDataGen(train_df,
      2                          X_col='path',
      3                          y_col='Pawpularity_box',
      4                          batch_size=batch_size, input_size=image_size)
      5 

NameError: name 'train_df' is not defined

## === cell 18
def conv_block(channels, kernel_size=(3,3), activation='relu',use_bn=True):
    model=tf.keras.models.Sequential()
    model.add(tf.keras.layers.Conv2D(channels, kernel_size=kernel_size, activation=None, padding='same'))
    if use_bn:
        model.add(tf.keras.layers.BatchNormalization())

    if activation=='relu':
        model.add(tf.keras.layers.ReLU())
    return model

def inception_module(x,filters_1x1,filters_3x3_reduce,filters_3x3,filters_5x5_reduce,
                     filters_5x5,filters_pool_proj, use_bn=True):
    conv_1x1 = conv_block(filters_1x1, (1, 1), use_bn=use_bn)(x)
    
    conv_3x3 = conv_block(filters_3x3_reduce, (1, 1), use_bn=use_bn)(x)
    conv_3x3 = conv_block(filters_3x3, (3, 3), use_bn=use_bn)(conv_3x3)

    conv_5x5 = conv_block(filters_5x5_reduce, (1, 1), use_bn=use_bn)(x)
    conv_5x5 = conv_block(filters_5x5, (5, 5), use_bn=use_bn)(conv_5x5)

    pool_proj = MaxPool2D((3, 3), strides=(1, 1), padding='same')(x)
    pool_proj = conv_block(filters_pool_proj, (1, 1), use_bn=use_bn)(pool_proj)

    output = tf.keras.layers.concatenate([conv_1x1, conv_3x3, conv_5x5, pool_proj], axis=3)
    return output

def tomasnet(image_size, classes_no, batch_size, activation):
    input_image = tf.keras.layers.Input(shape=image_size, batch_size=batch_size)
    x = conv_block(32, (7,7), activation='relu')(input_image)
    x = MaxPool2D((3, 3), padding='same', strides=(2, 2))(x)
    x = conv_block(32, (1,1))(x)
    x = conv_block(90, (3,3))(x)
    x = MaxPool2D((3, 3), padding='same', strides=(2, 2))(x)

    x = inception_module(x,
                     filters_1x1=32,
                     filters_3x3_reduce=48,
                     filters_3x3=64,
                     filters_5x5_reduce=8,
                     filters_5x5=16,
                     filters_pool_proj=16)


    x = MaxPool2D((3, 3), padding='same', strides=(2, 2))(x)

    x = inception_module(x,
                     filters_1x1=128,
                     filters_3x3_reduce=96,
                     filters_3x3=104,
                     filters_5x5_reduce=8,
                     filters_5x5=24,
                     filters_pool_proj=32)


    x1 = AveragePooling2D((5, 5), strides=3)(x)
    x1 = conv_block(64, (1, 1))(x1)
    x1 = Flatten()(x1)
    x1 = Dense(512, activation='relu')(x1)
    x1 = Dropout(0.7)(x1) #A dropout layer with 70% ratio of dropped outputs.

    x = inception_module(x,
                     filters_1x1=40,
                     filters_3x3_reduce=32,
                     filters_3x3=64,
                     filters_5x5_reduce=6,
                     filters_5x5=16,
                     filters_pool_proj=16)

    x = inception_module(x,
                     filters_1x1=128,
                     filters_3x3_reduce=128,
                     filters_3x3=256,
                     filters_5x5_reduce=24,
                     filters_5x5=64,
                     filters_pool_proj=64)

    x = inception_module(x,
                     filters_1x1=112,
                     filters_3x3_reduce=144,
                     filters_3x3=288,
                     filters_5x5_reduce=32,
                     filters_5x5=64,
                     filters_pool_proj=64)


    x2 = AveragePooling2D((5, 5), strides=3)(x)
    x2 = conv_block(5, (1, 1))(x2)
    x2 = Flatten()(x2)
    x2 = Dense(512, activation='relu')(x2)
    x2 = Dropout(0.7)(x2)

    x = inception_module(x,
                     filters_1x1=128,
                     filters_3x3_reduce=130,
                     filters_3x3=160,
                     filters_5x5_reduce=16,
                     filters_5x5=64,
                     filters_pool_proj=64)

    x = MaxPool2D((3, 3), padding='same', strides=(2, 2))(x)

    x = inception_module(x,
                     filters_1x1=128,
                     filters_3x3_reduce=80,
                     filters_3x3=160,
                     filters_5x5_reduce=16,
                     filters_5x5=64,
                     filters_pool_proj=64)

    x = inception_module(x,
                     filters_1x1=180,
                     filters_3x3_reduce=90,
                     filters_3x3=180,
                     filters_5x5_reduce=24,
                     filters_5x5=64,
                     filters_pool_proj=64)

    x = GlobalAveragePooling2D()(x)   #7x7 average pooling

    x = Dropout(0.4)(x)


    x = tf.keras.Model(inputs=input_image, outputs=x)
    x1 = tf.keras.Model(inputs=input_image, outputs=x1)
    x2 = tf.keras.Model(inputs=input_image, outputs=x2)

    combined = tf.keras.layers.concatenate([x.output, x1.output, x2.output])
    z = Dense(classes_no, activation=activation)(combined)
    return tf.keras.Model(input_image, outputs=z)


## === cell 19
def transition_block(x):
    c=x.shape[-1] # num channels
    x=tf.keras.layers.BatchNormalization()(x)
    x=tf.keras.layers.ReLU()(x)
    x=tf.keras.layers.Conv2D(c//2,(1,1))(x)
    x=tf.keras.layers.AveragePooling2D()(x)
    return x

def dense_block(x, k, rp, intermediate_channels=128):
    for _ in range(rp):
        x_branch=tf.keras.layers.BatchNormalization()(x)
        x_branch=tf.keras.layers.ReLU()(x_branch)
        x_branch=tf.keras.layers.Conv2D(intermediate_channels, (1,1), padding='same')(x_branch)

        x_branch=tf.keras.layers.BatchNormalization()(x_branch)
        x_branch=tf.keras.layers.ReLU()(x_branch)
        x_branch=tf.keras.layers.Conv2D(k, (3,3), padding='same')(x_branch)

        x=tf.keras.layers.concatenate([x, x_branch])
    return x


def densnet201(image_size, classes_no, batch_size, activation):
    input_image = tf.keras.layers.Input(shape=image_size, batch_size=batch_size)
    x = conv_block(64, (7,7), (2,2))(input_image)
    x = tf.keras.layers.MaxPool2D((3, 3), padding='same', strides=(2, 2))(x)
    x=dense_block(x, 32, 6)
    x=transition_block(x)
    x=dense_block(x, 32, 12)
    x=transition_block(x)
    x=dense_block(x, 32, 48)
    x=transition_block(x)
    x=dense_block(x, 32, 32)
    x=tf.keras.layers.GlobalAveragePooling2D()(x)

    x=tf.keras.layers.Dense(1000, 'relu')(x)
    x = tf.keras.layers.Dense(classes_no, activation=activation)(x)
    return tf.keras.models.Model(input_image, x)


## === cell 21
model = densnet201(image_size, classes_no, batch_size, 'linear')
model.compile(optimizer='adam', loss='mse', metrics=['RootMeanSquaredError'])


## --- ERROR in cell 21, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2556764828.py in <cell line: 0>()
----> 1 model = densnet201(image_size, classes_no, batch_size, 'linear')
      2 model.compile(optimizer='adam', loss='mse', metrics=['RootMeanSquaredError'])
      3 # model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

/tmp/ipykernel_11/1795604033.py in densnet201(image_size, classes_no, batch_size, activation)
     27     # dense block 1
     28     x=dense_block(x, 32, 6)
---> 29     x=transition_block(x)
     30     # dense block 2
     31     x=dense_block(x, 32, 12)

/tmp/ipykernel_11/1795604033.py in transition_block(x)
      4     x=tf.keras.layers.ReLU()(x)
      5     x=tf.keras.layers.Conv2D(c//2,(1,1))(x)
----> 6     x=tf.keras.layers.AveragePooling2D()(x)
      7     return x
      8 

TypeError: AveragePooling2D.__init__() missing 1 required positional argument: 'pool_size'

## === cell 26
model_tab = Sequential([
    
        tf.keras.layers.Input(shape=(14,)),
        tf.keras.layers.Dense(30, activation='relu'),
        tf.keras.layers.Dropout(0.05, noise_shape=None, seed=None),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(48, activation='relu'),
        tf.keras.layers.Dropout(0.1, noise_shape=None, seed=None),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(1, activation='linear')
        ]
        )
model_tab.compile(optimizer='adam', loss='mse', metrics=['RootMeanSquaredError'])


## === cell 27
print(x_train.shape)
print(y_train.shape)


## --- ERROR in cell 27, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1456976159.py in <cell line: 0>()
----> 1 print(x_train.shape)
      2 print(y_train.shape)

NameError: name 'x_train' is not defined

## === cell 28
model_hist = model_tab.fit(x_train, y_train, 
                         verbose=1,
                         validation_data=(x_validate, y_validate),
                         epochs=200,
                   callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=20, min_delta=0.0001)])


## --- ERROR in cell 28, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1321133198.py in <cell line: 0>()
----> 1 model_hist = model_tab.fit(x_train, y_train, 
      2                          verbose=1,
      3                          validation_data=(x_validate, y_validate),
      4                          epochs=200,
      5 #                        batch_size = 100,

NameError: name 'x_train' is not defined

## === cell 29
def prediction(r):
    img_org = cv2.imread(r['path'])
    img = cv2.resize(img_org, dsize=(300, 150), interpolation = cv2.INTER_AREA)
    img = img / 255
    return model.predict(img[np.newaxis])[0][0] ** 2

def prediction_from_box(r, model, lambda_bc_train):
    img_org = cv2.imread(r['path'])
    img = cv2.resize(img_org, dsize=(300, 150), interpolation = cv2.INTER_AREA)
    img = img / 255
    return special.inv_boxcox(model.predict(img[np.newaxis])[0][0], lambda_bc_train)


## === cell 30
test_df.columns


## === cell 31
tab_features = [ 'Subject Focus', 'Eyes', 'Face', 'Near', 'Action', 'Accessory',
       'Group', 'Collage', 'Human', 'Occlusion', 'Info', 'Blur', 'image_size', 'image_ratio']


## === cell 32
test_df_scaled = sc.transform (test_df[tab_features])
y_pred_tab = special.inv_boxcox(model_tab.predict(test_df_scaled), lambda_bc_train)


## --- ERROR in cell 32, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1211579320.py in <cell line: 0>()
----> 1 test_df_scaled = sc.transform (test_df[tab_features])
      2 y_pred_tab = special.inv_boxcox(model_tab.predict(test_df_scaled), lambda_bc_train)

NameError: name 'sc' is not defined

## === cell 33
test_df['pred_tab'] = y_pred_tab 


## --- ERROR in cell 33, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1202413850.py in <cell line: 0>()
----> 1 test_df['pred_tab'] = y_pred_tab
      2 # test_df['pred_den'] = test_df.apply(lambda r: prediction_from_box(r, model, lambda_bc_train), axis=1)
      3 # test_df['pred_tom'] = test_df.apply(lambda r: prediction_from_box(r, model1, lambda_bc_train), axis=1)

NameError: name 'y_pred_tab' is not defined

## === cell 35
test_df['Pawpularity'] = test_df.apply(lambda r: (r['pred_tab'] * 1), axis=1)


## --- ERROR in cell 35, traceback:
---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in get_loc(self, key)
   3804         try:
-> 3805             return self._engine.get_loc(casted_key)
   3806         except KeyError as err:

index.pyx in pandas._libs.index.IndexEngine.get_loc()

index.pyx in pandas._libs.index.IndexEngine.get_loc()

pandas/_libs/hashtable_class_helper.pxi in pandas._libs.hashtable.PyObjectHashTable.get_item()

pandas/_libs/hashtable_class_helper.pxi in pandas._libs.hashtable.PyObjectHashTable.get_item()

KeyError: 'pred_tab'

The above exception was the direct cause of the following exception:

KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_11/1617479527.py in <cell line: 0>()
----> 1 test_df['Pawpularity'] = test_df.apply(lambda r: (r['pred_tab'] * 1), axis=1)

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in apply(self, func, axis, raw, result_type, args, by_row, engine, engine_kwargs, **kwargs)
  10372             kwargs=kwargs,
  10373         )
> 10374         return op.apply().__finalize__(self, method="apply")
  10375 
  10376     def map(

/usr/local/lib/python3.11/dist-packages/pandas/core/apply.py in apply(self)
    914             return self.apply_raw(engine=self.engine, engine_kwargs=self.engine_kwargs)
    915 
--> 916         return self.apply_standard()
    917 
    918     def agg(self):

/usr/local/lib/python3.11/dist-packages/pandas/core/apply.py in apply_standard(self)
   1061     def apply_standard(self):
   1062         if self.engine == "python":
-> 1063             results, res_index = self.apply_series_generator()
   1064         else:
   1065             results, res_index = self.apply_series_numba()

/usr/local/lib/python3.11/dist-packages/pandas/core/apply.py in apply_series_generator(self)
   1079             for i, v in enumerate(series_gen):
   1080                 # ignore SettingWithCopy here in case the user mutates
-> 1081                 results[i] = self.func(v, *self.args, **self.kwargs)
   1082                 if isinstance(results[i], ABCSeries):
   1083                     # If we have a view on v, we need to make a copy because

/tmp/ipykernel_11/1617479527.py in <lambda>(r)
----> 1 test_df['Pawpularity'] = test_df.apply(lambda r: (r['pred_tab'] * 1), axis=1)

/usr/local/lib/python3.11/dist-packages/pandas/core/series.py in __getitem__(self, key)
   1119 
   1120         elif key_is_scalar:
-> 1121             return self._get_value(key)
   1122 
   1123         # Convert generator to list before going through hashable part

/usr/local/lib/python3.11/dist-packages/pandas/core/series.py in _get_value(self, label, takeable)
   1235 
   1236         # Similar to Index.get_value, but we do not fall back to positional
-> 1237         loc = self.index.get_loc(label)
   1238 
   1239         if is_integer(loc):

/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in get_loc(self, key)
   3810             ):
   3811                 raise InvalidIndexError(key)
-> 3812             raise KeyError(key) from err
   3813         except TypeError:
   3814             # If we have a listlike key, _check_indexing_error will raise

KeyError: 'pred_tab'

## === cell 36
test_df


## === cell 37
test_df[['Id', 'Pawpularity']].to_csv('submission.csv', index=False)


## --- ERROR in cell 37, traceback:
---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_11/1362724800.py in <cell line: 0>()
----> 1 test_df[['Id', 'Pawpularity']].to_csv('submission.csv', index=False)

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in __getitem__(self, key)
   4106             if is_iterator(key):
   4107                 key = list(key)
-> 4108             indexer = self.columns._get_indexer_strict(key, "columns")[1]
   4109 
   4110         # take() does not accept boolean indexers

/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in _get_indexer_strict(self, key, axis_name)
   6198             keyarr, indexer, new_indexer = self._reindex_non_unique(keyarr)
   6199 
-> 6200         self._raise_if_missing(keyarr, indexer, axis_name)
   6201 
   6202         keyarr = self.take(indexer)

/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in _raise_if_missing(self, key, indexer, axis_name)
   6250 
   6251             not_found = list(ensure_index(key)[missing_mask.nonzero()[0]].unique())
-> 6252             raise KeyError(f"{not_found} not in index")
   6253 
   6254     @overload

KeyError: "['Pawpularity'] not in index"
