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

20.45483

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "2")

import pandas as pd
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

train_csv = pd.read_csv("../input/petfinder-pawpularity-score/train.csv")
test_csv = pd.read_csv("../input/petfinder-pawpularity-score/test.csv")
submission = pd.read_csv("../input/petfinder-pawpularity-score/sample_submission.csv")


## === cell 1
train_csv


## === cell 2
train_csv.isnull().sum()


## === cell 3
train_csv.drop_duplicates()


## === cell 4
for i in train_csv.drop(['Id','Pawpularity'],axis=1):
    sns.countplot(train_csv[i])
    plt.show()


## === cell 5
sns.distplot(train_csv['Pawpularity'])


## === cell 6
test_csv


## === cell 7
test_csv.isnull().sum()


## === cell 8
submission


## === cell 9
os.chdir('../input/petfinder-pawpularity-score/train')

size_data = pd.DataFrame()
for file in os.listdir():
    imgg = cv2.imread(file)
    w,h,c = imgg.shape
    size_data=size_data.append([[w,h,c,imgg.size/3]])
size_data


## === cell 10
size_data[size_data[3] == size_data[3].min()]


## === cell 11
size_data[3].value_counts()


## === cell 12
size_data[size_data[3] == 691200]


## === cell 13
train_img = []
for i in os.listdir():
    file = cv2.imread(i)
    file=cv2.resize(file,(64,64), interpolation=cv2.INTER_AREA)
    train_img.append(file/255)
train_img[:5]


## === cell 14
train_img_name = []
for i in os.listdir():
    train_img_name.append(i)
train_img_name[:5]


## === cell 15
for name in train_img_name:
    if name[-4:] != '.jpg':
        print(name)


## === cell 16
train_csv_data = pd.DataFrame()
for img, name in zip(train_img, train_img_name):
    name=name[:-4]
    location = train_csv[train_csv['Id'] == name].index[0]
    train_csv_data= train_csv_data.append([train_csv.loc[location]])
train_csv_data


## === cell 17
train_csv_data=train_csv_data.reset_index().drop(['index'],axis=1)
train_csv_data


## === cell 18
image_1 = cv2.imread('./'+train_csv_data['Id'][0]+'.jpg')
plt.imshow(image_1)


## === cell 19
plt.imshow(train_img[0])


## === cell 20
image_2 = cv2.imread('./'+train_csv_data['Id'][1]+'.jpg')
plt.imshow(image_2)


## === cell 21
plt.imshow(train_img[1])


## === cell 22
os.chdir('../test')

for i in os.listdir():
    file = cv2.imread(i)
    print(file.shape)


## === cell 23
test_img = []
for i in os.listdir():
    file = cv2.imread(i)
    file=cv2.resize(file,(64,64), interpolation=cv2.INTER_AREA)
    test_img.append(file/255)
test_img[:5]


## === cell 24
test_img_name = []
for i in os.listdir():
    test_img_name.append(i)
test_img_name[:5]


## === cell 25
test_csv_data = pd.DataFrame()
for img, name in zip(test_img, test_img_name):
    name=name[:-4]
    location = test_csv[test_csv['Id'] == name].index[0]
    test_csv_data= test_csv_data.append([test_csv.loc[location]])
test_csv_data=test_csv_data.reset_index().drop(['index'],axis=1)
test_csv_data


## === cell 26
test_1 = cv2.imread('./'+test_csv_data['Id'][0]+'.jpg')
plt.imshow(test_1)


## === cell 27
plt.imshow(test_img[0])


## === cell 28
train_csv_x = train_csv_data.drop(['Id','Pawpularity'],axis=1)
train_y = train_csv_data['Pawpularity']

test_csv_x = test_csv_data.drop(['Id'],axis=1)


## === cell 29
csv_input = tf.keras.Input(shape = train_csv_x.shape[1:], name = 'CSV_Input')        ##
img_input = tf.keras.Input(shape = np.array(train_img).shape[1:], name = 'IMG_Input')##
csv_hidden1 = tf.keras.layers.Dense(200, activation='relu', name='CSV_Hidden1')(csv_input)   ##
csv_hidden2 = tf.keras.layers.Dense(200, activation='relu', name='CSV_Hidden2')(csv_hidden1)##
csv_hidden3 = tf.keras.layers.Dense(200, activation='relu', name='CSV_Hidden3')(csv_hidden2)##
csv_dropout = tf.keras.layers.Dropout(0.5, name ='CSV_Dropout')(csv_hidden3)               ##
csv_hidden4 = tf.keras.layers.Dense(200, activation='relu', name='CSV_Hidden4')(csv_dropout)##
csv_hidden5 = tf.keras.layers.Dense(200, activation='relu', name='CSV_Hidden5')(csv_hidden4)##
img_conv1 = tf.keras.layers.Conv2D(filters = 120, kernel_size = 5, strides= 1, padding = 'same',   ##
                                   activation = 'relu', name='IMG_Conv1')(img_input)              ##
img_pool1 = tf.keras.layers.MaxPool2D(3, name = 'IMG_Pool1')(img_conv1)                           ##
img_conv2 = tf.keras.layers.Conv2D(filters = 120, kernel_size = 4, strides= 1, padding = 'same',  ##
                                   activation = 'relu', name='IMG_Conv2')(img_pool1)              ##
img_conv3 = tf.keras.layers.Conv2D(filters = 120, kernel_size = 4, strides =1, padding = 'same',  ##
                                   activation = 'relu', name='IMG_Conv3')(img_conv2)              ##
img_pool2 = tf.keras.layers.MaxPool2D(3, name = 'IMG_Pool2')(img_conv3)                           ##
img_conv4 = tf.keras.layers.Conv2D(filters = 120, kernel_size = 3, strides =1, padding = 'same',  ##
                                   activation = 'relu', name='IMG_Conv4')(img_pool2)              ##
img_pool3 = tf.keras.layers.MaxPool2D(3, name = 'IMG_Pool3')(img_conv4)                           ##
img_conv5 = tf.keras.layers.Conv2D(filters = 120, kernel_size = 3, strides =1, padding = 'same',  ##
                                   activation = 'relu', name='IMG_Conv5')(img_pool3)              ##
img_conv6 = tf.keras.layers.Conv2D(filters = 120, kernel_size = 3, strides =1, padding = 'same',  ##
                                   activation = 'relu', name='IMG_Conv6')(img_conv5)              ##
img_pool4 = tf.keras.layers.MaxPool2D(2, name = 'IMG_Pool4')(img_conv6)                           ##
img_flatten = tf.keras.layers.Flatten(name = 'IMG_Flatten')(img_pool4)                            ##
img_dense1 = tf.keras.layers.Dense(300, activation = 'relu', name='IMG_Dense1')(img_flatten)     ##
img_dropout1 = tf.keras.layers.Dropout(0.5, name='IMG_Dropout1')(img_dense1)                      ##
img_dense2 = tf.keras.layers.Dense(300, activation = 'relu', name='IMG_Dense2')(img_dropout1)    ##
img_dropout2 = tf.keras.layers.Dropout(0.5, name='IMG_Dropout2')(img_dense2)                      ##
csv_output = tf.keras.layers.Dense(1, name = 'CSV_Output')(csv_hidden5)##
img_output = tf.keras.layers.Dense(1,name = 'IMG_Output')(img_dropout2)##
model = tf.keras.Model(inputs=[csv_input, img_input], outputs=[csv_output, img_output], name='Pythonash_model')##


## === cell 31
model.summary()


## === cell 32
os.chdir('../')
os.chdir('../')
os.chdir('../')
tf.keras.utils.plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True, rankdir='TB')


## === cell 33
learning_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
initial_learning_rate = 0.002,
decay_steps = 10000,
decay_rate = 0.9)

opt = tf.keras.optimizers.Adam(learning_rate = learning_schedule)
model.compile(loss=['mse','mse'], loss_weights=[0.5, 0.5], optimizer = opt, metrics = tf.keras.metrics.RootMeanSquaredError())

epoch_number = 100

check_1 = tf.keras.callbacks.ModelCheckpoint('pythonash_model.h5', save_best_only=True, verbose=2)


## === cell 34
model.fit( 
    x= [train_csv_x, np.array(train_img)], y = [train_y, train_y], epochs=epoch_number, 
    validation_split=0.2, verbose =2, workers=3, batch_size = 100, validation_batch_size = 100,
    callbacks = [check_1])


## === cell 35
best_model = tf.keras.models.load_model('pythonash_model.h5')
csv_result, img_result = best_model.predict([test_csv_x, np.array(test_img)])
final_result = pd.DataFrame(0.5 * csv_result + 0.5 * img_result)
final_result.columns =['Pawpularity']
final_result


## === cell 36
for ids, paw in zip(test_csv_data['Id'], final_result['Pawpularity']):
    location = submission[submission['Id'] == ids].index[0]
    submission['Pawpularity'].loc[location] = paw
submission
submission.to_csv('./working/submission.csv',index=False)
