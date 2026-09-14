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

3.10

# 2. Installed packages

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

# 3. Data file paths

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

# 4. Code solution

## === cell 0
import os

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "2")

import subprocess
import sys

subprocess.check_call(
    [sys.executable, "-m", "pip", "install", "-q", "protobuf==4.25.3"]
)

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
os.chdir("../input/petfinder-pawpularity-score/train")

rows = []
for file in os.listdir():
    if not os.path.isfile(file):
        continue
    imgg = cv2.imread(file)
    if imgg is None:
        continue
    w, h, c = imgg.shape
    rows.append([w, h, c, imgg.size / 3])

size_data = pd.DataFrame(rows)
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
    if not os.path.isfile(i):
        continue
    file = cv2.imread(i)
    if file is None:
        continue
    file = cv2.resize(file, (64, 64), interpolation=cv2.INTER_AREA)
    train_img.append(file / 255)
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
_rows = []
for img, name in zip(train_img, train_img_name):
    if (not isinstance(name, str)) or (not name.lower().endswith(".jpg")):
        continue
    if not os.path.isfile(name):
        continue

    img_id = name[:-4]
    matches = train_csv.index[train_csv["Id"] == img_id]
    if len(matches) == 0:
        continue

    location = matches[0]
    _rows.append(train_csv.loc[location])

train_csv_data = pd.DataFrame(_rows)
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
os.chdir("../test")

for i in os.listdir():
    if not os.path.isfile(i):
        continue
    file = cv2.imread(i)
    if file is None:
        continue
    print(file.shape)


## === cell 23
test_img = []
for i in os.listdir():
    if not os.path.isfile(i):
        continue
    file = cv2.imread(i)
    if file is None:
        continue
    file = cv2.resize(file, (64, 64), interpolation=cv2.INTER_AREA)
    test_img.append(file / 255)
test_img[:5]


## === cell 24
test_img_name = []
for i in os.listdir():
    test_img_name.append(i)
test_img_name[:5]


## === cell 25
test_csv_data = pd.DataFrame()
_rows = []
for img, name in zip(test_img, test_img_name):
    if (not isinstance(name, str)) or (not name.lower().endswith(".jpg")):
        continue
    img_id = name[:-4]

    matches = test_csv.index[test_csv["Id"] == img_id]
    if len(matches) == 0:
        continue

    location = matches[0]
    _rows.append(test_csv.loc[location])

test_csv_data = pd.DataFrame(_rows)
test_csv_data = test_csv_data.reset_index().drop(["index"], axis=1)
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


## === cell 30
csv_input = tf.keras.Input(shape = train_csv_x.shape[1:], name = 'CSV_Input')
img_input = tf.keras.Input(shape = np.array(train_img).shape[1:], name = 'IMG_Input')

csv_hidden1 = tf.keras.layers.Dense(200, activation='elu', kernel_initializer = 'he_normal', name='CSV_Hidden1')(csv_input)
csv_hidden2 = tf.keras.layers.Dense(200, activation='elu', kernel_initializer = 'he_normal', name='CSV_Hidden2')(csv_hidden1)
csv_hidden3 = tf.keras.layers.Dense(200, activation='elu', kernel_initializer = 'he_normal', name='CSV_Hidden3')(csv_hidden2)
csv_hidden4 = tf.keras.layers.Dense(200, activation='elu', kernel_initializer = 'he_normal', name='CSV_Hidden4')(csv_hidden3)
csv_hidden5 = tf.keras.layers.Dense(200, activation='elu', kernel_initializer = 'he_normal', name='CSV_Hidden5')(csv_hidden4)
csv_hidden6 = tf.keras.layers.Dense(200, activation='elu', kernel_initializer = 'he_normal', name='CSV_Hidden6')(csv_hidden5)
csv_dropout = tf.keras.layers.Dropout(0.5, name ='CSV_Dropout')(csv_hidden6)

img_conv1 = tf.keras.layers.Conv2D(120, 4, padding = 'same', activation = 'elu', kernel_initializer = 'he_normal',name='IMG_Conv1')(img_input)
img_conv2 = tf.keras.layers.Conv2D(120, 4, padding = 'same', activation = 'elu', kernel_initializer = 'he_normal',name='IMG_Conv2')(img_conv1)
img_pooling1 = tf.keras.layers.MaxPooling2D(4, name= 'IMG_Max1')(img_conv2)

img_conv3 = tf.keras.layers.Conv2D(120, 4, padding = 'same', activation = 'elu', kernel_initializer = 'he_normal',name='IMG_Conv3')(img_pooling1)
img_conv4 = tf.keras.layers.Conv2D(120, 4, padding = 'same', activation = 'elu', kernel_initializer = 'he_normal',name='IMG_Conv4')(img_conv3)
img_pooling2 = tf.keras.layers.MaxPooling2D(4, name= 'IMG_Max2')(img_conv4)

img_conv5 = tf.keras.layers.Conv2D(120, 4, padding = 'same', activation = 'elu', kernel_initializer = 'he_normal',name='IMG_Conv5')(img_pooling2)
img_conv6 = tf.keras.layers.Conv2D(120, 4, padding = 'same', activation = 'elu', kernel_initializer = 'he_normal',name='IMG_Conv6')(img_conv5)
img_pooling3 = tf.keras.layers.MaxPooling2D(3, name= 'IMG_Max3')(img_conv6)

img_dropout = tf.keras.layers.Dropout(0.5, name = 'IMG_Dropout')(img_pooling3)
img_conv7 = tf.keras.layers.Conv2D(120, 4, padding = 'same', activation = 'elu', kernel_initializer = 'he_normal',name='IMG_Conv7')(img_dropout)

img_hidden1 = tf.keras.layers.Dense(300, activation='elu', kernel_initializer = 'he_normal',name='IMG_hidden1' ,use_bias=False)(img_conv7)
img_dropout1 = tf.keras.layers.Dropout(0.5, name='IMG_Dropout1')(img_hidden1)

img_hidden2 = tf.keras.layers.Dense(300, activation='elu', kernel_initializer = 'he_normal',name='IMG_hidden2' ,use_bias=False)(img_dropout1)

img_gpool = tf.keras.layers.GlobalAvgPool2D(name = 'IMG_Gpool')(img_hidden2)

img_dropout2 = tf.keras.layers.Dropout(0.5, name='IMG_Dropout2')(img_gpool)



csv_output = tf.keras.layers.Dense(1, name = 'CSV_Output')(csv_dropout)
img_output = tf.keras.layers.Dense(1,name = 'IMG_Output')(img_dropout2)

model = tf.keras.Model(inputs=[csv_input, img_input], outputs=[csv_output, img_output], name='Pythonash_model')


## === cell 31
model.summary()


## === cell 32
os.chdir('../')
os.chdir('../')
os.chdir('../')
tf.keras.utils.plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True, rankdir='TB')


## === cell 33
learning_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=0.002, decay_steps=10000, decay_rate=0.97
)

opt = tf.keras.optimizers.Adam(learning_rate=learning_schedule)

rmse = tf.keras.metrics.RootMeanSquaredError()
model.compile(
    loss=["mse", "mse"],
    loss_weights=[1, 1],
    optimizer=opt,
    metrics=[rmse, rmse],
)

epoch_number = 20

check_1 = tf.keras.callbacks.ModelCheckpoint(
    "pythonash_model.h5", save_best_only=True, verbose=2
)


## === cell 34
model.fit(
    x=[train_csv_x, np.array(train_img)],
    y=[train_y, train_y],
    epochs=epoch_number,
    validation_split=0.2,
    verbose=2,
    batch_size=100,
    validation_batch_size=100,
    callbacks=[check_1],
)


## === cell 35
best_model = tf.keras.models.load_model('pythonash_model.h5')
csv_result, img_result = best_model.predict([test_csv_x, np.array(test_img)])
final_result = pd.DataFrame(0.5 * csv_result + 0.5 * img_result)
final_result.columns =['Pawpularity']
final_result


## --- ERROR in cell 35, traceback:
[0;31m---------------------------------------------------------------------------[0m
[0;31mTypeError[0m                                 Traceback (most recent call last)
[0;32m/tmp/ipykernel_11/25262455.py[0m in [0;36m<cell line: 0>[0;34m()[0m
[0;32m----> 1[0;31m [0mbest_model[0m [0;34m=[0m [0mtf[0m[0;34m.[0m[0mkeras[0m[0;34m.[0m[0mmodels[0m[0;34m.[0m[0mload_model[0m[0;34m([0m[0;34m'pythonash_model.h5'[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m      2[0m [0mcsv_result[0m[0;34m,[0m [0mimg_result[0m [0;34m=[0m [0mbest_model[0m[0;34m.[0m[0mpredict[0m[0;34m([0m[0;34m[[0m[0mtest_csv_x[0m[0;34m,[0m [0mnp[0m[0;34m.[0m[0marray[0m[0;34m([0m[0mtest_img[0m[0;34m)[0m[0;34m][0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m      3[0m [0mfinal_result[0m [0;34m=[0m [0mpd[0m[0;34m.[0m[0mDataFrame[0m[0;34m([0m[0;36m0.5[0m [0;34m*[0m [0mcsv_result[0m [0;34m+[0m [0;36m0.5[0m [0;34m*[0m [0mimg_result[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m      4[0m [0mfinal_result[0m[0;34m.[0m[0mcolumns[0m [0;34m=[0m[0;34m[[0m[0;34m'Pawpularity'[0m[0;34m][0m[0;34m[0m[0;34m[0m[0m
[1;32m      5[0m [0mfinal_result[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/keras/src/saving/saving_api.py[0m in [0;36mload_model[0;34m(filepath, custom_objects, compile, safe_mode)[0m
[1;32m    194[0m         )
[1;32m    195[0m     [0;32mif[0m [0mstr[0m[0;34m([0m[0mfilepath[0m[0;34m)[0m[0;34m.[0m[0mendswith[0m[0;34m([0m[0;34m([0m[0;34m".h5"[0m[0;34m,[0m [0;34m".hdf5"[0m[0;34m)[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 196[0;31m         return legacy_h5_format.load_model_from_hdf5(
[0m[1;32m    197[0m             [0mfilepath[0m[0;34m,[0m [0mcustom_objects[0m[0;34m=[0m[0mcustom_objects[0m[0;34m,[0m [0mcompile[0m[0;34m=[0m[0mcompile[0m[0;34m[0m[0;34m[0m[0m
[1;32m    198[0m         )

[0;32m/usr/local/lib/python3.11/dist-packages/keras/src/legacy/saving/legacy_h5_format.py[0m in [0;36mload_model_from_hdf5[0;34m(filepath, custom_objects, compile)[0m
[1;32m    153[0m             [0;31m# Compile model.[0m[0;34m[0m[0;34m[0m[0m
[1;32m    154[0m             model.compile(
[0;32m--> 155[0;31m                 **saving_utils.compile_args_from_training_config(
[0m[1;32m    156[0m                     [0mtraining_config[0m[0;34m,[0m [0mcustom_objects[0m[0;34m[0m[0;34m[0m[0m
[1;32m    157[0m                 )

[0;32m/usr/local/lib/python3.11/dist-packages/keras/src/legacy/saving/saving_utils.py[0m in [0;36mcompile_args_from_training_config[0;34m(training_config, custom_objects)[0m
[1;32m    141[0m         [0mloss_config[0m [0;34m=[0m [0mtraining_config[0m[0;34m.[0m[0mget[0m[0;34m([0m[0;34m"loss"[0m[0;34m,[0m [0;32mNone[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m    142[0m         [0;32mif[0m [0mloss_config[0m [0;32mis[0m [0;32mnot[0m [0;32mNone[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 143[0;31m             [0mloss[0m [0;34m=[0m [0m_deserialize_nested_config[0m[0;34m([0m[0mlosses[0m[0;34m.[0m[0mdeserialize[0m[0;34m,[0m [0mloss_config[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    144[0m             [0;31m# Ensure backwards compatibility for losses in legacy H5 files[0m[0;34m[0m[0;34m[0m[0m
[1;32m    145[0m             [0mloss[0m [0;34m=[0m [0m_resolve_compile_arguments_compat[0m[0;34m([0m[0mloss[0m[0;34m,[0m [0mloss_config[0m[0;34m,[0m [0mlosses[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/keras/src/legacy/saving/saving_utils.py[0m in [0;36m_deserialize_nested_config[0;34m(deserialize_fn, config)[0m
[1;32m    207[0m         }
[1;32m    208[0m     [0;32melif[0m [0misinstance[0m[0;34m([0m[0mconfig[0m[0;34m,[0m [0;34m([0m[0mtuple[0m[0;34m,[0m [0mlist[0m[0;34m)[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 209[0;31m         return [
[0m[1;32m    210[0m             [0m_deserialize_nested_config[0m[0;34m([0m[0mdeserialize_fn[0m[0;34m,[0m [0mobj[0m[0;34m)[0m [0;32mfor[0m [0mobj[0m [0;32min[0m [0mconfig[0m[0;34m[0m[0;34m[0m[0m
[1;32m    211[0m         ]

[0;32m/usr/local/lib/python3.11/dist-packages/keras/src/legacy/saving/saving_utils.py[0m in [0;36m<listcomp>[0;34m(.0)[0m
[1;32m    208[0m     [0;32melif[0m [0misinstance[0m[0;34m([0m[0mconfig[0m[0;34m,[0m [0;34m([0m[0mtuple[0m[0;34m,[0m [0mlist[0m[0;34m)[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m    209[0m         return [
[0;32m--> 210[0;31m             [0m_deserialize_nested_config[0m[0;34m([0m[0mdeserialize_fn[0m[0;34m,[0m [0mobj[0m[0;34m)[0m [0;32mfor[0m [0mobj[0m [0;32min[0m [0mconfig[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    211[0m         ]
[1;32m    212[0m [0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/keras/src/legacy/saving/saving_utils.py[0m in [0;36m_deserialize_nested_config[0;34m(deserialize_fn, config)[0m
[1;32m    200[0m         [0;32mreturn[0m [0;32mNone[0m[0;34m[0m[0;34m[0m[0m
[1;32m    201[0m     [0;32mif[0m [0m_is_single_object[0m[0;34m([0m[0mconfig[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 202[0;31m         [0;32mreturn[0m [0mdeserialize_fn[0m[0;34m([0m[0mconfig[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    203[0m     [0;32melif[0m [0misinstance[0m[0;34m([0m[0mconfig[0m[0;34m,[0m [0mdict[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m    204[0m         return {

[0;32m/usr/local/lib/python3.11/dist-packages/keras/src/losses/__init__.py[0m in [0;36mdeserialize[0;34m(name, custom_objects)[0m
[1;32m    153[0m         [0mA[0m [0mKeras[0m[0;31m [0m[0;31m`[0m[0mLoss[0m[0;31m`[0m [0minstance[0m [0;32mor[0m [0ma[0m [0mloss[0m [0mfunction[0m[0;34m.[0m[0;34m[0m[0;34m[0m[0m
[1;32m    154[0m     """
[0;32m--> 155[0;31m     return serialization_lib.deserialize_keras_object(
[0m[1;32m    156[0m         [0mname[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m
[1;32m    157[0m         [0mmodule_objects[0m[0;34m=[0m[0mALL_OBJECTS_DICT[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/keras/src/saving/serialization_lib.py[0m in [0;36mdeserialize_keras_object[0;34m(config, custom_objects, safe_mode, **kwargs)[0m
[1;32m    573[0m                 [0;32mreturn[0m [0mconfig[0m[0;34m[0m[0;34m[0m[0m
[1;32m    574[0m             [0;32mif[0m [0misinstance[0m[0;34m([0m[0mmodule_objects[0m[0;34m[[0m[0mconfig[0m[0;34m][0m[0;34m,[0m [0mtypes[0m[0;34m.[0m[0mFunctionType[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 575[0;31m                 return deserialize_keras_object(
[0m[1;32m    576[0m                     serialize_with_public_fn(
[1;32m    577[0m                         [0mmodule_objects[0m[0;34m[[0m[0mconfig[0m[0;34m][0m[0;34m,[0m [0mconfig[0m[0;34m,[0m [0mfn_module_name[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/keras/src/saving/serialization_lib.py[0m in [0;36mdeserialize_keras_object[0;34m(config, custom_objects, safe_mode, **kwargs)[0m
[1;32m    676[0m     [0;32mif[0m [0mclass_name[0m [0;34m==[0m [0;34m"function"[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m    677[0m         [0mfn_name[0m [0;34m=[0m [0minner_config[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 678[0;31m         return _retrieve_class_or_fn(
[0m[1;32m    679[0m             [0mfn_name[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m
[1;32m    680[0m             [0mregistered_name[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/keras/src/saving/serialization_lib.py[0m in [0;36m_retrieve_class_or_fn[0;34m(name, registered_name, module, obj_type, full_config, custom_objects)[0m
[1;32m    801[0m             [0;32mreturn[0m [0mobj[0m[0;34m[0m[0;34m[0m[0m
[1;32m    802[0m [0;34m[0m[0m
[0;32m--> 803[0;31m     raise TypeError(
[0m[1;32m    804[0m         [0;34mf"Could not locate {obj_type} '{name}'. "[0m[0;34m[0m[0;34m[0m[0m
[1;32m    805[0m         [0;34m"Make sure custom classes are decorated with "[0m[0;34m[0m[0;34m[0m[0m

[0;31mTypeError[0m: Could not locate function 'mse'. Make sure custom classes are decorated with `@keras.saving.register_keras_serializable()`. Full object config: {'module': 'keras.metrics', 'class_name': 'function', 'config': 'mse', 'registered_name': 'mse'}

## === cell 36
for ids, paw in zip(test_csv_data['Id'], final_result['Pawpularity']):
    location = submission[submission['Id'] == ids].index[0]
    submission['Pawpularity'].loc[location] = paw
submission
submission.to_csv('./working/submission.csv',index=False)
