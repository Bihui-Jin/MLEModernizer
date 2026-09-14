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
Given a dataset of images of scanned text that is noisy, remove the noise.

## Metric
Root mean squared error between the cleaned pixel intensities and the actual grayscale pixel intensities.

## Submission Format
Form the submission file by melting each images into a set of pixels, assigning each pixel an id of image_row_col (e.g. 1_2_1 is image 1, row 2, column 1). Intensity values range from 0 (black) to 1 (white). The file should contain a header and have the following format:

```
id,value
1_1_1,1
1_2_1,1
1_3_1,1
etc.
```

## Dataset
You are provided two sets of images, train and test. These images contain various styles of text, to which synthetic noise has been added to simulate real-world, messy artifacts. The training set includes the test without the noise (train_cleaned).

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
sklearn-pandas==2.2.0
tf_keras==2.18.0

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (59 lines)
            sampleSubmission.csv (5789881 lines)
            sampleSubmission.csv.zip (12.0 MB)
            test.zip (4.0 MB)
            train.zip (15.5 MB)
            train_cleaned.zip (5.2 MB)
            denoising-dirty-documents/
                description.md (59 lines)
                sampleSubmission.csv (5789881 lines)
                ... and 4 other files
                denoising-dirty-documents/
                test/
                    110.png (149.2 kB)
                    111.png (146.9 kB)
                    ... and 27 other files
                    test/
                train/
                    116.png (152.2 kB)
                    201.png (156.1 kB)
                    ... and 113 other files
                    train/
                train_cleaned/
                    173.png (60.4 kB)
                    47.png (35.5 kB)
                    ... and 113 other files
            test/
                110.png (149.2 kB)
                111.png (146.9 kB)
                ... and 27 other files
                test/
            train/
                116.png (152.2 kB)
                201.png (156.1 kB)
                ... and 113 other files
                train/
            train_cleaned/
                173.png (60.4 kB)
                47.png (35.5 kB)
                ... and 113 other files
        input/
            description.md (59 lines)
            sampleSubmission.csv (5789881 lines)
            sampleSubmission.csv.zip (12.0 MB)
            test.zip (4.0 MB)
            train.zip (15.5 MB)
            train_cleaned.zip (5.2 MB)
            denoising-dirty-documents/
                description.md (59 lines)
                sampleSubmission.csv (5789881 lines)
                ... and 4 other files
                denoising-dirty-documents/
                test/
                    110.png (149.2 kB)
                    111.png (146.9 kB)
                    ... and 27 other files
                    test/
                train/
                    116.png (152.2 kB)
                    201.png (156.1 kB)
                    ... and 113 other files
                    train/
                train_cleaned/
                    173.png (60.4 kB)
                    47.png (35.5 kB)
                    ... and 113 other files
            test/
                110.png (149.2 kB)
                111.png (146.9 kB)
                ... and 27 other files
                test/
                    110.png (149.2 kB)
                    111.png (146.9 kB)
                    ... and 27 other files
                    test/
            train/
                116.png (152.2 kB)
                201.png (156.1 kB)
                ... and 113 other files
                train/
                    116.png (152.2 kB)
                    201.png (156.1 kB)
                    ... and 113 other files
                    train/
            train_cleaned/
                173.png (60.4 kB)
                47.png (35.5 kB)
                ... and 113 other files
        working/
            denoising-dirty-documents/
                description.md (59 lines)
                sampleSubmission.csv (5789881 lines)
                ... and 4 other files
                denoising-dirty-documents/
                test/
                    110.png (149.2 kB)
                    111.png (146.9 kB)
                    ... and 27 other files
                    test/
                train/
                    116.png (152.2 kB)
                    201.png (156.1 kB)
                    ... and 113 other files
                    train/
                train_cleaned/
                    173.png (60.4 kB)
                    47.png (35.5 kB)
                    ... and 113 other files
```

-> data/denoising-dirty-documents/sampleSubmission.csv has 5789880 rows and 2 columns.
The columns are: id, value

-> data/sampleSubmission.csv has 5789880 rows and 2 columns.
The columns are: id, value

-> input/denoising-dirty-documents/sampleSubmission.csv has 5789880 rows and 2 columns.
The columns are: id, value

-> input/sampleSubmission.csv has 5789880 rows and 2 columns.
The columns are: id, value

-> working/denoising-dirty-documents/sampleSubmission.csv has 5789880 rows and 2 columns.
The columns are: id, value

# 5. Target score

0.09149

# 6. Current score

0.4711

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))



## === cell 1
import matplotlib.pyplot as plt
import matplotlib.image as img
import cv2

import os
import glob

from keras.preprocessing import image
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from keras.layers import Input, Dense, Activation, BatchNormalization, Flatten, Conv2D
from keras.layers import MaxPooling2D, Dropout, UpSampling2D

%matplotlib inline
plt.rcParams['figure.figsize'] = (10.0, 5.0) # set default size of plots


## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 2
!unzip ../input/denoising-dirty-documents/train.zip
!unzip ../input/denoising-dirty-documents/train_cleaned.zip


## === cell 3
import zipfile

zipfiles = ['train','test','train_cleaned']

for each_zip in zipfiles:
    with zipfile.ZipFile("../input/denoising-dirty-documents/"+each_zip+".zip","r") as z:
        z.extractall(".")


## === cell 4
bsb = img.imread('/kaggle/working/train/102.png')
plt.imshow(bsb, cmap=plt.cm.gray)


## === cell 5
def load_image_from_dir(img_path):
    file_list = glob.glob(img_path+'/*.png')
    img_list = np.empty((len(file_list), 258, 540, 1))
    for i, fig in enumerate(file_list):
        img = image.load_img(fig, color_mode='grayscale', target_size=(258, 540))
        img_array = image.img_to_array(img).astype('float32')
        img_array = img_array / 255.0
        img_list[i] = img_array
    
    return img_list

def train_test_split(data,random_seed=55,split=0.75):
    set_rdm = np.random.RandomState(seed=random_seed)
    dsize = len(data)
    ind = set_rdm.choice(dsize,dsize,replace=False)
    train_ind = ind[:int(0.75*dsize)]
    val_ind = ind[int(0.75*dsize):]
    return data[train_ind],data[val_ind]


## === cell 6
full_train = load_image_from_dir('/kaggle/working/train')
full_target = load_image_from_dir('/kaggle/working/train_cleaned')
test = load_image_from_dir('/kaggle/working/test')

train, val = train_test_split(full_train, random_seed=10, split=0.8)
target_train, target_val = train_test_split(full_target, random_seed=10, split=0.8)


## === cell 7
optimizer = Adam(lr=0.0001)


## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/1559990111.py in <cell line: 0>()
----> 1 optimizer = Adam(lr=0.0001)

/usr/local/lib/python3.11/dist-packages/keras/src/optimizers/adam.py in __init__(self, learning_rate, beta_1, beta_2, epsilon, amsgrad, weight_decay, clipnorm, clipvalue, global_clipnorm, use_ema, ema_momentum, ema_overwrite_frequency, loss_scale_factor, gradient_accumulation_steps, name, **kwargs)
     60         **kwargs,
     61     ):
---> 62         super().__init__(
     63             learning_rate=learning_rate,
     64             name=name,

/usr/local/lib/python3.11/dist-packages/keras/src/backend/tensorflow/optimizer.py in __init__(self, *args, **kwargs)
     19 class TFOptimizer(KerasAutoTrackable, base_optimizer.BaseOptimizer):
     20     def __init__(self, *args, **kwargs):
---> 21         super().__init__(*args, **kwargs)
     22         self._distribution_strategy = tf.distribute.get_strategy()
     23 

/usr/local/lib/python3.11/dist-packages/keras/src/optimizers/base_optimizer.py in __init__(self, learning_rate, weight_decay, clipnorm, clipvalue, global_clipnorm, use_ema, ema_momentum, ema_overwrite_frequency, loss_scale_factor, gradient_accumulation_steps, name, **kwargs)
     88             )
     89         if kwargs:
---> 90             raise ValueError(f"Argument(s) not recognized: {kwargs}")
     91 
     92         if name is None:

ValueError: Argument(s) not recognized: {'lr': 0.0001}

## === cell 8
input_layer = Input(shape=train[0].shape)

h = Conv2D(32, (3, 3), activation='relu', padding='same')(input_layer)
h = Conv2D(64, (3, 3), activation='relu', padding='same')(h)
h = MaxPooling2D((2, 2), padding='same')(h)



h = Conv2D(64, (3, 3), activation='relu', padding='same')(h)


h = Conv2D(32, (3, 3), activation='relu', padding='same')(h)
h = UpSampling2D((2, 2))(h)
output_layer = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(h)


## === cell 9
ae_model = Model(input_layer,output_layer)
ae_model.compile(loss='mse', optimizer=optimizer)
ae_model.summary()


## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/191433106.py in <cell line: 0>()
      1 ae_model = Model(input_layer,output_layer)
----> 2 ae_model.compile(loss='mse', optimizer=optimizer)
      3 ae_model.summary()

NameError: name 'optimizer' is not defined

## === cell 10
early_stopping = EarlyStopping(monitor='val_loss',
                               min_delta=0,
                               patience=10,
                               verbose=1, 
                               mode='auto')


## === cell 11
history = ae_model.fit(train, target_train,
                                     batch_size=20,
                                     epochs=200,
                                     validation_data=(val, target_val),
                                     callbacks=[early_stopping])


## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/3731746004.py in <cell line: 0>()
----> 1 history = ae_model.fit(train, target_train,
      2                                      batch_size=20,
      3                                      epochs=200,
      4                                      validation_data=(val, target_val),
      5                                      callbacks=[early_stopping])

/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py in error_handler(*args, **kwargs)
    120             # To get the full stack trace, call:
    121             # `keras.config.disable_traceback_filtering()`
--> 122             raise e.with_traceback(filtered_tb) from None
    123         finally:
    124             del filtered_tb

/usr/local/lib/python3.11/dist-packages/keras/src/trainers/trainer.py in _assert_compile_called(self, method_name)
   1047             else:
   1048                 msg += f"calling `{method_name}()`."
-> 1049             raise ValueError(msg)
   1050 
   1051     def _symbolic_build(self, iterator=None, data_batch=None):

ValueError: You must call `compile()` before using the model.

## === cell 12
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()


## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3711827419.py in <cell line: 0>()
----> 1 plt.plot(history.history['loss'])
      2 plt.plot(history.history['val_loss'])
      3 plt.title('Model loss')
      4 plt.ylabel('Loss')
      5 plt.xlabel('Epoch')

NameError: name 'history' is not defined

## === cell 13
preds = ae_model.predict(test)


## === cell 14
plt.imshow(test[22].reshape(258,540), cmap=plt.cm.gray)


## === cell 15
plt.imshow(preds[22].reshape(258,540), cmap=plt.cm.gray)


## === cell 16
ids = []
vals = []
file_list = glob.glob('/kaggle/working/test/*.png')
for i, f in enumerate(file_list):
    file = os.path.basename(f)
    imgid = int(file[:-4])
    test_img = cv2.imread(f, 0)
    img_shape = test_img.shape
    print('processing: {}'.format(imgid))
    print(img_shape)
    preds_reshaped = cv2.resize(preds[i], (img_shape[1], img_shape[0]))
    for r in range(img_shape[0]):
        for c in range(img_shape[1]):
            ids.append(str(imgid)+'_'+str(r + 1)+'_'+str(c + 1))
            vals.append(preds_reshaped[r, c])

print('Writing to csv file')
pd.DataFrame({'id': ids, 'value': vals}).to_csv('submission.csv', index=False)
