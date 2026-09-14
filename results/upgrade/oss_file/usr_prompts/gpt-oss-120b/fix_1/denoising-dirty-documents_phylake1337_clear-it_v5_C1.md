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
tf_keras==2.18.0
tqdm==4.67.1

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

0.05382

# 6. Current score

0.36689

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import keras, os
import tensorflow as tf
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from tqdm import tqdm
from tensorflow.keras.preprocessing.image import load_img

from keras.layers import Input, Dense, Activation, BatchNormalization, Flatten, Conv2D
from keras.layers import MaxPooling2D, Dropout, UpSampling2D


## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
train_zip_path = '/kaggle/input/denoising-dirty-documents/train.zip'
test_zip_path = '/kaggle/input/denoising-dirty-documents/test.zip'
sample_zip_path = '/kaggle/input/denoising-dirty-documents/sampleSubmission.csv.zip'
trainclean_zip_path = '/kaggle/input/denoising-dirty-documents/train_cleaned.zip'
extracting_path = '/kaggle/working'


## === cell 2
import zipfile
with zipfile.ZipFile(train_zip_path, 'r') as zip_ref:
    zip_ref.extractall(extracting_path)
    
with zipfile.ZipFile(test_zip_path, 'r') as zip_ref:
    zip_ref.extractall(extracting_path)
    
with zipfile.ZipFile(sample_zip_path, 'r') as zip_ref:
    zip_ref.extractall(extracting_path)
    
with zipfile.ZipFile(trainclean_zip_path, 'r') as zip_ref:
    zip_ref.extractall(extracting_path)


## === cell 3
img_arr = mpimg.imread(extracting_path + '/train/107.png')
h, w = img_arr.shape
print('Height: ', h,'- Width: ',w)
print(img_arr.dtype)


## === cell 4
image_names = os.listdir(extracting_path + '/train')
data_size = len(image_names)
X = np.zeros([data_size, 2], dtype=np.uint16)
for i in tqdm(range(data_size)):
    image_name = image_names[i]
    img_dir = os.path.join(extracting_path + '/train', image_name)
    img_pixels = mpimg.imread(img_dir)
    X[i] = img_pixels.shape

print('Number of training images:', data_size)
print('Differnet image hights: {}'.format(set(X[:,0])))
print('Differnet image widths: {}'.format(set(X[:,1])))


## === cell 5
def images_to_array(data_dir, label_dir=None, img_size=(h, w)):
    '''
    1- Read image samples from certain directory.
    2- Stack them into one big numpy array.
    -- And if there are labels images ..
    3- Read sample's label form the labels directory.
    4- Stack them into one big numpy array.
    5- Shuffle Data and label arrays.
    '''
    image_names = os.listdir(data_dir)
    data_size = len(image_names)
    X = np.zeros([data_size, img_size[0], img_size[1]], dtype=np.uint8)
    for i in tqdm(range(data_size)):
        image_name = image_names[i]
        img_dir = os.path.join(data_dir, image_name)
        img_pixels = load_img(img_dir, color_mode='grayscale', target_size=(h, w))
        X[i] = img_pixels
    X = X.reshape(data_size, h, w, 1) 
    
    if label_dir:
        label_names = os.listdir(label_dir)
        data_size = len(label_names)
        y = np.zeros([data_size, img_size[0], img_size[1]], dtype=np.uint8)
        for i in tqdm(range(data_size)):
            image_name = label_names[i]
            img_dir = os.path.join(label_dir, image_name)
            img_pixels = load_img(img_dir, color_mode='grayscale', target_size=(h, w))
            y[i] = img_pixels
        y = y.reshape(data_size, h, w, 1) 
        ind = np.random.permutation(data_size)
        X = X[ind]
        y = y[ind]
        print('Ouptut Data Size: ', X.shape)
        print('Ouptut Label Size: ', y.shape)
        return X/255., y/255.
    
    print('Ouptut Data Size: ', X.shape)
    return X/255.


## === cell 6
X, y = images_to_array(extracting_path + '/train', extracting_path + '/train_cleaned')


## === cell 7
val_split = int(.15 * data_size)
X_val, y_val = X[:val_split], y[:val_split]
X_train, y_train = X[val_split:], y[val_split:]
print('Train data shape: ', X_train.shape)
print('Test data shape: ', X_val.shape)


## === cell 8
samples = np.concatenate((X_train[:3], y_train[:3]), axis=0) 

f, ax = plt.subplots(2, 3, figsize=(20,10))
for i, img in enumerate(samples):
    ax[i//3, i%3].imshow(img[:,:,0], cmap='gray')
    ax[i//3, i%3].axis('off')
plt.show() 


## === cell 9
input_layer = Input(shape=(None, None, 1))
x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_layer)
x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2), padding='same')(x)

x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
output_layer = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)
model = keras.models.Model(inputs=[input_layer], outputs=[output_layer])

sgd = keras.optimizers.SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
rms = keras.optimizers.RMSprop(learning_rate=0.001, rho=0.9)
ada = keras.optimizers.Adagrad(learning_rate=0.01)

model.compile(optimizer = 'adam' , loss = "mean_squared_error")


## === cell 10
LR_callback = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', patience=4, verbose=10, factor=.4, min_lr=.00001)


## === cell 11
history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=70, batch_size=16, callbacks=[LR_callback])


## === cell 12
model.evaluate(X_val, y_val)


## === cell 13
test_samples, test_labels = X_val[:3], y_val[:3]
test_pred = model.predict(X_val[:3])

samples = np.concatenate((test_samples, test_labels, test_pred), axis=0) 

f, ax = plt.subplots(3, 3, figsize=(25,15))
for i, img in enumerate(samples):
    ax[i//3, i%3].imshow(img[:,:,0], cmap='gray')
    ax[i//3, i%3].axis('off')
plt.show() 


## === cell 14
image_names = sorted(os.listdir(extracting_path + '/test'))
data_size = len(image_names)
X_test = []
for i in tqdm(range(data_size)):
    image_name = image_names[i]
    img_dir = os.path.join(extracting_path + '/test', image_name)
    img_pixels = load_img(img_dir, color_mode='grayscale')
    w, h = img_pixels.size
    X_test.append(np.array(img_pixels).reshape(1, h, w, 1) / 255.)
    
print('Test sample shape: ', X_test[0].shape)
print('Test sample dtype: ', X_test[0].dtype)


## === cell 15
yh_test = []
for img in X_test:
    size = img.shape[1:3]
    yh_test.append(model.predict(img)[0, :, :, 0])


## === cell 16
f, ax = plt.subplots(1,2, figsize=(20,10))
ax[0].imshow(X_test[0][0,:,:,0], cmap='gray')
ax[0].axis('off')

ax[1].imshow(yh_test[0], cmap='gray')
ax[1].axis('off')
plt.show() 


## === cell 17
submit_vector = []
for img in yh_test:
    h, w = img.shape
    for i in range(w):
        for j in range(h):
            submit_vector.append(img[j,i])
print(len(submit_vector))


## === cell 18
sample_csv = pd.read_csv(extracting_path + '/sampleSubmission.csv')
sample_csv.head(10)


## === cell 19
c = 0
for img in yh_test:
    hi, wi = img.shape
    c += (hi * wi)


## === cell 20
id_col = sample_csv['id']
value_col = pd.Series(submit_vector, name='value')
submission = pd.concat([id_col, value_col], axis=1)
submission.head(10)


## === cell 21
submission.to_csv('Cleared.csv',index = False)
