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
Create a classifier to predict the severity of diabetic retinopathy.

## Metric
Quadratic weighted kappa, which measures the agreement between two ratings. This metric typically varies from 0 (random agreement between raters) to 1 (complete agreement between raters). In the event that there is less agreement between the raters than expected by chance, this metric may go below 0. The quadratic weighted kappa is calculated between the scores assigned by the human rater and the predicted scores.

Images have five possible ratings, 0,1,2,3,4.  Each image is characterized by a tuple *(e*,*e)*, which corresponds to its scores by *Rater A* (human) and *Rater B* (predicted).  The quadratic weighted kappa is calculated as follows. First, an N x N histogram matrix *O* is constructed, such that *O* corresponds to the number of images that received a rating *i* by *A* and a rating *j* by *B*. An *N-by-N* matrix of weights, *w*, is calculated based on the difference between raters' scores:

An *N-by-N* histogram matrix of expected ratings, *E*, is calculated, assuming that there is no correlation between rating scores.  This is calculated as the outer product between each rater's histogram vector of ratings, normalized such that *E* and *O* have the same sum.

## Submission Format
```
id_code,diagnosis
0005cfc8afb6,0
003f0afdcd15,0
etc.
```

## Dataset
You are provided with a large set of retina images taken using [fundus photography](https://en.wikipedia.org/wiki/Fundus_photography) under a variety of imaging conditions.

Labels are on a scale of 0 to 4:

> 0 - No DR
> 1 - Mild
> 2 - Moderate
> 3 - Severe
> 4 - Proliferative DR

Images may contain artifacts, be out of focus, underexposed, or overexposed. The images were gathered from multiple clinics using a variety of cameras over an extended period of time, which will introduce further variation.

- **train.csv** - the training labels
- **test.csv** - the test set (you must predict the `diagnosis` value for these variables)
- **sample_submission.csv** - a sample submission file in the correct format
- **train.zip** - the training set images
- **test.zip** - the public test set images

# 2. Python version

3.7

# 3. Installed packages

No external packages required in the script and installed.

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (118 lines)
            sample_submission.csv (368 lines)
            sample_submission.csv.zip (3.2 kB)
            test.csv (368 lines)
            test.csv.zip (2.9 kB)
            test.zip (160 Bytes)
            test_images.zip (902.9 MB)
            train.csv (3296 lines)
            train.csv.zip (27.5 kB)
            train.zip (162 Bytes)
            train_images.zip (7.7 GB)
            aptos2019-blindness-detection/
                description.md (118 lines)
                sample_submission.csv (368 lines)
                ... and 9 other files
                aptos2019-blindness-detection/
                test_images/
                    218c822a3dd9.png (5.7 MB)
                    0e82bcacc475.png (5.2 MB)
                    ... and 365 other files
                    test_images/
                train_images/
                    184a185e7447.png (337.5 kB)
                    c4aef0d88d1b.png (876.6 kB)
                    ... and 3293 other files
                    train_images/
            test_images/
                218c822a3dd9.png (5.7 MB)
                0e82bcacc475.png (5.2 MB)
                ... and 365 other files
                test_images/
            train_images/
                184a185e7447.png (337.5 kB)
                c4aef0d88d1b.png (876.6 kB)
                ... and 3293 other files
                train_images/
        input/
            description.md (118 lines)
            sample_submission.csv (368 lines)
            sample_submission.csv.zip (3.2 kB)
            test.csv (368 lines)
            test.csv.zip (2.9 kB)
            test.zip (160 Bytes)
            test_images.zip (902.9 MB)
            train.csv (3296 lines)
            train.csv.zip (27.5 kB)
            train.zip (162 Bytes)
            train_images.zip (7.7 GB)
            aptos2019-blindness-detection/
                description.md (118 lines)
                sample_submission.csv (368 lines)
                ... and 9 other files
                aptos2019-blindness-detection/
                test_images/
                    218c822a3dd9.png (5.7 MB)
                    0e82bcacc475.png (5.2 MB)
                    ... and 365 other files
                    test_images/
                train_images/
                    184a185e7447.png (337.5 kB)
                    c4aef0d88d1b.png (876.6 kB)
                    ... and 3293 other files
                    train_images/
            test_images/
                218c822a3dd9.png (5.7 MB)
                0e82bcacc475.png (5.2 MB)
                ... and 365 other files
                test_images/
                    218c822a3dd9.png (5.7 MB)
                    0e82bcacc475.png (5.2 MB)
                    ... and 365 other files
                    test_images/
            train_images/
                184a185e7447.png (337.5 kB)
                c4aef0d88d1b.png (876.6 kB)
                ... and 3293 other files
                train_images/
                    184a185e7447.png (337.5 kB)
                    c4aef0d88d1b.png (876.6 kB)
                    ... and 3293 other files
                    train_images/
        working/
            aptos2019-blindness-detection/
                description.md (118 lines)
                sample_submission.csv (368 lines)
                ... and 9 other files
                aptos2019-blindness-detection/
                test_images/
                    218c822a3dd9.png (5.7 MB)
                    0e82bcacc475.png (5.2 MB)
                    ... and 365 other files
                    test_images/
                train_images/
                    184a185e7447.png (337.5 kB)
                    c4aef0d88d1b.png (876.6 kB)
                    ... and 3293 other files
                    train_images/
```

-> data/aptos2019-blindness-detection/sample_submission.csv has 367 rows and 2 columns.
The columns are: id_code, diagnosis

-> data/aptos2019-blindness-detection/test.csv has 367 rows and 1 columns.
The columns are: id_code

-> data/aptos2019-blindness-detection/train.csv has 3295 rows and 2 columns.
The columns are: id_code, diagnosis

-> data/sample_submission.csv has 367 rows and 2 columns.
The columns are: id_code, diagnosis

-> data/test.csv has 367 rows and 1 columns.
The columns are: id_code

-> data/train.csv has 3295 rows and 2 columns.
The columns are: id_code, diagnosis

-> input/aptos2019-blindness-detection/sample_submission.csv has 367 rows and 2 columns.
The columns are: id_code, diagnosis

-> (stopped after 10 files for performance)

# 5. Target score

0.5556703829117215

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import random
from scipy import ndarray
import skimage as sk
from skimage import transform
from skimage import util
from copy import deepcopy


import os
print(os.listdir("../input"))



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
ImportError                               Traceback (most recent call last)
/tmp/ipykernel_11/1512867076.py in <cell line: 0>()
      6 import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
      7 import random
----> 8 from scipy import ndarray
      9 import skimage as sk
     10 from skimage import transform

ImportError: cannot import name 'ndarray' from 'scipy' (/usr/local/lib/python3.11/dist-packages/scipy/__init__.py)

## === cell 1
import tensorflow as tf
import keras
from keras import backend as K
import numpy as np
%matplotlib inline
import matplotlib.pyplot as plt
import cv2  # for image processing
import scipy.io
import os
print(tf.__version__)
print(keras.__version__)

## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 2
plt.rcParams.update({'axes.titlesize': 'small'})

## === cell 3
IMG_SIZE = 128

## === cell 4
import glob, pandas as pd

## === cell 5
!ls ../input/

## === cell 6
!ls ../input/aptos2019-blindness-detection

## === cell 7
pd.read_csv("../input/aptos2019-blindness-detection/train.csv").head()

## === cell 8
labels_df = pd.read_csv("../input/aptos2019-blindness-detection/train.csv").set_index('id_code')

## === cell 9
len(labels_df.index)

## === cell 10
labels_df['diagnosis'].value_counts()

## === cell 11
images = [f for f in glob.glob("../input/aptos2019-blindness-detection/train_images/" + "*.png")]
labels = [ labels_df.loc[f.split('/')[-1].split('.')[0].strip()].diagnosis for f in  images]

## === cell 12
train_images_ixs = set(random.choices(range(len(images)), k=6000))
train_images = []
train_labels = []
test_images = []
test_labels = []
for i in range(len(labels)):
    if i in train_images_ixs:
        train_images.append(images[i])
        train_labels.append(labels[i])
    else:
        test_images.append(images[i])
        test_labels.append(labels[i])

## === cell 13
print (len(train_labels), len(train_images))
print (len(test_labels), len(test_images))

## === cell 15
for img,lb in zip(images[:5], labels[:5]):
    print (img, lb)

## === cell 16
from PIL import Image
from matplotlib import pyplot as plt
import random

## === cell 17
label_text = ['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative DR']

## === cell 18
images_to_display = []
for lb in range(5):
    images_to_display += random.choices([ (images[ix], labels[ix]) for ix in range(len(images)) if labels[ix] == lb ] , k=10)

fig = plt.figure(figsize=(25, 16))
for ii, (img,label) in enumerate(images_to_display):
    ax = fig.add_subplot(5, 10, ii + 1, xticks=[], yticks=[])
    img = cv2.imread(img)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img)
    plt.text(0, img.shape[0], label_text[label], bbox=dict(facecolor='red', alpha=0.5))

## === cell 19
def adjust_gamma(image, gamma=1.0):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)

## === cell 20
def add_contrast(img, contrast):
        buf = img.copy()
        f = float(131 * (contrast + 127)) / (127 * (131 - contrast))
        alpha_c = f
        gamma_c = 127*(1-f)
        buf = cv2.addWeighted(buf, alpha_c, buf, 0, gamma_c)
        return buf

## === cell 21
def preproces_image(img):
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = adjust_gamma(img, 1.5)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = add_contrast(img, 20)
    return img
    

## === cell 22
fig = plt.figure(figsize=(25, 16))
for ii, (img,label) in enumerate(images_to_display):
    ax = fig.add_subplot(5, 10, ii + 1, xticks=[], yticks=[])
    img = cv2.imread(img)
    img_new = preproces_image(img)
    plt.imshow(img_new)
    plt.text(0, img_new.shape[0], label_text[label], bbox=dict(facecolor='red', alpha=0.5))

## === cell 23
def random_rotation(image_array: ndarray):
    random_degree = random.uniform(-180, 180)
    return sk.transform.rotate(image_array, random_degree)

def random_noise(image_array: ndarray):
    return sk.util.random_noise(image_array)

def horizontal_flip(image_array: ndarray):
    return image_array[:, ::-1]

## --- ERROR in cell 23, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4031249079.py in <cell line: 0>()
----> 1 def random_rotation(image_array: ndarray):
      2     # pick a random degree of rotation between 25% on the left and 25% on the right
      3     random_degree = random.uniform(-180, 180)
      4     return sk.transform.rotate(image_array, random_degree)
      5 

NameError: name 'ndarray' is not defined

## === cell 24
wts = [0.1, 0.4, 0.2 , 0.90, 0.60]

## === cell 25
img = cv2.imread(train_images[0])
img = preproces_image(img)
img = cv2.resize(img, (IMG_SIZE, IMG_SIZE) )
plt.imshow(img)

## === cell 26
num_of_class = 5

## === cell 27
def generate_training_images(cur_images, cur_tags, batch_size=500):
    while True:
        cur_batch = []
        cur_labels = []
        cur_wts = []
        for ix,image in enumerate(cur_images):
            label = cur_tags[ix]
            wt = wts[label]
            img = cv2.imread(image)
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            img = adjust_gamma(img, 1.5)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = add_contrast(img, 20)

            new_img = deepcopy(img)
            cur_batch.append(new_img)
            cur_labels.append(label)
            cur_wts.append(wt)
            if len(cur_batch) == batch_size:
                batch_imgs = np.stack(cur_batch, axis=0)
                batch_targets = keras.utils.np_utils.to_categorical(cur_labels, num_of_class )
                yield batch_imgs,batch_targets, np.array(cur_wts)
                cur_batch = []
                cur_labels = []
                cur_wts = []


            new_img = adjust_gamma(deepcopy(img), random.uniform(0.8, 1.8))
            cur_batch.append(new_img)
            cur_labels.append(label)
            cur_wts.append(wt)
            if len(cur_batch) == batch_size:
                batch_imgs = np.stack(cur_batch, axis=0)
                batch_targets = keras.utils.np_utils.to_categorical(cur_labels, num_of_class)
                yield batch_imgs,batch_targets, np.array(cur_wts)
                cur_batch = []
                cur_labels = []
                cur_wts = []

            new_img = horizontal_flip(deepcopy(img))
            cur_batch.append(new_img)
            cur_labels.append(label)
            cur_wts.append(wt)
            if len(cur_batch) == batch_size:
                batch_imgs = np.stack(cur_batch, axis=0)
                batch_targets = keras.utils.np_utils.to_categorical(cur_labels, num_of_class)
                yield batch_imgs,batch_targets, np.array(cur_wts)
                cur_batch = []
                cur_labels = []
                cur_wts = []
        batch_imgs = np.stack(cur_batch, axis=0)
        batch_targets = keras.utils.np_utils.to_categorical(cur_labels, num_of_class )
        yield batch_imgs,batch_targets,  np.array(cur_wts)

## === cell 28
def generate_testing_images(cur_images, cur_tags, batch_size=500):
    while True:
        cur_batch = []
        cur_labels = []
        cur_wts = []
        for ix,image in enumerate(cur_images):
            label = cur_tags[ix]
            wt = wts[label]
            img = cv2.imread(image)
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            img = adjust_gamma(img, 1.5)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = add_contrast(img, 20)

            new_img = deepcopy(img)
            cur_batch.append(new_img)
            cur_labels.append(label)
            cur_wts.append(wt)
            if len(cur_batch) == batch_size:
                batch_imgs = np.stack(cur_batch, axis=0)
                batch_targets = keras.utils.np_utils.to_categorical(cur_labels, num_of_class )
                yield batch_imgs,batch_targets
                cur_batch = []
                cur_labels = []
                cur_wts = []


            new_img = adjust_gamma(deepcopy(img), random.uniform(0.8, 1.8))
            cur_batch.append(new_img)
            cur_labels.append(label)
            cur_wts.append(wt)
            if len(cur_batch) == batch_size:
                batch_imgs = np.stack(cur_batch, axis=0)
                batch_targets = keras.utils.np_utils.to_categorical(cur_labels, num_of_class)
                yield batch_imgs,batch_targets
                cur_batch = []
                cur_labels = []
                cur_wts = []

            new_img = horizontal_flip(deepcopy(img))
            cur_batch.append(new_img)
            cur_labels.append(label)
            cur_wts.append(wt)
            if len(cur_batch) == batch_size:
                batch_imgs = np.stack(cur_batch, axis=0)
                batch_targets = keras.utils.np_utils.to_categorical(cur_labels, num_of_class)
                yield batch_imgs,batch_targets
                cur_batch = []
                cur_labels = []
                cur_wts = []
        batch_imgs = np.stack(cur_batch, axis=0)
        batch_targets = keras.utils.np_utils.to_categorical(cur_labels, num_of_class )
        yield batch_imgs,batch_targets

## === cell 29
print (len(train_images), len(train_labels))
train_labels[:1]

## === cell 30
for batch in generate_training_images(train_images, train_labels, 100):
    tmp_images, labels, wts  = batch
    print (tmp_images[0].shape, len(labels) )
    plt.imshow(tmp_images[10])
    break

## --- ERROR in cell 30, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2259757984.py in <cell line: 0>()
----> 1 for batch in generate_training_images(train_images, train_labels, 100):
      2     tmp_images, labels, wts  = batch
      3     print (tmp_images[0].shape, len(labels) )
      4     plt.imshow(tmp_images[10])
      5     break

/tmp/ipykernel_11/1304923287.py in generate_training_images(cur_images, cur_tags, batch_size)
     16             #img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
     17 
---> 18             new_img = deepcopy(img)
     19             #new_img += cv2.cvtColor(new_img, cv2.COLOR_BGR2GRAY)
     20             cur_batch.append(new_img)

NameError: name 'deepcopy' is not defined

## === cell 31
from keras.applications.densenet import DenseNet121
from keras.layers import Input
from keras.models import Model
from keras.layers import Dense
from keras.optimizers import Adam
from keras.models import load_model
from IPython.display import clear_output

## === cell 32
def reset_tf_session():
    curr_session = tf.get_default_session()
    if curr_session is not None:
        curr_session.close()
    K.clear_session()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    s = tf.InteractiveSession(config=config)
    K.set_session(s)
    return s

## === cell 33
input_shape = (IMG_SIZE, IMG_SIZE, 3)

## === cell 34
s = reset_tf_session()

## --- ERROR in cell 34, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/3764718488.py in <cell line: 0>()
----> 1 s = reset_tf_session()

/tmp/ipykernel_11/2157438071.py in reset_tf_session()
      1 def reset_tf_session():
----> 2     curr_session = tf.get_default_session()
      3     # close current session
      4     if curr_session is not None:
      5         curr_session.close()

AttributeError: module 'tensorflow' has no attribute 'get_default_session'

## === cell 37
INIT_LR = 5e-3  # initial learning rate
BATCH_SIZE = 200
EPOCHS = 200
def lr_scheduler(epoch):
    return min(INIT_LR * 0.9 ** epoch, 0.00001)

class LrHistory(keras.callbacks.Callback):
    def on_epoch_begin(self, epoch, logs={}):
        print("Learning rate:", K.get_value(model.optimizer.lr))



## === cell 38
len(test_images)
len(test_labels)

## === cell 39
class PlotLearning(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.i = 0
        self.x = []
        self.losses = []
        self.val_losses = []
        self.acc = []
        self.val_acc = []
        self.fig = plt.figure()
        
        self.logs = []

    def on_epoch_end(self, epoch, logs={}):
        if epoch%10 == 0 and epoch > 0:
            self.logs.append(logs)
            self.x.append(self.i)
            self.losses.append(logs.get('loss'))
            self.val_losses.append(logs.get('val_loss'))
            self.acc.append(logs.get('acc'))
            self.val_acc.append(logs.get('val_acc'))
            self.i += 1
            f, (ax1, ax2) = plt.subplots(1, 2, sharex=True)

            clear_output(wait=True)

            ax1.set_yscale('log')
            ax1.plot(self.x, self.losses, label="loss")
            ax1.plot(self.x, self.val_losses, label="val_loss")
            ax1.legend()

            ax2.plot(self.x, self.acc, label="accuracy")
            ax2.plot(self.x, self.val_acc, label="validation accuracy")
            ax2.legend()

            plt.show();
        
plot_losses = PlotLearning()

## === cell 41
!ls ../input/

## === cell 42
model = None
import os
model = load_model('../input/aptosv4/iris_trained_model_v3')
print ("loaded existing model weights")

## --- ERROR in cell 42, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/3257854404.py in <cell line: 0>()
      1 model = None
      2 import os
----> 3 model = load_model('../input/aptosv4/iris_trained_model_v3')
      4 print ("loaded existing model weights")

/usr/local/lib/python3.11/dist-packages/keras/src/saving/saving_api.py in load_model(filepath, custom_objects, compile, safe_mode)
    204         )
    205     else:
--> 206         raise ValueError(
    207             f"File format not supported: filepath={filepath}. "
    208             "Keras 3 only supports V3 `.keras` files and "

ValueError: File format not supported: filepath=../input/aptosv4/iris_trained_model_v3. Keras 3 only supports V3 `.keras` files and legacy H5 format files (`.h5` extension). Note that the legacy SavedModel format is not supported by `load_model()` in Keras 3. In order to reload a TensorFlow SavedModel as an inference-only layer in Keras 3, use `keras.layers.TFSMLayer(../input/aptosv4/iris_trained_model_v3, call_endpoint='serving_default')` (note that your `call_endpoint` might have a different name).

## === cell 46
predict_images = [f for f in glob.glob("../input/aptos2019-blindness-detection/test_images/" + "*.png")]


## === cell 47
len(predict_images)

## === cell 48
def generate_predict_images(cur_images):
    cur_batch = []
    cur_labels = []
    cur_wts = []
    for ix,image in enumerate(cur_images):
        img = cv2.imread(image)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img = adjust_gamma(img, 1.5)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = add_contrast(img, 20)
        cur_batch.append(img)
        cur_labels.append(image.split('/')[-1].split('.')[0])
    return cur_batch, cur_labels

## === cell 49
pred_batch, names = generate_predict_images(predict_images)

## === cell 50
predictions = model.predict(np.array(pred_batch) )

## --- ERROR in cell 50, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/1060642787.py in <cell line: 0>()
----> 1 predictions = model.predict(np.array(pred_batch) )

AttributeError: 'NoneType' object has no attribute 'predict'

## === cell 51
predictions[:5]

## --- ERROR in cell 51, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2667577773.py in <cell line: 0>()
----> 1 predictions[:5]

NameError: name 'predictions' is not defined

## === cell 52
fl = open('submission.csv', 'w')
fl.write("id_code,diagnosis\n")
for ix, row in enumerate(predictions):
    row = list(row)
    fl.write("{},{}\n".format(str(names[ix]), str(row.index(max(row ))) ))
fl.close()

## --- ERROR in cell 52, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3633634295.py in <cell line: 0>()
      1 fl = open('submission.csv', 'w')
      2 fl.write("id_code,diagnosis\n")
----> 3 for ix, row in enumerate(predictions):
      4     row = list(row)
      5     fl.write("{},{}\n".format(str(names[ix]), str(row.index(max(row ))) ))

NameError: name 'predictions' is not defined

## --- ERROR in outputing the csv:
Invalid submission: Submission DataFrame should not be empty
