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

3.9

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

-0.012600939397712

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
DATA_PATH = '../input/aptos2019-blindness-detection/'

## === cell 1
import os
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm


def preprocess_all(input_dir, output_dir, label_path=None, limit=np.inf, skip=0):
    if label_path is not None:
        labels = pd.read_csv(label_path)
        label_dict = dict(zip(labels['id_code'], labels['diagnosis']))
        del labels

    count = 0
    for filepath in tqdm(os.listdir(input_dir)[skip:]):
        if count >= limit:
            break

        input_path = os.path.join(input_dir, filepath)
        if not os.path.isfile(input_path):
            continue

        img = standard_crop(input_path, IMG_DIM=(512, 512), contrast_fnc=contrast_enhance, gray=True)
        img.save(os.path.join(output_dir, str(label_dict[filepath[:-4]]) if label_path else '', filepath))

        count += 1


import os
from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt

STANDARDIZE_CROP_RATIO = 0.792
OVERCROP_THRESHOLD = 25
ZERO_TOLERANCE = 2
BLUE_LAYER_IDX = 2
RED_LAYER_IDX = 0


def autocrop_scale(path, IMG_DIM=(512, 512), EPSILON=7, standardize_crop=False, return_crop_only=False):
    '''
    Loads the image file and automatically crops based on dark regions and rescales
        to a specified dimension ignoring aspect ratio

    :param path: str - path to file
    :param IMG_DIM: tuple(int,int)/None - final dimension of image (no rescale if None)
    :param EPSILON: int - minimum threshold for mean row/col pixel value
    :param standardize_crop: bool - if True, intentionally over-crop perfect circles to standardize to
                                    the images that are not perfect circles
    :param return_crop_only: bool - if True, does not resize and return numpy array
    :return: PIL.Image/np.ndarray
    '''

    data = path
    gray_data = data.mean(axis=2)

    limit_h = np.where(gray_data.mean(axis=0) >= EPSILON)[0]
    horizontal = (limit_h[0], limit_h[-1])
    limit_v = np.where(gray_data.mean(axis=1) >= EPSILON)[0]

    if standardize_crop and (abs((limit_h[-1] - limit_h[0]) - (limit_v[-1] - limit_v[0])) <= OVERCROP_THRESHOLD):
        crop_v = STANDARDIZE_CROP_RATIO * (limit_v[-1] - limit_v[0]) / 2
        center = (limit_v[-1] + limit_v[0]) / 2
        vertical = (int(center - crop_v), int(center + crop_v))
    else:
        vertical = (limit_v[0], limit_v[-1])

    new_data = data[vertical[0]:vertical[1] + 1, horizontal[0]:horizontal[1] + 1, :]

    if new_data.shape[0] < 100 or new_data.shape[1] < 100:
        new_data = data

    if return_crop_only:
        return new_data

    processed_img = Image.fromarray(new_data)
    if IMG_DIM is not None:
        processed_img = processed_img.resize(IMG_DIM)

    return processed_img


def standard_crop(path, IMG_DIM=(512, 512), ratio=4 / 3, cratio=592 / 386, contrast_fnc=None, **kwargs):
    '''
    Crops an image to an approximate form found in majority of the images in the test dataset.

    The values found here is derived from manual measurements with trigonometry.

    :param path: str - path of image
    :param IMG_DIM: tuple(int, int)/None - final dimension of image (no rescale if None)
    :param ratio: float - final aspect ratio of output image
    :param cratio: flaot - ratio of chord at the bottom to the radius
    :param contrast_fnc: None/function - contrast function
    :return: PIL.Image
    '''
    data = autocrop_scale(path, return_crop_only=True)  # crops away all excess background
    h, l = data.shape[0], data.shape[1]

    sample_column = data[:, 1, :]  # second column of data
    edge = np.where((data.mean(axis=2)[:, 0] > sample_column.mean() + 5 / sample_column.mean(axis=1).std()))
    edge = edge[0][edge[0] > h / 8]  # accepts if not near the corner
    if len(edge) > 0:
        r = np.sqrt((edge[0] - h / 2) ** 2 + (l / 2) ** 2)
    else:
        r = l / 2  # assumes length is the diameter

    delta_h = r - np.sqrt(r ** 2 - (cratio * r / 2) ** 2)  # height to be cropped away at both the top and bottom
    data = data[max(int(delta_h - (2 * r - h) / 2), 0):h - max(int(delta_h - (2 * r - h) / 2), 0), ...]

    h, l = data.shape[0], data.shape[1]
    delta_l = l - h * ratio  # length to be cropped away at both the left and right

    data = data[:, int(delta_l / 2):l - int(delta_l / 2), :]

    data = cv2.resize(data, dsize=IMG_DIM, interpolation=cv2.INTER_AREA)

    if contrast_fnc is not None:
        data = contrast_fnc(data, **kwargs)


    return data.reshape(*data.shape, 1)


def cropscale_all(input_dir, output_dir, cropping_fnc='standard', limit=np.inf, **kwargs):
    '''
    Calls autocrop_scale on all files in input_dir. This is non-recursive.

    :param input_dir: str - path to files to be processed
    :param output_dir: str - path to location for processed images to be stored
    :param cropping_fnc: "standard"/"autocrop" - type of cropping function to use
    :param limit: int - number of images to process
    :param kwargs: dict - arguments for cropping function
    :return: None
    '''
    if cropping_fnc == 'standard':
        cropping_fnc = lambda x: standard_crop(x, **kwargs)
    elif cropping_fnc == 'autocrop':
        cropping_fnc = lambda x: autocrop_scale(x, **kwargs)
    else:
        raise ValueError('Incorrect argument for cropping_fnc')

    count = 0
    for filepath in os.listdir(input_dir):
        if count >= limit:
            break

        input_path = os.path.join(input_dir, filepath)
        if not os.path.isfile(input_path):
            continue

        img = cropping_fnc(input_path)
        img.save(os.path.join(output_dir, filepath))
        count += 1


def contrast_enhance(img, sigma=10, gray=False):
    '''
    Contrast technique based on Kaggle notebook

    :param img: np.ndarray - image
    :param sigma: int - degree of contrast (affects efficiency of code)
    :param gray: bool - returns grayscale (i.e. 1 channel)
    :return: nd.ndarray
    '''
    if gray:
        img = img.mean(axis=2)
        img = np.dstack((img, img, img)).astype(np.uint8)

    return cv2.addWeighted(img, 4, cv2.GaussianBlur(img, (0, 0), sigma), -4, 128)


## === cell 3
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
def quadratic_kappa(actuals, preds, N=5):
    """This function calculates the Quadratic Kappa Metric used for Evaluation in the PetFinder competition
    at Kaggle. It returns the Quadratic Weighted Kappa metric score between the actual and the predicted values
    of adoption rating."""
    w = np.zeros((N, N))
    O = confusion_matrix(actuals, preds)
    for i in range(len(w)):
        for j in range(len(w)):
            w[i][j] = float(((i - j) ** 2) / (N - 1) ** 2)

    act_hist = np.zeros([N])
    for item in actuals:
        act_hist[item] += 1

    pred_hist = np.zeros([N])
    for item in preds:
        pred_hist[item] += 1

    E = np.outer(act_hist, pred_hist)
    E = E / E.sum()
    O = O / O.sum()

    num = 0
    den = 0
    for i in range(len(w)):
        for j in range(len(w)):
            num += w[i][j] * O[i][j]
            den += w[i][j] * E[i][j]
    return 1 - (num / den)

## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 4
model = load_model('../input/trial-1-model/trial_1_model.h5')

## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/2579628267.py in <cell line: 0>()
----> 1 model = load_model('../input/trial-1-model/trial_1_model.h5')

/usr/local/lib/python3.11/dist-packages/keras/src/saving/saving_api.py in load_model(filepath, custom_objects, compile, safe_mode)
    194         )
    195     if str(filepath).endswith((".h5", ".hdf5")):
--> 196         return legacy_h5_format.load_model_from_hdf5(
    197             filepath, custom_objects=custom_objects, compile=compile
    198         )

/usr/local/lib/python3.11/dist-packages/keras/src/legacy/saving/legacy_h5_format.py in load_model_from_hdf5(filepath, custom_objects, compile)
    114     opened_new_file = not isinstance(filepath, h5py.File)
    115     if opened_new_file:
--> 116         f = h5py.File(filepath, mode="r")
    117     else:
    118         f = filepath

/usr/local/lib/python3.11/dist-packages/h5py/_hl/files.py in __init__(self, name, mode, driver, libver, userblock_size, swmr, rdcc_nslots, rdcc_nbytes, rdcc_w0, track_order, fs_strategy, fs_persist, fs_threshold, fs_page_size, page_buf_size, min_meta_keep, min_raw_keep, locking, alignment_threshold, alignment_interval, meta_block_size, **kwds)
    562                                  fs_persist=fs_persist, fs_threshold=fs_threshold,
    563                                  fs_page_size=fs_page_size)
--> 564                 fid = make_fid(name, mode, userblock_size, fapl, fcpl, swmr=swmr)
    565 
    566             if isinstance(libver, tuple):

/usr/local/lib/python3.11/dist-packages/h5py/_hl/files.py in make_fid(name, mode, userblock_size, fapl, fcpl, swmr)
    236         if swmr and swmr_support:
    237             flags |= h5f.ACC_SWMR_READ
--> 238         fid = h5f.open(name, flags, fapl=fapl)
    239     elif mode == 'r+':
    240         fid = h5f.open(name, h5f.ACC_RDWR, fapl=fapl)

h5py/_objects.pyx in h5py._objects.with_phil.wrapper()

h5py/_objects.pyx in h5py._objects.with_phil.wrapper()

h5py/h5f.pyx in h5py.h5f.open()

FileNotFoundError: [Errno 2] Unable to synchronously open file (unable to open file: name = '../input/trial-1-model/trial_1_model.h5', errno = 2, error message = 'No such file or directory', flags = 0, o_flags = 0)

## === cell 5
submission_df = pd.read_csv(DATA_PATH + '/test.csv')
submission_df['filename'] = submission_df['id_code'].astype(str)+'.png'

submission = ImageDataGenerator(preprocessing_function=lambda x: standard_crop(x,IMG_DIM=(128,128), contrast_fnc=contrast_enhance, gray=False))
gen = submission.flow_from_dataframe(dataframe = submission_df,
                                       directory= DATA_PATH + "test_images",
                                       x_col="filename", color_mode = "grayscale",   
                                       batch_size = 32,
                                       shuffle=False,
                                       class_mode=None, 
                                       target_size=(128, 128), validate_filenames=False)

pred = model.predict_generator(gen, verbose=1)




## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3595080480.py in <cell line: 0>()
     11                                        target_size=(128, 128), validate_filenames=False)
     12 
---> 13 pred = model.predict_generator(gen, verbose=1)
     14 
     15 

NameError: name 'model' is not defined

## === cell 6
pred = np.argmax(pred, axis=1)
submission_df.drop(columns=['filename'], inplace= True)
submission_df['diagnosis'] = pred
submission_df.to_csv('submission.csv', index=False)

## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4140010424.py in <cell line: 0>()
----> 1 pred = np.argmax(pred, axis=1)
      2 submission_df.drop(columns=['filename'], inplace= True)
      3 submission_df['diagnosis'] = pred
      4 submission_df.to_csv('submission.csv', index=False)

NameError: name 'pred' is not defined

## === cell 7
submission_df.to_csv('../submission.csv', index=False)

## === cell 8
submission_df

## === cell 9
submission_df.shape
