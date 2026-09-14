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
Segment regions of salt in seismic images.

## Metric
Mean average precision at different intersection over union (IoU) thresholds. The IoU of a proposed set of object pixels and a set of true object pixels is calculated as:

$$\text{IoU}(A, B)=\frac{A \cap B}{A \cup B}$$

The metric sweeps over a range of IoU thresholds, at each point calculating an average precision value. The threshold values range from 0.5 to 0.95 with a step size of 0.05: `(0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95)`. In other words, at a threshold of 0.5, a predicted object is considered a "hit" if its intersection over union with a ground truth object is greater than 0.5.

At each threshold value 𝑡t, a precision value is calculated based on the number of true positives (TP), false negatives (FN), and false positives (FP) resulting from comparing the predicted object to all ground truth objects:

$$\frac{T P(t)}{T P(t)+F P(t)+F N(t)}$$

A true positive is counted when a single predicted object matches a ground truth object with an IoU above the threshold. A false positive indicates a predicted object had no associated ground truth object. A false negative indicates a ground truth object had no associated predicted object. The average precision of a single image is then calculated as the mean of the above precision values at each IoU threshold:

$$\frac{1}{\mid \text { thresholds } \mid} \sum_t \frac{T P(t)}{T P(t)+F P(t)+F N(t)}$$

## Submission Format
Use run-length encoding on the pixel values. Instead of submitting an exhaustive list of indices for your segmentation, you will submit pairs of values that contain a start position and a run length. E.g. '1 3' implies starting at pixel 1 and running a total of 3 pixels (1,2,3).

The competition format requires a space delimited list of pairs. For example, '1 3 10 5' implies pixels 1,2,3,10,11,12,13,14 are to be included in the mask. The pixels are one-indexed\
and numbered from top to bottom, then left to right: 1 is pixel (1,1), 2 is pixel (2,1), etc.

The metric checks that the pairs are sorted, positive, and the decoded pixel values are not duplicated. It also checks that no two predicted masks for the same image are overlapping.

The file should contain a header and have the following format. Each row in your submission represents a single predicted salt segmentation for the given image.

```
id,rle_mask
3e06571ef3,1 1
a51b08d882,1 1
c32590b06f,1 1
etc.
```

## Dataset
The data is a set of images chosen at various locations chosen at random in the subsurface. The images are 101 x 101 pixels and each pixel is classified as either salt or sediment. In addition to the seismic images, the depth of the imaged location is provided for each image.

# 2. Python version

3.9

# 3. Installed packages

No external packages required in the script and installed.

# 4. Data file paths

```
/
    kaggle/
        data/
            depths.csv (3001 lines)
            depths.csv.zip (25.0 kB)
            description.md (83 lines)
            sample_submission.csv (1001 lines)
            sample_submission.csv.zip (6.8 kB)
            test.zip (9.8 MB)
            train.csv (3001 lines)
            train.csv.zip (293.9 kB)
            train.zip (30.9 MB)
            test/
                images/
                    a05ae39815.png (10.5 kB)
                    9bf8a38b92.png (10.7 kB)
                    ... and 998 other files
                test/
            tgs-salt-identification-challenge/
                depths.csv (3001 lines)
                depths.csv.zip (25.0 kB)
                ... and 7 other files
                test/
                    images/
                        a05ae39815.png (10.5 kB)
                        9bf8a38b92.png (10.7 kB)
                        ... and 998 other files
                    test/
                tgs-salt-identification-challenge/
                train/
                    images/
                        66cf41c563.png (7.7 kB)
                        429bf7c665.png (8.2 kB)
                        ... and 2998 other files
                    masks/
                        6eeeda7f4a.png (230 Bytes)
                        5d600057f5.png (230 Bytes)
                        ... and 2998 other files
                    train/
            train/
                images/
                    66cf41c563.png (7.7 kB)
                    429bf7c665.png (8.2 kB)
                    ... and 2998 other files
                masks/
                    6eeeda7f4a.png (230 Bytes)
                    5d600057f5.png (230 Bytes)
                    ... and 2998 other files
                train/
        input/
            depths.csv (3001 lines)
            depths.csv.zip (25.0 kB)
            description.md (83 lines)
            sample_submission.csv (1001 lines)
            sample_submission.csv.zip (6.8 kB)
            test.zip (9.8 MB)
            train.csv (3001 lines)
            train.csv.zip (293.9 kB)
            train.zip (30.9 MB)
            test/
                images/
                    a05ae39815.png (10.5 kB)
                    9bf8a38b92.png (10.7 kB)
                    ... and 998 other files
                test/
                    images/
                        a05ae39815.png (10.5 kB)
                        9bf8a38b92.png (10.7 kB)
                        ... and 998 other files
                    test/
            tgs-salt-identification-challenge/
                depths.csv (3001 lines)
                depths.csv.zip (25.0 kB)
                ... and 7 other files
                test/
                    images/
                        a05ae39815.png (10.5 kB)
                        9bf8a38b92.png (10.7 kB)
                        ... and 998 other files
                    test/
                tgs-salt-identification-challenge/
                train/
                    images/
                        66cf41c563.png (7.7 kB)
                        429bf7c665.png (8.2 kB)
                        ... and 2998 other files
                    masks/
                        6eeeda7f4a.png (230 Bytes)
                        5d600057f5.png (230 Bytes)
                        ... and 2998 other files
                    train/
            train/
                images/
                    66cf41c563.png (7.7 kB)
                    429bf7c665.png (8.2 kB)
                    ... and 2998 other files
                masks/
                    6eeeda7f4a.png (230 Bytes)
                    5d600057f5.png (230 Bytes)
                    ... and 2998 other files
                train/
                    images/
                        66cf41c563.png (7.7 kB)
                        429bf7c665.png (8.2 kB)
                        ... and 2998 other files
                    masks/
                        6eeeda7f4a.png (230 Bytes)
                        5d600057f5.png (230 Bytes)
                        ... and 2998 other files
                    train/
        working/
            tgs-salt-identification-challenge/
                depths.csv (3001 lines)
                depths.csv.zip (25.0 kB)
                ... and 7 other files
                test/
                    images/
                        a05ae39815.png (10.5 kB)
                        9bf8a38b92.png (10.7 kB)
                        ... and 998 other files
                    test/
                tgs-salt-identification-challenge/
                train/
                    images/
                        66cf41c563.png (7.7 kB)
                        429bf7c665.png (8.2 kB)
                        ... and 2998 other files
                    masks/
                        6eeeda7f4a.png (230 Bytes)
                        5d600057f5.png (230 Bytes)
                        ... and 2998 other files
                    train/
```

-> data/depths.csv has 3000 rows and 2 columns.
The columns are: id, z

-> data/sample_submission.csv has 1000 rows and 2 columns.
The columns are: id, rle_mask

-> data/tgs-salt-identification-challenge/depths.csv has 3000 rows and 2 columns.
The columns are: id, z

-> data/tgs-salt-identification-challenge/sample_submission.csv has 1000 rows and 2 columns.
The columns are: id, rle_mask

-> data/tgs-salt-identification-challenge/train.csv has 3000 rows and 2 columns.
The columns are: id, rle_mask

-> data/train.csv has 3000 rows and 2 columns.
The columns are: id, rle_mask

-> input/depths.csv has 3000 rows and 2 columns.
The columns are: id, z

-> (stopped after 10 files for performance)

# 5. Target score

0.8134314149718772

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 3
import numpy as np
import pandas as pd
import tensorflow as tf 
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm

from sklearn.model_selection import train_test_split

from keras.models import Model
from keras.layers import  Conv2D, Conv2DTranspose, MaxPooling2D, Concatenate, concatenate, Dropout, BatchNormalization, Activation, Input, Add
    
from keras import models, layers, callbacks, optimizers
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from keras.preprocessing.image import load_img, img_to_array

## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 6
ROOT_DATA_DIR = '../input/tgs-salt-data/'

TRAIN_MASK_DIR = ROOT_DATA_DIR + 'train/masks/'
TRAIN_IMAGE_DIR = ROOT_DATA_DIR + 'train/images/'

TEST_IMAGE_DIR = ROOT_DATA_DIR + 'test/images/'

## === cell 9
train_df = pd.read_csv('../input/tgs-salt-identification-challenge/train.csv')
test_df = pd.read_csv('../input/tgs-salt-identification-challenge/sample_submission.csv')

depths_df = pd.read_csv(ROOT_DATA_DIR + 'depths.csv')

## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/3226259455.py in <cell line: 0>()
      2 test_df = pd.read_csv('../input/tgs-salt-identification-challenge/sample_submission.csv')
      3 
----> 4 depths_df = pd.read_csv(ROOT_DATA_DIR + 'depths.csv')

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in read_csv(filepath_or_buffer, sep, delimiter, header, names, index_col, usecols, dtype, engine, converters, true_values, false_values, skipinitialspace, skiprows, skipfooter, nrows, na_values, keep_default_na, na_filter, verbose, skip_blank_lines, parse_dates, infer_datetime_format, keep_date_col, date_parser, date_format, dayfirst, cache_dates, iterator, chunksize, compression, thousands, decimal, lineterminator, quotechar, quoting, doublequote, escapechar, comment, encoding, encoding_errors, dialect, on_bad_lines, delim_whitespace, low_memory, memory_map, float_precision, storage_options, dtype_backend)
   1024     kwds.update(kwds_defaults)
   1025 
-> 1026     return _read(filepath_or_buffer, kwds)
   1027 
   1028 

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in _read(filepath_or_buffer, kwds)
    618 
    619     # Create the parser.
--> 620     parser = TextFileReader(filepath_or_buffer, **kwds)
    621 
    622     if chunksize or iterator:

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in __init__(self, f, engine, **kwds)
   1618 
   1619         self.handles: IOHandles | None = None
-> 1620         self._engine = self._make_engine(f, self.engine)
   1621 
   1622     def close(self) -> None:

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in _make_engine(self, f, engine)
   1878                 if "b" not in mode:
   1879                     mode += "b"
-> 1880             self.handles = get_handle(
   1881                 f,
   1882                 mode,

/usr/local/lib/python3.11/dist-packages/pandas/io/common.py in get_handle(path_or_buf, mode, encoding, compression, memory_map, is_text, errors, storage_options)
    871         if ioargs.encoding and "b" not in ioargs.mode:
    872             # Encoding
--> 873             handle = open(
    874                 handle,
    875                 ioargs.mode,

FileNotFoundError: [Errno 2] No such file or directory: '../input/tgs-salt-data/depths.csv'

## === cell 11
train_df.head()

## === cell 12
train_df.info()

## === cell 14
depths_df.head()

## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1803677713.py in <cell line: 0>()
----> 1 depths_df.head()

NameError: name 'depths_df' is not defined

## === cell 15
depths_df.info()

## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2531767661.py in <cell line: 0>()
----> 1 depths_df.info()

NameError: name 'depths_df' is not defined

## === cell 18
def get_coverage(rle_mask):
    
    if pd.isna(rle_mask): # ảnh không có muối - rle_mask rỗng

        return 0
    
    else: # ảnh có muối        
        arr = rle_mask.split() # cắt chuỗi rle thành list        
        coverage = sum(int(x) for x in arr[1::2]) / (101**2) # cộng tổng các pixel (số ở vị trí lẻ)

        return np.round(coverage,1)

train_df['coverage'] = train_df['rle_mask'].map(get_coverage)
train_df.head()

## === cell 19
train_df.value_counts('coverage', sort=False, ascending=True)

## === cell 20
train_df.value_counts('coverage', sort=False, ascending=True).plot(kind ='bar', title='Phân bố mật độ muối trong ảnh' , rot=0);

## === cell 23
train_imgs = [load_img(TRAIN_IMAGE_DIR + image_name + '.png', color_mode='grayscale') for image_name in tqdm(train_df['id'])]
train_masks = [load_img(TRAIN_MASK_DIR + image_name + '.png', color_mode='grayscale') for image_name in tqdm(train_df['id'])]

train_imgs = [img_to_array(img, dtype=np.uint8)/255 for img in train_imgs]
train_masks = [img_to_array(mask, dtype=np.uint8)//255 for mask in train_masks]

## --- ERROR in cell 23, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/2944676521.py in <cell line: 0>()
      1 # Load image
----> 2 train_imgs = [load_img(TRAIN_IMAGE_DIR + image_name + '.png', color_mode='grayscale') for image_name in tqdm(train_df['id'])]
      3 train_masks = [load_img(TRAIN_MASK_DIR + image_name + '.png', color_mode='grayscale') for image_name in tqdm(train_df['id'])]
      4 
      5 # Chuyển từ PIL sang np và chuẩn hoá

/tmp/ipykernel_11/2944676521.py in <listcomp>(.0)
      1 # Load image
----> 2 train_imgs = [load_img(TRAIN_IMAGE_DIR + image_name + '.png', color_mode='grayscale') for image_name in tqdm(train_df['id'])]
      3 train_masks = [load_img(TRAIN_MASK_DIR + image_name + '.png', color_mode='grayscale') for image_name in tqdm(train_df['id'])]
      4 
      5 # Chuyển từ PIL sang np và chuẩn hoá

/usr/local/lib/python3.11/dist-packages/keras/src/utils/image_utils.py in load_img(path, color_mode, target_size, interpolation, keep_aspect_ratio)
    233         if isinstance(path, pathlib.Path):
    234             path = str(path.resolve())
--> 235         with open(path, "rb") as f:
    236             img = pil_image.open(io.BytesIO(f.read()))
    237     else:

FileNotFoundError: [Errno 2] No such file or directory: '../input/tgs-salt-data/train/images/000e218f21.png'

## === cell 25
fig, axes = plt.subplots(nrows=5, ncols=3, figsize=(15, 20));
axes[0][0].set_title('Image', size='large')
axes[0][1].set_title('Mask', size='large')
axes[0][2].set_title('Overlay', size='large')
for row in range(5):
    axes[row][0].imshow(train_imgs[row],cmap="gray")

    axes[row][1].imshow(train_masks[row],cmap="gray")

    axes[row][2].imshow(train_imgs[row],cmap="gray")
    axes[row][2].imshow(train_masks[row], alpha=0.2, cmap="Reds")
    axes[row][2].contour(train_masks[row].squeeze(), colors='red')

## --- ERROR in cell 25, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1519832224.py in <cell line: 0>()
      7     # Image
      8 #     axes[row][0].axis('off')
----> 9     axes[row][0].imshow(train_imgs[row],cmap="gray")
     10 
     11     # Mask

NameError: name 'train_imgs' is not defined

## === cell 28
x_train, x_valid, y_train, y_valid = train_test_split( np.array(train_imgs), 
                                                      np.array(train_masks), 
                                                      test_size = 0.2, 
                                                      stratify=train_df.coverage, 
                                                      random_state=1234)

## --- ERROR in cell 28, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3054652343.py in <cell line: 0>()
----> 1 x_train, x_valid, y_train, y_valid = train_test_split( np.array(train_imgs), 
      2                                                       np.array(train_masks),
      3                                                       test_size = 0.2,
      4                                                       stratify=train_df.coverage,
      5                                                       random_state=1234)

NameError: name 'train_imgs' is not defined

## === cell 31
x_train = np.append(x_train, [np.fliplr(x) for x in x_train], axis=0)
y_train = np.append(y_train, [np.fliplr(x) for x in y_train], axis=0)

## --- ERROR in cell 31, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1939030412.py in <cell line: 0>()
----> 1 x_train = np.append(x_train, [np.fliplr(x) for x in x_train], axis=0)
      2 y_train = np.append(y_train, [np.fliplr(x) for x in y_train], axis=0)

NameError: name 'x_train' is not defined

## === cell 35
def iou_vector(trues, preds):
    SMOOTH = 1e-10
    batch_size = trues.shape[0]
    
    metric = []

    for idx in range(batch_size):

        true, pred = trues[idx], preds[idx]

        intersection = np.logical_and(true, pred)
        union = np.logical_or(true,pred)
        
        iou = (np.sum(intersection) + SMOOTH) / (np.sum(union) + SMOOTH)
        
        thresholds = np.arange(0.5, 1, 0.05)
        s = []
        for thresh in thresholds:
            s.append(iou > thresh)
            
        metric.append(np.mean(s))

    return np.mean(metric)

def my_iou_metric(label, pred):
    
    return tf.py_function(iou_vector, [label, pred>0.5], tf.float64)

## === cell 39
def convolution_block(x, filters, size, strides=(1,1), padding='same', activation=True):
    x = Conv2D(filters, size, strides=strides, padding=padding)(x)
    x = BatchNormalization()(x)
    if activation == True:
        x = Activation('relu')(x)
    return x

def residual_block(blockInput, num_filters=16):
    x = convolution_block(blockInput, num_filters, (3,3) )
    x = convolution_block(x, num_filters, (3,3), activation=False)
    x = Add()([x, blockInput])
    x = Activation('relu')(x)
    return x

## === cell 41
def build_model(input_shape):
    start_feature = 32    
    DropoutRatio = 0.5
    skip_connections = []
    hyper_columns = []
    
    inputs = Input(input_shape)
    x = inputs

    for i in range(4):
        x = convolution_block(x, start_feature*(2**i), 3)
        x = residual_block(x, start_feature*(2**i))
        x = residual_block(x, start_feature*(2**i))

        skip_connections.append(x)

        x = MaxPooling2D((2,2))(x)


    x = convolution_block(x, start_feature*(2**4), 3)
    x = residual_block(x, start_feature*(2**4))
    x = residual_block(x, start_feature*(2**4))

    for i in reversed(range(4)):
        if x.shape[2]*2 != skip_connections[i].shape[2]:
            x = Conv2DTranspose(start_feature*(2**i), (3,3), strides=(2, 2), padding='valid')(x)
        else:
            x = Conv2DTranspose(start_feature*(2**i), (3,3), strides=(2, 2), padding='same')(x)

        x = concatenate([x, skip_connections[i]])
        x = Dropout(DropoutRatio)(x)

        x = convolution_block(x, start_feature*(2**i), 3)
        x = residual_block(x, start_feature*(2**i))
        x = residual_block(x, start_feature*(2**i))


    output = Conv2D(1, (1,1), padding='same', activation='sigmoid')(x)

    model = Model(inputs, output)

    return model

## === cell 44
model = build_model((101,101,1))
model.compile(loss='binary_crossentropy', optimizer="Adam", metrics=[my_iou_metric])

model.summary()

## === cell 47
early_stopping = EarlyStopping(monitor='val_my_iou_metric', mode = 'max',patience=15, verbose=1)

reduce_lr = ReduceLROnPlateau(monitor='val_my_iou_metric', mode='max', factor=0.5, patience=5,
                              min_lr=0.0001, verbose=1)

epochs = 100
steps_per_epoch = 50

history = model.fit(x_train, y_train,
                    validation_data = (x_valid, y_valid), 
                    epochs = epochs, 
                    steps_per_epoch = steps_per_epoch, 
                    callbacks = [early_stopping, reduce_lr],
                    verbose = 1)

## --- ERROR in cell 47, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1991793982.py in <cell line: 0>()
      9 steps_per_epoch = 50
     10 
---> 11 history = model.fit(x_train, y_train,
     12                     validation_data = (x_valid, y_valid),
     13                     epochs = epochs,

NameError: name 'x_train' is not defined

## === cell 49
fig, (ax_loss, ax_score) = plt.subplots(1, 2, figsize=(15,5));
ax_loss.set_ylim(0,1)
ax_loss.plot(history.epoch, history.history["loss"], label="Train loss")
ax_loss.plot(history.epoch, history.history["val_loss"], label="Validation loss")
ax_score.plot(history.epoch, history.history["my_iou_metric"], label="Train score")
ax_score.plot(history.epoch, history.history["val_my_iou_metric"], label="Validation score")


## --- ERROR in cell 49, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/268780954.py in <cell line: 0>()
      1 fig, (ax_loss, ax_score) = plt.subplots(1, 2, figsize=(15,5));
      2 ax_loss.set_ylim(0,1)
----> 3 ax_loss.plot(history.epoch, history.history["loss"], label="Train loss")
      4 ax_loss.plot(history.epoch, history.history["val_loss"], label="Validation loss")
      5 # ax_loss.legend()

NameError: name 'history' is not defined

## === cell 53
pred_valid = model.predict(x_valid[:5])

fig, axes = plt.subplots(nrows=5, ncols=4, figsize=(15, 20));
axes[0][0].set_title('Image', size='large')
axes[0][1].set_title('True mask', size='large')
axes[0][2].set_title('Predict mask', size='large')
axes[0][3].set_title('Overlay', size='large')
for row in range(5):
    axes[row][0].axis('off')
    axes[row][0].imshow(x_valid[row],cmap="gray")

    axes[row][1].axis('off')
    axes[row][1].imshow(y_valid[row],cmap="gray")

    axes[row][2].axis('off')
    axes[row][2].imshow(np.round(pred_valid[row]),cmap="gray")

    axes[row][3].axis('off')
    axes[row][3].imshow(x_valid[row],cmap="gray")
    axes[row][3].imshow(y_valid[row], alpha=0.2, cmap="Reds")
    axes[row][3].contour(y_valid[row].squeeze(), colors='red')
    axes[row][3].imshow(np.round(pred_valid[row]), alpha=0.2, cmap="Blues")
    axes[row][3].contour(np.round(pred_valid[row]).squeeze(), colors='blue')

## --- ERROR in cell 53, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1343774466.py in <cell line: 0>()
      1 # Predict valid
----> 2 pred_valid = model.predict(x_valid[:5])
      3 
      4 fig, axes = plt.subplots(nrows=5, ncols=4, figsize=(15, 20));
      5 axes[0][0].set_title('Image', size='large')

NameError: name 'x_valid' is not defined

## === cell 56
test_imgs = [load_img(TEST_IMAGE_DIR + image_name + '.png', color_mode='grayscale') for image_name in tqdm(test_df['id'])]
test_imgs = np.array([img_to_array(img, dtype=np.uint8)/255 for img in test_imgs])

pred_test = model.predict(test_imgs, verbose=1)

## --- ERROR in cell 56, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/3031591679.py in <cell line: 0>()
----> 1 test_imgs = [load_img(TEST_IMAGE_DIR + image_name + '.png', color_mode='grayscale') for image_name in tqdm(test_df['id'])]
      2 test_imgs = np.array([img_to_array(img, dtype=np.uint8)/255 for img in test_imgs])
      3 
      4 pred_test = model.predict(test_imgs, verbose=1)

/tmp/ipykernel_11/3031591679.py in <listcomp>(.0)
----> 1 test_imgs = [load_img(TEST_IMAGE_DIR + image_name + '.png', color_mode='grayscale') for image_name in tqdm(test_df['id'])]
      2 test_imgs = np.array([img_to_array(img, dtype=np.uint8)/255 for img in test_imgs])
      3 
      4 pred_test = model.predict(test_imgs, verbose=1)

/usr/local/lib/python3.11/dist-packages/keras/src/utils/image_utils.py in load_img(path, color_mode, target_size, interpolation, keep_aspect_ratio)
    233         if isinstance(path, pathlib.Path):
    234             path = str(path.resolve())
--> 235         with open(path, "rb") as f:
    236             img = pil_image.open(io.BytesIO(f.read()))
    237     else:

FileNotFoundError: [Errno 2] No such file or directory: '../input/tgs-salt-data/test/images/003c477d7c.png'

## === cell 57
test_flip = np.array([np.fliplr(x) for x in test_imgs])
pred_test_flip = model.predict(test_flip, verbose=1)

pred_test += np.array([ np.fliplr(x) for x in pred_test_flip])
pred_test/=2

## --- ERROR in cell 57, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/377701948.py in <cell line: 0>()
----> 1 test_flip = np.array([np.fliplr(x) for x in test_imgs])
      2 pred_test_flip = model.predict(test_flip, verbose=1)
      3 
      4 pred_test += np.array([ np.fliplr(x) for x in pred_test_flip])
      5 pred_test/=2

NameError: name 'test_imgs' is not defined

## === cell 60
def mask2rle(mask):
    pixels = mask.flatten(order='F')
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

## === cell 62
rle_masks = [mask2rle(np.round(x)) for x in pred_test]

## --- ERROR in cell 62, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/172858064.py in <cell line: 0>()
----> 1 rle_masks = [mask2rle(np.round(x)) for x in pred_test]

NameError: name 'pred_test' is not defined

## === cell 65
test_df["rle_mask"] = rle_masks
test_df.to_csv("submission.csv", index=False)

## --- ERROR in cell 65, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1218268559.py in <cell line: 0>()
----> 1 test_df["rle_mask"] = rle_masks
      2 test_df.to_csv("submission.csv", index=False)

NameError: name 'rle_masks' is not defined
