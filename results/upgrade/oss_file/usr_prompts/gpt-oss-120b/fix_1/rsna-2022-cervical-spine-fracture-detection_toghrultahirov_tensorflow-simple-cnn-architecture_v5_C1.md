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
Identify fractures in CT scans of the cervical spine (neck) at both the level of a single vertebrae and the entire patient.

## Metric
Weighted multi-label logarithmic loss. Each fracture sub-type is its own row for every exam, and you are expected to predict a probability for a fracture at each of the seven cervical vertebrae designated as C1, C2, C3, C4, C5, C6 and C7. There is also an any label, `patient_overall`, which indicates that a fracture of ANY kind described before exists in the examination. Fractures in the skull base, thoracic spine, ribs, and clavicles are ignored. The any label is weighted more highly than specific fracture level sub-types.

For each exam Id, you must submit a set of predicted probabilities (a separate row for each cervical level subtype). We then take the log loss for each predicted probability versus its true label.

The binary weighted log loss function for label j on exam i is specified as:

$$
L_{i j}=-w_j *\left[y_{i j} * \log \left(p_{i j}\right)+\left(1-y_{i j}\right) * \log \left(1-p_{i j}\right)\right]
$$

Finally, loss is averaged across all rows.

## Submission Format
There will be 8 rows per image Id. The label indicated by a particular row will look like [image Id]_[Sub-type Name], as follows. There is also a target column, `fractured`, indicating the probability of whether a fracture exists at the specified level. For each image ID in the test set, you must predict a probability for each of the different possible sub-types and the patient overall. The file should contain a header and have the following format:

```
row_id,fractured
1_C1,0
1_C2,0
1_C3,0
1_C4,0.6
1_C5,0
1_C6,0.9
1_C7,0.01
1_patient_overall,0.99
2_C1,0
etc.
```

## Dataset
**train.csv** Metadata for the train test set.

- `StudyInstanceUID` - The study ID. There is one unique study ID for each patient scan.
- `patient_overall` - One of the target columns. The patient level outcome, i.e. if any of the vertebrae are fractured.
- `C[1-7]` - The other target columns. Whether the given vertebrae is fractured. See [this diagram](https://en.wikipedia.org/wiki/Vertebral_column#/media/File:Gray_111_-_Vertebral_column-coloured.png) for the real location of each vertbrae in the spine.

**test.csv** Metadata for the test set prediction structure. Only the first few rows of the test set are available for download.

- `row_id` - The row ID. This will match the same column in the sample submission file.
- `StudyInstanceUID` - The study ID.
- `prediction_type` - Which one of the eight target columns needs a prediction in this row.

**[train/test]_images/[StudyInstanceUID]/[slice_number].dcm** The image data, organized with one folder per scan. Expect to see roughly 1,500 scans in the hidden test set.\

Each image is in [the dicom file format](https://www.dicomstandard.org/). The DICOM image files are ≤ 1 mm slice thickness, axial orientation, and bone kernel. Note that some of the DICOM files are JPEG compressed. You may require additional resources to read the pixel array of these files, such as GDCM and pylibjpeg.

**sample_submission.csv** A valid sample submission.

- `row_id` - The row ID. See the test.csv for what prediction needs to be filed in that row.
- `fractured` - The target column.

**train_bounding_boxes.csv** Bounding boxes for a subset of the training set.

**segmentations/** Pixel level annotations for a subset of the training set. This data is provided in the [nifti file format](https://nifti.nimh.nih.gov/).

A portion of the imaging datasets have been segmented automatically using a 3D UNET model, and radiologists modified and approved the segmentations. The provided segmentation labels have values of 1 to 7 for C1 to C7 (seven cervical vertebrae) and 8 to 19 for T1 to T12 (twelve thoracic vertebrae are located in the center of your upper and middle back), and 0 for everything else. As we focused on the cervical spine, all scans have C1 to C7 labels but not all thoracic labels.

Please be aware that the NIFTI files consist of segmentation in the sagittal plane, while the DICOM files are in the axial plane. Please use the NIFTI header information to determine the appropriate orientation such that the DICOM images and segmentation match. Otherwise, you run the risk of having the segmentations flipped in the Z axis and mirrored in the X axis.

# 2. Python version

3.10

# 3. Installed packages

cloudpathlib==0.21.1
cuda-pathfinder==1.3.2
geopandas==0.14.4
jmespath==1.0.1
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
nibabel==5.3.2
numpy==1.26.4
opencv-python==4.12.0.88
opencv-python-headless==4.12.0.88
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
path==17.1.1
path.py==12.5.0
pathos==0.3.2
pathspec==0.12.1
protobuf==6.33.0
pydicom==3.0.1
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
testpath==0.6.0
tqdm==4.67.1

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (276 lines)
            sample_submission.csv (14537 lines)
            sample_submission.csv.zip (50.3 kB)
            segmentations.zip (3.0 MB)
            test.csv (14537 lines)
            test.csv.zip (94.5 kB)
            test.zip (160 Bytes)
            test_images.zip (181.9 GB)
            train.csv (203 lines)
            train.csv.zip (1.2 kB)
            train.zip (162 Bytes)
            train_bounding_boxes.csv (691 lines)
            train_bounding_boxes.csv.zip (11.5 kB)
            train_images.zip (20.3 GB)
            rsna-2022-cervical-spine-fracture-detection/
                description.md (276 lines)
                sample_submission.csv (14537 lines)
                ... and 12 other files
                rsna-2022-cervical-spine-fracture-detection/
                segmentations/
                    1.2.826.0.1.3680043.12292.nii (89.4 MB)
                    1.2.826.0.1.3680043.24617.nii (307.2 MB)
                    ... and 7 other files
                test_images/
                    1.2.826.0.1.3680043.10001/
                        1.dcm (525.0 kB)
                        10.dcm (525.0 kB)
                        ... and 266 other files
                    1.2.826.0.1.3680043.10005/
                        1.dcm (525.1 kB)
                        10.dcm (525.1 kB)
                        ... and 257 other files
                    ... and 1816 other folders
                train_images/
                    1.2.826.0.1.3680043.10014/
                        1.dcm (240.9 kB)
                        10.dcm (250.8 kB)
                        ... and 256 other files
                    1.2.826.0.1.3680043.10058/
                        1.dcm (525.0 kB)
                        10.dcm (525.0 kB)
                        ... and 574 other files
                    ... and 201 other folders
            segmentations/
                1.2.826.0.1.3680043.12292.nii (89.4 MB)
                1.2.826.0.1.3680043.24617.nii (307.2 MB)
                ... and 7 other files
            test_images/
                1.2.826.0.1.3680043.10001/
                    1.dcm (525.0 kB)
                    10.dcm (525.0 kB)
                    ... and 266 other files
                1.2.826.0.1.3680043.10005/
                    1.dcm (525.1 kB)
                    10.dcm (525.1 kB)
                    ... and 257 other files
                ... and 1816 other folders
            train_images/
                1.2.826.0.1.3680043.10014/
                    1.dcm (240.9 kB)
                    10.dcm (250.8 kB)
                    ... and 256 other files
                1.2.826.0.1.3680043.10058/
                    1.dcm (525.0 kB)
                    10.dcm (525.0 kB)
                    ... and 574 other files
                ... and 201 other folders
        input/
            description.md (276 lines)
            sample_submission.csv (14537 lines)
            sample_submission.csv.zip (50.3 kB)
            segmentations.zip (3.0 MB)
            test.csv (14537 lines)
            test.csv.zip (94.5 kB)
            test.zip (160 Bytes)
            test_images.zip (181.9 GB)
            train.csv (203 lines)
            train.csv.zip (1.2 kB)
            train.zip (162 Bytes)
            train_bounding_boxes.csv (691 lines)
            train_bounding_boxes.csv.zip (11.5 kB)
            train_images.zip (20.3 GB)
            rsna-2022-cervical-spine-fracture-detection/
                description.md (276 lines)
                sample_submission.csv (14537 lines)
                ... and 12 other files
                rsna-2022-cervical-spine-fracture-detection/
                segmentations/
                    1.2.826.0.1.3680043.12292.nii (89.4 MB)
                    1.2.826.0.1.3680043.24617.nii (307.2 MB)
                    ... and 7 other files
                test_images/
                    1.2.826.0.1.3680043.10001/
                        1.dcm (525.0 kB)
                        10.dcm (525.0 kB)
                        ... and 266 other files
                    1.2.826.0.1.3680043.10005/
                        1.dcm (525.1 kB)
                        10.dcm (525.1 kB)
                        ... and 257 other files
                    ... and 1816 other folders
                train_images/
                    1.2.826.0.1.3680043.10014/
                        1.dcm (240.9 kB)
                        10.dcm (250.8 kB)
                        ... and 256 other files
                    1.2.826.0.1.3680043.10058/
                        1.dcm (525.0 kB)
                        10.dcm (525.0 kB)
                        ... and 574 other files
                    ... and 201 other folders
            segmentations/
                1.2.826.0.1.3680043.12292.nii (89.4 MB)
                1.2.826.0.1.3680043.24617.nii (307.2 MB)
                ... and 7 other files
            test_images/
                1.2.826.0.1.3680043.10001/
                    1.dcm (525.0 kB)
                    10.dcm (525.0 kB)
                    ... and 266 other files
                1.2.826.0.1.3680043.10005/
                    1.dcm (525.1 kB)
                    10.dcm (525.1 kB)
                    ... and 257 other files
                ... and 1816 other folders
            train_images/
                1.2.826.0.1.3680043.10014/
                    1.dcm (240.9 kB)
                    10.dcm (250.8 kB)
                    ... and 256 other files
                1.2.826.0.1.3680043.10058/
                    1.dcm (525.0 kB)
                    10.dcm (525.0 kB)
                    ... and 574 other files
                ... and 201 other folders
        working/
            rsna-2022-cervical-spine-fracture-detection/
                description.md (276 lines)
                sample_submission.csv (14537 lines)
                ... and 12 other files
                rsna-2022-cervical-spine-fracture-detection/
                segmentations/
                    1.2.826.0.1.3680043.12292.nii (89.4 MB)
                    1.2.826.0.1.3680043.24617.nii (307.2 MB)
                    ... and 7 other files
                test_images/
                    1.2.826.0.1.3680043.10001/
                        1.dcm (525.0 kB)
                        10.dcm (525.0 kB)
                        ... and 266 other files
                    1.2.826.0.1.3680043.10005/
                        1.dcm (525.1 kB)
                        10.dcm (525.1 kB)
                        ... and 257 other files
                    ... and 1816 other folders
                train_images/
                    1.2.826.0.1.3680043.10014/
                        1.dcm (240.9 kB)
                        10.dcm (250.8 kB)
                        ... and 256 other files
                    1.2.826.0.1.3680043.10058/
                        1.dcm (525.0 kB)
                        10.dcm (525.0 kB)
                        ... and 574 other files
                    ... and 201 other folders
```

-> data/rsna-2022-cervical-spine-fracture-detection/sample_submission.csv has 14536 rows and 2 columns.
The columns are: row_id, fractured

-> data/rsna-2022-cervical-spine-fracture-detection/test.csv has 14536 rows and 3 columns.
The columns are: StudyInstanceUID, prediction_type, row_id

-> data/rsna-2022-cervical-spine-fracture-detection/train.csv has 202 rows and 9 columns.
The columns are: StudyInstanceUID, patient_overall, C1, C2, C3, C4, C5, C6, C7

-> data/rsna-2022-cervical-spine-fracture-detection/train_bounding_boxes.csv has 690 rows and 6 columns.
The columns are: StudyInstanceUID, x, y, width, height, slice_number

-> data/sample_submission.csv has 14536 rows and 2 columns.
The columns are: row_id, fractured

-> data/test.csv has 14536 rows and 3 columns.
The columns are: StudyInstanceUID, prediction_type, row_id

-> data/train.csv has 202 rows and 9 columns.
The columns are: StudyInstanceUID, patient_overall, C1, C2, C3, C4, C5, C6, C7

-> data/train_bounding_boxes.csv has 690 rows and 6 columns.
The columns are: StudyInstanceUID, x, y, width, height, slice_number

-> (stopped after 10 files for performance)

# 5. Target score

8.0739

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 
import cv2 as cv
from path import Path
import os 
import glob
import tensorflow_hub as hub
import os 
import pydicom as dicom
from pydicom.pixel_data_handlers.util import apply_voi_lut
import tensorflow as tf
from tqdm import tqdm
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.utils import to_categorical
from pydicom import dcmread
import nibabel as nib


## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
train_df = pd.read_csv("../input/rsna-2022-cervical-spine-fracture-detection/train.csv")


## === cell 2
def load_dicom(path):
    img=dicom.dcmread(path)
    data=img.pixel_array
    data=data-np.min(data)
    if np.max(data) != 0:
        data=data/np.max(data)
    data=(data*255).astype(np.uint8)
    return data


## === cell 3
def listdirs(folder):
    return [d for d in os.listdir(folder) if os.path.isdir(os.path.join(folder, d))]


## === cell 4
train_dir='../input/rsna-2022-cervical-spine-fracture-detection/train_images'
patients = sorted(os.listdir(train_dir))
patients[:5]


## === cell 7
from pydicom.data import get_testdata_files
trainset=[]
trainlabel=[]
trainidt=[]
limit = 50 
for i in tqdm(range(len(train_df))): #there are 2019 rows, need much times, so just process 10 rows to test
    idt=train_df.loc[i,'StudyInstanceUID']
    
    path=os.path.join(train_dir,idt)   
    
    for im in os.listdir(path):
        
        
        dc = dicom.read_file(os.path.join(path,im))
        if dc.file_meta.TransferSyntaxUID.name =='JPEG Lossless, Non-Hierarchical, First-Order Prediction (Process 14 [Selection Value 1])':
            continue
        
        img=load_dicom(os.path.join(path,im)) 

        img=cv.resize(img,(64,64)) 
        image=img_to_array(img)
        image=image/255.0

        trainset+=[image]
        cur_label=[]
        cur_label.append(train_df.loc[i,'C1'])
        cur_label.append(train_df.loc[i,'C2'])
        cur_label.append(train_df.loc[i,'C3'])
        cur_label.append(train_df.loc[i,'C4'])
        cur_label.append(train_df.loc[i,'C5'])
        cur_label.append(train_df.loc[i,'C6'])
        cur_label.append(train_df.loc[i,'C7'])
        trainlabel+=[cur_label]
        trainidt+=[idt]
    i+=1
    if i==limit:
        break


## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/2727949716.py in <cell line: 0>()
     13 
     14 
---> 15         dc = dicom.read_file(os.path.join(path,im))
     16         if dc.file_meta.TransferSyntaxUID.name =='JPEG Lossless, Non-Hierarchical, First-Order Prediction (Process 14 [Selection Value 1])':
     17             continue

AttributeError: module 'pydicom' has no attribute 'read_file'

## === cell 8
y=np.array(trainlabel)
Y_train=y
X_train=np.array(trainset)


## === cell 9
test_df = ['1.2.826.0.1.3680043.22327', '1.2.826.0.1.3680043.25399', '1.2.826.0.1.3680043.5876']


## === cell 10
test_dir='../input/rsna-2022-cervical-spine-fracture-detection/test_images'
testset=[]
testidt=[]
for i in tqdm(range(len(test_df))):
    idt=test_df[i]
    path=os.path.join(test_dir,idt)   
    
    for im in os.listdir(path):
        dc = dicom.read_file(os.path.join(path,im))
        
        if dc.file_meta.TransferSyntaxUID.name =='JPEG Lossless, Non-Hierarchical, First-Order Prediction (Process 14 [Selection Value 1])':
            continue
        img=load_dicom(os.path.join(path,im)) 

        img=cv.resize(img,(64,64)) 
        image=img_to_array(img)
        image=image/255.0
        testset+=[image]
        testidt+=[idt]


## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/1345518116.py in <cell line: 0>()
      6     path=os.path.join(test_dir,idt)
      7 
----> 8     for im in os.listdir(path):
      9         dc = dicom.read_file(os.path.join(path,im))
     10 

FileNotFoundError: [Errno 2] No such file or directory: '../input/rsna-2022-cervical-spine-fracture-detection/test_images/1.2.826.0.1.3680043.22327'

## === cell 11
X_test = np.array(testset)


## === cell 12
from tensorflow.keras.layers import BatchNormalization, Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Activation


## === cell 13
def create_model():
    model = tf.keras.models.Sequential()
    model.add(BatchNormalization(input_shape = X_train.shape[1:]))
    model.add(Conv2D(32, (5, 5), padding='same', activation='elu'))
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
    model.add(Dropout(0.05))

    model = tf.keras.models.Sequential()
    model.add(BatchNormalization(input_shape = X_train.shape[1:]))
    model.add(Conv2D(64, (5, 5), padding='same', activation='elu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.05))

    model = tf.keras.models.Sequential()
    model.add(BatchNormalization(input_shape = X_train.shape[1:]))
    model.add(Conv2D(128, (5, 5), padding='same', activation='elu'))
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
    model.add(Dropout(0.05))
    
    model = tf.keras.models.Sequential()
    model.add(BatchNormalization(input_shape = X_train.shape[1:]))
    model.add(Conv2D(256, (5, 5), padding='same', activation='elu'))
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
    model.add(Dropout(0.25)) 
    
    model = tf.keras.models.Sequential()
    model.add(BatchNormalization(input_shape = X_train.shape[1:]))
    model.add(Conv2D(128, (5, 5), padding='same', activation='elu'))
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
    model.add(Dropout(0.05))   

    model = tf.keras.models.Sequential()
    model.add(BatchNormalization(input_shape = X_train.shape[1:]))
    model.add(Conv2D(64, (5, 5), padding='same', activation='elu'))
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
    model.add(Dropout(0.05))

    model.add(Flatten())
    model.add(Dense(32))
    model.add(Activation('elu'))
    model.add(Dropout(0.25))
    model.add(Dense(7))
    model.add(Activation('softmax'))

    return model


## === cell 17
model = create_model()
model.compile(
  optimizer=tf.keras.optimizers.Nadam(learning_rate=0.0005),
  loss='binary_crossentropy',
  metrics=['accuracy'])

callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor="loss", min_delta=0.001, 
                                     patience=10, restore_best_weights=True)
]


model.fit(
    X_train, Y_train,
    epochs=100,
    batch_size=64,
    callbacks=callbacks
)

model.save_weights('./fashion_mnist.h5', overwrite=True)


## --- ERROR in cell 17, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/3690444533.py in <cell line: 0>()
----> 1 model = create_model()
      2 model.compile(
      3   optimizer=tf.keras.optimizers.Nadam(learning_rate=0.0005),
      4   loss='binary_crossentropy',
      5   metrics=['accuracy'])

/tmp/ipykernel_11/895740493.py in create_model()
      1 def create_model():
      2     model = tf.keras.models.Sequential()
----> 3     model.add(BatchNormalization(input_shape = X_train.shape[1:]))
      4     model.add(Conv2D(32, (5, 5), padding='same', activation='elu'))
      5     model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

/usr/local/lib/python3.11/dist-packages/keras/src/models/sequential.py in add(self, layer, rebuild)
    120         self._layers.append(layer)
    121         if rebuild:
--> 122             self._maybe_rebuild()
    123         else:
    124             self.built = False

/usr/local/lib/python3.11/dist-packages/keras/src/models/sequential.py in _maybe_rebuild(self)
    139         if isinstance(self._layers[0], InputLayer) and len(self._layers) > 1:
    140             input_shape = self._layers[0].batch_shape
--> 141             self.build(input_shape)
    142         elif hasattr(self._layers[0], "input_shape") and len(self._layers) > 1:
    143             # We can build the Sequential model if the first layer has the

/usr/local/lib/python3.11/dist-packages/keras/src/layers/layer.py in build_wrapper(*args, **kwargs)
    226             with obj._open_name_scope():
    227                 obj._path = current_path()
--> 228                 original_build_method(*args, **kwargs)
    229             # Record build config.
    230             signature = inspect.signature(original_build_method)

/usr/local/lib/python3.11/dist-packages/keras/src/models/sequential.py in build(self, input_shape)
    185         for layer in self._layers[1:]:
    186             try:
--> 187                 x = layer(x)
    188             except NotImplementedError:
    189                 # Can happen if shape inference is not implemented.

/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py in error_handler(*args, **kwargs)
    120             # To get the full stack trace, call:
    121             # `keras.config.disable_traceback_filtering()`
--> 122             raise e.with_traceback(filtered_tb) from None
    123         finally:
    124             del filtered_tb

/usr/local/lib/python3.11/dist-packages/keras/src/backend/common/variables.py in _validate_shape(self, shape)
    207         shape = standardize_shape(shape)
    208         if None in shape:
--> 209             raise ValueError(
    210                 "Shapes used to initialize variables must be "
    211                 "fully-defined (no `None` dimensions). Received: "

ValueError: Shapes used to initialize variables must be fully-defined (no `None` dimensions). Received: shape=(None,) for variable path='sequential/batch_normalization/gamma'

## === cell 18
y_pred=model.predict(X_test)


## --- ERROR in cell 18, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/649594757.py in <cell line: 0>()
----> 1 y_pred=model.predict(X_test)

NameError: name 'model' is not defined

## === cell 19
result = pd.DataFrame(columns = train_df.columns, index = range(len(testidt)))


## === cell 20
for i in tqdm(range(len(testidt))):
    result.loc[i, 'StudyInstanceUID'] = testidt[i]
    rows = np.int64(y_pred[i]>0.07)
    result.loc[i, 'patient_overall'] = int(bool(np.sum(rows)))
    result.loc[i, 'C1'] = rows[0]
    result.loc[i, 'C2'] = rows[1]
    result.loc[i, 'C3'] = rows[2]
    result.loc[i, 'C4'] = rows[3]
    result.loc[i, 'C5'] = rows[4]
    result.loc[i, 'C6'] = rows[5]
    result.loc[i, 'C7'] = rows[6]


## === cell 21
means = result[['patient_overall', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7']].mean().to_dict()


## === cell 22
test_df2 = pd.read_csv('../input/rsna-2022-cervical-spine-fracture-detection/test.csv')
test_df2['fractured'] = test_df2['prediction_type'].map(means)

test_df2[['row_id','fractured']].to_csv('submission.csv', index=False, float_format='%.1g')


## --- ERROR in outputing the csv:
Invalid submission: Submission `fractured` values must be between 0 and 1.
