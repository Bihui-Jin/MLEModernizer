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
Predict whether a lesion is malignant (0 denotes **benign**, and 1 indicates **malignant**).

## Metric
Area under the ROC curve.

## Submission Format
For each `image_name` in the test set, you must predict the probability (`target`) that the sample is **malignant**. The file should contain a header and have the following format:

```
image_name,target
ISIC_0052060,0.7
ISIC_0052349,0.9
ISIC_0058510,0.8
ISIC_0073313,0.5
ISIC_0073502,0.5
etc.
```

## Dataset 
The images are provided in DICOM format.

Images are also provided in JPEG and TFRecord format (in the `jpeg` and `tfrecords` directories, respectively). Images in TFRecord format have been resized to a uniform 1024x1024.

Metadata is also provided outside of the DICOM format, in CSV files. See the `Columns` section for a description.

### Files
- **train.csv** - the training set
- **test.csv** - the test set
- **sample_submission.csv** - a sample submission file in the correct format

### Columns
- `image_name` - unique identifier, points to filename of related DICOM image
- `patient_id` - unique patient identifier
- `sex` - the sex of the patient (when unknown, will be blank)
- `age_approx` - approximate patient age at time of imaging
- `anatom_site_general_challenge` - location of imaged site
- `diagnosis` - detailed diagnosis information (train only)
- `benign_malignant` - indicator of malignancy of imaged lesion
- `target` - binarized version of the target variable

# 2. Python version

3.8

# 3. Installed packages

No external packages required in the script and installed.

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (176 lines)
            jpeg.zip (24.7 GB)
            sample_submission.csv (4143 lines)
            sample_submission.csv.zip (16.4 kB)
            test.csv (4143 lines)
            test.csv.zip (42.5 kB)
            test.zip (6.4 GB)
            tfrecords.zip (9.3 GB)
            train.csv (28985 lines)
            train.csv.zip (299.7 kB)
            train.zip (46.0 GB)
            jpeg/
                test/
                    ISIC_1440063.jpg (1.1 MB)
                    ISIC_0815802.jpg (853.1 kB)
                    ... and 4140 other files
                train/
                    ISIC_1845271.jpg (1.0 MB)
                    ISIC_1970027.jpg (138.4 kB)
                    ... and 28982 other files
            siim-isic-melanoma-classification/
                description.md (176 lines)
                jpeg.zip (24.7 GB)
                ... and 9 other files
                jpeg/
                    test/
                        ISIC_1440063.jpg (1.1 MB)
                        ISIC_0815802.jpg (853.1 kB)
                        ... and 4140 other files
                    train/
                        ISIC_1845271.jpg (1.0 MB)
                        ISIC_1970027.jpg (138.4 kB)
                        ... and 28982 other files
                siim-isic-melanoma-classification/
                test/
                    ISIC_0052212.dcm (1.5 MB)
                    ISIC_0076545.dcm (4.0 MB)
                    ... and 4140 other files
                    test/
                tfrecords/
                    test00-2071.tfrec (579.6 MB)
                    test01-2071.tfrec (583.5 MB)
                    ... and 14 other files
                train/
                    ISIC_0015719.dcm (2.4 MB)
                    ISIC_0068279.dcm (1.3 MB)
                    ... and 28982 other files
                    train/
            test/
                ISIC_0052212.dcm (1.5 MB)
                ISIC_0076545.dcm (4.0 MB)
                ... and 4140 other files
                test/
            tfrecords/
                test00-2071.tfrec (579.6 MB)
                test01-2071.tfrec (583.5 MB)
                ... and 14 other files
            train/
                ISIC_0015719.dcm (2.4 MB)
                ISIC_0068279.dcm (1.3 MB)
                ... and 28982 other files
                train/
        input/
            description.md (176 lines)
            jpeg.zip (24.7 GB)
            sample_submission.csv (4143 lines)
            sample_submission.csv.zip (16.4 kB)
            test.csv (4143 lines)
            test.csv.zip (42.5 kB)
            test.zip (6.4 GB)
            tfrecords.zip (9.3 GB)
            train.csv (28985 lines)
            train.csv.zip (299.7 kB)
            train.zip (46.0 GB)
            jpeg/
                test/
                    ISIC_1440063.jpg (1.1 MB)
                    ISIC_0815802.jpg (853.1 kB)
                    ... and 4140 other files
                train/
                    ISIC_1845271.jpg (1.0 MB)
                    ISIC_1970027.jpg (138.4 kB)
                    ... and 28982 other files
            siim-isic-melanoma-classification/
                description.md (176 lines)
                jpeg.zip (24.7 GB)
                ... and 9 other files
                jpeg/
                    test/
                        ISIC_1440063.jpg (1.1 MB)
                        ISIC_0815802.jpg (853.1 kB)
                        ... and 4140 other files
                    train/
                        ISIC_1845271.jpg (1.0 MB)
                        ISIC_1970027.jpg (138.4 kB)
                        ... and 28982 other files
                siim-isic-melanoma-classification/
                test/
                    ISIC_0052212.dcm (1.5 MB)
                    ISIC_0076545.dcm (4.0 MB)
                    ... and 4140 other files
                    test/
                tfrecords/
                    test00-2071.tfrec (579.6 MB)
                    test01-2071.tfrec (583.5 MB)
                    ... and 14 other files
                train/
                    ISIC_0015719.dcm (2.4 MB)
                    ISIC_0068279.dcm (1.3 MB)
                    ... and 28982 other files
                    train/
            test/
                ISIC_0052212.dcm (1.5 MB)
                ISIC_0076545.dcm (4.0 MB)
                ... and 4140 other files
                test/
                    ISIC_0052212.dcm (1.5 MB)
                    ISIC_0076545.dcm (4.0 MB)
                    ... and 4140 other files
                    test/
            tfrecords/
                test00-2071.tfrec (579.6 MB)
                test01-2071.tfrec (583.5 MB)
                ... and 14 other files
            train/
                ISIC_0015719.dcm (2.4 MB)
                ISIC_0068279.dcm (1.3 MB)
                ... and 28982 other files
                train/
                    ISIC_0015719.dcm (2.4 MB)
                    ISIC_0068279.dcm (1.3 MB)
                    ... and 28982 other files
                    train/
        working/
            siim-isic-melanoma-classification/
                description.md (176 lines)
                jpeg.zip (24.7 GB)
                ... and 9 other files
                jpeg/
                    test/
                        ISIC_1440063.jpg (1.1 MB)
                        ISIC_0815802.jpg (853.1 kB)
                        ... and 4140 other files
                    train/
                        ISIC_1845271.jpg (1.0 MB)
                        ISIC_1970027.jpg (138.4 kB)
                        ... and 28982 other files
                siim-isic-melanoma-classification/
                test/
                    ISIC_0052212.dcm (1.5 MB)
                    ISIC_0076545.dcm (4.0 MB)
                    ... and 4140 other files
                    test/
                tfrecords/
                    test00-2071.tfrec (579.6 MB)
                    test01-2071.tfrec (583.5 MB)
                    ... and 14 other files
                train/
                    ISIC_0015719.dcm (2.4 MB)
                    ISIC_0068279.dcm (1.3 MB)
                    ... and 28982 other files
                    train/
```

-> data/sample_submission.csv has 4142 rows and 2 columns.
The columns are: image_name, target

-> data/siim-isic-melanoma-classification/sample_submission.csv has 4142 rows and 2 columns.
The columns are: image_name, target

-> data/siim-isic-melanoma-classification/test.csv has 4142 rows and 5 columns.
The columns are: image_name, patient_id, sex, age_approx, anatom_site_general_challenge

-> data/siim-isic-melanoma-classification/train.csv has 28984 rows and 8 columns.
The columns are: image_name, patient_id, sex, age_approx, anatom_site_general_challenge, diagnosis, benign_malignant, target

-> data/test.csv has 4142 rows and 5 columns.
The columns are: image_name, patient_id, sex, age_approx, anatom_site_general_challenge

-> data/train.csv has 28984 rows and 8 columns.
The columns are: image_name, patient_id, sex, age_approx, anatom_site_general_challenge, diagnosis, benign_malignant, target

-> input/sample_submission.csv has 4142 rows and 2 columns.
The columns are: image_name, target

-> (stopped after 10 files for performance)

# 5. Target score

0.8663659563933698

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 4
!pip install -q efficientnet

import numpy as np
from numpy.random import shuffle
import pandas as pd 
import random
import matplotlib.pyplot as plt
import cv2
pd.set_option('expand_frame_repr', False)

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping
from tensorflow.keras.layers import Input, Dense, Flatten, Dropout, BatchNormalization, Concatenate, Activation, LeakyReLU, ReLU, GlobalAveragePooling2D
from tensorflow.keras.optimizers import RMSprop, SGD, Adam
import efficientnet.tfkeras as efn

import os
from tqdm import tqdm
import datetime
import copy

## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 7
TB_Log_Dir = r"/kaggle/working/{}/"       # TensorBoard logs directory
Labels_Dir = r"../input/siim-isic-melanoma-classification/{}.csv"
Images_Dir = r"../input/isicdenoised/ISIC/{}/{}"
Output_Dir = r"/kaggle/working/{}.h5"

Epochs = 4
Batch_Size = 32
Early_Stop = EarlyStopping(monitor="val_loss", mode="auto", verbose=1, patience=1, restore_best_weights=True)

Image_Size = (299, 299)
Class_Weight = {0: 1, 1: 3.32}

Train_Over_Sampel_Count = 12_000         # for data augmentation
Valid_Over_Sampel_Count = 3_254

## === cell 10
def data_augmentation(labels, target_length):
    counter = 0
    while counter < target_length:
        for i in tqdm(range(len(labels))):
            if counter >= target_length:                # if we produced enough labels
                continue
                
            row = labels[i]
            target = row[-2]
            
            if target == 0:
                continue
            
            mode = random.randint(0, 6)
            row_copy = copy.deepcopy(row)
            row_copy[-1] = mode                         # row_copy["argu_mode"] = mode      
            labels = np.concatenate([labels, [row_copy]])
            
            counter += 1
            
    shuffle(labels)
    return labels

## === cell 13

Labels = pd.read_csv(Labels_Dir.format("train"))
Labels = Labels.sample(frac=1).reset_index(drop=True)  

Labels["argu_mode"] = [None for i in range(len(Labels))]

Labels["sex"].replace(["male"], 1, inplace=True)
Labels["sex"].replace(["female"], 0, inplace=True)

Positive_Indexs = Labels.index[Labels['target'] == 1].tolist()     # Positive cases indexes
Negative_Indexs = Labels.index[Labels['target'] == 1].tolist()  

Labels = Labels.to_numpy()
Positive_Cases = Labels[tuple([Positive_Indexs])]
Negative_Cases = np.delete(Labels, (Positive_Indexs), axis=0)   

shuffle(Positive_Cases)
shuffle(Negative_Cases)

Place = int(len(Negative_Cases) * 0.1)

Train_Labels = np.concatenate([Positive_Cases[4:], Negative_Cases[Place:]])
Validation_Labels = np.concatenate([Positive_Cases[0: 4], Negative_Cases[:Place]])

Train_Labels = data_augmentation(Train_Labels, Train_Over_Sampel_Count)

shuffle(Train_Labels)
shuffle(Validation_Labels)

Train_Positive_Count = np.count_nonzero([row[-2] for row in Train_Labels])            # Positive cases count in Train labels
Valid_Positive_Count = np.count_nonzero([row[-2] for row in Validation_Labels])       # Positive cases count in Test labels

print("Train Labels:\n", Train_Labels, "\n\n",
      "Validation Labels:\n", Validation_Labels,
      "\n\n", "\tTrain:", Train_Positive_Count,
      "\tValidation", Valid_Positive_Count)

print(f"\nLen Validation_Data: {len(Validation_Labels)}\tLen Train Data: {len(Train_Labels)}")

## === cell 16
def data_generator(labels, imgs_dir, image_size, batch_size=32):
    images, targets = [], []
    
    while True:
        for index in range(len(labels)):
            Id, target, argu_mode = labels[index][1], labels[index][-2], labels[index][-1]
            
            image_path = f"{imgs_dir}/{Id}.jpg"
            image = cv2.imread(image_path, 1)
            image = cv2.resize(image, image_size)
            
            if argu_mode == 0:                  # augmentation mode 
                pass
                
            if argu_mode == 1:
                image = cv2.flip(image, 1)      # horizintal
            
            elif argu_mode == 2:
                image = cv2.flip(image, 0)      # vertical
                matrix = np.float32([[1, 0, 20], [0, 1, 20]])
                image = cv2.warpAffine(image, matrix, image_size)          # shifting image
            
            elif argu_mode == 3:
                image = cv2.flip(image, -1)     # both   
                matrix = np.float32([[1, 0, 33], [0, 1, 33]])
                image = cv2.warpAffine(image, matrix, image_size)          # shifting image
            
            elif argu_mode == 4:                # rotating image clockwise
                image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
                matrix = np.float32([[1, 0, 23], [0, 1, 25]])
                image = cv2.warpAffine(image, matrix, image_size)          # shifting image
            
            elif argu_mode == 5:                # rotating image counter clockwise
                image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
            
            elif argu_mode == 6:
                image = image[25: image_size[1] - 25, 25: image_size[0] - 25]    # Cropping image
                image = cv2.resize(image, (image_size[0], image_size[1]))
                matrix = np.float32([[1, 0, 32], [0, 1, 32]])
                image = cv2.warpAffine(image, matrix, image_size)                # shifting image
            
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = image / 255                # Normalizing image 
            image = np.float32(image)

            images.append(image)
            targets.append(target)
            
            if len(images) >= batch_size:
                yield np.array(images, dtype='float32'), np.array(targets, dtype='float32')
                images, targets = [], []

## === cell 17
Train_Gen = data_generator(
    Train_Labels,
    Images_Dir.format("Train", "hair-removed"),
    Image_Size,
    Batch_Size)

Validation_Gen = data_generator(
    Validation_Labels,
    Images_Dir.format("Train", "hair-removed"),
    Image_Size,
    Batch_Size)

## === cell 19
EfficientNetB1 = efn.EfficientNetB1(include_top=False, weights="imagenet", input_shape=(Image_Size[0], Image_Size[1], 3))
EfficientNetB2 = efn.EfficientNetB2(include_top=False, weights="imagenet", input_shape=(Image_Size[0], Image_Size[1], 3))
EfficientNetB5 = efn.EfficientNetB5(include_top=False, weights="imagenet", input_shape=(Image_Size[0], Image_Size[1], 3))
EfficientNetB0 = efn.EfficientNetB0(include_top=False, weights="imagenet", input_shape=(Image_Size[0], Image_Size[1], 3))

## --- ERROR in cell 19, traceback:
---------------------------------------------------------------------------
HTTPError                                 Traceback (most recent call last)
/usr/local/lib/python3.11/dist-packages/keras/src/utils/file_utils.py in get_file(fname, origin, untar, md5_hash, file_hash, cache_subdir, hash_algorithm, extract, archive_format, cache_dir, force_download)
    310             try:
--> 311                 urlretrieve(origin, download_target, DLProgbar())
    312             except urllib.error.HTTPError as e:

/usr/lib/python3.11/urllib/request.py in urlretrieve(url, filename, reporthook, data)
    240 
--> 241     with contextlib.closing(urlopen(url, data)) as fp:
    242         headers = fp.info()

/usr/lib/python3.11/urllib/request.py in urlopen(url, data, timeout, cafile, capath, cadefault, context)
    215         opener = _opener
--> 216     return opener.open(url, data, timeout)
    217 

/usr/lib/python3.11/urllib/request.py in open(self, fullurl, data, timeout)
    524             meth = getattr(processor, meth_name)
--> 525             response = meth(req, response)
    526 

/usr/lib/python3.11/urllib/request.py in http_response(self, request, response)
    633         if not (200 <= code < 300):
--> 634             response = self.parent.error(
    635                 'http', request, response, code, msg, hdrs)

/usr/lib/python3.11/urllib/request.py in error(self, proto, *args)
    562             args = (dict, 'default', 'http_error_default') + orig_args
--> 563             return self._call_chain(*args)
    564 

/usr/lib/python3.11/urllib/request.py in _call_chain(self, chain, kind, meth_name, *args)
    495             func = getattr(handler, meth_name)
--> 496             result = func(*args)
    497             if result is not None:

/usr/lib/python3.11/urllib/request.py in http_error_default(self, req, fp, code, msg, hdrs)
    642     def http_error_default(self, req, fp, code, msg, hdrs):
--> 643         raise HTTPError(req.full_url, code, msg, hdrs, fp)
    644 

HTTPError: HTTP Error 404: Not Found

During handling of the above exception, another exception occurred:

Exception                                 Traceback (most recent call last)
/tmp/ipykernel_11/3425362710.py in <cell line: 0>()
----> 1 EfficientNetB1 = efn.EfficientNetB1(include_top=False, weights="imagenet", input_shape=(Image_Size[0], Image_Size[1], 3))
      2 EfficientNetB2 = efn.EfficientNetB2(include_top=False, weights="imagenet", input_shape=(Image_Size[0], Image_Size[1], 3))
      3 EfficientNetB5 = efn.EfficientNetB5(include_top=False, weights="imagenet", input_shape=(Image_Size[0], Image_Size[1], 3))
      4 EfficientNetB0 = efn.EfficientNetB0(include_top=False, weights="imagenet", input_shape=(Image_Size[0], Image_Size[1], 3))

/usr/local/lib/python3.11/dist-packages/efficientnet/__init__.py in wrapper(*args, **kwargs)
     55         kwargs['models'] = tfkeras.models
     56         kwargs['utils'] = tfkeras.utils
---> 57         return func(*args, **kwargs)
     58 
     59     return wrapper

/usr/local/lib/python3.11/dist-packages/efficientnet/model.py in EfficientNetB1(include_top, weights, input_tensor, input_shape, pooling, classes, **kwargs)
    488         **kwargs
    489 ):
--> 490     return EfficientNet(
    491         1.0, 1.1, 240, 0.2,
    492         model_name='efficientnet-b1',

/usr/local/lib/python3.11/dist-packages/efficientnet/model.py in EfficientNet(width_coefficient, depth_coefficient, default_resolution, dropout_rate, drop_connect_rate, depth_divisor, blocks_args, model_name, include_top, weights, input_tensor, input_shape, pooling, classes, **kwargs)
    430             file_name = model_name + '_weights_tf_dim_ordering_tf_kernels_autoaugment_notop.h5'
    431             file_hash = IMAGENET_WEIGHTS_HASHES[model_name][1]
--> 432         weights_path = keras_utils.get_file(
    433             file_name,
    434             IMAGENET_WEIGHTS_PATH + file_name,

/usr/local/lib/python3.11/dist-packages/keras/src/utils/file_utils.py in get_file(fname, origin, untar, md5_hash, file_hash, cache_subdir, hash_algorithm, extract, archive_format, cache_dir, force_download)
    311                 urlretrieve(origin, download_target, DLProgbar())
    312             except urllib.error.HTTPError as e:
--> 313                 raise Exception(error_msg.format(origin, e.code, e.msg))
    314             except urllib.error.URLError as e:
    315                 raise Exception(error_msg.format(origin, e.errno, e.reason))

Exception: URL fetch failure on https://github.com/Callidior/keras-applications/releases/download/efficientnet/efficientnet-b1_weights_tf_dim_ordering_tf_kernels_autoaugment_notop.h5: 404 -- Not Found

## === cell 20
def build_lrfn(lr_start=0.00001, lr_max=0.0001, 
               lr_min=0.000001, lr_rampup_epochs=20, 
               lr_sustain_epochs=0, lr_exp_decay=.8):

    def lrfn(epoch):
        if epoch < lr_rampup_epochs:
            lr = (lr_max - lr_start) / lr_rampup_epochs * epoch + lr_start
        elif epoch < lr_rampup_epochs + lr_sustain_epochs:
            lr = lr_max
        else:
            lr = (lr_max - lr_min) * lr_exp_decay**(epoch - lr_rampup_epochs - lr_sustain_epochs) + lr_min
        return lr
    
    return lrfn

## === cell 21
lrfn = build_lrfn()
lr_schedule = tf.keras.callbacks.LearningRateScheduler(lrfn, verbose=1)

## === cell 22
def make_model(base_model):
    model = Sequential()
    model.add(base_model)
    model.add(GlobalAveragePooling2D())
    model.add(Dense(1, activation="sigmoid"))
    
    return model

## === cell 23
Model = make_model(EfficientNetB2)
Model.Name =f"efficentB2"
TB_Callback = TensorBoard(log_dir=TB_Log_Dir.format(Model.Name), histogram_freq=1)

SGD_Optimizer = SGD(lr=0.1)
Adam_Optimizer = Adam(lr=0.1)
    
Model.compile(
    loss="binary_crossentropy",
    optimizer="adam",
    metrics=["accuracy"]
)    

Model.summary()
    
Model.fit(
    Train_Gen,
    steps_per_epoch=len(Train_Labels)/Batch_Size,
    verbose=1,
    epochs=Epochs,
    validation_data=Validation_Gen,
    validation_steps=len(Validation_Labels)/Batch_Size,
    class_weight=Class_Weight,
    callbacks=[lr_schedule, Early_Stop, TB_Callback]
)

## --- ERROR in cell 23, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3178915173.py in <cell line: 0>()
----> 1 Model = make_model(EfficientNetB2)
      2 Model.Name =f"efficentB2"
      3 TB_Callback = TensorBoard(log_dir=TB_Log_Dir.format(Model.Name), histogram_freq=1)
      4 
      5 SGD_Optimizer = SGD(lr=0.1)

NameError: name 'EfficientNetB2' is not defined

## === cell 24
Model.save(Output_Dir.format(f"{Model.Name}--{str(datetime.now())}"))

## --- ERROR in cell 24, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1644355853.py in <cell line: 0>()
----> 1 Model.save(Output_Dir.format(f"{Model.Name}--{str(datetime.now())}"))

NameError: name 'Model' is not defined

## === cell 26
def predict(images, model, image_size=(299,299), is_rgb=True):
    predictions = []
    for image in images:
        image = cv2.resize(image, (299, 299))
                      
        if not is_rgb:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                      
        image = np.float32(image)
        image = image / 255             # Normalizing imgae
        image = np.reshape(image, (1, 299, 299, 3))
        predictions.append(model.predict(image))
        
    return predictions

## === cell 28
Sample_Submission = pd.read_csv(r"../input/siim-isic-melanoma-classification/sample_submission.csv")
Test_Labels = pd.read_csv(r"../input/siim-isic-melanoma-classification/test.csv")

print(Sample_Submission.tail())

My_Submission = {"image_name": [], "target": []}

for index in tqdm(range(len(Test_Labels))):
    image_name = Test_Labels["image_name"][index]
    
    image = cv2.imread(f"../input/isicdenoised/ISIC/Test/hair-removed/{image_name}.jpg")
    My_Submission["image_name"].append(image_name)
    My_Submission["target"].append(predict([image], Model, is_rgb=False)[0][0][0])

My_Submission = pd.DataFrame(My_Submission)
My_Submission.to_csv(r"submission.csv", index=False)
print("Sample submission:\n", Sample_Submission.head(), "\n\n", "My submission:\n", My_Submission.head(15), "\n", My_Submission.tail(15))

## --- ERROR in cell 28, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/519584918.py in <cell line: 0>()
     11     image = cv2.imread(f"../input/isicdenoised/ISIC/Test/hair-removed/{image_name}.jpg")
     12     My_Submission["image_name"].append(image_name)
---> 13     My_Submission["target"].append(predict([image], Model, is_rgb=False)[0][0][0])
     14 # =========
     15 

NameError: name 'Model' is not defined
