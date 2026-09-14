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
Develop a model to classify paddy leaf images into one of the nine disease categories or normal leaf.

## Metric
Categorization accuracy.

## Submission Format
```
image_id,label
200001.jpg,normal
200002.jpg,blast
etc.
```

## Dataset
**train.csv** - The training set

- `image_id` - Unique image identifier corresponds to image file names (.jpg) found in the train_images directory.
- `label` - Type of paddy disease, also the target class. There are ten categories, including the normal leaf.
- `variety` - The name of the paddy variety.
- `age` - Age of the paddy in days.

**sample_submission.csv** - Sample submission file.

**train_images** - Training images stored under different sub-directories corresponding to ten target classes. Filename corresponds to the `image_id` column of `train.csv`.

**test_images** - Test set images.

# 2. Python version

3.10

# 3. Installed packages

No external packages required in the script and installed.

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (72 lines)
            sample_submission.csv (2603 lines)
            sample_submission.csv.zip (7.4 kB)
            test.zip (160 Bytes)
            test_images.zip (205.2 MB)
            train.csv (7806 lines)
            train.csv.zip (40.1 kB)
            train.zip (162 Bytes)
            train_images.zip (614.5 MB)
            paddy-disease-classification/
                description.md (72 lines)
                sample_submission.csv (2603 lines)
                ... and 7 other files
                paddy-disease-classification/
                test_images/
                    102916.jpg (92.1 kB)
                    100596.jpg (83.9 kB)
                    ... and 2600 other files
                    test_images/
                train_images/
                    bacterial_leaf_blight/
                        109831.jpg (91.1 kB)
                        109785.jpg (81.8 kB)
                        ... and 356 other files
                    bacterial_leaf_streak/
                        100394.jpg (99.7 kB)
                        103308.jpg (104.3 kB)
                        ... and 295 other files
                    ... and 9 other folders
            test_images/
                102916.jpg (92.1 kB)
                100596.jpg (83.9 kB)
                ... and 2600 other files
                test_images/
            train_images/
                bacterial_leaf_blight/
                    109831.jpg (91.1 kB)
                    109785.jpg (81.8 kB)
                    ... and 356 other files
                bacterial_leaf_streak/
                    100394.jpg (99.7 kB)
                    103308.jpg (104.3 kB)
                    ... and 295 other files
                ... and 9 other folders
        input/
            description.md (72 lines)
            sample_submission.csv (2603 lines)
            sample_submission.csv.zip (7.4 kB)
            test.zip (160 Bytes)
            test_images.zip (205.2 MB)
            train.csv (7806 lines)
            train.csv.zip (40.1 kB)
            train.zip (162 Bytes)
            train_images.zip (614.5 MB)
            paddy-disease-classification/
                description.md (72 lines)
                sample_submission.csv (2603 lines)
                ... and 7 other files
                paddy-disease-classification/
                test_images/
                    102916.jpg (92.1 kB)
                    100596.jpg (83.9 kB)
                    ... and 2600 other files
                    test_images/
                train_images/
                    bacterial_leaf_blight/
                        109831.jpg (91.1 kB)
                        109785.jpg (81.8 kB)
                        ... and 356 other files
                    bacterial_leaf_streak/
                        100394.jpg (99.7 kB)
                        103308.jpg (104.3 kB)
                        ... and 295 other files
                    ... and 9 other folders
            test_images/
                102916.jpg (92.1 kB)
                100596.jpg (83.9 kB)
                ... and 2600 other files
                test_images/
                    102916.jpg (92.1 kB)
                    100596.jpg (83.9 kB)
                    ... and 2600 other files
                    test_images/
            train_images/
                bacterial_leaf_blight/
                    109831.jpg (91.1 kB)
                    109785.jpg (81.8 kB)
                    ... and 356 other files
                bacterial_leaf_streak/
                    100394.jpg (99.7 kB)
                    103308.jpg (104.3 kB)
                    ... and 295 other files
                ... and 9 other folders
        working/
            paddy-disease-classification/
                description.md (72 lines)
                sample_submission.csv (2603 lines)
                ... and 7 other files
                paddy-disease-classification/
                test_images/
                    102916.jpg (92.1 kB)
                    100596.jpg (83.9 kB)
                    ... and 2600 other files
                    test_images/
                train_images/
                    bacterial_leaf_blight/
                        109831.jpg (91.1 kB)
                        109785.jpg (81.8 kB)
                        ... and 356 other files
                    bacterial_leaf_streak/
                        100394.jpg (99.7 kB)
                        103308.jpg (104.3 kB)
                        ... and 295 other files
                    ... and 9 other folders
```

-> data/paddy-disease-classification/sample_submission.csv has 2602 rows and 2 columns.
The columns are: image_id, label

-> data/paddy-disease-classification/train.csv has 7805 rows and 4 columns.
The columns are: image_id, label, variety, age

-> data/sample_submission.csv has 2602 rows and 2 columns.
The columns are: image_id, label

-> data/train.csv has 7805 rows and 4 columns.
The columns are: image_id, label, variety, age

-> input/paddy-disease-classification/sample_submission.csv has 2602 rows and 2 columns.
The columns are: image_id, label

-> input/paddy-disease-classification/train.csv has 7805 rows and 4 columns.
The columns are: image_id, label, variety, age

-> (stopped after 10 files for performance)

# 5. Target score

0.97926267281106

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os
import pandas as pd
import numpy as np
import scipy.stats as ss
import tensorflow_hub as hub
import tensorflow as tf

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Concatenate
from tensorflow.keras.activations import relu, softmax
from tensorflow.keras.preprocessing.image import ImageDataGenerator

## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 3
m1 = tf.keras.models.load_model('../input/paddy-doc-ensemble-models/ensemble_estimators/effnet_b3.hdf5', custom_objects={'KerasLayer':hub.KerasLayer})
m2 = tf.keras.models.load_model('../input/paddy-doc-ensemble-models/ensemble_estimators/effnet_v2_m.hdf5', custom_objects={'KerasLayer':hub.KerasLayer})
m3 = tf.keras.models.load_model('../input/paddy-doc-ensemble-models/ensemble_estimators/xception.hdf5', custom_objects={'KerasLayer':hub.KerasLayer})
m4 = tf.keras.models.load_model('../input/paddydocoutputs/model_resnet150.hdf5', custom_objects={'KerasLayer':hub.KerasLayer})
m5 = tf.keras.models.load_model('../input/k/kasunpramodya/paddy-doctor-training/model_effnet_s.hdf5', custom_objects={'KerasLayer':hub.KerasLayer})

## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/660830188.py in <cell line: 0>()
----> 1 m1 = tf.keras.models.load_model('../input/paddy-doc-ensemble-models/ensemble_estimators/effnet_b3.hdf5', custom_objects={'KerasLayer':hub.KerasLayer})
      2 m2 = tf.keras.models.load_model('../input/paddy-doc-ensemble-models/ensemble_estimators/effnet_v2_m.hdf5', custom_objects={'KerasLayer':hub.KerasLayer})
      3 m3 = tf.keras.models.load_model('../input/paddy-doc-ensemble-models/ensemble_estimators/xception.hdf5', custom_objects={'KerasLayer':hub.KerasLayer})
      4 m4 = tf.keras.models.load_model('../input/paddydocoutputs/model_resnet150.hdf5', custom_objects={'KerasLayer':hub.KerasLayer})
      5 m5 = tf.keras.models.load_model('../input/k/kasunpramodya/paddy-doctor-training/model_effnet_s.hdf5', custom_objects={'KerasLayer':hub.KerasLayer})

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

FileNotFoundError: [Errno 2] Unable to synchronously open file (unable to open file: name = '../input/paddy-doc-ensemble-models/ensemble_estimators/effnet_b3.hdf5', errno = 2, error message = 'No such file or directory', flags = 0, o_flags = 0)

## === cell 5
test_loc = '../input/paddy-disease-classification/test_images'

test_data_300 = ImageDataGenerator(rescale=1.0/255).flow_from_directory(directory=test_loc,
                                                                    target_size=(300, 300),
                                                                    batch_size=16,
                                                                    classes=['.'],
                                                                    shuffle=False,
                                                                   )

## === cell 7
m1_p = m1.predict(test_data_300, verbose=1)
m2_p = m2.predict(test_data_300, verbose=1)
m3_p = m3.predict(test_data_300, verbose=1)
m4_p = m4.predict(test_data_300, verbose=1)
m5_p = m5.predict(test_data_300, verbose=1)

m1_predict_max = np.argmax(m1_p,axis=1)
m1_confidance = np.max(m1_p,axis=1)

m2_predict_max = np.argmax(m2_p,axis=1)
m2_confidance = np.max(m2_p,axis=1)

m3_predict_max = np.argmax(m3_p,axis=1)
m3_confidance = np.max(m3_p,axis=1)

m4_predict_max = np.argmax(m4_p,axis=1)
m4_confidance = np.max(m4_p,axis=1)

m5_predict_max = np.argmax(m5_p,axis=1)
m5_confidance = np.max(m5_p,axis=1)

## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4141731454.py in <cell line: 0>()
----> 1 m1_p = m1.predict(test_data_300, verbose=1)
      2 m2_p = m2.predict(test_data_300, verbose=1)
      3 m3_p = m3.predict(test_data_300, verbose=1)
      4 m4_p = m4.predict(test_data_300, verbose=1)
      5 m5_p = m5.predict(test_data_300, verbose=1)

NameError: name 'm1' is not defined

## === cell 8
class_indices = {'bacterial_leaf_blight': 0,
                 'bacterial_leaf_streak': 1,
                 'bacterial_panicle_blight': 2,
                 'blast': 3,
                 'brown_spot': 4,
                 'dead_heart': 5,
                 'downy_mildew': 6,
                 'hispa': 7,
                 'normal': 8,
                 'tungro': 9}

inverse_map = {v:k for k,v in class_indices.items()}

## === cell 9
pred_df = []
files=test_data_300.filenames

for preds in [m1_predict_max, m2_predict_max, m3_predict_max, m4_predict_max, m5_predict_max]:
    predictions = [inverse_map[k] for k in preds]
    temp = pd.DataFrame({"image_id":files,
                         "label":predictions})
    temp.image_id = temp.image_id.str.replace('./', '')
    pred_df.append(temp)

## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3907267116.py in <cell line: 0>()
      2 files=test_data_300.filenames
      3 
----> 4 for preds in [m1_predict_max, m2_predict_max, m3_predict_max, m4_predict_max, m5_predict_max]:
      5     predictions = [inverse_map[k] for k in preds]
      6     temp = pd.DataFrame({"image_id":files,

NameError: name 'm1_predict_max' is not defined

## === cell 10
s_full = pred_df[0]

for df in pred_df[1:]:
    s_full = pd.concat([s_full, df.drop('image_id', axis=1)], axis=1)
    
s_full.set_index('image_id', inplace=True)
s_full

## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
IndexError                                Traceback (most recent call last)
/tmp/ipykernel_11/231447905.py in <cell line: 0>()
----> 1 s_full = pred_df[0]
      2 
      3 for df in pred_df[1:]:
      4     s_full = pd.concat([s_full, df.drop('image_id', axis=1)], axis=1)
      5 

IndexError: list index out of range

## === cell 11
s_full_mode = ss.mode(s_full, axis=1)

## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3552784536.py in <cell line: 0>()
----> 1 s_full_mode = ss.mode(s_full, axis=1)

NameError: name 's_full' is not defined

## === cell 12
s_df = pd.DataFrame()
s_df['label'] = s_full_mode[0].reshape(1,-1)[0]
s_df['count'] = s_full_mode[1]
s_df

## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1859464721.py in <cell line: 0>()
      1 s_df = pd.DataFrame()
----> 2 s_df['label'] = s_full_mode[0].reshape(1,-1)[0]
      3 s_df['count'] = s_full_mode[1]
      4 s_df

NameError: name 's_full_mode' is not defined

## === cell 13
s_full['most'] = s_full_mode[0].reshape(1,-1)[0]
s_full

## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4160632301.py in <cell line: 0>()
----> 1 s_full['most'] = s_full_mode[0].reshape(1,-1)[0]
      2 s_full

NameError: name 's_full_mode' is not defined

## === cell 14
s_full.iloc[s_df[s_df['count']>2].index].index

## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1589726590.py in <cell line: 0>()
----> 1 s_full.iloc[s_df[s_df['count']>2].index].index

NameError: name 's_full' is not defined

## === cell 15
s_full.iloc[s_df[s_df['count']<2].index]

## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1460032817.py in <cell line: 0>()
----> 1 s_full.iloc[s_df[s_df['count']<2].index]

NameError: name 's_full' is not defined

## === cell 16
s_final = s_full[['most']]
s_final.iloc[s_df[s_df['count']<2].index] = pred_df[1].drop('image_id', axis=1).iloc[s_df[s_df['count']<2].index]
s_final

## --- ERROR in cell 16, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/355262842.py in <cell line: 0>()
----> 1 s_final = s_full[['most']]
      2 s_final.iloc[s_df[s_df['count']<2].index] = pred_df[1].drop('image_id', axis=1).iloc[s_df[s_df['count']<2].index]
      3 s_final

NameError: name 's_full' is not defined

## === cell 17
s = s_final.rename(columns={'most':'label'}).reset_index()
s

## --- ERROR in cell 17, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1716892173.py in <cell line: 0>()
----> 1 s = s_final.rename(columns={'most':'label'}).reset_index()
      2 s

NameError: name 's_final' is not defined

## === cell 18
s.to_csv('model_submission_v11.csv', index=False)

## --- ERROR in cell 18, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/154800766.py in <cell line: 0>()
----> 1 s.to_csv('model_submission_v11.csv', index=False)

NameError: name 's' is not defined

## === cell 19
if not os.path.isdir('ensemble_estimators'):
    os.mkdir('ensemble_estimators')
    
m1.save('ensemble_estimators/effnet_b3.hdf5')
m2.save('ensemble_estimators/effnet_v2_m.hdf5')
m3.save('ensemble_estimators/xception.hdf5')
m4.save('ensemble_estimators/resnet_150.hdf5')
m5.save('ensemble_estimators/effnet_v2_s.hdf5')

## --- ERROR in cell 19, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1018428729.py in <cell line: 0>()
      2     os.mkdir('ensemble_estimators')
      3 
----> 4 m1.save('ensemble_estimators/effnet_b3.hdf5')
      5 m2.save('ensemble_estimators/effnet_v2_m.hdf5')
      6 m3.save('ensemble_estimators/xception.hdf5')

NameError: name 'm1' is not defined
