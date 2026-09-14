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

0.976958525345622

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import pandas as pd
import numpy as np
import scipy.stats as ss
import tensorflow_hub as hub
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 3
s1 = pd.read_csv('../input/paddydocoutputs/model_submission_v5.csv')
s2 = pd.read_csv('../input/paddy-doctor-training/model_submission_v5.csv')
s3 = pd.read_csv('../input/k/kasunpramodya/paddy-doctor-training/model_submission_v5.csv')

## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/1900004569.py in <cell line: 0>()
----> 1 s1 = pd.read_csv('../input/paddydocoutputs/model_submission_v5.csv')
      2 s2 = pd.read_csv('../input/paddy-doctor-training/model_submission_v5.csv')
      3 s3 = pd.read_csv('../input/k/kasunpramodya/paddy-doctor-training/model_submission_v5.csv')

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

FileNotFoundError: [Errno 2] No such file or directory: '../input/paddydocoutputs/model_submission_v5.csv'

## === cell 5
m1 = tf.keras.models.load_model('../input/paddydocoutputs/model_xception.hdf5', custom_objects={'KerasLayer':hub.KerasLayer})
m3 = tf.keras.models.load_model('../input/k/kasunpramodya/paddy-doctor-training/model_effecientV2_b4.hdf5', custom_objects={'KerasLayer':hub.KerasLayer})

## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/1702101377.py in <cell line: 0>()
----> 1 m1 = tf.keras.models.load_model('../input/paddydocoutputs/model_xception.hdf5', custom_objects={'KerasLayer':hub.KerasLayer})
      2 m3 = tf.keras.models.load_model('../input/k/kasunpramodya/paddy-doctor-training/model_effecientV2_b4.hdf5', custom_objects={'KerasLayer':hub.KerasLayer})

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

FileNotFoundError: [Errno 2] Unable to synchronously open file (unable to open file: name = '../input/paddydocoutputs/model_xception.hdf5', errno = 2, error message = 'No such file or directory', flags = 0, o_flags = 0)

## === cell 6
back_bone = tf.keras.applications.Xception(weights=None, input_shape=(224,224,3), include_top=False)
input_layer = tf.keras.layers.Input(shape=(224,224,3))
x = back_bone(input_layer)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
output_layer = tf.keras.layers.Dense(10, activation='softmax')(x)

model = tf.keras.models.Model(input_layer,output_layer)
model.load_weights('../input/paddy-doctor-training/model_xception_weights.hdf5')

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
              loss=tf.keras.losses.categorical_crossentropy,
              metrics=['accuracy'])

## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/3968457534.py in <cell line: 0>()
      6 
      7 model = tf.keras.models.Model(input_layer,output_layer)
----> 8 model.load_weights('../input/paddy-doctor-training/model_xception_weights.hdf5')
      9 
     10 model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),

/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py in error_handler(*args, **kwargs)
    120             # To get the full stack trace, call:
    121             # `keras.config.disable_traceback_filtering()`
--> 122             raise e.with_traceback(filtered_tb) from None
    123         finally:
    124             del filtered_tb

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

FileNotFoundError: [Errno 2] Unable to synchronously open file (unable to open file: name = '../input/paddy-doctor-training/model_xception_weights.hdf5', errno = 2, error message = 'No such file or directory', flags = 0, o_flags = 0)

## === cell 8
test_loc = '../input/paddy-disease-classification/test_images'

test_data_224 = ImageDataGenerator(rescale=1.0/255).flow_from_directory(directory=test_loc,
                                                                    target_size=(224, 224),
                                                                    batch_size=32,
                                                                    classes=['.'],
                                                                    shuffle=False,
                                                                   )

test_data_300 = ImageDataGenerator(rescale=1.0/255).flow_from_directory(directory=test_loc,
                                                                    target_size=(300, 300),
                                                                    batch_size=16,
                                                                    classes=['.'],
                                                                    shuffle=False,
                                                                   )

## === cell 10
m1_p = m1.predict(test_data_224, verbose=1)
m2_p = model.predict(test_data_224, verbose=1)
m3_p = m3.predict(test_data_300, verbose=1)

m1_predict_max = np.argmax(m1_p,axis=1)
m1_confidance = np.max(m1_p,axis=1)

m2_predict_max = np.argmax(m2_p,axis=1)
m2_confidance = np.max(m2_p,axis=1)

m3_predict_max = np.argmax(m3_p,axis=1)
m3_confidance = np.max(m3_p,axis=1)

## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/655873478.py in <cell line: 0>()
----> 1 m1_p = m1.predict(test_data_224, verbose=1)
      2 m2_p = model.predict(test_data_224, verbose=1)
      3 m3_p = m3.predict(test_data_300, verbose=1)
      4 
      5 m1_predict_max = np.argmax(m1_p,axis=1)

NameError: name 'm1' is not defined

## === cell 11
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

## === cell 12
m1_p_df = pd.DataFrame({"image_id":test_data_224.filenames,
                        "label":[inverse_map[k] for k in m1_predict_max]})
m2_p_df = pd.DataFrame({"image_id":test_data_224.filenames,
                        "label":[inverse_map[k] for k in m2_predict_max]})
m3_p_df = pd.DataFrame({"image_id":test_data_300.filenames,
                        "label":[inverse_map[k] for k in m3_predict_max]})

m1_p_df.image_id = m1_p_df.image_id.str.replace('./', '')
m2_p_df.image_id = m2_p_df.image_id.str.replace('./', '')
m3_p_df.image_id = m3_p_df.image_id.str.replace('./', '')

## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2830115713.py in <cell line: 0>()
      1 m1_p_df = pd.DataFrame({"image_id":test_data_224.filenames,
----> 2                         "label":[inverse_map[k] for k in m1_predict_max]})
      3 m2_p_df = pd.DataFrame({"image_id":test_data_224.filenames,
      4                         "label":[inverse_map[k] for k in m2_predict_max]})
      5 m3_p_df = pd.DataFrame({"image_id":test_data_300.filenames,

NameError: name 'm1_predict_max' is not defined

## === cell 13
m1_p_df.set_index('image_id', inplace=True)
m2_p_df.set_index('image_id', inplace=True)
m3_p_df.set_index('image_id', inplace=True)

## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2551466709.py in <cell line: 0>()
----> 1 m1_p_df.set_index('image_id', inplace=True)
      2 m2_p_df.set_index('image_id', inplace=True)
      3 m3_p_df.set_index('image_id', inplace=True)

NameError: name 'm1_p_df' is not defined

## === cell 14
m1_p_df['conf'] = m1_confidance
m2_p_df['conf'] = m2_confidance
m3_p_df['conf'] = m3_confidance

## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1883655537.py in <cell line: 0>()
----> 1 m1_p_df['conf'] = m1_confidance
      2 m2_p_df['conf'] = m2_confidance
      3 m3_p_df['conf'] = m3_confidance

NameError: name 'm1_confidance' is not defined

## === cell 15
m1_p_df

## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2983581953.py in <cell line: 0>()
----> 1 m1_p_df

NameError: name 'm1_p_df' is not defined

## === cell 17
s_full = pd.concat([s1,s2.drop('image_id', axis=1),s3.drop('image_id', axis=1)], axis=1).set_index('image_id')
s_full

## --- ERROR in cell 17, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3420157244.py in <cell line: 0>()
----> 1 s_full = pd.concat([s1,s2.drop('image_id', axis=1),s3.drop('image_id', axis=1)], axis=1).set_index('image_id')
      2 s_full

NameError: name 's1' is not defined

## === cell 18
s_full_mode = ss.mode(s_full, axis=1)

## --- ERROR in cell 18, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3552784536.py in <cell line: 0>()
----> 1 s_full_mode = ss.mode(s_full, axis=1)

NameError: name 's_full' is not defined

## === cell 19
s_df = pd.DataFrame()
s_df['label'] = s_full_mode[0].reshape(1,-1)[0]
s_df['count'] = s_full_mode[1]
s_df

## --- ERROR in cell 19, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1859464721.py in <cell line: 0>()
      1 s_df = pd.DataFrame()
----> 2 s_df['label'] = s_full_mode[0].reshape(1,-1)[0]
      3 s_df['count'] = s_full_mode[1]
      4 s_df

NameError: name 's_full_mode' is not defined

## === cell 20
s_full['most'] = s_full_mode[0].reshape(1,-1)[0]
s_full

## --- ERROR in cell 20, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4160632301.py in <cell line: 0>()
----> 1 s_full['most'] = s_full_mode[0].reshape(1,-1)[0]
      2 s_full

NameError: name 's_full_mode' is not defined

## === cell 21
pd.concat([m1_p_df.loc[s_full.iloc[s_df[s_df['count']<2].index].index],
           m2_p_df.loc[s_full.iloc[s_df[s_df['count']<2].index].index],
           m3_p_df.loc[s_full.iloc[s_df[s_df['count']<2].index].index]], axis=1)

## --- ERROR in cell 21, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3717701807.py in <cell line: 0>()
----> 1 pd.concat([m1_p_df.loc[s_full.iloc[s_df[s_df['count']<2].index].index],
      2            m2_p_df.loc[s_full.iloc[s_df[s_df['count']<2].index].index],
      3            m3_p_df.loc[s_full.iloc[s_df[s_df['count']<2].index].index]], axis=1)

NameError: name 'm1_p_df' is not defined

## === cell 22
s_full.iloc[s_df[s_df['count']<2].index].index

## --- ERROR in cell 22, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1133888604.py in <cell line: 0>()
----> 1 s_full.iloc[s_df[s_df['count']<2].index].index

NameError: name 's_full' is not defined

## === cell 23
s_final = s_full[['most']]
s_final.iloc[s_df[s_df['count']<2].index] = [['brown_spot'], ['brown_spot'], ['brown_spot'], ['dead_heart'], ['normal'], ['hispa'], ['brown_spot'],
                                             ['normal'], ['tungro'],['blast'],['bacterial_leaf_streak'],['bacterial_leaf_streak'],['tungro'],['tungro']]
s_final

## --- ERROR in cell 23, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2837292666.py in <cell line: 0>()
----> 1 s_final = s_full[['most']]
      2 s_final.iloc[s_df[s_df['count']<2].index] = [['brown_spot'], ['brown_spot'], ['brown_spot'], ['dead_heart'], ['normal'], ['hispa'], ['brown_spot'],
      3                                              ['normal'], ['tungro'],['blast'],['bacterial_leaf_streak'],['bacterial_leaf_streak'],['tungro'],['tungro']]
      4 s_final

NameError: name 's_full' is not defined

## === cell 24
s = s_final.rename(columns={'most':'label'}).reset_index()
s

## --- ERROR in cell 24, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1716892173.py in <cell line: 0>()
----> 1 s = s_final.rename(columns={'most':'label'}).reset_index()
      2 s

NameError: name 's_final' is not defined

## === cell 25
s.to_csv('submission_v11.csv', index=False)

## --- ERROR in cell 25, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2975468099.py in <cell line: 0>()
----> 1 s.to_csv('submission_v11.csv', index=False)

NameError: name 's' is not defined
