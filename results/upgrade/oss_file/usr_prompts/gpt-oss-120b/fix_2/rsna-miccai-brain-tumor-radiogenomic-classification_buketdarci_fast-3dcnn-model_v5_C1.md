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
Predict the genetic subtype of glioblastoma using MRI (magnetic resonance imaging) scans to detect for the presence of MGMT promoter methylation.

## Metric
Area under the ROC curve between the predicted probability and the observed target.

## Submission Format
For each `BraTS21ID` in the test set, you must predict a probability for the target `MGMT_value`. The file should contain a header and have the following format:

```
BraTS21ID,MGMT_value
00001,0.5
00013,0.5
00015,0.5
etc.
```

## Dataset
- **train/** - folder containing the training files, with each top-level folder representing a subject. **NOTE:** There are some unexpected issues with the following three cases in the training dataset, participants can exclude the cases during training: `[00109, 00123, 00709]`. We have checked and confirmed that the testing dataset is free from such issues.
- **train_labels.csv** - file containing the target `MGMT_value` for each subject in the training data (e.g. the presence of MGMT promoter methylation)
- **test/** - the test files, which use the same structure as `train/`; your task is to predict the `MGMT_value` for each subject in the test data. **NOTE**: the total size of the rerun test set (Public and Private) is ~5x the size of the Public test set
- **sample_submission.csv** - a sample submission file in the correct format

# 2. Python version

3.9

# 3. Installed packages

geopandas==0.14.4
google-api-python-client==2.177.0
ipython==7.34.0
ipython-genutils==0.2.0
ipython_pygments_lexers==1.1.1
ipython-sql==0.5.0
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
protobuf==6.33.0
pydicom==3.0.1
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
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

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (202 lines)
            sample_submission.csv (60 lines)
            sample_submission.csv.zip (382 Bytes)
            test.zip (1.3 GB)
            train.zip (10.2 GB)
            train_labels.csv (527 lines)
            train_labels.csv.zip (1.4 kB)
            rsna-miccai-brain-tumor-radiogenomic-classification/
                description.md (202 lines)
                sample_submission.csv (60 lines)
                ... and 5 other files
                rsna-miccai-brain-tumor-radiogenomic-classification/
                test/
                    00002/
                        FLAIR/
                            ... (max depth reached)
                        T1w/
                            ... (max depth reached)
                        T1wCE/
                            ... (max depth reached)
                        T2w/
                            ... (max depth reached)
                    00019/
                        FLAIR/
                            ... (max depth reached)
                        T1w/
                            ... (max depth reached)
                        T1wCE/
                            ... (max depth reached)
                        T2w/
                            ... (max depth reached)
                    ... and 58 other folders
                train/
                    00000/
                        FLAIR/
                            ... (max depth reached)
                        T1w/
                            ... (max depth reached)
                        T1wCE/
                            ... (max depth reached)
                        T2w/
                            ... (max depth reached)
                    00003/
                        FLAIR/
                            ... (max depth reached)
                        T1w/
                            ... (max depth reached)
                        T1wCE/
                            ... (max depth reached)
                        T2w/
                            ... (max depth reached)
                    ... and 525 other folders
            test/
                00002/
                    FLAIR/
                        Image-387.dcm (525.4 kB)
                        Image-388.dcm (525.4 kB)
                        ... and 127 other files
                    T1w/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 29 other files
                    T1wCE/
                        Image-1.dcm (525.3 kB)
                        Image-10.dcm (525.3 kB)
                        ... and 127 other files
                    T2w/
                        Image-1.dcm (525.3 kB)
                        Image-10.dcm (525.3 kB)
                        ... and 382 other files
                00019/
                    FLAIR/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 127 other files
                    T1w/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 30 other files
                    T1wCE/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 127 other files
                    T2w/
                        Image-258.dcm (525.4 kB)
                        Image-259.dcm (525.4 kB)
                        ... and 127 other files
                ... and 58 other folders
            train/
                00000/
                    FLAIR/
                        Image-1.dcm (525.3 kB)
                        Image-10.dcm (525.3 kB)
                        ... and 398 other files
                    T1w/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 31 other files
                    T1wCE/
                        Image-1.dcm (525.3 kB)
                        Image-10.dcm (525.3 kB)
                        ... and 127 other files
                    T2w/
                        Image-1.dcm (525.3 kB)
                        Image-10.dcm (525.3 kB)
                        ... and 406 other files
                00003/
                    FLAIR/
                        Image-387.dcm (525.4 kB)
                        Image-388.dcm (525.4 kB)
                        ... and 127 other files
                    T1w/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 31 other files
                    T1wCE/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 127 other files
                    T2w/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 406 other files
                ... and 525 other folders
        input/
            description.md (202 lines)
            sample_submission.csv (60 lines)
            sample_submission.csv.zip (382 Bytes)
            test.zip (1.3 GB)
            train.zip (10.2 GB)
            train_labels.csv (527 lines)
            train_labels.csv.zip (1.4 kB)
            rsna-miccai-brain-tumor-radiogenomic-classification/
                description.md (202 lines)
                sample_submission.csv (60 lines)
                ... and 5 other files
                rsna-miccai-brain-tumor-radiogenomic-classification/
                test/
                    00002/
                        FLAIR/
                            ... (max depth reached)
                        T1w/
                            ... (max depth reached)
                        T1wCE/
                            ... (max depth reached)
                        T2w/
                            ... (max depth reached)
                    00019/
                        FLAIR/
                            ... (max depth reached)
                        T1w/
                            ... (max depth reached)
                        T1wCE/
                            ... (max depth reached)
                        T2w/
                            ... (max depth reached)
                    ... and 58 other folders
                train/
                    00000/
                        FLAIR/
                            ... (max depth reached)
                        T1w/
                            ... (max depth reached)
                        T1wCE/
                            ... (max depth reached)
                        T2w/
                            ... (max depth reached)
                    00003/
                        FLAIR/
                            ... (max depth reached)
                        T1w/
                            ... (max depth reached)
                        T1wCE/
                            ... (max depth reached)
                        T2w/
                            ... (max depth reached)
                    ... and 525 other folders
            test/
                00002/
                    FLAIR/
                        Image-387.dcm (525.4 kB)
                        Image-388.dcm (525.4 kB)
                        ... and 127 other files
                    T1w/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 29 other files
                    T1wCE/
                        Image-1.dcm (525.3 kB)
                        Image-10.dcm (525.3 kB)
                        ... and 127 other files
                    T2w/
                        Image-1.dcm (525.3 kB)
                        Image-10.dcm (525.3 kB)
                        ... and 382 other files
                00019/
                    FLAIR/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 127 other files
                    T1w/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 30 other files
                    T1wCE/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 127 other files
                    T2w/
                        Image-258.dcm (525.4 kB)
                        Image-259.dcm (525.4 kB)
                        ... and 127 other files
                ... and 58 other folders
            train/
                00000/
                    FLAIR/
                        Image-1.dcm (525.3 kB)
                        Image-10.dcm (525.3 kB)
                        ... and 398 other files
                    T1w/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 31 other files
                    T1wCE/
                        Image-1.dcm (525.3 kB)
                        Image-10.dcm (525.3 kB)
                        ... and 127 other files
                    T2w/
                        Image-1.dcm (525.3 kB)
                        Image-10.dcm (525.3 kB)
                        ... and 406 other files
                00003/
                    FLAIR/
                        Image-387.dcm (525.4 kB)
                        Image-388.dcm (525.4 kB)
                        ... and 127 other files
                    T1w/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 31 other files
                    T1wCE/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 127 other files
                    T2w/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 406 other files
                ... and 525 other folders
        working/
            rsna-miccai-brain-tumor-radiogenomic-classification/
                description.md (202 lines)
                sample_submission.csv (60 lines)
                ... and 5 other files
                rsna-miccai-brain-tumor-radiogenomic-classification/
                test/
                    00002/
                        FLAIR/
                            ... (max depth reached)
                        T1w/
                            ... (max depth reached)
                        T1wCE/
                            ... (max depth reached)
                        T2w/
                            ... (max depth reached)
                    00019/
                        FLAIR/
                            ... (max depth reached)
                        T1w/
                            ... (max depth reached)
                        T1wCE/
                            ... (max depth reached)
                        T2w/
                            ... (max depth reached)
                    ... and 58 other folders
                train/
                    00000/
                        FLAIR/
                            ... (max depth reached)
                        T1w/
                            ... (max depth reached)
                        T1wCE/
                            ... (max depth reached)
                        T2w/
                            ... (max depth reached)
                    00003/
                        FLAIR/
                            ... (max depth reached)
                        T1w/
                            ... (max depth reached)
                        T1wCE/
                            ... (max depth reached)
                        T2w/
                            ... (max depth reached)
                    ... and 525 other folders
```

-> data/rsna-miccai-brain-tumor-radiogenomic-classification/sample_submission.csv has 59 rows and 2 columns.
The columns are: BraTS21ID, MGMT_value

-> data/rsna-miccai-brain-tumor-radiogenomic-classification/train_labels.csv has 526 rows and 2 columns.
The columns are: BraTS21ID, MGMT_value

-> data/sample_submission.csv has 59 rows and 2 columns.
The columns are: BraTS21ID, MGMT_value

-> data/train_labels.csv has 526 rows and 2 columns.
The columns are: BraTS21ID, MGMT_value

-> input/rsna-miccai-brain-tumor-radiogenomic-classification/sample_submission.csv has 59 rows and 2 columns.
The columns are: BraTS21ID, MGMT_value

-> input/rsna-miccai-brain-tumor-radiogenomic-classification/train_labels.csv has 526 rows and 2 columns.
The columns are: BraTS21ID, MGMT_value

-> (stopped after 10 files for performance)

# 5. Target score

0.58439

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os, re, glob, numpy as np, pandas as pd, cv2, matplotlib.pyplot as plt
import pydicom
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks


def atoi(text):
    return int(text) if text.isdigit() else text


def natural_keys(text):
    return [atoi(c) for c in re.split(r"(\d+)", text)]


def load_dicom(path):
    ds = pydicom.dcmread(path)
    img = ds.pixel_array.astype(np.float32)
    if img.max() > 0:
        img = img / img.max()
    img = (img * 255).astype(np.uint8)
    return img




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
label_path = (
    "/kaggle/input/rsna-miccai-brain-tumor-radiogenomic-classification/train_labels.csv"
)
df_labels = pd.read_csv(label_path)
bad_ids = ["00109", "00123", "00709"]
df_labels = df_labels[~df_labels["BraTS21ID"].astype(str).isin(bad_ids)].reset_index(
    drop=True
)
labels = df_labels["MGMT_value"].values.astype(np.float32)




## === cell 2
train_root = "/kaggle/input/rsna-miccai-brain-tumor-radiogenomic-classification/train/"
subject_dirs = sorted(
    [d for d in glob.glob(os.path.join(train_root, "*")) if os.path.isdir(d)],
    key=natural_keys,
)
subject_dirs = [d for d in subject_dirs if os.path.basename(d) not in bad_ids]


def gather_modality_paths(modality):
    all_paths = []
    for sub in subject_dirs:
        pattern = os.path.join(sub, modality, "**", "*.dcm")
        files = sorted(glob.glob(pattern, recursive=True), key=natural_keys)
        all_paths.append(files)
    return all_paths


flair_paths = gather_modality_paths("FLAIR")
t1w_paths = gather_modality_paths("T1w")
t1wce_paths = gather_modality_paths("T1wCE")
t2w_paths = gather_modality_paths("T2w")




## === cell 3
def create_input(patch_lists, n_subjects):
    """
    Build a (n_subjects, 256, 256, 3) array.
    For each subject we take up to three central slices
    (or the first three if fewer are available) and
    average them to obtain a single 3‑channel image.
    """
    inputs = np.zeros((n_subjects, 256, 256, 3), dtype=np.uint8)
    for i, patches in enumerate(patch_lists):
        if len(patches) >= 3:
            start = len(patches) // 2 - 1
            idxs = [start, start + 1, start + 2]
        else:
            idxs = list(range(min(3, len(patches))))
        imgs = []
        for j in idxs:
            img = load_dicom(patches[j])
            img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA)
            imgs.append(img)
        if not imgs:  # very unlikely, but keep shape valid
            img_stack = np.zeros((256, 256), dtype=np.uint8)
        else:
            img_stack = np.mean(imgs, axis=0).astype(np.uint8)
        inputs[i] = np.stack([img_stack] * 3, axis=-1)
    return inputs


n_subj = len(subject_dirs)
flair_inputs = create_input(flair_paths, n_subj)




## === cell 4
flair_inputs = flair_inputs.astype(np.float32) / 255.0
flair_inputs = np.expand_dims(flair_inputs, -1)  # shape (N,256,256,3,1)

x_train, x_valid, y_train, y_valid = train_test_split(
    flair_inputs, labels, test_size=0.2, random_state=42, stratify=labels
)




## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_55/1568735287.py in <cell line: 0>()
      4 
      5 # train / validation split
----> 6 x_train, x_valid, y_train, y_valid = train_test_split(
      7     flair_inputs, labels, test_size=0.2, random_state=42, stratify=labels
      8 )

/usr/local/lib/python3.11/dist-packages/sklearn/model_selection/_split.py in train_test_split(test_size, train_size, random_state, shuffle, stratify, *arrays)
   2557         raise ValueError("At least one array required as input")
   2558 
-> 2559     arrays = indexable(*arrays)
   2560 
   2561     n_samples = _num_samples(arrays[0])

/usr/local/lib/python3.11/dist-packages/sklearn/utils/validation.py in indexable(*iterables)
    441 
    442     result = [_make_indexable(X) for X in iterables]
--> 443     check_consistent_length(*result)
    444     return result
    445 

/usr/local/lib/python3.11/dist-packages/sklearn/utils/validation.py in check_consistent_length(*arrays)
    395     uniques = np.unique(lengths)
    396     if len(uniques) > 1:
--> 397         raise ValueError(
    398             "Found input variables with inconsistent numbers of samples: %r"
    399             % [int(l) for l in lengths]

ValueError: Found input variables with inconsistent numbers of samples: [524, 526]

## === cell 5
def get_model(optimizer_name="Adam"):
    """Simple 3‑D CNN compatible with the input shape (256,256,3,1)."""
    inputs = keras.Input(shape=(256, 256, 3, 1))

    x = layers.Conv3D(32, kernel_size=3, activation="relu", padding="same")(inputs)
    x = layers.MaxPool3D(pool_size=2)(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv3D(64, kernel_size=3, activation="relu", padding="same")(x)
    x = layers.MaxPool3D(pool_size=2)(x)
    x = layers.BatchNormalization()(x)

    x = layers.GlobalAveragePooling3D()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)

    model = keras.Model(inputs, outputs)
    model.compile(
        optimizer=optimizer_name,
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
    )
    return model


model = get_model("Adam")




## === cell 6
ckpt_path = "/kaggle/working/best_model.h5"
ckpt = callbacks.ModelCheckpoint(
    ckpt_path, monitor="val_auc", save_best_only=True, mode="max"
)
lr_reduce = callbacks.ReduceLROnPlateau(
    monitor="val_auc", factor=0.5, patience=3, mode="max", verbose=1
)
early_stop = callbacks.EarlyStopping(
    monitor="val_auc", patience=6, mode="max", restore_best_weights=True
)

history = model.fit(
    x_train,
    y_train,
    validation_data=(x_valid, y_valid),
    epochs=30,
    batch_size=8,
    callbacks=[ckpt, lr_reduce, early_stop],
    verbose=2,
)




## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2437940007.py in <cell line: 0>()
     12 
     13 history = model.fit(
---> 14     x_train,
     15     y_train,
     16     validation_data=(x_valid, y_valid),

NameError: name 'x_train' is not defined

## === cell 7
val_pred = model.predict(x_valid).ravel()
val_auc = roc_auc_score(y_valid, val_pred)
print(f"Validation AUC: {val_auc:.5f}")




## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1051402476.py in <cell line: 0>()
      1 # evaluate on validation set (optional sanity check)
----> 2 val_pred = model.predict(x_valid).ravel()
      3 val_auc = roc_auc_score(y_valid, val_pred)
      4 print(f"Validation AUC: {val_auc:.5f}")
      5 

NameError: name 'x_valid' is not defined

## === cell 8
test_root = "/kaggle/input/rsna-miccai-brain-tumor-radiogenomic-classification/test/"
test_dirs = sorted(
    [d for d in glob.glob(os.path.join(test_root, "*")) if os.path.isdir(d)],
    key=natural_keys,
)


def gather_test_paths(modality):
    all_paths = []
    for sub in test_dirs:
        pattern = os.path.join(sub, modality, "**", "*.dcm")
        files = sorted(glob.glob(pattern, recursive=True), key=natural_keys)
        all_paths.append(files)
    return all_paths


flair_test_paths = gather_test_paths("FLAIR")
test_inputs = create_input(flair_test_paths, len(test_dirs))
test_inputs = test_inputs.astype(np.float32) / 255.0
test_inputs = np.expand_dims(test_inputs, -1)  # (N,256,256,3,1)

best_model = keras.models.load_model(ckpt_path)
test_pred = best_model.predict(test_inputs).ravel()




## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_55/3904139908.py in <cell line: 0>()
     23 
     24 # load best checkpoint
---> 25 best_model = keras.models.load_model(ckpt_path)
     26 test_pred = best_model.predict(test_inputs).ravel()
     27 

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

FileNotFoundError: [Errno 2] Unable to synchronously open file (unable to open file: name = '/kaggle/working/best_model.h5', errno = 2, error message = 'No such file or directory', flags = 0, o_flags = 0)

## === cell 9
sample_path = "/kaggle/input/rsna-miccai-brain-tumor-radiogenomic-classification/sample_submission.csv"
submission = pd.read_csv(sample_path, index_col="BraTS21ID")
submission = submission.loc[[os.path.basename(p) for p in test_dirs]]
submission["MGMT_value"] = test_pred
out_path = "/kaggle/working/submission.csv"
submission.to_csv(out_path)
print(f"Submission written to {out_path}")

## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_55/1783533628.py in <cell line: 0>()
      4 submission = pd.read_csv(sample_path, index_col="BraTS21ID")
      5 # ensure ordering matches test IDs
----> 6 submission = submission.loc[[os.path.basename(p) for p in test_dirs]]
      7 submission["MGMT_value"] = test_pred
      8 out_path = "/kaggle/working/submission.csv"

/usr/local/lib/python3.11/dist-packages/pandas/core/indexing.py in __getitem__(self, key)
   1189             maybe_callable = com.apply_if_callable(key, self.obj)
   1190             maybe_callable = self._check_deprecated_callable_usage(key, maybe_callable)
-> 1191             return self._getitem_axis(maybe_callable, axis=axis)
   1192 
   1193     def _is_scalar_access(self, key: tuple):

/usr/local/lib/python3.11/dist-packages/pandas/core/indexing.py in _getitem_axis(self, key, axis)
   1418                     raise ValueError("Cannot index with multidimensional key")
   1419 
-> 1420                 return self._getitem_iterable(key, axis=axis)
   1421 
   1422             # nested tuple slicing

/usr/local/lib/python3.11/dist-packages/pandas/core/indexing.py in _getitem_iterable(self, key, axis)
   1358 
   1359         # A collection of keys
-> 1360         keyarr, indexer = self._get_listlike_indexer(key, axis)
   1361         return self.obj._reindex_with_indexers(
   1362             {axis: [keyarr, indexer]}, copy=True, allow_dups=True

/usr/local/lib/python3.11/dist-packages/pandas/core/indexing.py in _get_listlike_indexer(self, key, axis)
   1556         axis_name = self.obj._get_axis_name(axis)
   1557 
-> 1558         keyarr, indexer = ax._get_indexer_strict(key, axis_name)
   1559 
   1560         return keyarr, indexer

/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in _get_indexer_strict(self, key, axis_name)
   6198             keyarr, indexer, new_indexer = self._reindex_non_unique(keyarr)
   6199 
-> 6200         self._raise_if_missing(keyarr, indexer, axis_name)
   6201 
   6202         keyarr = self.take(indexer)

/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in _raise_if_missing(self, key, indexer, axis_name)
   6247         if nmissing:
   6248             if nmissing == len(indexer):
-> 6249                 raise KeyError(f"None of [{key}] are in the [{axis_name}]")
   6250 
   6251             not_found = list(ensure_index(key)[missing_mask.nonzero()[0]].unique())

KeyError: "None of [Index(['00002', '00019', '00021', '00031', '00054', '00059', '00074', '00084',\n       '00132', '00140', '00155', '00177', '00185', '00214', '00236', '00241',\n       '00245', '00254', '00263', '00269', '00275', '00332', '00339', '00356',\n       '00399', '00413', '00459', '00470', '00495', '00496', '00499', '00511',\n       '00512', '00525', '00532', '00539', '00556', '00570', '00587', '00591',\n       '00601', '00613', '00622', '00623', '00650', '00656', '00657', '00661',\n       '00718', '00729', '00732', '00735', '00737', '00740', '00757', '00803',\n       '00823', '01001', '01009', 'test'],\n      dtype='object', name='BraTS21ID')] are in the [index]"
