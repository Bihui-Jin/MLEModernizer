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
Predict a patient’s severity of decline in lung function based on a CT scan of their lungs. Lung function is assessed based on output from a spirometer, which measures the forced vital capacity (`FVC`), i.e. the volume of air exhaled.

## Metric
A modified version of the Laplace Log Likelihood. 

For each true FVC measurement, you will predict both an FVC and a confidence measure (standard deviation 𝜎𝜎). The metric is computed as:

$$
\begin{gathered}
\sigma_{\text {clipped }}=\max (\sigma, 70), \\
\Delta=\min \left(\left|F V C_{\text {true }}-F V C_{\text {predicted }}\right|, 1000\right), \\
\text { metric }=-\frac{\sqrt{2} \Delta}{\sigma_{\text {clipped }}}-\ln \left(\sqrt{2} \sigma_{\text {clipped }}\right) .
\end{gathered}
$$

The error is thresholded at 1000 ml to avoid large errors adversely penalizing results, while the confidence values are clipped at 70 ml to reflect the approximate measurement uncertainty in FVC. The final score is calculated by averaging the metric across all test set `Patient_Week`s (three per patient). 

Metric values will be negative and higher is better.

## Submission Format
For each `Patient_Week`, you must predict the `FVC` and a confidence. You are asked to predict every patient's `FVC` measurement for every possible week. Those weeks which are not in the final three visits are ignored in scoring.

The file should contain a header and have the following format:

```
Patient_Week,FVC,Confidence
ID00002637202176704235138_1,2000,100
ID00002637202176704235138_2,2000,100
ID00002637202176704235138_3,2000,100
etc.

```

## Dataset
In the dataset, you are provided with a baseline chest CT scan and associated clinical information for a set of patients. A patient has an image acquired at time `Week = 0` and has numerous follow up visits over the course of approximately 1-2 years, at which time their `FVC` is measured.

- In the training set, you are provided with an anonymized, baseline CT scan and the entire history of FVC measurements.
- In the test set, you are provided with a baseline CT scan and only the initial FVC measurement. **You are asked to predict the final three `FVC` measurements for each patient, as well as a confidence value in your prediction.**

- **train.csv** - the training set, contains full history of clinical information
- **test.csv** - the test set, contains only the baseline measurement
- **train/** - contains the training patients' baseline CT scan in DICOM format
- **test/** - contains the test patients' baseline CT scan in DICOM format
- **sample_submission.csv** - demonstrates the submission format

**train.csv and test.csv**

- `Patient`a unique Id for each patient (also the name of the patient's DICOM folder)
- `Weeks`the relative number of weeks pre/post the baseline CT (may be negative)
- `FVC` - the recorded lung capacity in ml
- `Percent`a computed field which approximates the patient's FVC as a percent of the typical FVC for a person of similar characteristics
- `Age`
- `Sex`
- `SmokingStatus`

**sample submission.csv**

- `Patient_Week` - a unique Id formed by concatenating the `Patient` and `Weeks` columns (i.e. ABC_22 is a prediction for patient ABC at week 22)
- `FVC` - the predicted FVC in ml
- `Confidence` - a confidence value of your prediction (also has units of ml)

# 2. Python version

3.14

# 3. Installed packages

No external packages required in the script and installed.

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (122 lines)
            sample_submission.csv (1909 lines)
            sample_submission.csv.zip (5.7 kB)
            test.csv (19 lines)
            test.csv.zip (748 Bytes)
            test.zip (1.2 GB)
            train.csv (1395 lines)
            train.csv.zip (23.6 kB)
            train.zip (12.7 GB)
            osic-pulmonary-fibrosis-progression/
                description.md (122 lines)
                sample_submission.csv (1909 lines)
                ... and 7 other files
                osic-pulmonary-fibrosis-progression/
                test/
                    ID00014637202177757139317/
                        1.dcm (1.5 MB)
                        10.dcm (1.5 MB)
                        ... and 29 other files
                    ID00019637202178323708467/
                        1.dcm (525.5 kB)
                        10.dcm (525.5 kB)
                        ... and 27 other files
                    ... and 17 other folders
                train/
                    ID00007637202177411956430/
                        1.dcm (525.6 kB)
                        10.dcm (525.6 kB)
                        ... and 28 other files
                    ID00009637202177434476278/
                        1.dcm (1.2 MB)
                        10.dcm (1.2 MB)
                        ... and 392 other files
                    ... and 157 other folders
            test/
                ID00014637202177757139317/
                    1.dcm (1.5 MB)
                    10.dcm (1.5 MB)
                    ... and 29 other files
                ID00019637202178323708467/
                    1.dcm (525.5 kB)
                    10.dcm (525.5 kB)
                    ... and 27 other files
                ... and 17 other folders
            train/
                ID00007637202177411956430/
                    1.dcm (525.6 kB)
                    10.dcm (525.6 kB)
                    ... and 28 other files
                ID00009637202177434476278/
                    1.dcm (1.2 MB)
                    10.dcm (1.2 MB)
                    ... and 392 other files
                ... and 157 other folders
        input/
            description.md (122 lines)
            sample_submission.csv (1909 lines)
            sample_submission.csv.zip (5.7 kB)
            test.csv (19 lines)
            test.csv.zip (748 Bytes)
            test.zip (1.2 GB)
            train.csv (1395 lines)
            train.csv.zip (23.6 kB)
            train.zip (12.7 GB)
            osic-pulmonary-fibrosis-progression/
                description.md (122 lines)
                sample_submission.csv (1909 lines)
                ... and 7 other files
                osic-pulmonary-fibrosis-progression/
                test/
                    ID00014637202177757139317/
                        1.dcm (1.5 MB)
                        10.dcm (1.5 MB)
                        ... and 29 other files
                    ID00019637202178323708467/
                        1.dcm (525.5 kB)
                        10.dcm (525.5 kB)
                        ... and 27 other files
                    ... and 17 other folders
                train/
                    ID00007637202177411956430/
                        1.dcm (525.6 kB)
                        10.dcm (525.6 kB)
                        ... and 28 other files
                    ID00009637202177434476278/
                        1.dcm (1.2 MB)
                        10.dcm (1.2 MB)
                        ... and 392 other files
                    ... and 157 other folders
            test/
                ID00014637202177757139317/
                    1.dcm (1.5 MB)
                    10.dcm (1.5 MB)
                    ... and 29 other files
                ID00019637202178323708467/
                    1.dcm (525.5 kB)
                    10.dcm (525.5 kB)
                    ... and 27 other files
                ... and 17 other folders
            train/
                ID00007637202177411956430/
                    1.dcm (525.6 kB)
                    10.dcm (525.6 kB)
                    ... and 28 other files
                ID00009637202177434476278/
                    1.dcm (1.2 MB)
                    10.dcm (1.2 MB)
                    ... and 392 other files
                ... and 157 other folders
        working/
            osic-pulmonary-fibrosis-progression/
                description.md (122 lines)
                sample_submission.csv (1909 lines)
                ... and 7 other files
                osic-pulmonary-fibrosis-progression/
                test/
                    ID00014637202177757139317/
                        1.dcm (1.5 MB)
                        10.dcm (1.5 MB)
                        ... and 29 other files
                    ID00019637202178323708467/
                        1.dcm (525.5 kB)
                        10.dcm (525.5 kB)
                        ... and 27 other files
                    ... and 17 other folders
                train/
                    ID00007637202177411956430/
                        1.dcm (525.6 kB)
                        10.dcm (525.6 kB)
                        ... and 28 other files
                    ID00009637202177434476278/
                        1.dcm (1.2 MB)
                        10.dcm (1.2 MB)
                        ... and 392 other files
                    ... and 157 other folders
```

-> data/osic-pulmonary-fibrosis-progression/sample_submission.csv has 1908 rows and 3 columns.
The columns are: Patient_Week, FVC, Confidence

-> data/osic-pulmonary-fibrosis-progression/test.csv has 18 rows and 7 columns.
The columns are: Patient, Weeks, FVC, Percent, Age, Sex, SmokingStatus

-> data/osic-pulmonary-fibrosis-progression/train.csv has 1394 rows and 7 columns.
The columns are: Patient, Weeks, FVC, Percent, Age, Sex, SmokingStatus

-> data/sample_submission.csv has 1908 rows and 3 columns.
The columns are: Patient_Week, FVC, Confidence

-> data/test.csv has 18 rows and 7 columns.
The columns are: Patient, Weeks, FVC, Percent, Age, Sex, SmokingStatus

-> data/train.csv has 1394 rows and 7 columns.
The columns are: Patient, Weeks, FVC, Percent, Age, Sex, SmokingStatus

-> (stopped after 10 files for performance)

# 5. Target score

-18.975826401998745

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 1
import os, glob
from tqdm import tqdm

import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt

import tensorflow as tf
import keras
from keras import layers
from keras.optimizers import Adam
from keras.losses import MeanSquaredError

import pydicom

IMG_WIDTH, IMG_HEIGHT, IMG_DEPTH = 128, 128, 64
MIN_WEEK, MAX_WEEK = -12, 133
SIGMA_MIN, DELTA_MAX = 70, 1000
SQRT_2 = tf.constant(tf.sqrt(2.))

import kagglehub
DIR = '../input/osic-pulmonary-fibrosis-progression'
SUBMISSION_DIR = '.'


## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 2
DEVs = tf.config.list_physical_devices()
tf.config.set_visible_devices(DEVs)
print(tf.config.get_visible_devices())

## === cell 4
df = pd.read_csv(f'{DIR}/train.csv') \
    .reset_index(drop=True) \
    .groupby(['Patient', 'Weeks']) \
    .agg({
        'FVC': 'mean',
        'Percent': 'mean',
        'Age': 'first',
        'Sex': 'first',
        'SmokingStatus': 'first',
    }) \
    .reset_index()

df

## === cell 5
def path_scans(patient_id):
    paths = glob.glob(f'{DIR}/train/{patient_id}/*.dcm')
    keys = (int(path.split('/')[-1].split('.')[0]) for path in paths)
    paths_keys = sorted(zip(paths, keys), key=lambda x: x[1])
    paths = [elem[0] for elem in paths_keys]

    img = pydicom.dcmread(paths[0])
    return paths, (img.Columns, img.Rows), len(paths)

def patients(df):
    def fvc_agg(g):
        pos = g[g['Weeks'] >= 0].sort_values(by='Weeks', ascending=True)
        neg = g[g['Weeks'] < 0].sort_values(by='Weeks', ascending=False)

        fvc = 0
        if pos.iloc[0]['Weeks'] == 0:
            fvc = pos.iloc[0]['FVC']
        else:
            if neg.shape[0] > 0:
                (x1, y1) = neg.iloc[0][['Weeks', 'FVC']]
                (x2, y2) = pos.iloc[0][['Weeks', 'FVC']]

                fvc = (x2 * y1 - x1 * y2) / (x2 - x1)
            else:
                (x1, y1) = pos.iloc[0][['Weeks', 'FVC']]
                fvc = y1

        fvc_full = pos.iloc[0]['FVC'] / pos.iloc[0]['Percent'] * 100
        return pd.Series({'FVC_0': fvc, 'FVC_full': fvc_full, 'Ratio_0': fvc / fvc_full})

    df_fvc_0 = df.groupby(['Patient'])[['Weeks', 'FVC', 'Percent']].apply(fvc_agg).reset_index()
    df_patients = df[['Patient', 'Age', 'Sex', 'SmokingStatus']].drop_duplicates().reset_index(drop=True)
    df_patients[['ScanDim', 'ScanDepth']] = df_patients['Patient'].apply(lambda _id: path_scans(_id)[1:]).apply(pd.Series)
    return df_patients.merge(df_fvc_0, on='Patient').set_index('Patient')

df_patients = patients(df)
df_patients

## === cell 6
def eda_scans():
    fig, axes = plt.subplots(ncols=2, nrows=1, figsize=(10, 5))
    df_patients.value_counts('ScanDim').plot.barh(ax=axes[0])
    df_patients['ScanDepth'].hist(ax=axes[1], bins=25)

    avg_depth = df_patients['ScanDepth'].mean()
    axes[1].axvline(avg_depth, color='r')

    axes[0].set_title('Image dimensions (W x H)')
    axes[1].set_title('Image scan depths (D)')
    plt.show()

eda_scans()


## === cell 8
def eda():
    fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(20, 5))
    ages = df_patients['Age']
    ages_m = df_patients.loc[df_patients['Sex'] ==   'Male', 'Age']
    ages_f = df_patients.loc[df_patients['Sex'] == 'Female', 'Age']
    sexes = df_patients['Sex'].value_counts()
    statuses = df_patients['SmokingStatus'].value_counts()

    axes[0].hist(x=ages)
    axes[1].hist(x=[ages_m, ages_f], label=['Male', 'Female'])
    axes[1].legend()
    axes[2].bar(x=sexes.index, height=sexes.values, color=['tab:blue', 'tab:orange'])
    axes[3].bar(x=statuses.index, height=statuses.values)

    axes[0].set_title('Distribution of age')
    axes[1].set_title('Distribution of age for each gender')
    axes[2].set_title('Distribution of gender')
    axes[3].set_title('Distribution of SmokingStatus')
    plt.show()

eda()

## === cell 9
def progression(status, color, ax=None):
    patients_with_status = df_patients.loc[df_patients['SmokingStatus'] == status] \
        .sample(10, replace=True).index

    for patient_id in patients_with_status:
        df_patient = df.loc[df['Patient'] == patient_id]

        weeks, fvcs = df_patient[['Weeks', 'Percent']].T.values
        ax.set_xlabel('Weeks')
        ax.set_ylabel('Percent')
        ax.plot(weeks, fvcs, color=color)
        ax.tick_params(axis='y', labelcolor=color)

fig, ax1 = plt.subplots()
progression('Ex-smoker', color='tab:orange', ax=ax1)
progression('Never smoked', color='tab:blue', ax=ax1)
progression('Currently smokes', color='tab:red', ax=ax1)

fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.show()

## === cell 10
df_fvc_ratios = df.pivot(index='Patient', columns=['Weeks'], values=['FVC']) \
    .droplevel(0, axis=1) \
    .reindex(columns=range(MIN_WEEK, MAX_WEEK + 1)) \
    .interpolate(axis='columns') \
    .bfill(axis=1)

df_fvc_ratios

## === cell 11
class OSICDataset(tf.keras.utils.Sequence):
    def __init__(self, mode, DIR, csv_file,
                 depth=64,
                 batch_size=1,
                 input_size=(128, 128, 64),
                 shuffle=True):

        self.mode = mode
        self.df = pd.read_csv(f'{DIR}/{csv_file}') \
            .reset_index(drop=True) \
            .groupby(['Patient', 'Weeks']) \
            .agg({
                'FVC': 'mean',
                'Percent': 'mean',
                'Age': 'first',
                'Sex': 'first',
                'SmokingStatus': 'first',
            }) \
            .reset_index()

        self.df_fvc_ratios = self._fvc_ratios()
        self.df_patients = self._patients()

        self.depth = depth
        self.batch_size = batch_size
        self.input_size = input_size
        self.shuffle = shuffle
        self.n = len(self.df_patients)

    def _patients(self):
        def __path_scans(patient_id):
            paths = glob.glob(f'{DIR}/{self.mode}/{patient_id}/*.dcm')

            keys = (int(path.split('/')[-1].split('.')[0]) for path in paths)
            paths_keys = sorted(zip(paths, keys), key=lambda x: x[1])
            paths = [elem[0] for elem in paths_keys]

            img = pydicom.dcmread(paths[0])
            return paths, (img.Columns, img.Rows), len(paths)

        def __fvc_full(g):



            fvc_full = g.iloc[0]['FVC'] / g.iloc[0]['Percent'] * 100
            return pd.Series({'FVC_full': fvc_full})

        df_fvc_full = self.df.groupby(['Patient'])[['Weeks', 'FVC', 'Percent']] \
            .apply(__fvc_full).reset_index()
        df_patients = self.df[['Patient', 'Age', 'Sex', 'SmokingStatus']] \
            .drop_duplicates().reset_index(drop=True)
        df_patients[['ScanDim', 'ScanDepth']] = df_patients['Patient'] \
            .apply(lambda _id: __path_scans(_id)[1:]) \
            .apply(pd.Series)

        df_patients = df_patients.merge(df_fvc_full, on='Patient').set_index('Patient')
        return df_patients

    def _fvc_ratios(self):
        return self.df.pivot(index='Patient', columns=['Weeks'], values=['FVC']) \
            .droplevel(0, axis=1) \
            .reindex(columns=range(MIN_WEEK, MAX_WEEK + 1)) \
            .interpolate(axis='columns') \
            .bfill(axis=1)

    def on_epoch_end(self):
        pass

    def __getitem__(self, idx):
        sta = idx * self.batch_size
        fin = min(sta + self.batch_size, len(self.df_patients))
        df_patients = self.df_patients.iloc[sta:fin]

        imgs_3d = None
        for patient_id in df_patients.index:
            paths = glob.glob(f'{DIR}/{self.mode}/{patient_id}/*.dcm')
            keys = (int(path.split('/')[-1].split('.')[0]) for path in paths)
            paths = sorted(zip(paths, keys), key=lambda x: x[1])

            img_3d = None
            for idx in range(self.depth):
                pos = int(idx * len(paths) / self.depth)
                filename = paths[pos][0]
                img = pydicom.dcmread(filename).pixel_array.astype(np.float32)
                img = img / (np.max(img) + 0.001)
                img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
                img = np.expand_dims(img, -1)

                if img_3d is None:
                    img_3d = img
                else:
                    img_3d = np.concatenate((img_3d, img), axis=-1)

            img_3d = np.expand_dims(img_3d, 0)
            if imgs_3d is None:
                imgs_3d = img_3d
            else:
                imgs_3d = np.concatenate((imgs_3d, img_3d), axis=0)

        if self.mode == 'train':
            df_fvc_ratios = self.df_fvc_ratios.loc[df_patients.index.tolist()]
            return (imgs_3d, df_patients, df_fvc_ratios)
        else:
            return (imgs_3d, df_patients)

    def __len__(self):
        return (self.n + self.batch_size - 1) // self.batch_size

osic_dataset = OSICDataset('train', DIR, 'train.csv')
osic_dataset

## === cell 13

def cnn_3d_regression_features(width, height, depth):
    """Build a 3D convolutional neural network model."""

    inputs = keras.Input((width, height, depth, 1))

    x = layers.Conv3D(filters=64, kernel_size=3, activation="relu")(inputs)
    x = layers.MaxPool3D(pool_size=2)(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv3D(filters=64, kernel_size=3, activation="relu")(inputs)
    x = layers.MaxPool3D(pool_size=2)(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv3D(filters=64, kernel_size=3, activation="relu")(x)
    x = layers.MaxPool3D(pool_size=2)(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv3D(filters=128, kernel_size=3, activation="relu")(x)
    x = layers.MaxPool3D(pool_size=2)(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv3D(filters=256, kernel_size=3, activation="relu")(x)
    x = layers.MaxPool3D(pool_size=2)(x)
    x = layers.BatchNormalization()(x)

    x = layers.GlobalAveragePooling3D()(x)
    feature_vector = layers.Dense(units=256, activation="relu", name="feature_vector")(x)
    x = layers.Dropout(0.3)(feature_vector)

    outputs = layers.Dense(units=1, name="fvc_0")(x)

    model = keras.Model(inputs, [outputs, feature_vector], name="cnn_3d_regression_features")
    return model

cnn_3d_model = cnn_3d_regression_features(width=IMG_WIDTH, height=IMG_HEIGHT, depth=IMG_DEPTH)
cnn_3d_model.summary()

## === cell 14
class OSICTimeSeriesDataset(OSICDataset):
    def __init__(self, mode, DIR, csv_file, **kwargs):
        super().__init__(mode, DIR, csv_file, **kwargs)

    def __getitem__(self, idx):
        if self.mode == 'train':
            (img_3d, df_patients, df_fvc_ratios) = super().__getitem__(idx)
            return img_3d, df_fvc_ratios.to_numpy()
        else:
            (img_3d, df_patients) = super().__getitem__(idx)
            return img_3d, df_patients

osic_time_series_dataset = OSICTimeSeriesDataset('train', DIR, 'train.csv', batch_size=8)
print(osic_time_series_dataset[0][0].shape)
print(osic_time_series_dataset[0][1].shape)

## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
RuntimeError                              Traceback (most recent call last)
/tmp/ipykernel_11/2187057846.py in <cell line: 0>()
     12 
     13 osic_time_series_dataset = OSICTimeSeriesDataset('train', DIR, 'train.csv', batch_size=8)
---> 14 print(osic_time_series_dataset[0][0].shape)
     15 print(osic_time_series_dataset[0][1].shape)

/tmp/ipykernel_11/2187057846.py in __getitem__(self, idx)
      5     def __getitem__(self, idx):
      6         if self.mode == 'train':
----> 7             (img_3d, df_patients, df_fvc_ratios) = super().__getitem__(idx)
      8             return img_3d, df_fvc_ratios.to_numpy()
      9         else:

/tmp/ipykernel_11/4036977981.py in __getitem__(self, idx)
    101                 pos = int(idx * len(paths) / self.depth)
    102                 filename = paths[pos][0]
--> 103                 img = pydicom.dcmread(filename).pixel_array.astype(np.float32)
    104                 img = img / (np.max(img) + 0.001)
    105                 img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))

/usr/local/lib/python3.11/dist-packages/pydicom/dataset.py in pixel_array(self)
   2191             that iterates through the image frames.
   2192         """
-> 2193         self.convert_pixel_data()
   2194         return cast("numpy.ndarray", self._pixel_array)
   2195 

/usr/local/lib/python3.11/dist-packages/pydicom/dataset.py in convert_pixel_data(self, handler_name)
   1724             # Use 'pydicom.pixels' backend
   1725             opts["decoding_plugin"] = name
-> 1726             self._pixel_array = pixel_array(self, **opts)
   1727             self._pixel_id = get_image_pixel_ids(self)
   1728         else:

/usr/local/lib/python3.11/dist-packages/pydicom/pixels/utils.py in pixel_array(src, ds_out, specific_tags, index, raw, decoding_plugin, **kwargs)
   1428 
   1429         opts = as_pixel_options(ds, **kwargs)
-> 1430         return decoder.as_array(
   1431             ds,
   1432             index=index,

/usr/local/lib/python3.11/dist-packages/pydicom/pixels/decoders/base.py in as_array(self, src, index, validate, raw, decoding_plugin, **kwargs)
    980             cast(
    981                 dict[str, "DecodeFunction"],
--> 982                 self._validate_plugins(decoding_plugin),
    983             ),
    984         )

/usr/local/lib/python3.11/dist-packages/pydicom/pixels/common.py in _validate_plugins(self, plugin)
    255         missing = "\n".join([f"\t{s}" for s in self.missing_dependencies])
    256         if self._decoder:
--> 257             raise RuntimeError(
    258                 f"Unable to decompress '{self.UID.name}' pixel data because all "
    259                 f"plugins are missing dependencies:\n{missing}"

RuntimeError: Unable to decompress 'JPEG Lossless, Non-Hierarchical, First-Order Prediction (Process 14 [Selection Value 1])' pixel data because all plugins are missing dependencies:
	gdcm - requires gdcm>=3.0.10
	pylibjpeg - requires pylibjpeg>=2.0 and pylibjpeg-libjpeg>=2.1

## === cell 15
def full_model(width, height, depth, activation=['linear', 'linear'], hidden_units=64, dense_units=112):
    """Build a 3D convolutional neural network model."""


    cnn_3d_model = cnn_3d_regression_features(width, height, depth)
    inputs = cnn_3d_model.input
    _, x = cnn_3d_model.outputs
    x = layers.Reshape((1, 256))(x)
    x = layers.LSTM(hidden_units, activation=activation[0])(x)
    outputs = layers.Dense(units=dense_units, activation='relu', name="output")(x)

    model = keras.Model(inputs, outputs, name="full_model")
    return model

model = full_model(width=IMG_WIDTH, height=IMG_HEIGHT, depth=IMG_DEPTH, dense_units=MAX_WEEK-MIN_WEEK+1)
model.summary()

## === cell 16
@keras.saving.register_keras_serializable()
def competition_metric(y_true, y_pred):
    """OSIC Competition Metric"""
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    print(f'y_true.shape = {y_true.shape}')
    print(f'y_pred.shape = {y_pred.shape}')
    print(f'y_true = {y_true[0]}')
    print(f'y_pred = {y_pred[0]}')

    sigma = y_pred[:, 2] - y_pred[:, 0]
    fvc_pred = y_pred[:, 1]

    sigma_clipped = tf.maximum(sigma, SIGMA_MIN)
    delta = tf.abs(y_true[:, 0] - fvc_pred)
    delta_clipped = tf.minimum(delta, DELTA_MAX)

    metric = (delta_clipped / sigma_clipped) * SQRT_2 + \
                tf.math.log(sigma_clipped * SQRT_2)

    return tf.keras.backend.mean(metric)







## === cell 17
tf.keras.backend.clear_session()
with tf.device('/GPU:0'):
    model.compile(optimizer='adam', loss='mean_absolute_error', metrics=[competition_metric], run_eagerly=True)
    history = model.fit(osic_time_series_dataset, epochs=1)
    model.save('251117-cnn-rnn.keras')

## --- ERROR in cell 17, traceback:
---------------------------------------------------------------------------
RuntimeError                              Traceback (most recent call last)
/tmp/ipykernel_11/860958506.py in <cell line: 0>()
      2 with tf.device('/GPU:0'):
      3     model.compile(optimizer='adam', loss='mean_absolute_error', metrics=[competition_metric], run_eagerly=True)
----> 4     history = model.fit(osic_time_series_dataset, epochs=1)
      5     model.save('251117-cnn-rnn.keras')

/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py in error_handler(*args, **kwargs)
    120             # To get the full stack trace, call:
    121             # `keras.config.disable_traceback_filtering()`
--> 122             raise e.with_traceback(filtered_tb) from None
    123         finally:
    124             del filtered_tb

/tmp/ipykernel_11/2187057846.py in __getitem__(self, idx)
      5     def __getitem__(self, idx):
      6         if self.mode == 'train':
----> 7             (img_3d, df_patients, df_fvc_ratios) = super().__getitem__(idx)
      8             return img_3d, df_fvc_ratios.to_numpy()
      9         else:

/tmp/ipykernel_11/4036977981.py in __getitem__(self, idx)
    101                 pos = int(idx * len(paths) / self.depth)
    102                 filename = paths[pos][0]
--> 103                 img = pydicom.dcmread(filename).pixel_array.astype(np.float32)
    104                 img = img / (np.max(img) + 0.001)
    105                 img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))

/usr/local/lib/python3.11/dist-packages/pydicom/dataset.py in pixel_array(self)
   2191             that iterates through the image frames.
   2192         """
-> 2193         self.convert_pixel_data()
   2194         return cast("numpy.ndarray", self._pixel_array)
   2195 

/usr/local/lib/python3.11/dist-packages/pydicom/dataset.py in convert_pixel_data(self, handler_name)
   1724             # Use 'pydicom.pixels' backend
   1725             opts["decoding_plugin"] = name
-> 1726             self._pixel_array = pixel_array(self, **opts)
   1727             self._pixel_id = get_image_pixel_ids(self)
   1728         else:

/usr/local/lib/python3.11/dist-packages/pydicom/pixels/utils.py in pixel_array(src, ds_out, specific_tags, index, raw, decoding_plugin, **kwargs)
   1428 
   1429         opts = as_pixel_options(ds, **kwargs)
-> 1430         return decoder.as_array(
   1431             ds,
   1432             index=index,

/usr/local/lib/python3.11/dist-packages/pydicom/pixels/decoders/base.py in as_array(self, src, index, validate, raw, decoding_plugin, **kwargs)
    980             cast(
    981                 dict[str, "DecodeFunction"],
--> 982                 self._validate_plugins(decoding_plugin),
    983             ),
    984         )

/usr/local/lib/python3.11/dist-packages/pydicom/pixels/common.py in _validate_plugins(self, plugin)
    255         missing = "\n".join([f"\t{s}" for s in self.missing_dependencies])
    256         if self._decoder:
--> 257             raise RuntimeError(
    258                 f"Unable to decompress '{self.UID.name}' pixel data because all "
    259                 f"plugins are missing dependencies:\n{missing}"

RuntimeError: Unable to decompress 'JPEG Lossless, Non-Hierarchical, First-Order Prediction (Process 14 [Selection Value 1])' pixel data because all plugins are missing dependencies:
	gdcm - requires gdcm>=3.0.10
	pylibjpeg - requires pylibjpeg>=2.0 and pylibjpeg-libjpeg>=2.1

## === cell 18
plt.plot(history.history['loss'])
plt.title('Model Loss per epochs')
plt.xticks(range(len(history.history['loss'])))
plt.xlabel('Epoch')
plt.ylabel('MAE')
plt.savefig('loss_per_epochs.png')
plt.show()

## --- ERROR in cell 18, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1057039776.py in <cell line: 0>()
----> 1 plt.plot(history.history['loss'])
      2 plt.title('Model Loss per epochs')
      3 plt.xticks(range(len(history.history['loss'])))
      4 plt.xlabel('Epoch')
      5 plt.ylabel('MAE')

NameError: name 'history' is not defined

## === cell 19
x, y = osic_time_series_dataset[0]
y_pred = model.predict(x)

plt.plot(y_pred[0])
plt.plot(y[0])
plt.show()

## --- ERROR in cell 19, traceback:
---------------------------------------------------------------------------
RuntimeError                              Traceback (most recent call last)
/tmp/ipykernel_11/841503638.py in <cell line: 0>()
----> 1 x, y = osic_time_series_dataset[0]
      2 y_pred = model.predict(x)
      3 
      4 plt.plot(y_pred[0])
      5 plt.plot(y[0])

/tmp/ipykernel_11/2187057846.py in __getitem__(self, idx)
      5     def __getitem__(self, idx):
      6         if self.mode == 'train':
----> 7             (img_3d, df_patients, df_fvc_ratios) = super().__getitem__(idx)
      8             return img_3d, df_fvc_ratios.to_numpy()
      9         else:

/tmp/ipykernel_11/4036977981.py in __getitem__(self, idx)
    101                 pos = int(idx * len(paths) / self.depth)
    102                 filename = paths[pos][0]
--> 103                 img = pydicom.dcmread(filename).pixel_array.astype(np.float32)
    104                 img = img / (np.max(img) + 0.001)
    105                 img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))

/usr/local/lib/python3.11/dist-packages/pydicom/dataset.py in pixel_array(self)
   2191             that iterates through the image frames.
   2192         """
-> 2193         self.convert_pixel_data()
   2194         return cast("numpy.ndarray", self._pixel_array)
   2195 

/usr/local/lib/python3.11/dist-packages/pydicom/dataset.py in convert_pixel_data(self, handler_name)
   1724             # Use 'pydicom.pixels' backend
   1725             opts["decoding_plugin"] = name
-> 1726             self._pixel_array = pixel_array(self, **opts)
   1727             self._pixel_id = get_image_pixel_ids(self)
   1728         else:

/usr/local/lib/python3.11/dist-packages/pydicom/pixels/utils.py in pixel_array(src, ds_out, specific_tags, index, raw, decoding_plugin, **kwargs)
   1428 
   1429         opts = as_pixel_options(ds, **kwargs)
-> 1430         return decoder.as_array(
   1431             ds,
   1432             index=index,

/usr/local/lib/python3.11/dist-packages/pydicom/pixels/decoders/base.py in as_array(self, src, index, validate, raw, decoding_plugin, **kwargs)
    980             cast(
    981                 dict[str, "DecodeFunction"],
--> 982                 self._validate_plugins(decoding_plugin),
    983             ),
    984         )

/usr/local/lib/python3.11/dist-packages/pydicom/pixels/common.py in _validate_plugins(self, plugin)
    255         missing = "\n".join([f"\t{s}" for s in self.missing_dependencies])
    256         if self._decoder:
--> 257             raise RuntimeError(
    258                 f"Unable to decompress '{self.UID.name}' pixel data because all "
    259                 f"plugins are missing dependencies:\n{missing}"

RuntimeError: Unable to decompress 'JPEG Lossless, Non-Hierarchical, First-Order Prediction (Process 14 [Selection Value 1])' pixel data because all plugins are missing dependencies:
	gdcm - requires gdcm>=3.0.10
	pylibjpeg - requires pylibjpeg>=2.0 and pylibjpeg-libjpeg>=2.1

## === cell 21
osic_time_series_dataset_test = OSICTimeSeriesDataset('test', DIR, 'test.csv', batch_size=2)
osic_time_series_dataset_test

## === cell 22
patients_tests = []

for idx in range(len(osic_time_series_dataset_test)):
    imgs, patients = osic_time_series_dataset_test[idx]
    with tf.device('/GPU:0'):
        y_test_pred = model.predict(imgs)
    patients_test = patients.reset_index()[['Patient', 'FVC_full']]

    df_y_test_pred = pd.DataFrame(y_test_pred)
    df_y_test_pred.columns += MIN_WEEK

    patients_test = pd.concat([patients_test, df_y_test_pred], axis=1) \
        .melt(id_vars=['Patient', 'FVC_full'], var_name='Week', value_name='Percent')
    patients_test['FVC'] = patients_test['FVC_full'] * patients_test['Percent'] / 100
    patients_test['Confidence'] = 100

    patients_test['Patient_Week'] = patients_test['Patient'] + '_' + patients_test['Week'].astype('str')
    patients_tests.append(patients_test)

df_patients_test = pd.concat(patients_tests).set_index('Patient_Week')
df_patients_test

## === cell 23
df_patients_test[['FVC', 'Confidence']] \
    .to_csv(f'{SUBMISSION_DIR}/submission.csv')

## --- ERROR in outputing the csv:
Invalid submission: Patient_Week ID00014637202177757139317_-12 in submission does not exist in answers
