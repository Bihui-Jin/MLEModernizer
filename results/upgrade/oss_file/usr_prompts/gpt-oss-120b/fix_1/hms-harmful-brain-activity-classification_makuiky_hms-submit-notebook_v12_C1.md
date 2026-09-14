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
Detect and classify harmful brain activity in electroencephalography (EEG) data: seizure (SZ), generalized periodic discharges (GPD), lateralized periodic discharges (LPD), lateralized rhythmic delta activity (LRDA), generalized rhythmic delta activity (GRDA), or "other".

## Metric
Kullback Liebler divergence between the predicted probability and the observed target.

## Submission Format
For each `eeg_id` in the test set, you must predict a probability for each of the `vote` columns. The file should contain a header and have the following format:

```
eeg_id,seizure_vote,lpd_vote,gpd_vote,lrda_vote,grda_vote,other_vote\
0,0.166,0.166,0.167,0.167,0.167,0.167\
1,0.166,0.166,0.167,0.167,0.167,0.167\
etc.
```

Your total predicted probabilities for each row must sum to one or your submission will fail.

## Dataset
**train.csv** Metadata for the train set. The expert annotators reviewed 50 second long EEG samples plus matched spectrograms covering 10 a minute window centered at the same time and labeled the central 10 seconds. Many of these samples overlapped and have been consolidated. `train.csv` provides the metadata that allows you to extract the original subsets that the raters annotated.

- `eeg_id` - A unique identifier for the entire EEG recording.
- `eeg_sub_id` - An ID for the specific 50 second long subsample this row's labels apply to.
- `eeg_label_offset_seconds` - The time between the beginning of the consolidated EEG and this subsample.
- `spectrogram_id` - A unique identifier for the entire EEG recording.
- `spectrogram_sub_id` - An ID for the specific 10 minute subsample this row's labels apply to.
- `spectogram_label_offset_seconds` - The time between the beginning of the consolidated spectrogram and this subsample.
- `label_id` - An ID for this set of labels.
- `patient_id` - An ID for the patient who donated the data.
- `expert_consensus` - The consensus annotator label. Provided for convenience only.
- `[seizure/lpd/gpd/lrda/grda/other]_vote` - The count of annotator votes for a given brain activity class. The full names of the activity classes are as follows: `lpd`: lateralized periodic discharges, `gpd`: generalized periodic discharges, `lrd`: lateralized rhythmic delta activity, and `grda`: generalized rhythmic delta activity . A detailed explanations of these patterns is [available here.](https://www.acns.org/UserFiles/file/ACNSStandardizedCriticalCareEEGTerminology_rev2021.pdf)

**test.csv** Metadata for the test set. As there are no overlapping samples in the test set, many columns in the train metadata don't apply.

- `eeg_id`
- `spectrogram_id`
- `patient_id`

**sample_submission.csv**

- `eeg_id`
- `[seizure/lpd/gpd/lrda/grda/other]_vote` - The target columns. Your predictions must be probabilities. Note that the test samples had between 3 and 20 annotators.

**train_eegs/** EEG data from one or more overlapping samples. Use the metadata in train.csv to select specific annotated subsets. The column names are [the names of the individual electrode locations for EEG leads](https://en.wikipedia.org/wiki/10%E2%80%9320_system_%28EEG%29), with one exception. The EKG column is for an electrocardiogram lead that records data from the heart. All of the EEG data (for both train and test) was collected at a frequency of 200 samples per second.

**test_eegs/** Exactly 50 seconds of EEG data.

train_spectrograms/ Spectrograms assembled EEG data. Use the metadata in train.csv to select specific annotated subsets. The column names indicate the frequency in hertz and the recording regions of the EEG electrodes. The latter are abbreviated as LL = left lateral; RL = right lateral; LP = left parasagittal; RP = right parasagittal.

**test_spectrograms/** Spectrograms assembled using exactly 10 minutes of EEG data.

**example_figures/** Larger copies of the example case images used on the overview tab.

# 2. Python version

3.12

# 3. Installed packages

geopandas==0.14.4
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
pillow==11.3.0
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

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (166 lines)
            example_figures.zip (14.8 MB)
            sample_submission.csv (9851 lines)
            sample_submission.csv.zip (19.0 kB)
            test.csv (9851 lines)
            test.csv.zip (22.8 kB)
            test_eegs.zip (1.5 GB)
            test_spectrograms.zip (346.5 MB)
            train.csv (96951 lines)
            train.csv.zip (1.6 MB)
            train_eegs.zip (14.4 GB)
            train_spectrograms.zip (3.2 GB)
            example_figures/
                Sample01.pdf (914.3 kB)
                Sample02.pdf (703.4 kB)
                ... and 18 other files
            hms-harmful-brain-activity-classification/
                description.md (166 lines)
                example_figures.zip (14.8 MB)
                ... and 10 other files
                example_figures/
                    Sample01.pdf (914.3 kB)
                    Sample02.pdf (703.4 kB)
                    ... and 18 other files
                hms-harmful-brain-activity-classification/
                test_eegs/
                    1001717358.parquet (3.1 MB)
                    1003353736.parquet (972.5 kB)
                    ... and 1691 other files
                test_spectrograms/
                    1002209002.parquet (713.8 kB)
                    1005228554.parquet (648.5 kB)
                    ... and 1112 other files
                train_eegs/
                    1000913311.parquet (980.2 kB)
                    1001369401.parquet (1.2 MB)
                    ... and 15394 other files
                train_spectrograms/
                    1000086677.parquet (564.7 kB)
                    1000189855.parquet (672.7 kB)
                    ... and 10022 other files
            test_eegs/
                1001717358.parquet (3.1 MB)
                1003353736.parquet (972.5 kB)
                ... and 1691 other files
            test_spectrograms/
                1002209002.parquet (713.8 kB)
                1005228554.parquet (648.5 kB)
                ... and 1112 other files
            train_eegs/
                1000913311.parquet (980.2 kB)
                1001369401.parquet (1.2 MB)
                ... and 15394 other files
            train_spectrograms/
                1000086677.parquet (564.7 kB)
                1000189855.parquet (672.7 kB)
                ... and 10022 other files
        input/
            description.md (166 lines)
            example_figures.zip (14.8 MB)
            sample_submission.csv (9851 lines)
            sample_submission.csv.zip (19.0 kB)
            test.csv (9851 lines)
            test.csv.zip (22.8 kB)
            test_eegs.zip (1.5 GB)
            test_spectrograms.zip (346.5 MB)
            train.csv (96951 lines)
            train.csv.zip (1.6 MB)
            train_eegs.zip (14.4 GB)
            train_spectrograms.zip (3.2 GB)
            example_figures/
                Sample01.pdf (914.3 kB)
                Sample02.pdf (703.4 kB)
                ... and 18 other files
            hms-harmful-brain-activity-classification/
                description.md (166 lines)
                example_figures.zip (14.8 MB)
                ... and 10 other files
                example_figures/
                    Sample01.pdf (914.3 kB)
                    Sample02.pdf (703.4 kB)
                    ... and 18 other files
                hms-harmful-brain-activity-classification/
                test_eegs/
                    1001717358.parquet (3.1 MB)
                    1003353736.parquet (972.5 kB)
                    ... and 1691 other files
                test_spectrograms/
                    1002209002.parquet (713.8 kB)
                    1005228554.parquet (648.5 kB)
                    ... and 1112 other files
                train_eegs/
                    1000913311.parquet (980.2 kB)
                    1001369401.parquet (1.2 MB)
                    ... and 15394 other files
                train_spectrograms/
                    1000086677.parquet (564.7 kB)
                    1000189855.parquet (672.7 kB)
                    ... and 10022 other files
            test_eegs/
                1001717358.parquet (3.1 MB)
                1003353736.parquet (972.5 kB)
                ... and 1691 other files
            test_spectrograms/
                1002209002.parquet (713.8 kB)
                1005228554.parquet (648.5 kB)
                ... and 1112 other files
            train_eegs/
                1000913311.parquet (980.2 kB)
                1001369401.parquet (1.2 MB)
                ... and 15394 other files
            train_spectrograms/
                1000086677.parquet (564.7 kB)
                1000189855.parquet (672.7 kB)
                ... and 10022 other files
        working/
            hms-harmful-brain-activity-classification/
                description.md (166 lines)
                example_figures.zip (14.8 MB)
                ... and 10 other files
                example_figures/
                    Sample01.pdf (914.3 kB)
                    Sample02.pdf (703.4 kB)
                    ... and 18 other files
                hms-harmful-brain-activity-classification/
                test_eegs/
                    1001717358.parquet (3.1 MB)
                    1003353736.parquet (972.5 kB)
                    ... and 1691 other files
                test_spectrograms/
                    1002209002.parquet (713.8 kB)
                    1005228554.parquet (648.5 kB)
                    ... and 1112 other files
                train_eegs/
                    1000913311.parquet (980.2 kB)
                    1001369401.parquet (1.2 MB)
                    ... and 15394 other files
                train_spectrograms/
                    1000086677.parquet (564.7 kB)
                    1000189855.parquet (672.7 kB)
                    ... and 10022 other files
```

-> data/hms-harmful-brain-activity-classification/sample_submission.csv has 9850 rows and 7 columns.
The columns are: eeg_id, seizure_vote, lpd_vote, gpd_vote, lrda_vote, grda_vote, other_vote

-> data/hms-harmful-brain-activity-classification/test.csv has 9850 rows and 3 columns.
The columns are: spectrogram_id, eeg_id, patient_id

-> data/hms-harmful-brain-activity-classification/train.csv has 96950 rows and 15 columns.
The columns are: eeg_id, eeg_sub_id, eeg_label_offset_seconds, spectrogram_id, spectrogram_sub_id, spectrogram_label_offset_seconds, label_id, patient_id, expert_consensus, seizure_vote, lpd_vote, gpd_vote, lrda_vote, grda_vote, other_vote

-> data/sample_submission.csv has 9850 rows and 7 columns.
The columns are: eeg_id, seizure_vote, lpd_vote, gpd_vote, lrda_vote, grda_vote, other_vote

-> data/test.csv has 9850 rows and 3 columns.
The columns are: spectrogram_id, eeg_id, patient_id

-> data/train.csv has 96950 rows and 15 columns.
The columns are: eeg_id, eeg_sub_id, eeg_label_offset_seconds, spectrogram_id, spectrogram_sub_id, spectrogram_label_offset_seconds, label_id, patient_id, expert_consensus, seizure_vote, lpd_vote, gpd_vote, lrda_vote, grda_vote, other_vote

-> (stopped after 10 files for performance)

# 5. Target score

0.9209651751737288

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

import os

import tensorflow as tf
from tensorflow import keras

from PIL import Image
import gc



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 2
EEG_TEST_PATH = '/kaggle/input/hms-harmful-brain-activity-classification/test_eegs/'
SPEC_TEST_PATH = '/kaggle/input/hms-harmful-brain-activity-classification/test_spectrograms/'

if not os.path.exists('/kaggle/working/test_eegs_img/'):
    os.makedirs('/kaggle/working/test_eegs_img/')
EEG_IMG_TEST_PATH = '/kaggle/working/test_eegs_img/'

if not os.path.exists('/kaggle/working/test_spec_img/'):
    os.makedirs('/kaggle/working/test_spec_img/')
SPEC_IMG_TEST_PATH='/kaggle/working/test_spec_img/'

META_TEST = '/kaggle/input/hms-harmful-brain-activity-classification/test.csv'

## === cell 4
eeg_zone={
    'Cz-Pz' : ['Cz', 'Pz'],
    'Fz-Cz' : ['Fz', 'Cz'],
    
    'P4-O2' : ['P4', 'O2'],
    'C4-P4' : ['C4', 'P4'],
    'F4-C4' : ['F4', 'C4'],
    'Fp2-F4': ['Fp2', 'F4'],
    
    'P3-O1' : ['P3', 'O1'],
    'C3-P3' : ['C3', 'P3'],
    'F3-C3' : ['F3', 'C3'],
    'Fp1-F3' : ['Fp1', 'F3'],
    
    'T6-O2' : ['T6','O2'],
    'T4-T6' : ['T4', 'T6'],
    'F8-T4' : ['F8', 'T4'],
    'Fp2-F8' : ['Fp2', 'F8'],
    
    'T5-O1' : ['T5', 'O1'],
    'T3-T5' : ['T3', 'T5'],
    'F7-T3' : ['F7', 'T3'],
    'Fp1-F7' : ['Fp1', 'F7'],
}

## === cell 5
def generate_eeg(eegid,input_path=EEG_TEST_PATH,eeg_zone=eeg_zone,linewidth=0.2, eeg_out=EEG_IMG_TEST_PATH):
    eeg = pd.read_parquet(f'{input_path}{eegid}.parquet')
    
    ysticks=[]
    labels= []
    cicles= 0
    relpos= 0
    fig, ax = plt.subplots(1,1, figsize=(3,3), sharex= True)
    
    for key, values in eeg_zone.items():
        ax.plot(eeg.index/200, eeg[values[0]]-eeg[values[1]]+relpos,color='black',linewidth=linewidth)
        
        ysticks.append(relpos)
        labels.append(key)
        if cicles==1 or cicles==5 or cicles==9 or cicles==13:
            relpos+=200
        else:
            relpos+=40
            
        cicles+=1
       
    ax.set_xticks([])
    ax.set_yticks([])
    
    ax.set_xlim(0,50)
    save_path= f'{eeg_out}{eegid}.jpeg' 
    fig.savefig(save_path,bbox_inches='tight', dpi=100)
    plt.close()
    
    return  save_path

## === cell 7
spec_zones=['LL','RL','LP','RP']

def generate_spectrogram(specid,
                         input_path=SPEC_TEST_PATH, spec_out=SPEC_IMG_TEST_PATH,
                         spec_zones=spec_zones,
                        output_filter=0.5):
    
    spec = pd.read_parquet(f'{input_path}{specid}.parquet')
    spec = spec.fillna(0)
    spec=spec.set_index('time')
    spec=spec.T
    spec['column']=spec.index.str.split('_', expand=True)
    
    spec['freq'] = spec.column.apply(lambda x: x[1]).astype(float)
    spec['brainreg'] = spec.column.apply(lambda x: x[0]).astype(str)
    
    spec=spec.drop('column', axis=1)
    spec.set_index('freq',inplace=True)
    subspec=dict()
    for zone in spec_zones:
        subspec[f'{zone}_sub']=spec[spec.brainreg==zone]
        subspec[f'{zone}_sub']= subspec[f'{zone}_sub'].drop('brainreg', axis=1)
    
    
    fig, ax = plt.subplots(nrows=len(spec_zones), figsize=(3,3), sharex=True)
    for row in range(len(spec_zones)):
        data=subspec[f'{spec_zones[row]}_sub']
        ax[row].imshow(data, cmap='turbo', 
                       aspect='auto', 
                       origin='lower', 
                       extent=[data.columns.min(),data.columns.max(),data.index.min(),data.index.max()],
                      vmin=0,vmax=data.max().max()*output_filter)
        
        ax[row].set_xticks([])
        ax[row].set_yticks([])
        
    plt.subplots_adjust(hspace=0.01)
    
    save_path= f'{spec_out}{specid}.jpeg'
    fig.savefig(save_path,bbox_inches='tight', dpi=100)
    plt.close()
    return save_path

## === cell 8
metadata = pd.read_csv(META_TEST)
train_metadata = pd.read_csv('/kaggle/input/hms-harmful-brain-activity-classification/train.csv')

## === cell 9
def drop_images(paths):
    for path in paths:
        os.remove(path)

## === cell 10
def preprocess_data(metadata,row,train=False):
    if train:
        eeg_path = generate_eeg(eegid=metadata.loc[row].eeg_id,input_path='/kaggle/input/hms-harmful-brain-activity-classification/train_eegs/')
        spec_path = generate_spectrogram(specid=metadata.loc[row].spectrogram_id,output_filter=0.7,input_path='/kaggle/input/hms-harmful-brain-activity-classification/train_spectrograms/')
    else:
        eeg_path=generate_eeg(eegid=metadata.loc[row].eeg_id)
        spec_path=generate_spectrogram(specid=metadata.loc[row].spectrogram_id,output_filter=0.7)
    
    
    eeg_img = Image.open(eeg_path)
    spec_img = Image.open(spec_path)
    
    eeg_img_tr = np.array(eeg_img)
    spec_img_tr = np.array(spec_img)
        
    
        
    eeg_img_tr =tf.convert_to_tensor(eeg_img)
    spec_img_tr =tf.convert_to_tensor(spec_img)
    
    eeg_img_tr =tf.image.resize(eeg_img_tr, [224,224])
    spec_img_tr = tf.image.resize(spec_img_tr, [240,240])
    
    eeg_img_tr = tf.expand_dims(eeg_img_tr, axis=0)
    spec_img_tr = tf.expand_dims(spec_img_tr, axis=0)
    
    drop_images(paths=[eeg_path,spec_path])
    eeg_id = metadata.loc[row].eeg_id
    del metadata
    return eeg_img_tr,spec_img_tr, eeg_id

## === cell 11
model = tf.keras.models.load_model("/kaggle/input/hms-models/best_model_st_6_ft.keras")

## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_10/3776192834.py in <cell line: 0>()
----> 1 model = tf.keras.models.load_model("/kaggle/input/hms-models/best_model_st_6_ft.keras")

/usr/local/lib/python3.11/dist-packages/keras/src/saving/saving_api.py in load_model(filepath, custom_objects, compile, safe_mode)
    198         )
    199     elif str(filepath).endswith(".keras"):
--> 200         raise ValueError(
    201             f"File not found: filepath={filepath}. "
    202             "Please ensure the file is an accessible `.keras` "

ValueError: File not found: filepath=/kaggle/input/hms-models/best_model_st_6_ft.keras. Please ensure the file is an accessible `.keras` zip file.

## === cell 12
submission={
    'eeg_id' : [],
    'seizure_vote' : [],
    'lpd_vote' : [],
    'gpd_vote' : [],
    'lrda_vote' : [],
    'grda_vote' : [],
    'other_vote' : [],
}
iter_dict = ['seizure_vote','lpd_vote','gpd_vote','lrda_vote','grda_vote','other_vote']

%matplotlib agg
for row in metadata.index:
    
    eeg_img, spec_img, eeg_id = preprocess_data(metadata,row)
    prediction = model.predict([eeg_img,spec_img],verbose=1)
    prediction = prediction.squeeze()
    submission['eeg_id'].append(eeg_id)
    counter=0
    for illness in iter_dict:
        submission[illness].append(prediction[counter])
        counter+=1
        
pdsubmit=pd.DataFrame(submission)

## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_10/3839087891.py in <cell line: 0>()
     14 
     15     eeg_img, spec_img, eeg_id = preprocess_data(metadata,row)
---> 16     prediction = model.predict([eeg_img,spec_img],verbose=1)
     17     prediction = prediction.squeeze()
     18     submission['eeg_id'].append(eeg_id)

NameError: name 'model' is not defined

## === cell 13
pdsubmit

## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_10/3690183632.py in <cell line: 0>()
----> 1 pdsubmit

NameError: name 'pdsubmit' is not defined

## === cell 14
pdsubmit.to_csv('submission.csv',index=False)

## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_10/312508593.py in <cell line: 0>()
----> 1 pdsubmit.to_csv('submission.csv',index=False)

NameError: name 'pdsubmit' is not defined
