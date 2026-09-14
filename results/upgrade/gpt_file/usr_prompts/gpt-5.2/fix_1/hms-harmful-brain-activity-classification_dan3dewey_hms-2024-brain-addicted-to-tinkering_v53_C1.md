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
pyarrow==19.0.1
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
sklearn-pandas==2.2.0

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

1.015208464728662

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 4
NUM_CLUSTS = 9      # 6 to 10

SMOOTH_WIDTH = 5    # Odd>1: 3,5,7,9,...
TRAIN_DOWNSEL = 23  # small values for code test; set to 1 to output all.
VALID_DOWNSEL = 47  #  "
USE_PREPROC = True   # Read in saved meta and features frames

USE_LR1 = True
LR1_C = 0.10     # smaller --> fewer non-zero coeff.s
USE_LR2 = True
LR2_C = 0.10
LR_BLUR = 0.10



above_dir = "../input/hms-harmful-brain-activity-classification/"
above_dir_preproc = "../input/hms-2024-brain-data/"


## === cell 6
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

import pyarrow
import pyarrow.parquet as pq
import pyarrow.dataset as pads

from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import LogisticRegression

## === cell 8
HBA_number = 6
HBA_names = ["seizure", "lpd", "gpd", "lrda", "grda", "other"]
HBA_expert_names = ["Seizure","LPD","GPD","LRDA","GRDA","Other"]
iHBA_of_expert = {"Seizure":0,"LPD":1,"GPD":2,"LRDA":3,"GRDA":4,"Other":5}
HBA_votes = ["seizure_vote", "lpd_vote", "gpd_vote",
             "lrda_vote", "grda_vote", "other_vote"]
HBA_probs = ["seizure_prob", "lpd_prob", "gpd_prob",
             "lrda_prob", "grda_prob", "other_prob"]
the4chains = ["LL","RL","LP","RP"]

np.set_printoptions(precision=6, suppress=True)

## === cell 9
def kld_score(solution, submission):
    '''
    Calculate the average KL divergence score.
    Ignores the "row id" assumed in the first column.
    '''
    sumsum = 0.0
    for prob_col in solution.columns.values:
        sumsum += np.nansum(-1.0*solution[prob_col] *
                        np.log(submission[prob_col] / solution[prob_col]))
    return sumsum/(len(solution))

## === cell 10
def read_hms_meta():
    '''
    Read in the train.csv and test.csv files.
    Add total_vote, _prob columns, and vote entropy to train_meta.
    Add extra cols to test to allow the same processing as train:
        eeg[spectro]_sub_id, eeg[spectro]_label_offset_seconds, label_id
    Make various plots of the train_meta values.
    '''

    test_meta = pd.read_csv(above_dir+"test.csv")
    test_meta_len = len(test_meta)
    print("Test has length", test_meta_len)
    test_meta["eeg_sub_id"] = 0
    test_meta["eeg_label_offset_seconds"] = 0.0
    test_meta["spectrogram_sub_id"] = 0
    test_meta["spectrogram_label_offset_seconds"] = 0.0
    test_meta["label_id"] = test_meta.eeg_id
    if test_meta_len > 1:
        REAL_TEST = True
    else:
        REAL_TEST = False
        print("  --> not the real LB test data.\n")
        pass
  
    train_meta = pd.read_csv(above_dir+"train.csv")
    train_meta_len = len(train_meta)
    print("Train has length", train_meta_len, " with:")
    
    train_meta["total_vote"] = ( train_meta["seizure_vote"] +
                    train_meta["lpd_vote"] + train_meta["gpd_vote"] +
                    train_meta["lrda_vote"] + train_meta["grda_vote"] +
                                    train_meta["other_vote"] )
    train_meta["max_vote"] = np.max(np.array([train_meta["seizure_vote"] ,
                    train_meta["lpd_vote"] , train_meta["gpd_vote"] ,
                    train_meta["lrda_vote"] , train_meta["grda_vote"] ,
                    train_meta["other_vote"]]), axis=0)
    
    for this_col in ["label_id","eeg_id","spectrogram_id",
                     "patient_id","total_vote"]:
        print("   ", len(train_meta[this_col].unique()),
            "unique "+this_col+" values.")
    
    print("\nHistogram of the total votes")
    plt.figure(figsize=(6,3))
    plt.hist(train_meta["total_vote"],bins=55,log=True)
    plt.title("Histogram of Total Votes")
    plt.show()
    
    allvt = train_meta.expert_consensus.value_counts()
    less9 = train_meta[train_meta["total_vote"] < 9].expert_consensus.value_counts()
    more9 = train_meta[train_meta["total_vote"] > 9].expert_consensus.value_counts()
    vnot3 = train_meta[train_meta["total_vote"] != 3].expert_consensus.value_counts()
    
    for col_pre in HBA_names:
        train_meta[col_pre + "_prob"] = (train_meta[col_pre + "_vote"] / 
                                     train_meta["total_vote"] )
        if False:
            plt.figure(figsize=(6,1.5))
            plt.hist(train_meta[col_pre + "_prob"],bins=20,range=(0.02,1))
            plt.ylim(0,len(train_meta)/5)
            plt.title("Histogram of   "+col_pre+"_prob")
            plt.show()
        
    print("Calculating voting entropy values ...")
    def calc_entropy(row):
        the_probs = np.clip(row[16:21+1].values.astype(float), 1.e-8,1.0)
        return np.nansum(the_probs * -1*np.log(the_probs))
    train_meta["entropy"] = train_meta.apply(calc_entropy, axis=1)
    
    return train_meta, test_meta


## === cell 11
def prob_prob_scatter(name1, name2, probs2plot, clust_ids, iclust_order=[0]):
    '''
    Make a prob1 vs prob2 scatter plot.
    - name1, name2 are 2 of the 6 HBA expert_consensus labels.
    - probs2plot is 6-column dataframe, e.g., train_meta[HBA_probs]
    - clust_ids is an array of cluster id integers, e.g., train_meta["clust_id"]
    Include an x at the cluster centers in the chosen axes.
    Use sqrt scaling to emphasize lower values.
    External: HBA_probs, iHBA_of_expert[ ], clust_centers
    '''
    hba_clrs = ["orange","blue","red","black","green","purple",
              "green","red","blue","orange"]  # up to 10 clusters
    if len(iclust_order) > 2:
        kmclrs = hba_clrs.copy()
        for iord, iclust in enumerate(iclust_order):
            kmclrs[iclust] = hba_clrs[iord]
   
    clstclrs = []
    for ilab in clust_ids:
        clstclrs.append(kmclrs[ilab])
        
    ixax = iHBA_of_expert[name1]
    iyax = iHBA_of_expert[name2]
    lenprob = len(probs2plot)
    plt.figure(figsize=(5,5))
    plt.scatter(np.sqrt(probs2plot[HBA_probs[ixax]]) + 
                0.04*(np.random.rand(lenprob)-0.5),
            np.sqrt(probs2plot[HBA_probs[iyax]]) + 
                0.04*(np.random.rand(lenprob)-0.5),
           s=3, c=clstclrs, alpha=0.02)
    for iclust in range(0,len(clust_centers)):
        plt.plot(np.sqrt([clust_centers[iclust,ixax]]),
                 np.sqrt([clust_centers[iclust,iyax]]),
                 c=kmclrs[iclust],marker="x",markersize=15)
    plt.xlabel("sqrt( "+name1+" )")
    plt.ylabel("sqrt( "+name2+" )")
    plt.show()
    return kmclrs  # returns cluster colors appropriate for iclust

## === cell 12
def assemble_features(meta_frame, traintest="train", smooth_width=5,
                     SHOW_PLOT=True):
    '''
    Create a dataframe of spectrogram features from the meta_frame rows.
    Will include clust_id (i.e, the y) if it is in the input meta_frame.
    Assumes these are available: above_dir, num_clusts
    '''
    freqs = np.array(range(100))*0.19525 + 0.59
    spect_trend = 150.0/(1.0**2.3+freqs**(2.3))
    freqs[0] = 0.0   # separate the (unsmoothed) first bin from others
    freqs4 = np.array(4*list(freqs))
    spect_trend4 = np.array(4*list(spect_trend))
    if SHOW_PLOT:
        plt.figure(figsize=(10,8))
    feats_frame=[]
    last_spectro_id_str = "starting"
    print_every_nth = max([100, 100*int(0.5+len(meta_frame.index)/(100.0*15))])
    for irow in meta_frame.index:
        this_row = meta_frame.loc[irow]
        spectro_id_str = str(int(this_row.spectrogram_id)) # make sure int
        if spectro_id_str != last_spectro_id_str:
            if traintest != 'test':
                spectro_file = above_dir+"train_spectrograms/"+spectro_id_str+".parquet"
            else:
                spectro_file = above_dir+"test_spectrograms/"+spectro_id_str+".parquet"
            pads_spectro = pads.dataset(spectro_file)
            this_spectro = pads_spectro.to_table().to_pandas()
        last_spectro_id_str = spectro_id_str
        loc_offset = int(this_row.spectrogram_label_offset_seconds/2)
        middle4s = ((this_spectro.iloc[loc_offset+148, 1:] +
                     this_spectro.iloc[loc_offset+149, 1:] +
                     this_spectro.iloc[loc_offset+150, 1:] +
                     this_spectro.iloc[loc_offset+151, 1:])/
                        (4.0*spect_trend4)) # 4 copies of trend
        middle4s = np.clip(middle4s,0.001,1000.0)
        middle4s = middle4s.replace([np.nan, -np.inf, np.inf], 0.001)
        ratio4s = ((this_spectro.iloc[loc_offset+148, 1:] +
                    this_spectro.iloc[loc_offset+149, 1:] +
                    this_spectro.iloc[loc_offset+150, 1:] +
                    this_spectro.iloc[loc_offset+151, 1:]) /
                       (this_spectro.iloc[loc_offset+149-56, 1:] +
                        this_spectro.iloc[loc_offset+149-40, 1:] +
                        this_spectro.iloc[loc_offset+149-24, 1:] +
                        this_spectro.iloc[loc_offset+149+56, 1:] +
                        this_spectro.iloc[loc_offset+149+40, 1:] +
                        this_spectro.iloc[loc_offset+149+24, 1:]))
        ratio4s = np.clip((6./4.)*ratio4s,0.01,100.0)
        ratio4s = ratio4s.replace([np.nan, -np.inf, np.inf], 1.0)
        ratio4spre = np.log10(ratio4s)
        spect_mean = np.mean(middle4s)  # over all 4 spectra
        spect_median = np.median(middle4s)
        the4means=[]
        the4medians = []
        for ispec in range(4):
            ibeg = ([0,100,200,300])[ispec]
            iend = ibeg+100
            the4means.append(np.mean(middle4s[ibeg:iend]))
            the4medians.append(np.median(middle4s[ibeg:iend]))
        middle4spre = np.log10(middle4s / spect_mean)
        middle4s = middle4spre.rolling(smooth_width, min_periods=smooth_width, 
                                        center=True, closed=None).mean()
        ratio4s = ratio4spre.rolling(smooth_width, min_periods=smooth_width, 
                                        center=True, closed=None).mean()
        for ioff in range(0, 400, 100):
            for ibin in range(int((smooth_width-1)/2)):  # assumes width is odd
                middle4s[ibin+ioff] = middle4spre[ibin+ioff]
                ratio4s[ibin+ioff] = ratio4spre[ibin+ioff]
        baseinds = np.insert(np.arange(int((smooth_width-1)/2), 100, smooth_width),
                             0, 0)
        select_inds = np.concatenate((baseinds,100+baseinds,
                                      200+baseinds,300+baseinds))
        freqs4ds = freqs4[select_inds]
        middle4sds = middle4s[select_inds]
        ratio4sds = ratio4s[select_inds]
        middle_feats = middle4sds.to_frame().T
        ratio_feats = ratio4sds.to_frame().T
        oldcols = ratio_feats.columns
        newcols = []
        for oldcol in oldcols:
            newcols.append("r"+oldcol)
        ratio_feats.columns = newcols
        these_feats = pd.concat([middle_feats, ratio_feats], axis=1)
        spect_mean = np.log10(spect_mean)
        spect_median = np.log10(spect_median)
        the4means = np.log10(the4means)
        the4medians = np.log10(the4medians)
        these_feats["Mean"] = spect_mean
        these_feats["Median"] = spect_median
        for ispec in range(4):
            these_feats[the4chains[ispec]+"mean"] = the4means[ispec]
            these_feats[the4chains[ispec]+"median"] = the4medians[ispec]
        if "clust_id" in meta_frame.columns:
            these_feats["clust_id"] = this_row.clust_id
        if len(feats_frame) == 0:
            feats_frame = these_feats.copy()
        else:
            feats_frame = pd.concat([feats_frame, these_feats])
        if len(feats_frame) % print_every_nth == 0:
            print("... {} done...".format(len(feats_frame)))
        if SHOW_PLOT:
            this_clr = "blue"
            title_start = "Middle-8s Feature Values of "+traintest
            plt.title(title_start+" (smooth={})".format(smooth_width))
            if "clust_id" in meta_frame.columns:
                this_clust = this_row.clust_id
                this_clr = kmclrs[this_clust]
                plt.title(title_start+" (color-coded by the "+
                          "{} clusters, smooth={})".format(
                              num_clusts, smooth_width))
            the_alpha = np.clip(0.05*350/len(meta_frame),0.003,0.5)
            plt.plot(freqs4ds+smooth_width*0.15*(np.random.rand(len(freqs4ds))-0.5),
                     middle4sds, ".",c=this_clr,markersize=10,alpha=the_alpha)
            plt.plot(-1.2+smooth_width*0.15*(np.random.rand(4)-0.5),
                     the4means+(0.0*(np.random.rand(4)-0.5)), ".",
                     c=this_clr,markersize=10,alpha=the_alpha)
            plt.ylim(-1.5,1.5)
            plt.xlabel("<-- Means are band < 0"+20*" "+"Frequency Bands (Hz)"+50*" ")
            plt.ylabel("Amplitude (/ref-spectrum, /mean, and smoothed n={})".format(
                                            smooth_width))
    if SHOW_PLOT:
        plt.savefig("middle8s_"+traintest+"_features.png")
        plt.show()
    return feats_frame.reset_index().drop(columns=["index"])
   

## === cell 13
def find_best_tamed_kl():
    '''
    Adjust the taming fraction for each cluster center to optimize KL
    Assumed inputs in environment:
        submission, pred_ids, solution
    Assumed useful values available:
        clust_centers, NUM_CLUSTS, HBA_number, HBA_votes
    '''
    mean_all_probs = np.array([0.208319, 0.132120, 0.128532,
                           0.138913, 0.179294, 0.212822])
    tamed_fracs = 0.0 * np.ones(NUM_CLUSTS)
    tamed_centers = clust_centers.copy()
    for iclust in range(NUM_CLUSTS):
        tamed_centers[iclust,:] = mean_all_probs
    for iclust in range(NUM_CLUSTS):
        last_kl = 10.0
        for this_frac in np.arange(0.03, 1.00, 0.05):  # 0.03--0.98
            tamed_fracs[iclust] = this_frac
            this_cent = (tamed_fracs[iclust]*clust_centers[iclust,:] +
                         (1.0-tamed_fracs[iclust])*mean_all_probs)
            tamed_centers[iclust,:] = this_cent

            for iprob in range(HBA_number):
                this_col_probs = tamed_centers[: , iprob]
                submission[HBA_votes[iprob]] = this_col_probs[pred_ids]
            this_kl = kld_score(solution, submission)
            if (this_kl < last_kl):
                best_fracs = tamed_fracs.copy()
                best_centers = tamed_centers.copy()
                last_kl = this_kl
            else:
                tamed_fracs[iclust] = best_fracs[iclust]
                tamed_centers[iclust, :] = best_centers[iclust, :]
                break
    return best_fracs, best_centers

## === cell 15
train_meta, test_meta = read_hms_meta()

## === cell 16
plt.figure(figsize=(5,2))
plt.hist(train_meta["spectrogram_sub_id"],bins=55,log=True)
plt.title("Histogram of spectrogram_sub_id")
plt.show()

plt.figure(figsize=(5,2))
plt.hist(train_meta["eeg_sub_id"],bins=55,log=True)
plt.title("Histogram of eeg_sub_id")
plt.show()

## === cell 18
num_clusts = NUM_CLUSTS   # 6 to 10

clust_rows_bool =(train_meta.eeg_sub_id < 200)

## === cell 19
prob_vectors = train_meta.loc[clust_rows_bool, HBA_probs]

print("\nUsing {} HBA samples for clustering.".format(len(prob_vectors)))
print("These include {} unique eeg_ids".format(
                train_meta.loc[clust_rows_bool, "eeg_id"].nunique()),
                "and {} unique patient ids.".format(
                train_meta.loc[clust_rows_bool, "patient_id"].nunique()))
plt.figure(figsize=(6,3))
plt.hist(train_meta.loc[clust_rows_bool,"total_vote"],bins=55,log=True)
plt.title("Histogram of total_vote in HBA samples clustered")
plt.show()
    
prob_array = np.array(prob_vectors)
kmeans = KMeans(n_clusters = num_clusts, init = 'k-means++', n_init = 10,
                    max_iter = 300, random_state=None)
kmeans.fit(prob_array)

clust_centers = kmeans.cluster_centers_
for iclust in range(HBA_number):
    clust_centers[iclust ,:] = (clust_centers[iclust ,:]/
                            np.sum(clust_centers[iclust ,:]))
print("cluster centers:")
print(clust_centers)

iclust_of_order = []
for icol in range(HBA_number):
    iclust_of_order.append(np.argmax(clust_centers[:,icol]))
clust_by_max = np.argsort(-1*np.max(clust_centers,axis=1))
for iord in range(HBA_number,num_clusts):
    iclust_of_order.append(clust_by_max[iord])    
    
kmnames = HBA_expert_names.copy()
for ihyb in range(1,(num_clusts - HBA_number)+1):
    kmnames.append("Hybrid-"+str(ihyb))


## === cell 20

train_meta["clust_id"] = kmeans.predict(np.array(train_meta[HBA_probs]))

all_probs = train_meta[HBA_probs]
all_ids = train_meta["clust_id"]

if True:
    kmclrs = prob_prob_scatter("Seizure","GPD", all_probs, all_ids, iclust_of_order)
    
    kmclrs = prob_prob_scatter("LPD","GRDA", all_probs, all_ids, iclust_of_order)

    kmclrs = prob_prob_scatter("LRDA","Other", all_probs, all_ids, iclust_of_order)

clust_counts = train_meta.clust_id.value_counts()
    
iorder_of_clust = num_clusts*[-1]
for iord, iclust in enumerate(iclust_of_order):
    iorder_of_clust[iclust] = iord
    if iord == 0: print("The close-to-unit-vector cluster centers:")
    if iord == 6: print("The Hybrid cluster centers:")
    plt.figure(figsize=(5,1))
    plt.bar(HBA_expert_names, clust_centers[iclust,:],color=kmclrs[iclust])
    plt.ylim(-0.01,1.01)
    plt.title(kmnames[iord]+"  kmclust={} has {} samples".format(
        iclust,clust_counts[iclust]), size='medium')
    if iord < 5: plt.xticks([])
    plt.show()

## === cell 22
solution_train = train_meta[["eeg_id"] + HBA_votes]
for col_pre in HBA_names:
    solution_train.loc[:, col_pre + "_vote"] = train_meta[col_pre + "_prob"]

submission_train = solution_train.copy()
clust_ids = train_meta["clust_id"]
if False:
    clust_ids = np.random.choice(9, size=len(train_meta),
                             replace=True, p=None)
for iprob in range(HBA_number):
    this_col_probs = clust_centers[: , iprob]
    submission_train[HBA_votes[iprob]] = this_col_probs[clust_ids]

print("Score if HBA samples are correctly assigned cluster prob.s:",
      np.round(kld_score(solution_train, submission_train),4))

## === cell 24
smooth_width = 9   # Here smooth_width is just for looking,

spectro_meta = train_meta[clust_rows_bool].copy()
every_nth = 97


freqs = np.array(range(100))*0.19525 + 0.59
downsel_freqs = np.insert(freqs[int((smooth_width-1)/2):100:smooth_width], 0, freqs[0])

spect_trend = 150.0/(1.0**2.3+freqs**(2.3))


## === cell 25
if NUM_CLUSTS <10:
    fig = plt.figure(figsize=(10,10))
    gs = fig.add_gridspec(3, 3, hspace=0, wspace=0)
    (ax0, ax1, ax2), (ax3, ax4, ax5), (ax6, ax7, ax8) = gs.subplots(sharex='col', sharey='row')
    pltaxs = [ax0, ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8]
else:
    fig = plt.figure(figsize=(10,13.5))
    gs = fig.add_gridspec(4, 3, hspace=0, wspace=0)
    (ax0, ax1, ax2), (ax3, ax4, ax5), (ax6, ax7, ax8), (ax9, ax10, ax11) = gs.subplots(sharex='col', sharey='row')
    pltaxs = [ax0, ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9, ax10, ax11]
all_medians = []
all_means = []
all_clusts = []
print("Plotting {} x 4 processed spectra".format(int(len(spectro_meta)/every_nth)))
print_every_nth = max([100, 100*int(0.5+(len(spectro_meta)/every_nth)/(100.0*15))])
for ilocrow in range(0,len(spectro_meta),every_nth):
    this_row = spectro_meta.iloc[ilocrow]
    this_clust = this_row.clust_id
    if True:
        spectro_id_str = str(this_row.spectrogram_id)
        spectro_file = above_dir + "train_spectrograms/"+spectro_id_str+".parquet"
        pads_spectro = pads.dataset(spectro_file)
        this_spectro = pads_spectro.to_table().to_pandas()
        loc_offset = int(this_row.spectrogram_label_offset_seconds/2)
        for ispec in range(4):
            ibeg = ([1,101,201,301])[ispec]
            iend = ibeg+100
            middle4s = ((this_spectro.iloc[loc_offset+148, ibeg:iend] +
                        this_spectro.iloc[loc_offset+149, ibeg:iend] +
                        this_spectro.iloc[loc_offset+150, ibeg:iend] +
                        this_spectro.iloc[loc_offset+151, ibeg:iend])
                            /(4.0*spect_trend))
            middle4s = np.clip(middle4s,0.001,1000.0)
            middle4s = middle4s.replace([np.nan, -np.inf, np.inf], 0.001)
            if False:  # ***   ***
                ratio4s = ((this_spectro.iloc[loc_offset+148, ibeg:iend] +
                            this_spectro.iloc[loc_offset+149, ibeg:iend] +
                            this_spectro.iloc[loc_offset+150, ibeg:iend] +
                            this_spectro.iloc[loc_offset+151, ibeg:iend]) /
                       (this_spectro.iloc[loc_offset+149-56, ibeg:iend] +
                        this_spectro.iloc[loc_offset+149-40, ibeg:iend] +
                        this_spectro.iloc[loc_offset+149-24, ibeg:iend] +
                        this_spectro.iloc[loc_offset+149+56, ibeg:iend] +
                        this_spectro.iloc[loc_offset+149+40, ibeg:iend] +
                        this_spectro.iloc[loc_offset+149+24, ibeg:iend]))
                ratio4s = np.clip((6./4.)*ratio4s,0.01,100.0)
                ratio4s = ratio4s.replace([np.nan, -np.inf, np.inf], 1.0)
                ratio4s = np.log10(ratio4s)
            spect_mean = np.log10(np.mean(middle4s))
            all_means.append(spect_mean)
            spect_median = np.log10(np.median(middle4s))
            all_medians.append(spect_median)
            all_clusts.append(this_clust)
            middle4s = np.log10(middle4s)
            middle4spre = middle4s - spect_mean
            middle4s = middle4spre.rolling(smooth_width, min_periods=smooth_width, 
                                        center=True, closed=None).mean()
            for ibin in range(int((smooth_width-1)/2)):  # assumes width is odd
                middle4s[ibin] = middle4spre[ibin]
            pltaxs[iorder_of_clust[this_clust]].plot((freqs), middle4s,
                               c=kmclrs[this_clust],lw=3,alpha=0.01)
    if (ilocrow/every_nth + 1) % print_every_nth == 0:
        print("... {} done...".format(int(ilocrow/every_nth)+1))

for iax in range(NUM_CLUSTS):
    pltaxs[iax].text(7.5, 0.8, kmnames[iax], fontsize=16)
    pltaxs[iax].plot([0.0,20.0],[0.0,0.0],c='black',lw=3,alpha=0.2)
    pltaxs[iax].plot(downsel_freqs, len(downsel_freqs)*[0.0],'.k')
    pltaxs[iax].set_ylim(-1.0,1.0)  # log already taken
    pltaxs[iax].set_xlim(0.0,20.5)
    if (iax == 3):
        pltaxs[iax].set_ylabel("log10[ Spectra /RefSpectrum /Mean"+
                               " & smoothed]")
    if (iax == 6) or (iax == 7) or (iax == 8):
        pltaxs[iax].set_xlabel("Frequency (Hz)")


fig.suptitle("Spectra of middle 8s (color-coded by the {} clusters, smooth={})".format(
                                          num_clusts, smooth_width))
plt.savefig("middle8s_spectra.png")
plt.show()

## === cell 26

print("\nMedian of the Means(below): {:.4f}".format(np.median(all_means)))
plt.figure(figsize=(6,2))
plt.hist(np.clip(all_means,-2.0,2.0),bins=100)
plt.xlim(-2.05,2.05)
plt.xlabel("Mean")
plt.title("Histogram of the Means of the log(Spectra/Ref-spect)")
plt.show()

print("\nMedian of the Medians(below): {:.4f}".format(np.median(all_medians)))
plt.figure(figsize=(6,2))
plt.hist(np.clip(all_medians,-2.0,2.0),bins=100)
plt.xlim(-2.05,2.05)
plt.xlabel("log10( Median )")
plt.title("Histogram of the Medians of the log(Spectra/Ref-spect)")
plt.show()

mmclrs = []
for ilab in all_clusts:
    mmclrs.append(kmclrs[ilab])
plt.figure(figsize=(3,3))
plt.scatter(all_medians, all_means, s=2,c=mmclrs,alpha=0.3)
plt.xlim(-1.,1); plt.ylim(-1.,1)
plt.xlabel("Median")
plt.ylabel("Mean")
plt.show()

## === cell 28
train_rows_bool = (((train_meta.eeg_sub_id < 33+1) &
                     (train_meta.eeg_sub_id % 5 == 3)) |   # id=3,8,13,18,23,28,33
                    (train_meta.eeg_sub_id == 0) &
                      ((train_meta.eeg_id%23)%8 > 1)) # include odd and even eeg_ids
print("Number of Training rows:",sum(train_rows_bool))

valid_rows_bool = (((train_meta.eeg_sub_id < 44+1) &
                    (train_meta.eeg_sub_id % 19 == 6)) |  # id=6,25,44
                    (train_meta.eeg_sub_id == 0) &
                      ((train_meta.eeg_id%23)%8 < 2)) # include odd and even eeg_ids
print("Number of Validation rows:",sum(valid_rows_bool))

## === cell 30
if USE_PREPROC:
    Xy_train_meta = pd.read_csv(above_dir_preproc+"Xy_train_meta_v47.csv")
    Xy_train_feats = pd.read_csv(above_dir_preproc+"Xy_train_feats_v47.csv")
    Xy_train_meta["clust_id"] = kmeans.predict(np.array(Xy_train_meta[HBA_probs]))
    Xy_train_feats["clust_id"] = Xy_train_meta["clust_id"]
    SMOOTH_WIDTH = 5 
else:
    Xy_train_meta = (train_meta[train_rows_bool])[::TRAIN_DOWNSEL].copy()
    Xy_train_meta = Xy_train_meta.reset_index().drop(columns=["index"])
    print("Number of samples used for training =",len(Xy_train_meta))

    Xy_train_feats = assemble_features(Xy_train_meta, traintest="train",
                            smooth_width=SMOOTH_WIDTH, SHOW_PLOT=False)
    Xy_train_meta.to_csv("Xy_train_meta.csv", header=True,
                    index=False, float_format='%.6f')
    Xy_train_feats.to_csv("Xy_train_feats.csv", header=True,
                    index=False, float_format='%.6f')

## --- ERROR in cell 30, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/2732133715.py in <cell line: 0>()
      2 if USE_PREPROC:
      3     # Restore previously saved data frames "/kaggle/input/
----> 4     Xy_train_meta = pd.read_csv(above_dir_preproc+"Xy_train_meta_v47.csv")
      5     Xy_train_feats = pd.read_csv(above_dir_preproc+"Xy_train_feats_v47.csv")
      6     # Assign the current clusters instead of ones in the file

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

FileNotFoundError: [Errno 2] No such file or directory: '../input/hms-2024-brain-data/Xy_train_meta_v47.csv'

## === cell 31
Xy_train_feats

## --- ERROR in cell 31, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/860130366.py in <cell line: 0>()
      1 # The training features
----> 2 Xy_train_feats

NameError: name 'Xy_train_feats' is not defined

## === cell 33
if USE_PREPROC:
    Xy_valid_meta = pd.read_csv(above_dir_preproc+"Xy_valid_meta_v47.csv")
    Xy_valid_feats = pd.read_csv(above_dir_preproc+"Xy_valid_feats_v47.csv")
    Xy_valid_meta["clust_id"] = kmeans.predict(np.array(Xy_valid_meta[HBA_probs]))
    Xy_valid_feats["clust_id"] = Xy_valid_meta["clust_id"]
else:
    Xy_valid_meta = (train_meta[valid_rows_bool])[::VALID_DOWNSEL].copy()
    Xy_valid_meta = Xy_valid_meta.reset_index().drop(columns=["index"])
    print("Number of samples used for Validation =",len(Xy_valid_meta))

    Xy_valid_feats = assemble_features(Xy_valid_meta, traintest="validation",
                            smooth_width=SMOOTH_WIDTH, SHOW_PLOT=True)
    Xy_valid_meta.to_csv("Xy_valid_meta.csv", header=True,
                    index=False, float_format='%.6f')
    Xy_valid_feats.to_csv("Xy_valid_feats.csv", header=True,
                    index=False, float_format='%.6f')

## --- ERROR in cell 33, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/1773841022.py in <cell line: 0>()
      2 if USE_PREPROC:
      3     # Restore previously saved data frames "/kaggle/input/
----> 4     Xy_valid_meta = pd.read_csv(above_dir_preproc+"Xy_valid_meta_v47.csv")
      5     Xy_valid_feats = pd.read_csv(above_dir_preproc+"Xy_valid_feats_v47.csv")
      6     # Assign the current clusters instead of ones in the file

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

FileNotFoundError: [Errno 2] No such file or directory: '../input/hms-2024-brain-data/Xy_valid_meta_v47.csv'

## === cell 34
Xy_valid_feats

## --- ERROR in cell 34, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/520913205.py in <cell line: 0>()
      1 # The validation features
----> 2 Xy_valid_feats

NameError: name 'Xy_valid_feats' is not defined

## === cell 36


X = Xy_train_feats.drop(columns=["clust_id"])
y = Xy_train_feats.clust_id

Xlr = X.drop(columns=X.columns[-10:])

if USE_LR1:
    Xlr1 = Xlr.iloc[:, 0:int(len(Xlr.columns)/2)]
    lrmodel1 = LogisticRegression(penalty='l1', C=LR1_C, solver='saga', max_iter=1500,
                                multi_class='multinomial', # ovr or multinomial
                                n_jobs=-1).fit(Xlr1,y)

    plt.figure(figsize=(8,4))
    plt.plot(lrmodel1.coef_.T,'-',alpha=0.5)
    plt.plot(lrmodel1.coef_.T,'.',alpha=1.0)
    plt.title("LR1 coefficients, C="+str(LR1_C)+" (colored by cluster)")
    plt.show()

    print("\nLR1 model score for X,y = {:.1f}%\n".format(100*lrmodel1.score(Xlr1, y)))

if USE_LR2:
    Xlr2 = Xlr.iloc[:, int(len(Xlr.columns)/2):]
    lrmodel2 = LogisticRegression(penalty='l1', C=LR2_C, solver='saga', max_iter=1500,
                                multi_class='multinomial', # ovr or multinomial
                                n_jobs=-1).fit(Xlr2,y)

    plt.figure(figsize=(8,4))
    plt.plot(lrmodel2.coef_.T,'-',alpha=0.5)
    plt.plot(lrmodel2.coef_.T,'.',alpha=1.0)
    plt.title("LR2 coefficients, C="+str(LR2_C)+" (colored by cluster)")
    plt.show()

    print("\nLR2 model score for X,y = {:.1f}%\n".format(100*lrmodel2.score(Xlr2, y)))

## --- ERROR in cell 36, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2121349158.py in <cell line: 0>()
      8 # Train the LR model
      9 
---> 10 X = Xy_train_feats.drop(columns=["clust_id"])
     11 y = Xy_train_feats.clust_id
     12 

NameError: name 'Xy_train_feats' is not defined

## === cell 37

Xy_train_wLRfeats = Xy_train_feats.copy()
X = Xy_train_feats.drop(columns=["clust_id"])
Xlr = X.drop(columns=X.columns[-10:])
if USE_LR1:
    Xlr1 = Xlr.iloc[:, 0:int(len(Xlr.columns)/2)]
    lrprobas = lrmodel1.predict_proba(Xlr1)
    for iadd in range(NUM_CLUSTS):
        Xy_train_wLRfeats["lr"+str(iadd)] = (lrprobas[: , iadd] +
                            LR_BLUR*(np.random.rand(len(lrprobas))-0.5))
    plt.figure(figsize=(7,2.5))
    plt.hist(np.max(lrmodel1.predict_proba(Xlr1),axis=1),bins=50)
    plt.xlim(-0.5*LR_BLUR, 1.01)
    plt.title("Training max LR1 proba values (pre-blur)")
    plt.show()
if USE_LR2:
    Xlr2 = Xlr.iloc[:, int(len(Xlr.columns)/2):]
    lrprobas = lrmodel2.predict_proba(Xlr2)
    for iadd in range(NUM_CLUSTS):
        Xy_train_wLRfeats["rlr"+str(iadd)] = (lrprobas[: , iadd] +
                            LR_BLUR*(np.random.rand(len(lrprobas))-0.5))
    plt.figure(figsize=(7,2.5))
    plt.hist(np.max(lrmodel2.predict_proba(Xlr2),axis=1),bins=50)
    plt.xlim(-0.5*LR_BLUR, 1.01)
    plt.title("Training max LR2 proba values (pre-blur)")
    plt.show()


Xy_valid_wLRfeats = Xy_valid_feats.copy()
X = Xy_valid_feats.drop(columns=["clust_id"])
Xlr = X.drop(columns=X.columns[-10:])
if USE_LR1:
    Xlr1 = Xlr.iloc[:, 0:int(len(Xlr.columns)/2)]
    lrprobas = lrmodel1.predict_proba(Xlr1)
    for iadd in range(NUM_CLUSTS):
        Xy_valid_wLRfeats["lr"+str(iadd)] = (lrprobas[: , iadd] +
                            LR_BLUR*(np.random.rand(len(lrprobas))-0.5))
    plt.figure(figsize=(7,2.5))
    plt.hist(np.max(lrmodel1.predict_proba(Xlr1),axis=1),bins=50)
    plt.xlim(-0.5*LR_BLUR, 1.01)
    plt.title("Validation max LR1 proba values (pre-blur)")
    plt.show()
if USE_LR2:
    Xlr2 = Xlr.iloc[:, int(len(Xlr.columns)/2):]
    lrprobas = lrmodel2.predict_proba(Xlr2)
    for iadd in range(NUM_CLUSTS):
        Xy_valid_wLRfeats["rlr"+str(iadd)] = (lrprobas[: , iadd] +
                            LR_BLUR*(np.random.rand(len(lrprobas))-0.5))
    plt.figure(figsize=(7,2.5))
    plt.hist(np.max(lrmodel2.predict_proba(Xlr2),axis=1),bins=50)
    plt.xlim(-0.5*LR_BLUR, 1.01)
    plt.title("Validation max LR2 proba values (pre-blur)")
    plt.show()

## --- ERROR in cell 37, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/261132494.py in <cell line: 0>()
      3 
      4 # Train: Add these values as new columns in Xy_train_wLRfeats
----> 5 Xy_train_wLRfeats = Xy_train_feats.copy()
      6 X = Xy_train_feats.drop(columns=["clust_id"])
      7 Xlr = X.drop(columns=X.columns[-10:])

NameError: name 'Xy_train_feats' is not defined

## === cell 39

X = Xy_train_wLRfeats.drop(columns=["clust_id"])
y = Xy_train_wLRfeats.clust_id

ave_oob = []
nfits=3
for ifit in range(nfits):
    rfmodel = RandomForestClassifier(
                    n_estimators=300,  # was 100
                    min_samples_leaf=5,  # was 9
                    max_features=0.2,
                    max_samples=0.9,      # was 0.8
                    oob_score=True, class_weight="balanced_subsample",
                    n_jobs=-1, verbose=0, random_state=None).fit(X,y)
    ave_oob.append(rfmodel.oob_score_)
    
sort_inds = (rfmodel.feature_importances_.argsort())
plt.figure(figsize=(4,7))
plt.barh(rfmodel.feature_names_in_[sort_inds], rfmodel.feature_importances_[sort_inds])
plt.ylim(len(sort_inds)-40,len(sort_inds)+0.2)
plt.title("Feature Importances (top 40)")
plt.show()

print("\nRF model ave OOB score = {:.1f}% +/- {:.1f}".format(
        100*np.mean(ave_oob),100*np.std(ave_oob)))

print("\nRF model score for X,y = {:.1f}%\n".format(100*rfmodel.score(X, y)))

## --- ERROR in cell 39, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2209118696.py in <cell line: 0>()
     11 #    monotonic_cst=None)
     12 
---> 13 X = Xy_train_wLRfeats.drop(columns=["clust_id"])
     14 y = Xy_train_wLRfeats.clust_id
     15 

NameError: name 'Xy_train_wLRfeats' is not defined

## === cell 40

Xy_train_meta["pred_id"] = rfmodel.predict(Xy_train_wLRfeats.drop(columns=["clust_id"]))

maxprobs_train = np.max(rfmodel.predict_proba(X),axis=1)
plt.figure(figsize=(7,2.5))
plt.hist(maxprobs_train, bins=50)
plt.xlim(0.,1.)
plt.title("Histogram of max(proba) for model-training samples")
plt.show()

solution = Xy_train_meta[["eeg_id"] + HBA_votes]
for col_pre in HBA_names:
    solution.loc[:, col_pre + "_vote"] = Xy_train_meta[col_pre + "_prob"]

submission = solution.copy()
pred_ids = Xy_train_meta["pred_id"] # putting clust_id gives the ideal value


best_fracs, best_centers = find_best_tamed_kl()

for iprob in range(HBA_number):
    this_col_probs = best_centers[: , iprob]
    submission[HBA_votes[iprob]] = this_col_probs[pred_ids]
this_kl = kld_score(solution, submission)
print("Tamed fractions:\n", best_fracs, "\nand centers:\n", best_centers)
print("\nKL from tamed centers: {:.4f}".format(this_kl))

## --- ERROR in cell 40, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1823717333.py in <cell line: 0>()
      2 
      3 # Make and add the predicted clusters to Xy_train_meta
----> 4 Xy_train_meta["pred_id"] = rfmodel.predict(Xy_train_wLRfeats.drop(columns=["clust_id"]))
      5 
      6 # How high is the probability of the assigned (maximum prob) class for each X?

NameError: name 'rfmodel' is not defined

## === cell 42
Xy_valid_meta["pred_id"] = rfmodel.predict(Xy_valid_wLRfeats.drop(columns=["clust_id"]))

maxprobs_valid = np.max(rfmodel.predict_proba(
                    Xy_valid_wLRfeats.drop(columns=["clust_id"])),axis=1)
plt.figure(figsize=(7,2.5))
plt.hist(maxprobs_valid, bins=50)
plt.xlim(0.,1.)
plt.title("Histogram of max(proba) for validation samples")
plt.show()

solution = Xy_valid_meta[["eeg_id"] + HBA_votes]
for col_pre in HBA_names:
    solution.loc[:, col_pre + "_vote"] = Xy_valid_meta[col_pre + "_prob"]

submission = solution.copy()
pred_ids = Xy_valid_meta["pred_id"]

best_fracs, best_centers = find_best_tamed_kl()

for iprob in range(HBA_number):
    this_col_probs = best_centers[: , iprob]
    submission[HBA_votes[iprob]] = this_col_probs[pred_ids]
this_kl = kld_score(solution, submission)
print("Tamed fractions:\n", best_fracs, "\nand centers:\n", best_centers)
print("\nKL from tamed centers: {:.4f}".format(this_kl))

## --- ERROR in cell 42, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3907961707.py in <cell line: 0>()
      1 # Make and add the predictions to Xy_valid_meta
----> 2 Xy_valid_meta["pred_id"] = rfmodel.predict(Xy_valid_wLRfeats.drop(columns=["clust_id"]))
      3 # For reference, randomly assign pred_ids to see zero-accuracy result
      4 ##Xy_valid_meta["pred_id"] = np.random.choice(num_clusts,
      5 ##                        size=len(Xy_valid_meta),replace=True, p=None)

NameError: name 'rfmodel' is not defined

## === cell 45
if False:
    X = Xy_train_wLRfeats.drop(columns=["clust_id"])
    y = Xy_train_wLRfeats.clust_id
    solution = Xy_valid_meta[["eeg_id"] + HBA_votes]
    for col_pre in HBA_names:
        solution.loc[:, col_pre + "_vote"] = Xy_valid_meta[col_pre + "_prob"]
    submission = solution.copy()

if False:
    rfmodscan = RandomForestClassifier(
                    n_estimators=300,  # was 100
                    min_samples_leaf=5,  # was 9
                    max_features=0.2,
                    max_samples=0.9,      # was 0.8
                    oob_score=True, class_weight="balanced_subsample",
                    n_jobs=-1, verbose=0, random_state=None).fit(X,y)
    print("   _ _ _ scan value =",str(scan_this),":")
    print("RF train score = {:.1f}%".format(100*rfmodscan.score(X, y)))
    Xy_valid_meta["pred_id"] = rfmodscan.predict(Xy_valid_wLRfeats.drop(
                                    columns=["clust_id"]))
    pred_ids = Xy_valid_meta["pred_id"]
    best_fracs, best_centers = find_best_tamed_kl()
    for iprob in range(HBA_number):
        this_col_probs = best_centers[: , iprob]
        submission[HBA_votes[iprob]] = this_col_probs[pred_ids]
    this_kl = kld_score(solution, submission)
    print("Validation KL with tamed centers: {:.4f}".format(this_kl))

## === cell 48

Xy_test_feats = assemble_features(test_meta, traintest="test",
                            smooth_width=SMOOTH_WIDTH, SHOW_PLOT=False)

Xy_test_wLRfeats = Xy_test_feats.copy()
Xlr = Xy_test_feats.drop(columns=Xy_test_feats.columns[-10:])
if USE_LR1:
    Xlr1 = Xlr.iloc[:, 0:int(len(Xlr.columns)/2)]
    lrprobas = lrmodel1.predict_proba(Xlr1)
    for iadd in range(NUM_CLUSTS):
        Xy_test_wLRfeats["lr"+str(iadd)] = (lrprobas[: , iadd] +
                            LR_BLUR*(np.random.rand(len(lrprobas))-0.5))
if USE_LR2:
    Xlr2 = Xlr.iloc[:, int(len(Xlr.columns)/2):]
    lrprobas = lrmodel2.predict_proba(Xlr2)
    for iadd in range(NUM_CLUSTS):
        Xy_test_wLRfeats["rlr"+str(iadd)] = (lrprobas[: , iadd] +
                            LR_BLUR*(np.random.rand(len(lrprobas))-0.5))
    
pred_ids = rfmodel.predict(Xy_test_wLRfeats)

test_submit = test_meta[["eeg_id"]].copy()
for new_col in HBA_votes:
    test_submit[new_col] = 1/HBA_number
for iprob in range(HBA_number):
    this_col_probs = best_centers[: , iprob]
    test_submit[HBA_votes[iprob]] = this_col_probs[pred_ids]    
    
print(test_submit)

test_submit.to_csv("submission.csv", header=True, 
                        index=False, na_rep='', float_format='%.6f')


## --- ERROR in cell 48, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/32968763.py in <cell line: 0>()
     11 if USE_LR1:
     12     Xlr1 = Xlr.iloc[:, 0:int(len(Xlr.columns)/2)]
---> 13     lrprobas = lrmodel1.predict_proba(Xlr1)
     14     for iadd in range(NUM_CLUSTS):
     15         Xy_test_wLRfeats["lr"+str(iadd)] = (lrprobas[: , iadd] +

NameError: name 'lrmodel1' is not defined
