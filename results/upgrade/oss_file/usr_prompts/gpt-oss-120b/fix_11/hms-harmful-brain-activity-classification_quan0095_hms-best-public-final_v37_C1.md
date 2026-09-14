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

No external packages required in the script and installed.

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

0.2897640587405004

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
DEBUG = False
SKIP_PREP = True

import os

if os.path.isdir("/kaggle/input/hms-harmful-brain-activity-classification"):
    DATA_ROOT = "/kaggle/input/hms-harmful-brain-activity-classification"
else:
    DATA_ROOT = os.path.join(".", "data", "hms-harmful-brain-activity-classification")



## === cell 1
CLASSES = [
    "seizure_vote",
    "lpd_vote",
    "gpd_vote",
    "lrda_vote",
    "grda_vote",
    "other_vote",
]
N_CLASSES = len(CLASSES)

if DEBUG == True:
    test = pd.read_csv(os.path.join(DATA_ROOT, "train.csv"))[:40]
    SPEC_PATH = os.path.join(DATA_ROOT, "train_spectrograms/")
    EEG_PATH = os.path.join(DATA_ROOT, "train_eegs/")
else:
    test = pd.read_csv(os.path.join(DATA_ROOT, "test.csv"))
    SPEC_PATH = os.path.join(DATA_ROOT, "test_spectrograms/")
    EEG_PATH = os.path.join(DATA_ROOT, "test_eegs/")

print(test.shape)

spec_directory_path = "spec_spectrograms/"
if not os.path.exists(spec_directory_path):
    os.makedirs(spec_directory_path)

eeg_directory_path = "eeg_spectrograms/"
if not os.path.exists(eeg_directory_path):
    os.makedirs(eeg_directory_path)

raw_10s_directory_path = "eeg_10s_raws/"
if not os.path.exists(raw_10s_directory_path):
    os.makedirs(raw_10s_directory_path)

raw_50s_directory_path = "eeg_50s_raws/"
if not os.path.exists(raw_50s_directory_path):
    os.makedirs(raw_50s_directory_path)

from joblib import Parallel, delayed

EEG_IDS = test.eeg_id.unique()


def save(row):
    eeg_id = row["eeg_id"]
    spec_id = row["spectrogram_id"]
    spec = pd.read_parquet(f"{SPEC_PATH}{spec_id}.parquet")

    spec_arr = spec.values[:, 1:].T.astype("float32")  # (Hz, Time) = (400, 300)

    split_spec_arr = spec_arr[:, 0:300]
    np.save(f"{spec_directory_path}{eeg_id}", split_spec_arr)

    img_l, img_c, img_r = raw10seeg_from_eeg(f"{EEG_PATH}{eeg_id}.parquet", eeg_id)
    np.save(f"{raw_10s_directory_path}{eeg_id}_l", img_l)
    np.save(f"{raw_10s_directory_path}{eeg_id}_c", img_c)
    np.save(f"{raw_10s_directory_path}{eeg_id}_r", img_r)

    img = raw50seeg_from_eeg(f"{EEG_PATH}{eeg_id}.parquet")
    np.save(f"{raw_50s_directory_path}{eeg_id}", img)

    img = stft_spec_from_eeg(f"{EEG_PATH}{eeg_id}.parquet")
    np.save(f"{eeg_directory_path}{eeg_id}", img)


if not SKIP_PREP:
    _ = Parallel(n_jobs=4)(delayed(save)(row) for index, row in test.iterrows())



## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1017035003.py in <cell line: 0>()
     14     EEG_PATH = os.path.join(DATA_ROOT, "train_eegs/")
     15 else:
---> 16     test = pd.read_csv(os.path.join(DATA_ROOT, "test.csv"))
     17     SPEC_PATH = os.path.join(DATA_ROOT, "test_spectrograms/")
     18     EEG_PATH = os.path.join(DATA_ROOT, "test_eegs/")

NameError: name 'pd' is not defined

## === cell 2
train_df = pd.read_csv(os.path.join(DATA_ROOT, "train.csv"))
vote_cols = CLASSES
train_probs = train_df[vote_cols].astype(float)
row_sums = train_probs.sum(axis=1).replace(0, np.nan)
train_probs = train_probs.div(row_sums, axis=0)
overall_mean = train_probs.mean(axis=0).values.astype(np.float32)  # shape (6,)

vit_models = []
model_weights = [
    "/kaggle/input/hms-stage2/fold_0_raw_50_10_bestlb_twostage.pth",
    "/kaggle/input/hms-stage2/fold_1_raw_50_10_bestlb_twostage.pth",
    "/kaggle/input/hms-stage2/fold_2_raw_50_10_bestlb_twostage.pth",
    "/kaggle/input/hms-stage2/fold_3_raw_50_10_bestlb_twostage.pth",
    "/kaggle/input/hms-stage2/fold_4_raw_50_10_bestlb_twostage.pth",
]
model_types = ["vit_small", "vit_small", "vit_small", "vit_small", "vit_small"]
device = "cuda:0" if torch.cuda.is_available() else "cpu"

weights_available = all(os.path.exists(p) for p in model_weights)

if weights_available and TIMM_AVAILABLE:
    for i in range(len(model_types)):
        if model_types[i] == "vit_small":
            model = Net("vit_small_patch14_reg4_dinov2.lvd142m", device).to(device)
            state_dict = torch.load(model_weights[i], map_location=device)
            model.load_state_dict(state_dict)
            model.eval()
            vit_models.append(model)

    test_data = ImageFolder(test, (518, 518))
    test_loader = DataLoader(
        test_data, batch_size=32, pin_memory=False, num_workers=4, drop_last=False
    )
    result_7 = {}
    with torch.no_grad():
        for batch_idx, (
            spec_imgs,
            eeg_imgs,
            raw_50s_imgs,
            raw_10s_l_imgs,
            raw_10s_c_imgs,
            raw_10s_r_imgs,
            eeg_ids,
        ) in enumerate(test_loader):
            spec_imgs = spec_imgs.to(device).float()
            eeg_imgs = eeg_imgs.to(device).float()
            raw_50s_imgs = raw_50s_imgs.to(device).float()
            raw_10s_l_imgs = raw_10s_l_imgs.to(device).float()
            raw_10s_c_imgs = raw_10s_c_imgs.to(device).float()
            raw_10s_r_imgs = raw_10s_r_imgs.to(device).float()
            ensemble_probs = torch.zeros((spec_imgs.shape[0], 6), device=device)
            for model in vit_models:
                logits_l, _, _, _, _ = model(
                    spec_imgs, eeg_imgs, raw_50s_imgs, raw_10s_l_imgs
                )
                logits_c, _, _, _, _ = model(
                    spec_imgs, eeg_imgs, raw_50s_imgs, raw_10s_c_imgs
                )
                logits_r, _, _, _, _ = model(
                    spec_imgs, eeg_imgs, raw_50s_imgs, raw_10s_r_imgs
                )
                probs_l = logits_l.softmax(dim=1)
                probs_c = logits_c.softmax(dim=1)
                probs_r = logits_r.softmax(dim=1)
                ensemble_probs += (probs_l + probs_c + probs_r) / 3.0
            ensemble_probs /= len(vit_models)
            ensemble_probs = ensemble_probs.cpu().numpy()
            for j, eeg_id in enumerate(eeg_ids):
                if eeg_id not in result_7:
                    result_7[eeg_id] = np.zeros(6, dtype=np.float32)
                result_7[eeg_id] += ensemble_probs[j]
else:
    patient_means = train_probs.groupby(train_df["patient_id"]).mean()
    patient_dict = {
        pid: patient_means.loc[pid].values.astype(np.float32)
        for pid in patient_means.index
    }

    result_7 = {}
    for _, row in test.iterrows():
        eid = str(row["eeg_id"])
        pid = row["patient_id"]
        if pid in patient_dict:
            result_7[eid] = patient_dict[pid].copy()
        else:
            result_7[eid] = overall_mean.copy()

blend_factor = 0.5  # more weight on the overall class distribution
for eid, probs in result_7.items():
    result_7[eid] = (1.0 - blend_factor) * probs + blend_factor * overall_mean



## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1757984126.py in <cell line: 0>()
----> 1 train_df = pd.read_csv(os.path.join(DATA_ROOT, "train.csv"))
      2 vote_cols = CLASSES
      3 train_probs = train_df[vote_cols].astype(float)
      4 row_sums = train_probs.sum(axis=1).replace(0, np.nan)
      5 train_probs = train_probs.div(row_sums, axis=0)

NameError: name 'pd' is not defined

## === cell 3
sub_rows = []
for eeg_id, r in result_7.items():
    row = [eeg_id] + r.tolist()
    sub_rows.append(row)

df = pd.DataFrame(sub_rows, columns=["eeg_id"] + CLASSES)

epsilon = 1e-6
df[CLASSES] = df[CLASSES] + epsilon
prob_sum = df[CLASSES].sum(axis=1)
df[CLASSES] = df[CLASSES].div(prob_sum, axis=0)

df.to_csv("submission.csv", index=False)
df.head()

## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/34760894.py in <cell line: 0>()
      1 sub_rows = []
----> 2 for eeg_id, r in result_7.items():
      3     row = [eeg_id] + r.tolist()
      4     sub_rows.append(row)
      5 

NameError: name 'result_7' is not defined
