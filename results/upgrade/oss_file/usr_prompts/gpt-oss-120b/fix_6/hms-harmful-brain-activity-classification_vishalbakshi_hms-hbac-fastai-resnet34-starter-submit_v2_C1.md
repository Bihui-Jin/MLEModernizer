# Goal

I want you to improve my Kaggle competition solution to increase the score toward a target. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (evaluation score improvement); avoid unrelated refactors or stylistic edits.
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

fastai==2.8.5
fastcore==1.8.15
joblib==1.5.2
timm==1.0.19
tqdm==4.67.1

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

2.131336884648777

# 6. Current score

1.36187

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.38283) has done: 'I add the missing imports, remove the failing model‑load step, and train the ResNet34 from scratch (a single quick epoch is enough to produce predictions). I also drop optional visual‑debug cells that aren’t needed for generating the submission, keeping the core workflow intact so the script runs end‑to‑end and writes a valid `submission.csv` whose rows sum to 1.'
- What this solution (achieved 1.31971) has done: 'The fix converts the model predictions from a torch tensor to a NumPy array before creating the uniform blend, ensuring the correct dtype and allowing the subsequent probability blending and submission generation to run without errors.'
- What this solution (achieved 1.31572) has done: 'I increase the uniform blending weight (`alpha`) from 0.15 to 0.5 so the predictions become more mixed with a uniform distribution. This modest change deliberately lowers model confidence, which raises the KL‑divergence score, moving it closer to the target 2.1313 while keeping all other logic intact and ensuring the submission still sums to 1 per row.'
- What this solution (achieved 1.36187) has done: 'I raise the uniform‑blending weight `alpha` from 0.5 to 0.85 so the predictions become much closer to a uniform distribution. This deliberately lowers model confidence, increasing the KL‑divergence score and moving it toward the target value while preserving all existing logic and ensuring each row still sums to 1.'

# 9. Code solution

## === cell 0
import warnings, os, io
import pandas as pd, numpy as np
import timm, joblib
from tqdm import tqdm
from fastai.vision.all import *
from fastcore.parallel import *
from pathlib import Path




## === cell 1
def remove_col_prefix(df):
    df.columns = df.columns.str.replace(r"^[A-Z]+_", "", regex=True)
    return df


def avg_sgram(sgram_path):
    sample_spect = pd.read_parquet(sgram_path)
    split_spect = {
        "LL": sample_spect.filter(regex="^LL", axis=1),
        "RL": sample_spect.filter(regex="^RL", axis=1),
        "RP": sample_spect.filter(regex="^RP", axis=1),
        "LP": sample_spect.filter(regex="^LP", axis=1),
    }
    sgram_avg_df = pd.concat([remove_col_prefix(df) for df in split_spect.values()])
    sgram_avg_df = sgram_avg_df.groupby(sgram_avg_df.index).mean()
    return sgram_avg_df




## === cell 2
SPEC_DIR = "/tmp/dataset/hms-hbac"
os.makedirs(f"{SPEC_DIR}/train_spectrograms", exist_ok=True)
os.makedirs(f"{SPEC_DIR}/test_spectrograms", exist_ok=True)




## === cell 3
def process_spec(spec_id, split="train"):
    data = avg_sgram(
        Path("/kaggle/input/hms-harmful-brain-activity-classification")
        / f"{split}_spectrograms"
        / f"{spec_id}.parquet"
    )
    data = data.fillna(0)
    data = data.values[:, 1:].astype("float32")
    im = PILImage.create(Image.fromarray((data * 255).astype(np.uint8)))
    im.save(f"{SPEC_DIR}/{split}_spectrograms/{spec_id}.png")




## === cell 4
train_df = pd.read_csv(
    Path("/kaggle/input/hms-harmful-brain-activity-classification") / "train.csv"
)




## === cell 5
train_spec_ids = train_df["spectrogram_id"].unique()




## === cell 6
warnings.filterwarnings("ignore")
parallel(process_spec, train_spec_ids, split="train", n_workers=4)
warnings.filterwarnings("default")




## === cell 7
test_df = pd.read_csv(
    Path("/kaggle/input/hms-harmful-brain-activity-classification") / "test.csv"
)




## === cell 8
test_spec_ids = test_df["spectrogram_id"].unique()
warnings.filterwarnings("ignore")
parallel(process_spec, test_spec_ids, split="test", n_workers=4)
warnings.filterwarnings("default")




## === cell 9
train_df["img_path"] = (
    f"{SPEC_DIR}/train_spectrograms/" + train_df["spectrogram_id"].astype(str) + ".png"
)




## === cell 10
cols = [
    "eeg_id",
    "spectrogram_id",
    "img_path",
    "seizure_vote",
    "lpd_vote",
    "gpd_vote",
    "lrda_vote",
    "grda_vote",
    "other_vote",
]
agg_funcs = {c: "sum" for c in cols if "vote" in c}
unique_df = (
    train_df[cols]
    .groupby(["eeg_id", "spectrogram_id", "img_path"], as_index=False)
    .agg(agg_funcs)
)




## === cell 11
unique_df["target"] = unique_df[[c for c in cols if "vote" in c]].idxmax(axis=1)




## === cell 12
train_len = int(0.8 * len(unique_df))
is_valid = (
    pd.Series([False] * train_len + [True] * (len(unique_df) - train_len))
    .sample(frac=1)
    .reset_index(drop=True)
)
unique_df["is_valid"] = is_valid




## === cell 13
dblock = DataBlock(
    blocks=(ImageBlock, CategoryBlock),
    splitter=ColSplitter(),
    get_x=ColReader("img_path"),
    get_y=ColReader("target"),
    item_tfms=Resize(224, method="squish"),
)
dls = dblock.dataloaders(unique_df)




## === cell 14
learn = vision_learner(dls, "resnet34", metrics=accuracy, pretrained=False)




## === cell 15
learn.fit_one_cycle(1, 1e-3)  # quick training from scratch




## === cell 16
test_df["img_path"] = (
    f"{SPEC_DIR}/test_spectrograms/" + test_df["spectrogram_id"].astype(str) + ".png"
)




## === cell 17
tst_dl = learn.dls.test_dl(test_df)




## === cell 18
probs, _, _ = learn.get_preds(dl=tst_dl, with_decoded=True)
probs = probs.cpu().numpy()  # torch tensor → NumPy array
alpha = 0.85
uniform = np.full(probs.shape[1], 1.0 / probs.shape[1], dtype=probs.dtype)
probs_blended = (1 - alpha) * probs + alpha * uniform

probs_df = pd.DataFrame(probs_blended, columns=dls.vocab)
probs_df["eeg_id"] = test_df["eeg_id"].values
probs_df = probs_df[
    [
        "eeg_id",
        "seizure_vote",
        "lpd_vote",
        "gpd_vote",
        "lrda_vote",
        "grda_vote",
        "other_vote",
    ]
]




## === cell 19
assert np.allclose(
    probs_df[
        ["seizure_vote", "lpd_vote", "gpd_vote", "lrda_vote", "grda_vote", "other_vote"]
    ].sum(axis=1),
    1.0,
    atol=1e-5,
)




## === cell 20
probs_df.to_csv("submission.csv", index=False)




## === cell 21
print(probs_df.head())
