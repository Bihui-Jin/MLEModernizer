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
timm==1.0.19

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

1.8536354903337964

# 6. Current score

1.41148

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.41191) has done: 'The submission merge is expanding rows because `probs_df` contains multiple predictions per `eeg_id` (test has repeated `eeg_id` values across different `spectrogram_id`s), which breaks the required row count. I fix this by aggregating predictions to one row per `eeg_id` (mean over duplicates) before merging with `sample_submission.csv`, keeping the same model and TTA inference. I also make the validation split deterministic and correct for `ColSplitter` by explicitly using the `"is_valid"` column, which is score-neutral but avoids silent splitting issues. Finally, I keep the probability normalization/clipping so every row sums to 1 and the CSV is always valid.'
- What this solution (achieved 1.41148) has done: 'Your current score (1.41191, lower-is-better) is better than the target (1.8536), so we should *slightly degrade* performance to move closer to the target band with minimal, low-risk changes. The smallest legitimate lever that preserves your model/training/inference core logic is probability calibration at submission time: blending your model probabilities with a uniform distribution (equivalent to adding mild label-smoothing only at inference). This keeps the submission valid (rows sum to 1) and should monotonically move KL toward a weaker baseline as the blend increases. I also make the class-column alignment explicit by mapping from `dls.vocab -> VOTE_COLS` so we never accidentally permute probabilities (this is stability/semantic correctness, not an optimization).'

# 9. Code solution

## === cell 0
import warnings
import os
import numpy as np
import pandas as pd
from PIL import Image

import timm
from fastai.vision.all import *
from fastcore.parallel import *

path = Path("/kaggle/input/hms-harmful-brain-activity-classification")
print("Using data path:", path)
print("Exists:", path.exists())
print("Top-level files:", (path.ls()[:10] if path.exists() else "N/A"))



## === cell 1
set_seed(42, reproducible=False)



## === cell 2
VOTE_COLS = [
    "seizure_vote",
    "lpd_vote",
    "gpd_vote",
    "lrda_vote",
    "grda_vote",
    "other_vote",
]



## === cell 3
SPEC_DIR = "/tmp/dataset/hms-hbac"
os.makedirs(os.path.join(SPEC_DIR, "train_spectrograms"), exist_ok=True)
os.makedirs(os.path.join(SPEC_DIR, "test_spectrograms"), exist_ok=True)




## === cell 4
def process_spec(spec_id, split="train"):
    data = pd.read_parquet(path / f"{split}_spectrograms" / f"{spec_id}.parquet")
    data = data.fillna(0)

    arr = data.values[:, 1:].T.astype(np.float32)  # (freq_bins, time_steps)

    mn, mx = float(arr.min()), float(arr.max())
    if not np.isfinite(mn) or not np.isfinite(mx) or mx <= mn:
        img = np.zeros(arr.shape, dtype=np.uint8)
    else:
        img = (arr - mn) / (mx - mn)
        img = (img * 255.0).clip(0, 255).astype(np.uint8)

    pil_img = Image.fromarray(img, mode="L")
    im = PILImage.create(pil_img)
    im.save(f"{SPEC_DIR}/{split}_spectrograms/{spec_id}.png")




## === cell 5
def process_missing_specs(spec_ids, split="train", n_workers=4):
    out_dir = Path(SPEC_DIR) / f"{split}_spectrograms"
    missing = []
    for sid in spec_ids:
        if not (out_dir / f"{sid}.png").exists():
            missing.append(sid)
    if len(missing) == 0:
        return
    warnings.filterwarnings("ignore")
    parallel(process_spec, missing, split=split, n_workers=n_workers)
    warnings.filterwarnings("default")




## === cell 6
df = pd.read_csv(path / "train.csv")
print(df.shape)
df.head(3)



## === cell 7
spec_ids = df["spectrogram_id"].unique()



## === cell 8
len(spec_ids)



## === cell 9
process_missing_specs(spec_ids, split="train", n_workers=4)



## === cell 10
PILImage.create(Path("/tmp/dataset/hms-hbac/train_spectrograms").ls()[0])



## === cell 11
test_df = pd.read_csv(path / "test.csv")
print(test_df.shape)
test_df.head(3)



## === cell 12
spec_ids = test_df["spectrogram_id"].unique()
len(spec_ids)



## === cell 13
process_missing_specs(spec_ids, split="test", n_workers=4)



## === cell 14
PILImage.create(Path("/tmp/dataset/hms-hbac/test_spectrograms").ls()[0])



## === cell 15
df["img_path"] = (
    "/tmp/dataset/hms-hbac/train_spectrograms/"
    + df["spectrogram_id"].astype(str)
    + ".png"
)

cols = ["eeg_id", "spectrogram_id", "img_path"] + VOTE_COLS
agg_funcs = {c: "sum" for c in cols if c in VOTE_COLS}

unique_df = (
    df[cols]
    .groupby(["eeg_id", "spectrogram_id", "img_path"], as_index=False)
    .agg(agg_funcs)
)
unique_df["target"] = unique_df[VOTE_COLS].idxmax(axis=1)

unique_df = unique_df.sample(frac=1.0, random_state=42).reset_index(drop=True)
cut = int(0.8 * len(unique_df))
unique_df["is_valid"] = False
unique_df.loc[cut:, "is_valid"] = True

unique_df.head(3)



## === cell 16
dblock = DataBlock(
    blocks=(ImageBlock, CategoryBlock),
    splitter=ColSplitter(col="is_valid"),
    get_x=ColReader("img_path"),
    get_y=ColReader("target"),
    item_tfms=Resize(224, method="squish"),
)

dls = dblock.dataloaders(unique_df)



## === cell 17
dls.show_batch(nrows=1, ncols=3)



## === cell 18
dls.vocab



## === cell 19
learn = vision_learner(dls, "resnet34", metrics=accuracy, pretrained=False)



## === cell 20
weights_path = Path(
    "/kaggle/input/hms-hbac-resnet34/pytorch/2/1/hms_hbac_resnet34_stacked.pth"
)
try:
    if weights_path.exists():
        learn.load(str(weights_path).replace(".pth", ""))
    else:
        learn.load(
            "/kaggle/input/hms-hbac-resnet34/pytorch/2/1/hms_hbac_resnet34_stacked"
        )
except Exception as e:
    print(
        f"[WARN] Could not load pretrained weights; proceeding with current model state. Error: {e}"
    )



## === cell 21
test_df["img_path"] = (
    "/tmp/dataset/hms-hbac/test_spectrograms/"
    + test_df["spectrogram_id"].astype(str)
    + ".png"
)
tst_dl = learn.dls.test_dl(test_df)
tst_dl.show_batch()



## === cell 22
valid = learn.dls.valid
preds, targs = learn.get_preds(dl=valid)



## === cell 23
accuracy(preds, targs)



## === cell 24
tta_preds, _ = learn.tta(dl=valid)



## === cell 25
accuracy(tta_preds, targs)



## === cell 26
probs, _ = learn.tta(dl=tst_dl)

probs_df = pd.DataFrame(probs, columns=dls.vocab)
probs_df["eeg_id"] = test_df["eeg_id"].values

for c in VOTE_COLS:
    if c not in probs_df.columns:
        probs_df[c] = 0.0

probs_df = probs_df[["eeg_id"] + VOTE_COLS]

probs_df = probs_df.groupby("eeg_id", as_index=False)[VOTE_COLS].mean()

p = probs_df[VOTE_COLS].to_numpy(dtype=np.float64)
p = np.nan_to_num(p, nan=0.0, posinf=0.0, neginf=0.0)
p = np.clip(p, 1e-12, None)
p = p / p.sum(axis=1, keepdims=True)
probs_df[VOTE_COLS] = p.astype(np.float32)

probs_df.head()



## === cell 27
sample_sub = pd.read_csv(path / "sample_submission.csv")
sub = sample_sub[["eeg_id"]].merge(probs_df, on="eeg_id", how="left")

miss = sub[VOTE_COLS].isna().any(axis=1)
if miss.any():
    sub.loc[miss, VOTE_COLS] = 1.0 / len(VOTE_COLS)

ALPHA_TO_UNIFORM = (
    0.20  # small, monotonic degradation lever; adjust if needed after one submission
)
p = sub[VOTE_COLS].to_numpy(dtype=np.float64)
p = np.clip(np.nan_to_num(p, nan=0.0, posinf=0.0, neginf=0.0), 1e-12, None)
p = p / p.sum(axis=1, keepdims=True)

u = np.full_like(p, 1.0 / len(VOTE_COLS), dtype=np.float64)
p = (1.0 - ALPHA_TO_UNIFORM) * p + ALPHA_TO_UNIFORM * u

p = np.clip(p, 1e-12, None)
p = p / p.sum(axis=1, keepdims=True)
sub[VOTE_COLS] = p.astype(np.float32)

assert len(sub) == len(sample_sub), "Submission row count must match sample_submission."
assert (
    list(sub.columns) == ["eeg_id"] + VOTE_COLS
), "Submission columns must match required schema."
row_sums = sub[VOTE_COLS].sum(axis=1).values
print("Row sum min/max:", row_sums.min(), row_sums.max())

sub.to_csv("submission.csv", index=False)
print(pd.read_csv("submission.csv").head())
print("Wrote submission.csv with shape:", sub.shape)
print("ALPHA_TO_UNIFORM used:", ALPHA_TO_UNIFORM)
