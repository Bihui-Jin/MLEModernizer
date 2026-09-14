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

1.33389

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plan

- What this solution (achieved 1.33389) has done: 'I remove the stray natural-language “cell 0” content that’s being executed as Python (causing the Unicode quote SyntaxError) and keep only runnable code cells starting at 1. Then I fix inference by using `learn.get_preds(with_decoded=True)` (or unpacking only two outputs) so it matches fastai’s actual return signature and the pipeline can reach submission generation. Finally, I make the submission construction robust to class-order/column-name mismatches by explicitly mapping from the model vocab to the required `*_vote` columns, normalizing rows to sum to 1, and always writing a valid `submission.csv`.'

# 9. Code solution

## === cell 0
import warnings, os
import timm
from fastai.vision.all import *
from fastcore.parallel import *
import pandas as pd
import numpy as np
from PIL import Image

path = Path("/kaggle/input/hms-harmful-brain-activity-classification")
path.ls()



## === cell 1
set_seed(42, reproducible=True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False



## === cell 2
SPEC_DIR = "/tmp/dataset/hms-hbac"
os.makedirs(f"{SPEC_DIR}/train_spectrograms", exist_ok=True)
os.makedirs(f"{SPEC_DIR}/test_spectrograms", exist_ok=True)




## === cell 3
def process_spec(spec_id, split="train"):
    data = pd.read_parquet(path / f"{split}_spectrograms" / f"{spec_id}.parquet")
    data = data.fillna(0)
    data = data.values[:, 1:]  # drop time column
    data = data.T.astype("float32")

    im = PILImage.create(Image.fromarray((data * 255).astype(np.uint8)))
    im.save(f"{SPEC_DIR}/{split}_spectrograms/{spec_id}.png")




## === cell 4
df = pd.read_csv(path / "train.csv")
df.head(3)



## === cell 5
spec_ids = df["spectrogram_id"].unique()
len(spec_ids)



## === cell 6
warnings.filterwarnings("ignore")
parallel(process_spec, spec_ids, split="train", n_workers=4)
warnings.filterwarnings("default")



## === cell 7
PILImage.create(Path(f"{SPEC_DIR}/train_spectrograms").ls()[0])



## === cell 8
test_df = pd.read_csv(path / "test.csv")
test_df.head(3)



## === cell 9
spec_ids = test_df["spectrogram_id"].unique()
len(spec_ids)



## === cell 10
warnings.filterwarnings("ignore")
parallel(process_spec, spec_ids, split="test", n_workers=4)
warnings.filterwarnings("default")



## === cell 11
PILImage.create(Path(f"{SPEC_DIR}/test_spectrograms").ls()[0])



## === cell 12
df["img_path"] = (
    f"{SPEC_DIR}/train_spectrograms/" + df["spectrogram_id"].astype(str) + ".png"
)

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
    df[cols]
    .groupby(["eeg_id", "spectrogram_id", "img_path"], as_index=False)
    .agg(agg_funcs)
)
unique_df["target"] = unique_df[[c for c in cols if "vote" in c]].idxmax(axis=1)

n = len(unique_df)
perm = np.random.RandomState(42).permutation(n)
cut = int(0.8 * n)
is_valid = np.zeros(n, dtype=bool)
is_valid[perm[cut:]] = True
unique_df["is_valid"] = is_valid

unique_df.head(3)



## === cell 13
dblock = DataBlock(
    blocks=(ImageBlock, CategoryBlock),
    splitter=ColSplitter(col="is_valid"),
    get_x=ColReader("img_path"),
    get_y=ColReader("target"),
    item_tfms=Resize(224, method="squish"),
)

dls = dblock.dataloaders(unique_df, bs=64)



## === cell 14
dls.show_batch(nrows=1, ncols=3)



## === cell 15
dls.vocab



## === cell 16
learn = vision_learner(dls, "resnet34", metrics=accuracy, pretrained=False)



## === cell 17
ckpt_base = Path("/kaggle/input/hms-hbac-resnet34/pytorch/2/1")
ckpt_name = "hms_hbac_resnet34_stacked"
ckpt_path = ckpt_base / f"{ckpt_name}.pth"

if ckpt_path.exists():
    learn.load(str(ckpt_base / ckpt_name))
else:
    learn.fit_one_cycle(1, 1e-3)



## === cell 18
test_df["img_path"] = (
    f"{SPEC_DIR}/test_spectrograms/" + test_df["spectrogram_id"].astype(str) + ".png"
)
tst_dl = learn.dls.test_dl(test_df)
tst_dl.show_batch(max_n=6)



## === cell 19
probs, _ = learn.get_preds(dl=tst_dl)
probs = probs.float()

probs = torch.clamp(probs, min=1e-9)
probs = probs / probs.sum(dim=1, keepdim=True)

sub_cols = [
    "eeg_id",
    "seizure_vote",
    "lpd_vote",
    "gpd_vote",
    "lrda_vote",
    "grda_vote",
    "other_vote",
]

probs_np = probs.cpu().numpy()
vocab = list(map(str, dls.vocab))

probs_df = pd.DataFrame(probs_np, columns=vocab)
probs_df.insert(0, "eeg_id", test_df["eeg_id"].values)

for c in sub_cols[1:]:
    if c not in probs_df.columns:
        probs_df[c] = 0.0

probs_df = probs_df[sub_cols]

row_sum = probs_df[sub_cols[1:]].sum(axis=1).values
row_sum = np.where(row_sum == 0, 1.0, row_sum)
probs_df[sub_cols[1:]] = probs_df[sub_cols[1:]].div(row_sum, axis=0)

probs_df.head()



## === cell 20
row_sums = probs_df[sub_cols[1:]].sum(axis=1).values
float(np.max(np.abs(row_sums - 1.0))), float(np.min(row_sums)), float(np.max(row_sums))



## === cell 21
probs_df.to_csv("submission.csv", index=False)

chk = pd.read_csv("submission.csv")
print(chk.shape)
print(chk.columns.tolist())
print(float((chk.drop(columns=["eeg_id"]).sum(1).sub(1).abs().max())))
print(chk.head(5))
