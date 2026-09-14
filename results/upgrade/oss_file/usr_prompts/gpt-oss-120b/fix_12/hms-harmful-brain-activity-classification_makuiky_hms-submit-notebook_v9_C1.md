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

0.9568163852605625

# 6. Current score

1.41937

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.39779) has done: 'I replace the failing TensorFlow import with a safe try‑except and fall back to a simple baseline that predicts the overall class distribution computed from the training votes. This avoids the missing model file and the protobuf error, guarantees that probabilities sum to 1, and writes a correct `submission.csv` file.'
- What this solution (achieved 1.39779) has done: 'I added a safe import for matplotlib.pyplot, computed per‑`eeg_id` vote distributions from the training metadata, and made the submission logic use these per‑id probabilities when available (falling back to the overall baseline otherwise). This keeps the original fallback to the baseline when the model cannot be loaded, fixes the missing pyplot import, and provides more tailored predictions to lower the KL‑divergence score toward the target.'
- What this solution (achieved 1.39779) has done: 'I fixed the import failures by keeping the safe‑import logic and added a simple probability smoothing step: per‑`eeg_id` vote distributions are blended with the overall baseline using a Laplace‑style prior. This reduces over‑confidence on IDs with few votes, which lowers the KL‑divergence and moves the score closer to the target while preserving the original workflow.'
- What this solution (achieved 1.41937) has done: 'The update adds the missing imports, safely handles optional TensorFlow and matplotlib availability, and ensures all variables are defined before they are used. This fixes the NameError issues, guarantees the submission CSV is written correctly, and keeps the original baseline‑per‑ID probability logic (which already aims to lower the KL‑divergence score).'
- What this solution (achieved 1.41937) has done: 'The fix disables TensorFlow entirely (avoiding the protobuf import error) and raises the Laplace smoothing factor so low‑vote `eeg_id`s rely more on the overall class distribution, which should lower the KL‑divergence toward the target. No core modeling logic is changed, and the script still writes a valid `submission.csv` with correctly normalised probabilities.'
- What this solution (achieved 1.41937) has done: 'The fix corrects a variable typo in the EEG image generator, safely disables TensorFlow loading, and increases the Laplace smoothing factor to blend per‑ID vote distributions more heavily with the overall class baseline—this reduces over‑confidence on small‑vote IDs and lowers the KL‑divergence score while preserving the core workflow.'
- What this solution (achieved 1.41937) has done: 'I remove the problematic TensorFlow import (setting `tf = None` directly) to stop the protobuf error, and increase the Laplace smoothing factor to pull noisy per‑ID predictions closer to the overall class distribution, which should lower the KL‑divergence toward the target score. The rest of the workflow stays unchanged, ensuring a valid `submission.csv` is written.'
- What this solution (achieved 1.41937) has done: 'I increase the Laplace‑style smoothing factor used when blending per‑eeg‑id vote distributions with the overall class baseline. By raising `smoothing` from 10.0 to 30.0 the predictions rely more on the global distribution, which reduces over‑confidence on noisy small‑vote IDs and should lower the KL‑divergence score toward the target without altering any core modeling logic.'
- What this solution (achieved 1.41937) has done: 'The update reduces the Laplace smoothing factor from 30.0 to 5.0, allowing the per‑`eeg_id` vote distributions to influence the predictions more strongly while still keeping a modest regularisation term. This change keeps the original workflow intact but should lower the KL‑divergence score, moving it closer to the target.'
- What this solution (achieved 1.41937) has done: 'To reduce over‑confidence on noisy `eeg_id` distributions we increase the Laplace smoothing factor (more weight to the global baseline) and clip very small probabilities before normalising, which safely lowers the KL‑divergence without altering the core workflow.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd
from PIL import Image

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None

tf = None

EEG_TEST_PATH = "/kaggle/input/hms-harmful-brain-activity-classification/test_eegs/"
SPEC_TEST_PATH = (
    "/kaggle/input/hms-harmful-brain-activity-classification/test_spectrograms/"
)
EEG_IMG_TEST_PATH = "/kaggle/working/test_eegs_img/"
SPEC_IMG_TEST_PATH = "/kaggle/working/test_spec_img/"

os.makedirs(EEG_IMG_TEST_PATH, exist_ok=True)
os.makedirs(SPEC_IMG_TEST_PATH, exist_ok=True)

META_TEST = "/kaggle/input/hms-harmful-brain-activity-classification/test.csv"
META_TRAIN = "/kaggle/input/hms-harmful-brain-activity-classification/train.csv"



## === cell 1
metadata_test = pd.read_csv(META_TEST)
metadata_train = pd.read_csv(META_TRAIN)



## === cell 2
eeg_zone = {
    "Cz-Pz": ["Cz", "Pz"],
    "Fz-Cz": ["Fz", "Cz"],
    "P4-O2": ["P4", "O2"],
    "C4-P4": ["C4", "P4"],
    "F4-C4": ["F4", "C4"],
    "Fp2-F4": ["Fp2", "F4"],
    "P3-O1": ["P3", "O1"],
    "C3-P3": ["C3", "P3"],
    "F3-C3": ["F3", "C3"],
    "Fp1-F3": ["Fp1", "F3"],
    "T6-O2": ["T6", "O2"],
    "T4-T6": ["T4", "T6"],
    "F8-T4": ["F8", "T4"],
    "Fp2-F8": ["Fp2", "F8"],
    "T5-O1": ["T5", "O1"],
    "T3-T5": ["T3", "T5"],
    "F7-T3": ["F7", "T3"],
    "Fp1-F7": ["Fp1", "F7"],
}




## === cell 3
def generate_eeg(
    eegid,
    input_path=EEG_TEST_PATH,
    eeg_zone=eeg_zone,
    linewidth=0.2,
    eeg_out=EEG_IMG_TEST_PATH,
):
    """Create a simple EEG line‑plot image for a given eeg_id."""
    if plt is None:
        raise RuntimeError("matplotlib is required for EEG image generation.")
    eeg = pd.read_parquet(f"{input_path}{egd}.parquet")
    relpos = 0
    cicles = 0
    fig, ax = plt.subplots(1, 1, figsize=(3, 3), sharex=True)
    for key, values in eeg_zone.items():
        ax.plot(
            eeg.index / 200,
            eeg[values[0]] - eeg[values[1]] + relpos,
            color="black",
            linewidth=linewidth,
        )
        if cicles in (1, 5, 9, 13):
            relpos += 200
        else:
            relpos += 40
        cicles += 1
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(0, 50)
    save_path = f"{eeg_out}{eegid}.jpeg"
    fig.savefig(save_path, bbox_inches="tight", dpi=100)
    plt.close()
    return save_path


spec_zones = ["LL", "RL", "LP", "RP"]


def generate_spectrogram(
    specid,
    input_path=SPEC_TEST_PATH,
    spec_out=SPEC_IMG_TEST_PATH,
    spec_zones=spec_zones,
    output_filter=0.5,
):
    """Create a spectrogram image for a given spectrogram_id."""
    if plt is None:
        raise RuntimeError("matplotlib is required for spectrogram image generation.")
    spec = pd.read_parquet(f"{input_path}{specid}.parquet")
    spec = spec.fillna(0).set_index("time").T
    spec["column"] = spec.index.str.split("_", expand=True)
    spec["freq"] = spec.column.apply(lambda x: float(x[1]))
    spec["brainreg"] = spec.column.apply(lambda x: x[0])
    spec = spec.drop("column", axis=1).set_index("freq")
    subspec = {}
    for zone in spec_zones:
        sub = spec[spec.brainreg == zone].drop("brainreg", axis=1)
        subspec[f"{zone}_sub"] = sub
    fig, ax = plt.subplots(nrows=len(spec_zones), figsize=(3, 3), sharex=True)
    for row, zone in enumerate(spec_zones):
        data = subspec[f"{zone}_sub"]
        ax[row].imshow(
            data,
            cmap="turbo",
            aspect="auto",
            origin="lower",
            extent=[
                data.columns.min(),
                data.columns.max(),
                data.index.min(),
                data.index.max(),
            ],
            vmin=0,
            vmax=data.max().max() * output_filter,
        )
        ax[row].set_xticks([])
        ax[row].set_yticks([])
    plt.subplots_adjust(hspace=0.01)
    save_path = f"{spec_out}{specid}.jpeg"
    fig.savefig(save_path, bbox_inches="tight", dpi=100)
    plt.close()
    return save_path




## === cell 4
model_path = "/kaggle/input/hms-models/sexto_modelo_impr.keras"
model = None  # TensorFlow model disabled; will use baseline logic

vote_cols = [
    "seizure_vote",
    "lpd_vote",
    "gpd_vote",
    "lrda_vote",
    "grda_vote",
    "other_vote",
]

train_votes = metadata_train[vote_cols].astype(float)
total_votes = train_votes.sum(axis=0)  # sum per class over all rows
baseline_probs = (total_votes / total_votes.sum()).values  # numpy array

grouped = metadata_train.groupby("eeg_id")[vote_cols].sum()
group_sums = grouped.sum(axis=1).replace(0, np.nan)
per_id_probs = grouped.div(group_sums, axis=0).fillna(0)
per_id_probs = per_id_probs[vote_cols]

per_id_counts = group_sums  # number of votes aggregated per eeg_id

smoothing = 30.0



## === cell 5
submission = {
    "eeg_id": [],
    "seizure_vote": [],
    "lpd_vote": [],
    "gpd_vote": [],
    "lrda_vote": [],
    "grda_vote": [],
    "other_vote": [],
}
iter_cols = vote_cols

for _, row in metadata_test.iterrows():
    eeg_id = row["eeg_id"]
    if model is not None and tf is not None:
        eeg_path = generate_eeg(eegid=eeg_id)
        spec_path = generate_spectrogram(specid=row["spectrogram_id"])

        eeg_img = Image.open(eeg_path)
        spec_img = Image.open(spec_path)

        eeg_tensor = tf.convert_to_tensor(np.array(eeg_img))
        spec_tensor = tf.convert_to_tensor(np.array(spec_img))
        eeg_tensor = tf.expand_dims(eeg_tensor, axis=0)
        spec_tensor = tf.expand_dims(spec_tensor, axis=0)

        os.remove(eeg_path)
        os.remove(spec_path)

        preds = model.predict([eeg_tensor, spec_tensor], verbose=0).squeeze()
        preds = np.clip(preds, 0, None)
        if preds.sum() == 0:
            preds = baseline_probs
        else:
            preds = preds / preds.sum()
    else:
        if eeg_id in per_id_probs.index:
            count = per_id_counts.get(eeg_id, 0.0)
            id_probs = per_id_probs.loc[eeg_id].values
            preds = (id_probs * count + baseline_probs * smoothing) / (
                count + smoothing
            )
        else:
            preds = baseline_probs

        if preds.sum() == 0:
            preds = baseline_probs
        else:
            preds = preds / preds.sum()

    eps = 1e-6
    preds = np.clip(preds, eps, None)
    preds = preds / preds.sum()

    submission["eeg_id"].append(eeg_id)
    for col, val in zip(iter_cols, preds):
        submission[col].append(val)

pdsubmit = pd.DataFrame(submission)



## === cell 6
output_path = "submission.csv"
pdsubmit.to_csv(output_path, index=False)
print(f"Submission written to {output_path}")
