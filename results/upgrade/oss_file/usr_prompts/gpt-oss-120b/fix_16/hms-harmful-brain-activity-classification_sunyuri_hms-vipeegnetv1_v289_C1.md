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

3.13

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

0.3103161734374802

# 6. Current score

1.41937

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.41937) has done: 'The fix corrects the wrong `np` import and replaces the placeholder uniform predictions with class‑frequency‑based probabilities derived from the training set, which yields a better baseline without altering the core model logic.'
- What this solution (achieved 1.41937) has done: 'I make the TensorFlow import optional (skipping all TF‑related code when we are only generating a submission) to prevent the protobuf‑related crash, and I replace the simple overall class‑frequency baseline with a per‑`eeg_id` probability lookup (falling back to the global frequencies for unseen ids) to move the KL‑divergence score closer to the target.'
- What this solution (achieved 1.41937) has done: 'I add a tiny amount of Laplace smoothing to the per‑eeg vote counts and then blend those smoothed probabilities with the overall global class frequencies. This keeps the original per‑eeg lookup logic while reducing overly confident predictions, which should lower the KL‑divergence toward the target score. No new libraries or major logic changes are introduced, and the script still writes a valid `submission.csv`.'
- What this solution (achieved 1.41937) has done: 'I keep the overall baseline logic unchanged but make two small adjustments that are expected to lower the KL‑divergence: (1) use a lighter Laplace smoothing (adding 0.1 instead of 1) so per‑eeg probabilities stay closer to the observed vote ratios, and (2) increase the weight of the per‑eeg signal (`alpha`) from 0.2 to 0.6 so the model leans more on the specific information available for each eeg_id. After blending we re‑normalise the probabilities to guarantee they sum to 1 for every row. These minimal changes keep the core workflow intact while moving the score toward the target.'
- What this solution (achieved 1.41937) has done: 'I lower the reliance on per‑eeg statistics (which over‑fit the training set) and increase Laplace smoothing so the predictions are closer to the overall class distribution, which should markedly reduce KL‑divergence toward the target while keeping the original workflow unchanged.'
- What this solution (achieved 1.41937) has done: 'I slightly increase the reliance on per‑eeg statistics (raise `alpha` from 0.2 to 0.6) and reduce the Laplace smoothing (lower `smooth_const` from 0.5 to 0.1). This keeps the original workflow intact while making the predictions more tailored to each `eeg_id`, which should lower the KL‑divergence and move the score closer to the target. No other logic is changed, and the script still writes a valid `submission.csv`.'
- What this solution (achieved 1.41937) has done: 'I lower the reliance on per‑eeg statistics and increase smoothing so the predictions stay closer to the overall class distribution, which should reduce the KL‑divergence (lower is better). This is done by setting a larger Laplace constant and decreasing the blending factor `alpha`. No other logic is changed, and the script still writes a valid `submission.csv`.'
- What this solution (achieved 1.41937) has done: 'The current baseline relies heavily on the global class frequencies (α = 0.1) and uses a large Laplace smoothing of 1.0, which makes the per‑`eeg_id` signal too weak and overly smooth. To lower the KL‑divergence we increase the weight of the per‑eeg statistics (α → 0.5) and reduce the smoothing constant (→ 0.1). These small adjustments keep the original workflow unchanged while giving the model a stronger, less‑smoothed per‑eeg signal, bringing the score closer to the target.'
- What this solution (achieved 1.41937) has done: 'The changes lower the reliance on per‑eeg statistics (α → 0.05) and increase Laplace smoothing (constant → 1.0) so the predicted distributions stay much closer to the global class frequencies, which reduces the KL‑divergence toward the target while keeping the original workflow unchanged.'
- What this solution (achieved 1.41937) has done: 'The update lowers the KL‑divergence by removing the overly‑smoothed per‑EEG blending, which was pushing many predictions toward a uniform distribution. We set a tiny Laplace constant (0.1) and eliminate the blending factor (α = 0) so the submission uses the global class frequencies for every test record, keeping predictions valid and summed to one while moving the score closer to the target.'
- What this solution (achieved 1.41937) has done: 'I increase the blending factor `alpha` so the predictions use a mix of per‑`eeg_id` statistics and the global class distribution. This adds useful signal without altering the core workflow, and should lower the KL‑divergence (lower is better) toward the target score.'

# 9. Code solution

## === cell 0
"""
Created on Tue Oct 22 20:48:49 2024

@author: yuri

email: syuri@tju.edu.cn
"""

import os
import warnings
import pandas as pd
import numpy as np

warnings.filterwarnings("ignore")

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

PLATFORM = "kaggle"  # *** local kaggle *** local training or online testing
NEEDTRAIN = False  # training flag – False for submission generation only

DATATYPE = ["eeg"]  # *** spe, eeg, stft, img *** the data type used
print(DATATYPE)

LOAD_MODELS_FROM = "models20241118b"  # path of trained model weights for testing

if PLATFORM == "local":
    LOAD_MODELS_FROM = f"./input/{LOAD_MODELS_FROM}"
    LOAD_DATA_FROM = "./input/hms-harmful-brain-activity-classification"
elif PLATFORM == "kaggle":
    LOAD_MODELS_FROM = f"/kaggle/input/{LOAD_MODELS_FROM}"
    LOAD_DATA_FROM = "/kaggle/input/hms-harmful-brain-activity-classification"

SEED = 2024
np.random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)

TF_AVAILABLE = False
if NEEDTRAIN:
    try:
        import tensorflow as tf
        from tensorflow.keras import optimizers, backend as K
        from tensorflow.keras.models import clone_model
        from tensorflow.python.framework.ops import reset_default_graph

        try:
            _ = tf.config.list_physical_devices()
            TF_AVAILABLE = True
        except Exception as tf_err:
            print(f"TensorFlow environment problem: {tf_err}")
            TF_AVAILABLE = False
    except Exception as e:
        print(f"TensorFlow import failed ({e}); proceeding without it.")
        TF_AVAILABLE = False

df = pd.read_csv(os.path.join(LOAD_DATA_FROM, "train.csv"))
TARGETS = df.columns[-6:]  # last six columns are the vote targets
print("Train shape:", df.shape)
print("Targets", list(TARGETS))

global_counts = df[TARGETS].sum()
global_prob = global_counts / global_counts.sum()  # pandas Series, sums to 1

smooth_const = 0.1

eeg_counts = df.groupby("eeg_id")[list(TARGETS)].sum()
eeg_counts_smooth = eeg_counts + smooth_const
eeg_sum_smooth = eeg_counts_smooth.sum(axis=1)
eeg_prob = eeg_counts_smooth.div(eeg_sum_smooth, axis=0)  # per‑eeg probabilities

alpha = 0.6

eeg_prob = eeg_prob * alpha + global_prob * (1.0 - alpha)

eeg_prob = eeg_prob.fillna(global_prob)

eeg_prob_dict = {
    str(eeg_id): row.values.astype(np.float32) for eeg_id, row in eeg_prob.iterrows()
}

test = pd.read_csv(os.path.join(LOAD_DATA_FROM, "test.csv"))
num_samples = test.shape[0]

preds_all = np.empty((num_samples, len(TARGETS)), dtype=np.float32)

for idx, eeg_id in enumerate(test["eeg_id"].values):
    prob_vec = eeg_prob_dict.get(str(eeg_id), global_prob.values.astype(np.float32))
    preds_all[idx] = prob_vec

preds_all = preds_all / preds_all.sum(axis=1, keepdims=True)

sub = pd.DataFrame({"eeg_id": test["eeg_id"].values})
sub[TARGETS] = preds_all
sub.to_csv("submission.csv", index=False)

print("Submission shape", sub.shape)
print(sub.head())
