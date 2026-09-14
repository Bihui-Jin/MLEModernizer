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

0.3538838820208729

# 6. Current score

0.79884

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.41937) has done: 'I replace the failing script with a concise pipeline that loads the train and test metadata, computes class‑probability estimates from the vote counts (using per‑eeg_id distributions when available and falling back to global priors), normalizes them so each row sums to 1, and writes a correctly‑formatted `submission.csv`. This removes the protobuf monkey‑patch, undefined model code, and heavy preprocessing while still providing sensible predictions that respect the competition’s requirement of probabilistic outputs.'
- What this solution (achieved 1.68479) has done: 'I keep the existing simple vote‑based approach but add a fallback – patient‑level probability estimates – so that rows missing an exact eeg_id still get a more informed distribution rather than the global prior. This small enrichment should lower the KL divergence toward the target without altering the core logic.'
- What this solution (achieved 1.05318) has done: 'I add a tiny Laplace smoothing term to every predicted class probability before the final normalization. This prevents any zero probabilities (which heavily penalize KL‑divergence) while keeping the original vote‑based logic intact, so the model’s core behavior stays the same but the evaluation score should move closer to the lower target.'
- What this solution (achieved 0.788) has done: 'I replace the uniform epsilon smoothing with a small Dirichlet‑style smoothing that adds a fraction of the global class prior to every predicted probability. This keeps the original vote‑based logic (eeg_id → patient → global) while preventing zero probabilities in a more informed way, which should lower the KL‑divergence and move the score closer to the target.'
- What this solution (achieved 0.90499) has done: 'I keep the original vote‑based probability logic but increase the Dirichlet‑style smoothing strength from 0.05 to 1.0. A stronger smoothing moves each prediction closer to the global class prior, which reduces overly extreme probabilities that cause high KL divergence and brings the score nearer to the target lower value. No other parts of the pipeline are altered.'
- What this solution (achieved 1.1708) has done: 'I increase the Dirichlet smoothing strength so each prediction is pulled further toward the global class prior, which reduces overly extreme probabilities and therefore lowers the KL‑divergence toward the target value. The change is limited to adjusting the `alpha` constant and adding a brief comment; the core logic and data handling remain untouched.'
- What this solution (achieved 0.77245) has done: 'I replace the fixed‑strength Dirichlet smoothing with a count‑aware blending: each row’s empirical class distribution (per‑eeg or per‑patient) is combined with the global prior proportionally to the number of votes seen for that entity. This keeps strong predictions where many votes exist while pulling scarce predictions toward the prior, which should lower the KL‑divergence and move the score closer to the target. The rest of the pipeline remains unchanged.'
- What this solution (achieved 0.7852) has done: 'I keep the overall vote‑based pipeline but improve the probability blending: instead of using only the per‑eeg distribution (or falling back to patient or global), I blend the per‑eeg, per‑patient and global priors together, weighting each by its observed vote count. This uses more information while preserving the original logic. I also lower the Dirichlet pseudo‑count (α) to 2.0 so the model relies a bit more on the observed data, which should reduce the KL‑divergence and move the score closer to the target.'
- What this solution (achieved 1.41937) has done: 'The fix removes the patient‑level blending, which can introduce noisy estimates, and relies only on the per‑eeg vote distribution blended with a global prior using the same Dirichlet‑style pseudo‑count α. This keeps the core vote‑based logic while giving a smoother, more reliable probability estimate, expected to lower the KL‑divergence and move the score closer to the target.'
- What this solution (achieved 1.41937) has done: 'A small adjustment replaces the raw vote count with a logarithmic‑scaled weight when blending the per‑eeg distribution with the global prior. This reduces the influence of very high‑count EEGs (which can make predictions over‑confident) and pulls them slightly toward the more stable global prior, moving the KL‑divergence lower toward the target while keeping the original vote‑based logic unchanged.'
- What this solution (achieved 1.41937) has done: 'I reduce over‑confident per‑EEG predictions by limiting the influence of the vote count and strengthening the Dirichlet prior. Specifically, I cap the count‑derived weight at 5.0 and raise the pseudo‑count α to 5.0, so every row is blended more toward the global class prior while still respecting the observed votes. This modest smoothing should lower the KL‑divergence, moving the score closer to the target without changing the overall logic.'
- What this solution (achieved 0.7852) has done: 'I replace the fixed‑log weight and the overly strong Dirichlet smoothing with a count‑aware blending that uses the actual vote counts (or patient‑level counts when an EEG‑id is missing) and a modest pseudo‑count α = 2.0. This lets rows with many votes keep their empirical distribution while sparse rows are gently pulled toward the global prior, which should lower the KL‑divergence and move the score closer to the target.'
- What this solution (achieved 0.79884) has done: 'I increase the Dirichlet pseudo‑count `alpha` from 2.0 to a larger value (e.g., 50) so that each prediction is blended much more strongly toward the global class prior. This reduces overly confident per‑eeg or per‑patient estimates, which should lower the KL‑divergence and move the score closer to the target while keeping the original vote‑based logic unchanged.'

# 9. Code solution

## === cell 0
import os
import pandas as pd
import numpy as np

if os.path.isdir("/kaggle/input"):
    DATA_ROOT = "/kaggle/input/hms-harmful-brain-activity-classification"
else:
    DATA_ROOT = "./input/hms-harmful-brain-activity-classification"

train_path = os.path.join(DATA_ROOT, "train.csv")
test_path = os.path.join(DATA_ROOT, "test.csv")
train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

TARGETS = train_df.columns[-6:]

vote_sums_eeg = train_df.groupby("eeg_id")[list(TARGETS)].sum()
total_votes_eeg = vote_sums_eeg.sum(axis=1)
prob_per_eeg = vote_sums_eeg.div(total_votes_eeg.replace(0, np.nan), axis=0).fillna(0)

vote_sums_pat = train_df.groupby("patient_id")[list(TARGETS)].sum()
total_votes_pat = vote_sums_pat.sum(axis=1)
prob_per_pat = vote_sums_pat.div(total_votes_pat.replace(0, np.nan), axis=0).fillna(0)

global_counts = train_df[TARGETS].sum()
global_prior = global_counts / global_counts.sum()

alpha = 50.0

prob_rows = []
for _, row in test_df.iterrows():
    eid = row["eeg_id"]
    pid = row["patient_id"]

    if eid in prob_per_eeg.index:
        prob = prob_per_eeg.loc[eid].values
        cnt = total_votes_eeg.loc[eid]
    elif pid in prob_per_pat.index:
        prob = prob_per_pat.loc[pid].values
        cnt = total_votes_pat.loc[pid]
    else:
        prob = np.zeros(len(TARGETS))
        cnt = 0.0

    blended = (cnt * prob + alpha * global_prior.values) / (cnt + alpha)
    prob_rows.append(blended)

prob_array = np.vstack(prob_rows)

prob_array = prob_array / prob_array.sum(axis=1, keepdims=True)

sub = pd.DataFrame({"eeg_id": test_df["eeg_id"]})
sub[TARGETS] = prob_array

output_path = "submission.csv"
sub.to_csv(output_path, index=False)
print(f"Submission saved to {output_path} with shape {sub.shape}")
