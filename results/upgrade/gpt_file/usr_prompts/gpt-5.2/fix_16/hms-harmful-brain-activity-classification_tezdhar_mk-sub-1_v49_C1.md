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
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
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

0.3092453848929784

# 6. Current score

0.78004

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.40995) has done: 'The runtime failure is caused by `from src.settings import TARGET_COLS` after you removed the `hms-mk-codes` path, so `src` is no longer importable at merge time. I make the merge step self-contained by deriving `TARGET_COLS` directly from `sample_submission.csv`, and I also make the merge robust to missing fold/version files and ensure the final probabilities are valid (non-negative and row-sum=1). This keeps the core inference logic unchanged (you still generate the same per-fold CSVs), but fixes the crash and guarantees a valid `submission.csv` is written. Finally, I keep your intended ensemble weights and simply normalize safely to avoid submission format failures.'
- What this solution (achieved 1.41937) has done: 'I remove the failing offline `pip install` steps and add checks so the script doesn’t depend on unavailable Kaggle datasets (`hms-mk-codes`, `hms-mk-codesv2`, `hms-mk-data`). Since those model assets aren’t present, I keep your ensemble/merge logic intact but make it robust: it use any fold/version prediction CSVs that exist, otherwise it fall back to a safe baseline built from `train.csv` vote priors (better than uniform, so it should move KL score down from 1.40995 toward your target). Finally, I guarantee the written `/kaggle/working/submission.csv` is valid (correct columns, non-negative, row sums = 1).'
- What this solution (achieved 0.81104) has done: 'I fix the shape blow-up in `merge_preds` that happens because `test_eeg_ids` contains duplicates (sample_submission repeats `eeg_id`), causing merges to expand rows and making `preds` far longer than `sol`. The minimal correction is to build `sol` from the unique `eeg_id` list (matching the actual required submission length), and to also deduplicate any incoming prediction CSVs before merging. I also make the final output strictly follow `sample_submission.csv` order/length by reindexing to its `eeg_id` list and re-normalizing probabilities, ensuring the submission always passes Kaggle’s validation. No modeling/inference logic is changed; this is purely to make the merge and submission generation correct and stable.'
- What this solution (achieved 0.80863) has done: 'To move your KL score down from 0.81104 toward the 0.309 target (lower is better), the smallest safe lever is improving probability calibration/priors without changing your inference/core ensemble logic. Your current `dirichlet_alpha` smoothing is relatively strong and tends to flatten predictions, which typically worsens KL here; I reduce it to a much smaller value to keep probabilities valid but less uniform. I also slightly reduce the uniform-mix in the global prior fallback so the fallback distribution is closer to the empirical class distribution (usually better for KL). Finally, I keep the exact same merge/inference flow and submission alignment checks.'
- What this solution (achieved 0.78004) has done: 'Your current score (0.80863, lower-is-better) is still far from the target (0.3092), so we should improve (decrease) KL with the smallest low-risk change. The core issue is that when model prediction CSVs are missing or partially missing, the code falls back to near-uniform rows (inside `merge_preds` for NaNs), which is usually very bad for KL on this competition; we instead fill missing rows using patient-aware priors (shrunk to global), which keeps semantics the same (still just merging existing predictions) but provides a much better “default” distribution. Additionally, we apply a tiny prior-mix to all predictions (including when model CSVs exist) to reduce overconfident zeros that get heavily penalized by KL, without flattening too much. Finally, we keep submission alignment/normalization exactly as you already do and still write `/kaggle/working/submission.csv`.'
- What this solution (achieved 0.78005) has done: 'Your current KL (0.78004, lower-is-better) is still far above the target (0.3092), so we should reduce KL with a minimal, low-risk calibration tweak rather than changing any model/inference code. The safest lever here is the post-merge probability smoothing: replace the tiny additive Dirichlet term with a slightly larger “floor” (prevents near-zero probabilities that get heavily punished by KL) and slightly increase the prior mix so missing/weak rows are pulled closer to patient-aware priors. I keep your ensemble/merge logic, file discovery, and submission alignment identical, only adjusting these two scalar hyperparameters and making the probability floor explicit and stable. This should move the score downward toward the target without rewriting core logic.'
- What this solution (achieved 0.78004) has done: 'Your KL (0.78005, lower-is-better) is still far above the target (0.3092), so we should reduce KL with a minimal, low-risk change focused on probability calibration rather than altering any model/inference logic. The strongest lever in your current pipeline is the post-merge smoothing: your current `prior_mix_all` and additive smoothing are likely over-flattening good model rows while still not fully protecting against near-zeros. I (1) switch the additive smoothing to an explicit per-row probability floor (renormalized) which is more directly aligned with avoiding KL blow-ups from tiny probabilities, and (2) reduce the global pull (`prior_mix_all`) slightly so we don’t wash out informative ensemble predictions. Everything else (file discovery, ensembling, patient-aware fallback, submission alignment) stays the same, and we still write a valid `/kaggle/working/submission.csv`.'
- What this solution (achieved 0.78004) has done: 'We keep your ensemble/merge logic intact and only tune the post-merge probability calibration knobs that most affect KL: the probability floor (to avoid near-zero penalties) and the global pull toward patient-aware priors (to reduce overconfident mistakes). Your current `prob_floor=2e-4` is likely still too small for this metric, so we increase it modestly, and we also increase `prior_mix_all` slightly so rows are gently regularized toward patient priors without turning them uniform. No inference code paths, file discovery, weights, or submission alignment semantics are changed; we still write `/kaggle/working/submission.csv` with correct order/row-sum=1.'
- What this solution (achieved 0.78004) has done: 'We keep your ensemble/merge pipeline exactly the same and only adjust the two post-merge calibration knobs that most directly affect KL: the per-class probability floor (to prevent near-zero probabilities that are harshly penalized) and the global pull toward patient-aware priors (to reduce overconfident errors). Your current settings (`prob_floor=8e-4`, `prior_mix_all=0.070`) can still produce very peaked distributions; we increase both slightly so predictions are more conservative without collapsing toward uniform. All file discovery, weights/versions, patient-aware fallback, alignment to `sample_submission.csv`, and row-sum normalization stay unchanged, and the script still writes `/kaggle/working/submission.csv`.'
- What this solution (achieved 0.78004) has done: 'We keep your ensemble/merge pipeline unchanged and only adjust the post-merge calibration knobs that most directly affect KL: the probability floor and the global pull toward patient-aware priors. Your current predictions can still be too peaky for KL (tiny probabilities are heavily penalized), so we increase `prob_floor` modestly and slightly increase `prior_mix_all` to regularize toward patient priors without collapsing toward uniform. We also apply the same probability-floor step after the final expansion back to `sample_submission` order to guarantee no near-zeros slip back in due to merging/fallback. This is a minimal change that should reduce KL (move downward from 0.78004 toward 0.3092) without changing any inference/model logic.'
- What this solution (achieved 0.78004) has done: 'We keep your ensemble/merge pipeline identical and only adjust the post-merge calibration knobs that directly affect KL: (1) slightly increase the probability floor to better protect against near-zero probabilities (which KL punishes heavily), and (2) slightly reduce the global pull toward patient priors so we don’t over-flatten informative model rows. This is the smallest, lowest-risk change that plausibly moves KL down from 0.780 toward your 0.309 target (lower is better) without touching any model inference or file discovery logic. We also keep the final “expand back to sample_submission order” normalization/flooring consistent with the updated floor.'
- What this solution (achieved 0.78004) has done: 'Your current KL (0.78004, lower-is-better) is still far above the target (0.3092), so we should reduce KL with the smallest low-risk changes that don’t alter your ensemble/inference flow. The biggest likely issue is over-regularization: the current very large `FINAL_PROB_FLOOR=6e-3` and relatively high `FINAL_PRIOR_MIX_ALL=0.09` can wash out informative model predictions and push everything toward a bland distribution, which often hurts KL here. I keep the exact same merge/ensemble logic and file discovery, but reduce the final probability floor and slightly reduce the global prior mix so predictions remain valid (no near-zeros) while being less flattened. I also ensure the same floor is applied consistently only once at the end (still the same semantics: clamp + renormalize), to avoid double-flattening.'

# 9. Code solution

## === cell 0
import os, subprocess, shlex
import numpy as np
import pandas as pd


def run_cmd(cmd: str, raise_on_error: bool = True):
    print(cmd)
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout)
        print(r.stderr)
        if raise_on_error:
            raise RuntimeError(f"Command failed with exit code {r.returncode}: {cmd}")
    return r.stdout, r.returncode




## === cell 1
DATA_PATH = "/kaggle/input/hms-harmful-brain-activity-classification"
OUT_PATH = "/kaggle/working"
OUT_PATH2 = "/kaggle/working/v2"

SAMPLE_SUB_PATH = f"{DATA_PATH}/sample_submission.csv"
TRAIN_CSV_PATH = f"{DATA_PATH}/train.csv"
TEST_CSV_PATH = f"{DATA_PATH}/test.csv"

sample_sub = pd.read_csv(SAMPLE_SUB_PATH)
TARGET_COLS = [c for c in sample_sub.columns if c != "eeg_id"]

test_eeg_ids_all = sample_sub["eeg_id"].copy().reset_index(drop=True)
test_eeg_ids_unique = pd.Index(test_eeg_ids_all).drop_duplicates().to_numpy()

print("TARGET_COLS:", TARGET_COLS)
print("sample_sub shape:", sample_sub.shape)
print("sample_sub unique eeg_id:", sample_sub["eeg_id"].nunique())
print("test eeg_id count (from sample_submission):", len(test_eeg_ids_all))
print("test unique eeg_id:", len(test_eeg_ids_unique))



## === cell 2
MK_CODES = "/kaggle/input/hms-mk-codes"
MK_DATA = "/kaggle/input/hms-mk-data"
MK_CODES_V2 = "/kaggle/input/hms-mk-codesv2"

print("Exists MK_CODES:", os.path.exists(MK_CODES))
print("Exists MK_DATA:", os.path.exists(MK_DATA))
print("Exists MK_CODES_V2:", os.path.exists(MK_CODES_V2))




## === cell 3
def maybe_run_original_inference():
    if not (os.path.exists(MK_CODES) and os.path.exists(MK_DATA)):
        print(
            "Skipping v0/v3 inference: missing /kaggle/input/hms-mk-codes or /kaggle/input/hms-mk-data"
        )
        return

    convert_cmd = (
        f"cd {shlex.quote(MK_CODES)} && "
        f"python -m src.convert_parquet_to_npy --data_dir={shlex.quote(DATA_PATH)} --out_dir={shlex.quote(OUT_PATH)}"
    )
    run_cmd(convert_cmd, raise_on_error=False)

    base = (
        f"cd {shlex.quote(MK_CODES)} && python -m test "
        f"paths.data_dir={shlex.quote(DATA_PATH)} "
        f"data.test_eegs_dir={shlex.quote(OUT_PATH)} "
        f"hydra=test +model.test_output_dir={shlex.quote(OUT_PATH)} "
        f"+model.net.pretrained=False"
    )
    cmds = [
        f"{base} ckpt_path=/kaggle/input/hms-mk-data/fold0_levit_pseudo.ckpt experiment=conv1d_tfm2d_pseudo && mv /kaggle/working/submission.csv /kaggle/working/submission_fold0_v0.csv",
        f"{base} ckpt_path=/kaggle/input/hms-mk-data/fold1_levit_pseudo.ckpt experiment=conv1d_tfm2d_pseudo && mv /kaggle/working/submission.csv /kaggle/working/submission_fold1_v0.csv",
        f"{base} ckpt_path=/kaggle/input/hms-mk-data/fold2_levit_pseudo.ckpt experiment=conv1d_tfm2d_pseudo && mv /kaggle/working/submission.csv /kaggle/working/submission_fold2_v0.csv",
        f"{base} ckpt_path=/kaggle/input/hms-mk-data/fold3_levit_pseudo.ckpt experiment=conv1d_tfm2d_pseudo && mv /kaggle/working/submission.csv /kaggle/working/submission_fold3_v0.csv",
        f"{base} ckpt_path=/kaggle/input/hms-mk-data/fold4_levit_pseudo.ckpt experiment=conv1d_tfm2d_pseudo && mv /kaggle/working/submission.csv /kaggle/working/submission_fold4_v0.csv",
        f"{base} ckpt_path=/kaggle/input/hms-mk-data/fold0_effb3_sim_pseudo.ckpt experiment=conv1d_effv2_pseudo && mv /kaggle/working/submission.csv /kaggle/working/submission_fold0_v3.csv",
        f"{base} ckpt_path=/kaggle/input/hms-mk-data/fold1_effb3_sim_pseudo.ckpt experiment=conv1d_effv2_pseudo && mv /kaggle/working/submission.csv /kaggle/working/submission_fold1_v3.csv",
        f"{base} ckpt_path=/kaggle/input/hms-mk-data/fold2_effb3_sim_pseudo.ckpt experiment=conv1d_effv2_pseudo && mv /kaggle/working/submission.csv /kaggle/working/submission_fold2_v3.csv",
        f"{base} ckpt_path=/kaggle/input/hms-mk-data/fold3_effb3_sim_pseudo.ckpt experiment=conv1d_effv2_pseudo && mv /kaggle/working/submission.csv /kaggle/working/submission_fold3_v3.csv",
        f"{base} ckpt_path=/kaggle/input/hms-mk-data/fold4_effb3_sim_pseudo.ckpt experiment=conv1d_effv2_pseudo && mv /kaggle/working/submission.csv /kaggle/working/submission_fold4_v3.csv",
    ]
    for c in cmds:
        run_cmd(c, raise_on_error=False)


def maybe_run_original_inference_v2():
    if not (os.path.exists(MK_CODES_V2) and os.path.exists(MK_DATA)):
        print(
            "Skipping v4/v5 inference: missing /kaggle/input/hms-mk-codesv2 or /kaggle/input/hms-mk-data"
        )
        return

    convert_cmd2 = (
        f"cd {shlex.quote(MK_CODES_V2)} && "
        f"python -m src.convert_parquet_to_npy --data_dir={shlex.quote(DATA_PATH)} --out_dir={shlex.quote(OUT_PATH2)}"
    )
    run_cmd(convert_cmd2, raise_on_error=False)

    base2 = (
        f"cd {shlex.quote(MK_CODES_V2)} && python -m test "
        f"paths.data_dir={shlex.quote(DATA_PATH)} "
        f"data.test_dataset._target_=src.nn_datasets.components.eegdataset_clean.HMSTestDataKG "
        f"data.test_dataset.eeg_dir={shlex.quote(OUT_PATH2)}/test_eegs "
        f"hydra=test +model.test_output_dir={shlex.quote(OUT_PATH)} "
        f"data.num_workers=2 +model.net.pretrained=False"
    )
    cmds2 = [
        f"{base2} ckpt_path=/kaggle/input/hms-mk-data/clean_xcit_pseudo_fold0.ckpt experiment=clean_tfm_pseudo && mv /kaggle/working/submission.csv /kaggle/working/submission_fold0_v4.csv",
        f"{base2} ckpt_path=/kaggle/input/hms-mk-data/clean_xcit_pseudo_fold1.ckpt experiment=clean_tfm_pseudo && mv /kaggle/working/submission.csv /kaggle/working/submission_fold1_v4.csv",
        f"{base2} ckpt_path=/kaggle/input/hms-mk-data/clean_xcit_pseudo_fold2.ckpt experiment=clean_tfm_pseudo && mv /kaggle/working/submission.csv /kaggle/working/submission_fold2_v4.csv",
        f"{base2} ckpt_path=/kaggle/input/hms-mk-data/clean_xcit_pseudo_fold3.ckpt experiment=clean_tfm_pseudo && mv /kaggle/working/submission.csv /kaggle/working/submission_fold3_v4.csv",
        f"{base2} ckpt_path=/kaggle/input/hms-mk-data/clean_xcit_pseudo_fold4.ckpt experiment=clean_tfm_pseudo && mv /kaggle/working/submission.csv /kaggle/working/submission_fold4_v4.csv",
        f"{base2} ckpt_path=/kaggle/input/hms-mk-data/clean_effb1_pseudo_fold0.ckpt experiment=clean_effb1_pseudo && mv /kaggle/working/submission.csv /kaggle/working/submission_fold0_v5.csv",
        f"{base2} ckpt_path=/kaggle/input/hms-mk-data/clean_effb1_pseudo_fold1.ckpt experiment=clean_effb1_pseudo && mv /kaggle/working/submission.csv /kaggle/working/submission_fold1_v5.csv",
        f"{base2} ckpt_path=/kaggle/input/hms-mk-data/clean_effb1_pseudo_fold2.ckpt experiment=clean_effb1_pseudo && mv /kaggle/working/submission.csv /kaggle/working/submission_fold2_v5.csv",
        f"{base2} ckpt_path=/kaggle/input/hms-mk-data/clean_effb1_pseudo_fold3.ckpt experiment=clean_effb1_pseudo && mv /kaggle/working/submission.csv /kaggle/working/submission_fold3_v5.csv",
        f"{base2} ckpt_path=/kaggle/input/hms-mk-data/clean_effb1_pseudo_fold4.ckpt experiment=clean_effb1_pseudo && mv /kaggle/working/submission.csv /kaggle/working/submission_fold4_v5.csv",
    ]
    for c in cmds2:
        run_cmd(c, raise_on_error=False)


maybe_run_original_inference()
maybe_run_original_inference_v2()




## === cell 4
def train_prior_probs(
    train_csv_path: str, target_cols: list[str], eps: float = 1e-12
) -> np.ndarray:
    train = pd.read_csv(train_csv_path, usecols=target_cols)
    sums = train[target_cols].sum(axis=0).to_numpy(dtype=np.float64)
    probs = sums / max(sums.sum(), eps)
    probs = np.clip(probs, eps, None)
    probs = probs / probs.sum()
    return probs


def patient_prior_probs(
    train_csv_path: str,
    test_csv_path: str,
    target_cols: list[str],
    eps: float = 1e-12,
    shrink_to_global: float = 0.15,
) -> pd.DataFrame:
    usecols = ["patient_id"] + target_cols
    tr = pd.read_csv(train_csv_path, usecols=usecols)
    te = pd.read_csv(test_csv_path, usecols=["eeg_id", "patient_id"])

    g = tr[target_cols].sum(axis=0).to_numpy(np.float64)
    g = np.clip(g, eps, None)
    g = g / g.sum()

    grp = tr.groupby("patient_id")[target_cols].sum()
    grp_sum = grp.sum(axis=1).to_numpy(np.float64).reshape(-1, 1)
    pp = grp.to_numpy(np.float64) / np.clip(grp_sum, eps, None)

    pp = (1.0 - shrink_to_global) * pp + shrink_to_global * g.reshape(1, -1)
    pp = np.clip(pp, eps, None)
    pp = pp / pp.sum(axis=1, keepdims=True)

    patient_df = pd.DataFrame(pp, columns=target_cols)
    patient_df["patient_id"] = grp.index.values

    te2 = te.merge(patient_df, on="patient_id", how="left")
    miss = te2[target_cols].isna().any(axis=1)
    if miss.any():
        te2.loc[miss, target_cols] = g.reshape(1, -1)

    te2 = te2.drop_duplicates(subset=["eeg_id"], keep="first")
    return te2[["eeg_id"] + target_cols]


def merge_preds(
    folds=(0, 1, 2, 3, 4),
    versions=("v4", "v5"),
    weights=(0.5, 0.5),
    workdir="/kaggle/working",
    eps=1e-12,
    prior_uniform_mix: float = 0.005,
    dirichlet_alpha: float = 0.0,
    prior_mix_all: float = 0.045,
    patient_shrink_to_global: float = 0.15,
    prob_floor: float = 2e-4,
):
    sol = pd.DataFrame({"eeg_id": test_eeg_ids_unique.copy()})

    weights = list(weights)
    versions = list(versions)
    if len(weights) != len(versions):
        raise ValueError("weights and versions must have the same length")

    te_patient = patient_prior_probs(
        TRAIN_CSV_PATH,
        TEST_CSV_PATH,
        TARGET_COLS,
        eps=eps,
        shrink_to_global=float(patient_shrink_to_global),
    )
    te_patient = sol.merge(te_patient, on="eeg_id", how="left")
    prior_preds = te_patient[TARGET_COLS].to_numpy(np.float64)

    gprior = train_prior_probs(TRAIN_CSV_PATH, TARGET_COLS, eps=eps)
    u = np.full_like(gprior, 1.0 / len(TARGET_COLS), dtype=np.float64)
    gprior = (1.0 - float(prior_uniform_mix)) * gprior + float(prior_uniform_mix) * u
    gprior = np.clip(gprior, eps, None)
    gprior = gprior / gprior.sum()

    if np.isnan(prior_preds).any():
        nan_rows = np.isnan(prior_preds).any(axis=1)
        prior_preds[nan_rows] = gprior.reshape(1, -1)

    acc = np.zeros((len(sol), len(TARGET_COLS)), dtype=np.float64)
    used_weight = 0.0
    used_files = []

    for fold in folds:
        for w, v in zip(weights, versions):
            fp = f"{workdir}/submission_fold{fold}_{v}.csv"
            if not os.path.exists(fp):
                continue
            df = pd.read_csv(fp)
            if not set(["eeg_id"] + TARGET_COLS).issubset(df.columns):
                continue

            df = df[["eeg_id"] + TARGET_COLS].copy()
            df = df.drop_duplicates(subset=["eeg_id"], keep="first")
            df = sol[["eeg_id"]].merge(df, on="eeg_id", how="left")

            vals = df[TARGET_COLS].to_numpy(np.float64)

            if np.isnan(vals).any():
                nan_rows = np.isnan(vals).any(axis=1)
                vals[nan_rows] = prior_preds[nan_rows]

            acc += vals * float(w)
            used_weight += float(w)
            used_files.append(fp)

    if used_weight <= 0:
        preds = prior_preds.copy()
        print(
            "No fold/version CSVs found; using patient-aware priors (shrunk to global)."
        )
    else:
        preds = acc / used_weight
        print(
            f"Ensembled {len(used_files)} prediction files with total weight={used_weight:.3f}."
        )
        print("First few used files:", used_files[:5])

    preds = np.clip(preds, eps, None)

    pf = float(prob_floor)
    if pf > 0:
        preds = np.maximum(preds, pf)
        preds = preds / preds.sum(axis=1, keepdims=True)

    if dirichlet_alpha > 0:
        preds = preds + float(dirichlet_alpha)
        preds = preds / preds.sum(axis=1, keepdims=True)

    pmix = float(prior_mix_all)
    if pmix > 0:
        preds = (1.0 - pmix) * preds + pmix * prior_preds
        preds = np.clip(preds, eps, None)
        preds = preds / preds.sum(axis=1, keepdims=True)

    sol[TARGET_COLS] = preds
    sol = sol[["eeg_id"] + TARGET_COLS]
    return sol




## === cell 5
FINAL_PROB_FLOOR = 1.5e-3

FINAL_PRIOR_MIX_ALL = 0.06

sol_unique = merge_preds(
    folds=(0, 1, 2, 3, 4),
    versions=("v0", "v3", "v4", "v5"),
    weights=(0.3, 0.3, 0.2, 0.2),
    prior_uniform_mix=0.005,
    dirichlet_alpha=0.0,
    prior_mix_all=FINAL_PRIOR_MIX_ALL,
    patient_shrink_to_global=0.15,
    prob_floor=FINAL_PROB_FLOOR,
)

sol = pd.DataFrame({"eeg_id": test_eeg_ids_all.copy()}).merge(
    sol_unique, on="eeg_id", how="left"
)

if sol[TARGET_COLS].isna().any(axis=1).any():
    gprior = train_prior_probs(TRAIN_CSV_PATH, TARGET_COLS, eps=1e-12)
    miss = sol[TARGET_COLS].isna().any(axis=1)
    sol.loc[miss, TARGET_COLS] = gprior.reshape(1, -1)

vals = sol[TARGET_COLS].to_numpy(np.float64)
vals = np.clip(vals, 1e-12, None)
vals = vals / vals.sum(axis=1, keepdims=True)

vals = np.maximum(vals, float(FINAL_PROB_FLOOR))
vals = vals / vals.sum(axis=1, keepdims=True)
sol[TARGET_COLS] = vals



## === cell 6
out_file = "/kaggle/working/submission.csv"
sol.to_csv(out_file, index=False)
print(f"Wrote {out_file}")
print(sol.shape)
print(sol.head())



## === cell 7
assert list(sol.columns) == ["eeg_id"] + TARGET_COLS
vals = sol[TARGET_COLS].to_numpy(np.float64)
row_sums = vals.sum(axis=1)
assert np.all(np.isfinite(vals))
assert np.all(vals >= 0)
assert np.max(np.abs(row_sums - 1.0)) < 1e-6
assert len(sol) == len(sample_sub)
assert (sol["eeg_id"].to_numpy() == sample_sub["eeg_id"].to_numpy()).all()
print(
    "Submission looks valid: correct columns, non-negative probs, row-sum=1, and matches sample_submission order/length."
)
