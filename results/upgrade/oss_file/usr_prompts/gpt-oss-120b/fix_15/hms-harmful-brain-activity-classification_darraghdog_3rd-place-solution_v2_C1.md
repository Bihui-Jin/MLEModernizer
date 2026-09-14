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
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
pytorch-ignite==0.5.3
pytorch-lightning==2.5.5
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
scipy==1.15.3
sklearn-pandas==2.2.0
torch==2.6.0+cu124
torchao==0.10.0
torchaudio==2.6.0+cu124
torchdata==0.11.0
torchinfo==1.8.0
torchmetrics==1.8.2
torchsummary==1.5.1
torchtune==0.6.1
torchvision==0.21.0+cu124
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

0.275967

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 1.42007) has done: 'I fixed the pipeline by adding safe fallback handling for the missing configuration modules and weight files. Each model generation is now wrapped in a try‑except block; if loading fails, a uniform‑probability tensor is created instead. This ensures `PREDS` is always populated, allowing the weighted averaging, bias correction, and normalization steps to run without errors, and a valid `submission.csv` is written that respects the required format and row‑sum constraint.'
- What this solution (achieved 1.4147) has done: 'Implemented a lightweight fix that drops uniform‑fallback predictions from the ensemble and slightly scales back the class‑bias correction.  
- Detects uniform predictions (standard deviation ≈ 0) and excludes their weights, then re‑normalises the remaining weights before averaging.  
- Scales the bias vector by 0.5 to avoid over‑compensating, which empirically nudges KL‑divergence toward the target while keeping the original pipeline intact.'
- What this solution (achieved 1.41224) has done: 'I slightly reduce the class‑bias correction and add a modest temperature scaling ( > 1 ) before the softmax. This flattens over‑confident logits, which should lower the KL divergence and move the score toward the target while keeping the overall pipeline unchanged.'
- What this solution (achieved 1.40995) has done: 'I adjust the ensemble handling so it leans less on the hand‑crafted bias and temperature adjustments that were over‑compensating the predictions. First, I replace the original weighted averaging with equal weights for all models (so no single model dominates). Then I remove the class‑bias correction entirely and set the temperature scaling back to 1 (no flattening). These small, focused changes keep the overall pipeline intact while moving the KL‑divergence closer to the target lower‑score region.'
- What this solution (achieved 1.41115) has done: 'I keep the overall pipeline unchanged but improve the ensemble weighting and calibration: use the originally provided model weights (normalised to sum 1) instead of forcing uniform weights, add a modest bias correction and apply a slight temperature scaling (> 1) to soften over‑confident predictions. These minimal tweaks are expected to lower the KL‑divergence toward the target while preserving the core logic.'
- What this solution (achieved 1.40995) has done: 'I adjust the ensemble to use equal weights (removing the hand‑crafted imbalance), turn off the class‑bias correction, and increase the temperature scaling to 2.0. These minimal changes keep the original pipeline intact while flattening predictions, which should lower the KL‑divergence toward the target score.'
- What this solution (achieved 1.41177) has done: 'I adjust the ensemble weighting to use the provided model weights (instead of equal weighting), re‑introduce a modest class‑bias correction, and lower the temperature scaling to 1.5. These small calibrations should make the predictions less over‑flattened and move the KL‑divergence closer to the target lower score while keeping the core pipeline unchanged.'
- What this solution (achieved 1.40995) has done: 'I adjust the calibration steps to bring the KL‑divergence closer to the target.  
- Replace the weighted‑average that uses the provided model weights with simple equal weights for all valid models (removes any over‑emphasis).  
- Remove the class‑bias correction by setting its scale to 0.0 (eliminates a source of systematic shift).  
- Increase the temperature scaling to 2.5 to flatten the logits more, which typically reduces over‑confident errors and lowers the KL score.  
These minimal changes keep the core pipeline intact while addressing the main sources of calibration error.'
- What this solution (achieved 1.41224) has done: 'I restore the original model‐weighting (using the provided normalized WEIGHTS), re‑introduce a modest class‑bias correction, and remove the aggressive temperature flattening. These minimal tweaks keep the pipeline intact while making the ensemble predictions sharper and better calibrated, which should lower the KL‑divergence toward the target score.'
- What this solution (achieved 1.40995) has done: 'I reduce the class‑bias correction to zero and apply a mild temperature < 1 (sharpening the logits) which should improve calibration and move the KL‑divergence lower toward the target while keeping the original pipeline unchanged.'
- What this solution (achieved 1.41146) has done: 'I slightly adjust the calibration steps to bring the KL divergence closer to the target: use uniform ensemble weights (instead of the original hand‑crafted ones), re‑introduce a modest bias correction, and apply a gentle temperature > 1 to flatten the logits. These small changes keep the overall pipeline intact while moving the score toward the lower‑score target.'
- What this solution (achieved 1.40995) has done: 'The update restores the original model‑weighting (using the provided normalized WEIGHTS), removes the class‑bias correction, and eliminates temperature scaling so the ensemble predictions are combined as intended and kept in their natural confidence range, which should lower the KL‑divergence toward the target score.'
- What this solution (achieved 1.41224) has done: 'I add a modest class‑bias correction and increase the temperature scaling to flatten the logits before the softmax. These small calibration tweaks keep the original ensemble logic unchanged while moving the KL‑divergence lower toward the target score.'

# 9. Code solution

## === cell 0
COMP_FOLDER = "/kaggle/input/hms-harmful-brain-activity-classification/"

train_df = pd.read_csv(COMP_FOLDER + "train.csv")
test_df = pd.read_csv(COMP_FOLDER + "test.csv")
sample_submission = pd.read_csv(COMP_FOLDER + "sample_submission.csv")

EEG_FOLDER = "/kaggle/input/hms-harmful-brain-activity-classification/test_eegs/"

PUBLIC_RUN = len(test_df) == 1
N_CORES = mp.cpu_count()
MIXED_PRECISION = False

RAM_CHECK = False
OOF_CHECK = False

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

if PUBLIC_RUN is False:
    RAM_CHECK = False
    OOF_CHECK = False

if OOF_CHECK is True:
    train_df = pd.read_csv("/kaggle/input/hms-aws-bucket/train_folded_17k.csv")
    EEG_FOLDER = "/kaggle/input/hms-harmful-brain-activity-classification/train_eegs/"
    test_df = train_df[train_df["fold"] == 0].copy()

if RAM_CHECK is True:
    train_df = pd.read_csv("/kaggle/input/hms-aws-bucket/train_folded_17k.csv")
    EEG_FOLDER = "/kaggle/input/hms-harmful-brain-activity-classification/train_eegs/"
    test_df = train_df[train_df["fold"] == 0].copy()
    test_df = test_df.head(2640).copy()

print(train_df.shape)
print(test_df.shape)

TARGETS = [
    "seizure_vote",
    "lpd_vote",
    "gpd_vote",
    "lrda_vote",
    "grda_vote",
    "other_vote",
]
if TARGETS[0] not in test_df.columns:
    test_df[TARGETS] = 1
    test_df["eeg_label_offset_seconds"] = 0
    test_df["spectrogram_label_offset_seconds"] = 0

class_counts = train_df[TARGETS].sum()
prior_probs = (class_counts / class_counts.sum()).values  # shape (6,)




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1903198107.py in <cell line: 0>()
      1 COMP_FOLDER = "/kaggle/input/hms-harmful-brain-activity-classification/"
      2 
----> 3 train_df = pd.read_csv(COMP_FOLDER + "train.csv")
      4 test_df = pd.read_csv(COMP_FOLDER + "test.csv")
      5 sample_submission = pd.read_csv(COMP_FOLDER + "sample_submission.csv")

NameError: name 'pd' is not defined

## === cell 1
def generate(name="cfg_1", weights_dir="./"):
    cfg = get_cfg(name)
    cfg.pretrained = False
    cfg.data_folder = EEG_FOLDER
    state_dict_fps = sorted(glob.glob(weights_dir, recursive=True))
    if not state_dict_fps:
        base_dir = os.path.dirname(weights_dir.rstrip("*"))
        state_dict_fps = sorted(
            glob.glob(os.path.join(base_dir, "**/*.pth"), recursive=True)
        )
    test_dl, batch_to_device = get_dl(cfg)
    if test_dl is None:
        print(f"Fallback: generating prior‑based predictions for {name}")
        return (
            torch.tensor(prior_probs, dtype=torch.float32)
            .unsqueeze(0)
            .repeat(len(test_df), 1)
        )
    print("\n".join(state_dict_fps))
    nets = get_nets(cfg, state_dict_fps, test_dl.dataset)
    preds = []
    with torch.inference_mode():
        for batch in tqdm(test_dl):
            batch = batch_to_device(batch, DEVICE)
            outs = [net(batch) for net in nets]
            if not outs:
                continue
            preds += [torch.stack([out["logits"] for out in outs], dim=0).mean(0).cpu()]
    if not preds:  # all nets failed – use prior
        print(f"All models failed for {name}, using prior.")
        return (
            torch.tensor(prior_probs, dtype=torch.float32)
            .unsqueeze(0)
            .repeat(len(test_df), 1)
        )
    preds = torch.cat(preds, dim=0).float()
    print("preds", preds.shape, ", test_df", test_df.shape)
    gc.collect()
    torch.cuda.empty_cache()
    return preds




## === cell 2
CLASS_BIAS = [0.012535, 0.03458, 0.01761, 0.05957, -0.02608, -0.0982]




## === cell 3
BIAS_SCALE = 0.0
postproc = BIAS_SCALE * torch.tensor(CLASS_BIAS).unsqueeze(0)




## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1158955682.py in <cell line: 0>()
      1 # Reduce bias influence to zero (bias correction removed)
      2 BIAS_SCALE = 0.0
----> 3 postproc = BIAS_SCALE * torch.tensor(CLASS_BIAS).unsqueeze(0)
      4 
      5 

NameError: name 'torch' is not defined

## === cell 4
TEMPERATURE = 1.0
preds = preds / TEMPERATURE




## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/101227973.py in <cell line: 0>()
      1 # Remove temperature scaling (set to 1.0)
      2 TEMPERATURE = 1.0
----> 3 preds = preds / TEMPERATURE
      4 
      5 

NameError: name 'preds' is not defined

## === cell 5
sub = pd.DataFrame({"eeg_id": test_df.eeg_id.values})
sub[TARGETS] = preds
sub.to_csv("submission.csv", index=False)
print("Submission shape", sub.shape)
sub.head()

## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3737351718.py in <cell line: 0>()
----> 1 sub = pd.DataFrame({"eeg_id": test_df.eeg_id.values})
      2 sub[TARGETS] = preds
      3 sub.to_csv("submission.csv", index=False)
      4 print("Submission shape", sub.shape)
      5 sub.head()

NameError: name 'pd' is not defined
