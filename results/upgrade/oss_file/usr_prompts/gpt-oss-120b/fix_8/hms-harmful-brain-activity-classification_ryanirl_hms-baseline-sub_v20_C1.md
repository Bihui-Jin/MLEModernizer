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

albumentations==2.0.8
cudf-polars-cu12==25.6.0
geopandas==0.14.4
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
polars==1.25.0
pytorch-ignite==0.5.3
pytorch-lightning==2.5.5
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

0.5958003092721826

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 1.41937) has done: 'The fix introduces safe imports for the custom models, provides a simple fallback model that returns uniform log‑probabilities, computes global class probabilities from the training data to use when no pretrained models are loaded, and corrects the way predictions are placed into the submission DataFrame. These changes ensure the script runs end‑to‑end, produces a valid `submission.csv` whose rows sum to 1, and moves the score toward the target without altering the overall pipeline logic.'
- What this solution (achieved 1.68479) has done: 'The changes add a patient‑wise fallback probability (using the training vote distribution per patient) and fix the test‑row handling by converting the test metadata to pandas. This provides more informed predictions than the uniform global baseline, helping lower the KL‑divergence toward the target while keeping the original model ensemble logic unchanged.'
- What this solution (achieved 1.68479) has done: 'Implemented missing imports, path definitions, data loading, and utility functions; added safe fallback implementations for model loading and spectrogram reading; ensured all variables (torch, nn, np, pd, os, device, df_train, df_test_pd, EEG_DIR, SPEC_DIR, patient_probs, GLOBAL_PROBS) are defined; corrected submission generation so rows sum to 1 and a valid `submission.csv` is written. This fixes runtime errors and yields a functional pipeline while preserving the original ensemble logic.'

# 9. Code solution

## === cell 0
try:
    sys.path.append("/kaggle/input/hms-models/")
    from eeg_cnn_rnn_att import EegModel as EegModel
    from spc_cnn_att import SpectrogramCnnModel as SpcModel
except Exception as e:
    print("Custom model imports failed:", e)

    class SpcModel(nn.Module):
        """Fallback model that returns uniform log‑probabilities."""

        def __init__(self):
            super().__init__()
            self.logits = torch.log(torch.full((1, 6), 1.0 / 6.0))

        def forward(self, x):
            batch_size = x.shape[0]
            return self.logits.expand(batch_size, -1)


def load_model(path, model):
    return model




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_54/1671456592.py in <cell line: 0>()
      1 try:
----> 2     sys.path.append("/kaggle/input/hms-models/")
      3     from eeg_cnn_rnn_att import EegModel as EegModel

NameError: name 'sys' is not defined

During handling of the above exception, another exception occurred:

NameError                                 Traceback (most recent call last)
/tmp/ipykernel_54/1671456592.py in <cell line: 0>()
      6     print("Custom model imports failed:", e)
      7 
----> 8     class SpcModel(nn.Module):
      9         """Fallback model that returns uniform log‑probabilities."""
     10 

NameError: name 'nn' is not defined

## === cell 1
models_spc = []
model_paths = sorted(
    glob.glob("/kaggle/input/hms-models/final_spec_kaggle/final_spec_kaggle/*")
)
for model_path in model_paths:
    model = SpcModel()
    try:
        model = load_model(model_path, model)
    except Exception as e:
        print(f"Failed to load model from {model_path}: {e}")
    model = model.to(device)
    model.eval()
    models_spc.append(model)

print("Number of spectrogram models loaded:", len(models_spc))




## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_54/3243270949.py in <cell line: 0>()
      1 models_spc = []
      2 model_paths = sorted(
----> 3     glob.glob("/kaggle/input/hms-models/final_spec_kaggle/final_spec_kaggle/*")
      4 )
      5 for model_path in model_paths:

NameError: name 'glob' is not defined

## === cell 2
BASE_DIR = "/kaggle/input/hms-harmful-brain-activity-classification"

df_train = pd.read_csv(os.path.join(BASE_DIR, "train.csv"))
df_test_pd = pd.read_csv(os.path.join(BASE_DIR, "test.csv"))

LABELS = [
    "seizure_vote",
    "lpd_vote",
    "gpd_vote",
    "lrda_vote",
    "grda_vote",
    "other_vote",
]

patient_group = df_train.groupby("patient_id")[LABELS].sum()
patient_probs = patient_group.div(patient_group.sum(axis=1), axis=0).to_dict(
    orient="index"
)

vote_sums = np.array([df_train[col].sum() for col in LABELS], dtype=np.float64)
GLOBAL_PROBS = vote_sums / vote_sums.sum()
print("Global class probabilities (fallback):", GLOBAL_PROBS)

EEG_DIR = os.path.join(BASE_DIR, "test_eegs")
SPEC_DIR = os.path.join(BASE_DIR, "test_spectrograms")


def compute_kaggle_spec_from_file(filepath):
    df = pd.read_parquet(filepath)
    return df.values.astype(np.float32)


spec_transforms = A.Compose([A.Resize(height=96, width=224)])


def proc_kspec(x):
    x = x.copy()
    x = x[:, 2:98]  # retain frequency band
    x[np.isnan(x) | np.isinf(x)] = 0
    x = x.clip(np.exp(-4), np.exp(7))
    x = np.log(x)
    x = x - x.mean(axis=(1, 2), keepdims=True)
    x = x / (x.std(axis=(1, 2), keepdims=True) + 1e-5)
    x = x.transpose(1, 2, 0)  # H, W, C
    x = spec_transforms(image=x)["image"]
    x = x.transpose(2, 0, 1)  # C, H, W
    x = x.reshape(4, 96, 224)  # 4 channels expected by the fallback model
    return x


_MODEL_WEIGHT = 0.0
_FALLBACK_WEIGHT = 1.0


def _fallback_probs(patient_id):
    """Return a normalized fallback probability vector for a patient."""
    prob_dict = patient_probs.get(patient_id)
    if prob_dict is None:
        probs = GLOBAL_PROBS.copy()
    else:
        probs = np.array(list(prob_dict.values()))
    probs = probs / probs.sum()
    return probs




## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_54/3310152883.py in <cell line: 0>()
      1 BASE_DIR = "/kaggle/input/hms-harmful-brain-activity-classification"
      2 
----> 3 df_train = pd.read_csv(os.path.join(BASE_DIR, "train.csv"))
      4 df_test_pd = pd.read_csv(os.path.join(BASE_DIR, "test.csv"))
      5 

NameError: name 'pd' is not defined

## === cell 3
@torch.no_grad()
def gen_ensemble_pred(row):
    """Return a probability vector (length 6) for a test row."""
    eeg_id = row["eeg_id"]
    spc_id = row["spectrogram_id"]
    patient_id = row["patient_id"]

    eeg_filepath = os.path.join(EEG_DIR, f"{eeg_id}.parquet")
    spc_filepath = os.path.join(SPEC_DIR, f"{spc_id}.parquet")

    try:
        kspec = compute_kaggle_spec_from_file(spc_filepath)
        kspec = proc_kspec(kspec)
        kspec = torch.tensor(kspec, device=device).unsqueeze(0)
    except Exception as e:
        return _fallback_probs(patient_id)

    preds = []
    for model in models_spc:
        try:
            preds.append(model(kspec).exp().cpu().numpy().reshape(-1))
        except Exception:
            continue

    if not preds:
        return _fallback_probs(patient_id)

    model_pred = np.mean(preds, axis=0)
    model_pred = model_pred / model_pred.sum()

    fallback = _fallback_probs(patient_id)
    blended = _MODEL_WEIGHT * model_pred + _FALLBACK_WEIGHT * fallback
    blended = blended / blended.sum()
    return blended




## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_54/396541414.py in <cell line: 0>()
----> 1 @torch.no_grad()
      2 def gen_ensemble_pred(row):
      3     """Return a probability vector (length 6) for a test row."""
      4     eeg_id = row["eeg_id"]
      5     spc_id = row["spectrogram_id"]

NameError: name 'torch' is not defined

## === cell 4
preds_final = []
for i in tqdm(range(len(df_test_pd)), desc="Generating predictions"):
    row = df_test_pd.iloc[i]
    pred = gen_ensemble_pred(row)
    preds_final.append(pred)




## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_54/3424709196.py in <cell line: 0>()
      1 preds_final = []
----> 2 for i in tqdm(range(len(df_test_pd)), desc="Generating predictions"):
      3     row = df_test_pd.iloc[i]
      4     pred = gen_ensemble_pred(row)
      5     preds_final.append(pred)

NameError: name 'tqdm' is not defined

## === cell 5
preds_array = np.vstack(preds_final)  # shape (num_rows, 6)

df_sub = pd.DataFrame({"eeg_id": df_test_pd["eeg_id"].tolist()})
df_sub[LABELS] = preds_array
df_sub[LABELS] = df_sub[LABELS].div(df_sub[LABELS].sum(axis=1), axis=0)

output_path = "submission.csv"
df_sub.to_csv(output_path, index=False)
print(f"Submission written to {output_path}")
df_sub.head()

## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_54/3048016424.py in <cell line: 0>()
----> 1 preds_array = np.vstack(preds_final)  # shape (num_rows, 6)
      2 
      3 df_sub = pd.DataFrame({"eeg_id": df_test_pd["eeg_id"].tolist()})
      4 df_sub[LABELS] = preds_array
      5 df_sub[LABELS] = df_sub[LABELS].div(df_sub[LABELS].sum(axis=1), axis=0)

NameError: name 'np' is not defined
