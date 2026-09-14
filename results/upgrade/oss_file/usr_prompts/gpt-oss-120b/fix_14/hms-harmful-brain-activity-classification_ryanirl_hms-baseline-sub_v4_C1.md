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

cudf-polars-cu12==25.6.0
geopandas==0.14.4
librosa==0.11.0
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
timm==1.0.19
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

0.4881291529503589

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 1.40989) has done: 'The fix updates **NewModel** to use a single‑channel input (the model was created with `in_chans=4` but receives one channel at a time), which resolves the channel‑size runtime error. No other logic is changed, so the pipeline now runs end‑to‑end and produces a correctly‑shaped CSV submission.'
- What this solution (achieved 1.39902) has done: 'I import missing modules, ensure dataframes are correctly loaded as pandas objects for later indexing, add the tqdm import, and fix the variable references so the pipeline runs end‑to‑end and writes a valid `submission.csv` with probabilities that sum to 1.'
- What this solution (achieved 1.39613) has done: 'The fix adds necessary imports, defines the data paths, loads the train and test tables, and sets the EEG directory so the later functions can locate the parquet files. It also ensures `pl` (polars) and `pd` (pandas) are available, eliminating the NameError issues and allowing the pipeline to run end‑to‑end and write a proper `submission.csv` with probabilities that sum to one.'
- What this solution (achieved 1.39779) has done: 'The fix adds all missing imports, defines the device and data directories, loads the train and test metadata, ensures the `pl` (polars) and `tqdm` packages are available, and corrects variable names so the model ensemble can run and produce a proper `submission.csv` whose rows sum to 1.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import timm
from timm.layers import SelectAdaptivePool2d
from tqdm import tqdm
import concurrent.futures

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BASE_PATH = os.path.join("data", "hms-harmful-brain-activity-classification")
TRAIN_PATH = os.path.join(BASE_PATH, "train.csv")
TEST_PATH = os.path.join(BASE_PATH, "test.csv")

df_train = pd.read_csv(TRAIN_PATH)
df_test = pd.read_csv(TEST_PATH)

LABELS = [
    "seizure_vote",
    "lpd_vote",
    "gpd_vote",
    "lrda_vote",
    "grda_vote",
    "other_vote",
]

train_votes = df_train[LABELS].values.astype(float)
row_sums = train_votes.sum(axis=1, keepdims=True)
row_sums[row_sums == 0] = 1.0
class_prior = train_votes / row_sums
class_prior = class_prior.mean(axis=0)
class_prior = class_prior / class_prior.sum()

alpha = 0.2

models = []




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_55/2874511611.py in <cell line: 0>()
     18 
     19 # Load metadata tables
---> 20 df_train = pd.read_csv(TRAIN_PATH)
     21 df_test = pd.read_csv(TEST_PATH)
     22 

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in read_csv(filepath_or_buffer, sep, delimiter, header, names, index_col, usecols, dtype, engine, converters, true_values, false_values, skipinitialspace, skiprows, skipfooter, nrows, na_values, keep_default_na, na_filter, verbose, skip_blank_lines, parse_dates, infer_datetime_format, keep_date_col, date_parser, date_format, dayfirst, cache_dates, iterator, chunksize, compression, thousands, decimal, lineterminator, quotechar, quoting, doublequote, escapechar, comment, encoding, encoding_errors, dialect, on_bad_lines, delim_whitespace, low_memory, memory_map, float_precision, storage_options, dtype_backend)
   1024     kwds.update(kwds_defaults)
   1025 
-> 1026     return _read(filepath_or_buffer, kwds)
   1027 
   1028 

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in _read(filepath_or_buffer, kwds)
    618 
    619     # Create the parser.
--> 620     parser = TextFileReader(filepath_or_buffer, **kwds)
    621 
    622     if chunksize or iterator:

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in __init__(self, f, engine, **kwds)
   1618 
   1619         self.handles: IOHandles | None = None
-> 1620         self._engine = self._make_engine(f, self.engine)
   1621 
   1622     def close(self) -> None:

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in _make_engine(self, f, engine)
   1878                 if "b" not in mode:
   1879                     mode += "b"
-> 1880             self.handles = get_handle(
   1881                 f,
   1882                 mode,

/usr/local/lib/python3.11/dist-packages/pandas/io/common.py in get_handle(path_or_buf, mode, encoding, compression, memory_map, is_text, errors, storage_options)
    871         if ioargs.encoding and "b" not in ioargs.mode:
    872             # Encoding
--> 873             handle = open(
    874                 handle,
    875                 ioargs.mode,

FileNotFoundError: [Errno 2] No such file or directory: 'data/hms-harmful-brain-activity-classification/train.csv'

## === cell 1
class NewModel(nn.Module):
    """
    Adjusted to accept a single‑channel image per slice.
    The original model was instantiated with `in_chans=4` but the forward
    method feeds one channel at a time, causing a channel‑mismatch error.
    Setting `in_chans=1` aligns the architecture with the actual input shape.
    """

    def __init__(self, pretrained=False):
        super().__init__()
        self.m0 = timm.create_model(
            "fastvit_t8.apple_in1k",
            pretrained=pretrained,
            num_classes=6,
            in_chans=1,
            features_only=True,
        )
        self.pool_0 = SelectAdaptivePool2d(
            pool_type="avg", flatten=True, input_fmt="NCHW"
        )
        self.fc = nn.Sequential(nn.Linear(384 * 4, 6), nn.Sigmoid())

    def forward(self, x):
        x0 = self.pool_0(self.m0(x[:, 0:1])[-1])
        x1 = self.pool_0(self.m0(x[:, 1:2])[-1])
        x2 = self.pool_0(self.m0(x[:, 2:3])[-1])
        x3 = self.pool_0(self.m0(x[:, 3:4])[-1])
        embed = torch.concat([x0, x1, x2, x3], dim=1)
        out = self.fc(embed)
        out = out + 0.001
        out = out / out.sum(axis=1, keepdim=True)
        return out, embed

    @torch.no_grad()
    def predict(self, x):
        x = torch.Tensor(x).unsqueeze(0).to(device)
        pred, _ = self.forward(x)
        return pred.detach().cpu().numpy().reshape(-1)




## === cell 2
if models and isinstance(models[0], NewModel):
    def gen_ensemble_pred(models, row):
        preds = []
        for m in models:
            dummy_input = np.zeros((4, 224, 224), dtype=np.float32)  # shape placeholder
            pred = m.predict(dummy_input)
            preds.append(pred)
        return np.mean(preds, axis=0)

    def _process_index(idx):
        """Generate blended prediction for a single test row."""
        raw_pred = gen_ensemble_pred(models, df_test.iloc[idx])
        blended = alpha * raw_pred + (1 - alpha) * class_prior
        blended = blended / (blended.sum() + 1e-12)
        return idx, blended

    n_test = len(df_test)
    preds_array = np.empty((n_test, len(LABELS)), dtype=np.float32)

    max_workers = min(8, (os.cpu_count() or 1))
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(
            tqdm(
                executor.map(_process_index, range(n_test)),
                total=n_test,
                desc="Generating predictions",
            )
        )
        for idx, blended in results:
            preds_array[idx] = blended

    preds_final = preds_array
else:
    preds_final = np.tile(class_prior, (len(df_test), 1))



## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/856026061.py in <cell line: 0>()
      1 # Prediction generation (uses class prior if no models are provided)
----> 2 if models and isinstance(models[0], NewModel):
      3     # Stub for ensemble prediction – replace with actual implementation if models exist
      4     def gen_ensemble_pred(models, row):
      5         # Simple average of model predictions (placeholder)

NameError: name 'models' is not defined

## === cell 3
submission = pd.DataFrame(preds_final, columns=LABELS)
submission.insert(0, "eeg_id", df_test["eeg_id"].values)

row_sums = submission[LABELS].sum(axis=1)
submission[LABELS] = submission[LABELS].div(row_sums, axis=0)

submission_path = "submission.csv"
submission.to_csv(submission_path, index=False)
print(f"Submission saved to {submission_path}")

## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3776414183.py in <cell line: 0>()
      1 # Build submission DataFrame
----> 2 submission = pd.DataFrame(preds_final, columns=LABELS)
      3 submission.insert(0, "eeg_id", df_test["eeg_id"].values)
      4 
      5 # Ensure each row sums to 1 (small numerical adjustment)

NameError: name 'preds_final' is not defined
