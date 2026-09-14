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
geopandas==0.14.4
librosa==0.11.0
mne==1.10.2
numpy==1.26.4
opencv-python==4.12.0.88
opencv-python-headless==4.12.0.88
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
pytorch-ignite==0.5.3
pytorch-lightning==2.5.5
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

0.4102354133265912

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 1.41937) has done: 'I guard the weight‑loading steps so they are only executed when the expected directories exist, and fall back to a simple baseline that uses class‑wise vote frequencies from the training data. This removes the `FileNotFoundError` and the “Is a directory” error, ensures the prediction array has the correct shape, and writes a valid `submission.csv` where each row sums to 1.'
- What this solution (achieved 1.68479) has done: 'I add a lightweight patient‑wise prior fallback so that when no model weights are found the script uses per‑patient vote distributions (or the overall class distribution if a patient is unseen). This keeps the core pipeline unchanged, guarantees a valid probability matrix, and should lower the KL divergence toward the target score.'
- What this solution (achieved 1.68479) has done: 'I add a lightweight patient‑wise prior that is combined with any model predictions instead of using the raw model output alone. By averaging the model predictions with the per‑patient (or overall) vote distribution we keep the existing pipeline but bias the results toward a more reliable prior, which is expected to lower the KL divergence and move the score closer to the target (lower is better).'
- What this solution (achieved 1.68479) has done: 'I keep the overall pipeline unchanged but adjust the blending of model predictions with the patient‑wise priors. Since the current score is much higher than the target (lower is better), giving more weight to the reliable priors should reduce the KL divergence. I introduce a modest blend factor (30 % model, 70 % prior) and then renormalize, which is a minimal change aimed at moving the score toward the target.'
- What this solution (achieved 1.68479) has done: 'I reduce the influence of the model predictions by setting the blend weight to 0.0, so the final submission relies solely on the patient‑wise or overall priors (which are more reliable given the current high KL score). This minimal change keeps the core pipeline intact while moving the score closer to the target (0.4102) by eliminating noisy model contributions.'
- What this solution (achieved 1.41937) has done: 'I replace the patient‑wise prior with a simple overall class prior for every test sample, eliminating the noisy per‑patient distributions that are inflating the KL divergence. This minimal change keeps the rest of the pipeline unchanged and should lower the score toward the target while still producing a valid CSV submission.'
- What this solution (achieved 1.68479) has done: 'I added all missing imports (os, pandas, numpy, torch, torch.nn, DataLoader, tqdm, albumentations, copy, random, mne, gc, timm, torchaudio) and corrected the weight‑listing logic. I also ensured the model classes can be defined without errors, converted the predictions list to a NumPy array before normalising, and kept the blend weight at 0.0 so the submission relies only on the reliable class‑wise priors. Finally, the script now writes a proper `submission.csv` with rows that sum to 1.'
- What this solution (achieved 1.68479) has done: 'I increase the blend between model predictions and the patient/overall priors from 0 % model to a modest 50 % model weight. This keeps the core pipeline unchanged while allowing the learned models to contribute useful signal, which should lower the KL‑divergence toward the target. The change is limited to the blend weight setting and includes a short comment explaining the intent.'

# 9. Code solution

## === cell 0
import os
import copy
import random
import gc

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchaudio
import timm
import albumentations as A
from tqdm.auto import tqdm
import mne

CFG = {
    "data": "./data/hms-harmful-brain-activity-classification/test.csv",
    "train": "./data/hms-harmful-brain-activity-classification/train.csv",
    "weights_spec_dir": "./weights/spec",
    "weights_eeg_dir": "./weights/eeg",
    "batch_size": 64,
    "num_worker": 0,
    "flip": False,  # no test‑time flipping
}


def list_weights(dir_path):
    if os.path.isdir(dir_path):
        return [os.path.join(dir_path, x) for x in sorted(os.listdir(dir_path))]
    return []


CFG["weights_spec"] = list_weights(CFG["weights_spec_dir"])
CFG["weights_eeg"] = list_weights(CFG["weights_eeg_dir"])
CFG




## === cell 1
class AlaskaDataIter:
    def __init__(
        self, df, training_flag=False, shuffle=False, use_eeg=False, ll=0, rr=20
    ):
        self.ll = ll
        self.rr = rr
        self.training_flag = training_flag
        self.shuffle = shuffle
        self.df = df
        self.use_eeg = use_eeg

        self.train_trans = A.Compose([A.HorizontalFlip(p=0.5)])

        TARS = {"Seizure": 0, "LPD": 1, "GPD": 2, "LRDA": 3, "GRDA": 4, "Other": 5}
        self.TARS2 = {x: y for y, x in TARS.items()}

        self.eeg_nms = [
            "Fp1",
            "F3",
            "C3",
            "P3",
            "F7",
            "T3",
            "T5",
            "O1",
            "Fz",
            "Cz",
            "Pz",
            "Fp2",
            "F4",
            "C4",
            "P4",
            "F8",
            "T4",
            "T6",
            "O2",
            "EKG",
        ]
        self.LL = ["Fp1", "F7", "T3", "T5", "O1"]
        self.RR = ["Fp2", "F8", "T4", "T6", "O2"]
        self.LP = ["Fp1", "F3", "C3", "P3", "O1"]
        self.RP = ["Fp2", "F4", "C4", "P4", "O2"]
        self.leads_dict = {value: index for index, value in enumerate(self.eeg_nms)}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return self.single_map_func(self.df.iloc[idx], self.training_flag)

    def brain_lead(self, waves):
        waves = copy.deepcopy(waves)
        brain_leads = [self.LL, self.LP, self.RP, self.RR]  # corrected fourth entry
        leads = []
        for combine in brain_leads:
            for i in range(len(combine) - 1):
                tmp = (
                    waves[self.leads_dict[combine[i]]]
                    - waves[self.leads_dict[combine[i + 1]]]
                )
                leads.append(tmp)
        return np.concatenate(leads, axis=0)

    def single_map_func(self, dp, is_training):
        if self.use_eeg:
            eeg_path = f"./data/hms-harmful-brain-activity-classification/test_eegs/{dp['eeg_id']}.parquet"
            eeg = pd.read_parquet(eeg_path)
            offset = 0
            if random.random() < 1.0 and is_training:
                offset += random.uniform(-1, 1)
                offset = np.clip(offset, a_min=0, a_max=10)  # dummy bounds
            eeg = eeg.iloc[int(offset) * 200 : (int(offset) + 50) * 200]
            waves = eeg.values.T
            for i in range(waves.shape[0]):
                m = np.nanmean(waves[i])
                if np.isnan(waves[i]).mean() < 1:
                    waves[i] = np.nan_to_num(waves[i], nan=m)
                else:
                    waves[i] = 0
            waves = mne.filter.filter_data(waves, 200, self.ll, self.rr, verbose=False)
            data = self.brain_lead(waves)
        else:
            spec_path = f"./data/hms-harmful-brain-activity-classification/test_spectrograms/{dp['spectrogram_id']}.parquet"
            spec = pd.read_parquet(spec_path).values[:, 1:]
            images = []
            for region in range(4):
                img = spec[0:300, region * 100 : (region + 1) * 100].T
                img = np.clip(img, np.exp(-4), np.exp(8))
                img = np.log(img)
                img = np.nan_to_num(img, nan=0.0)
                images.append(img)
            images = np.stack(images, -1)
            data = np.transpose(images, [2, 0, 1])
        return data.astype(np.float32)




## === cell 2
class NetSpec(nn.Module):
    def __init__(self, num_classes=6):
        super().__init__()
        self.model = timm.create_model("efficientnet_b5", pretrained=False, in_chans=3)
        self.fc = nn.Linear(2048, num_classes, bias=True)
        self.dropout = nn.Dropout(0.5)
        self.avg = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        bs = x.size(0)
        x1 = torch.cat([x[:, i : i + 1, :, :] for i in range(4)], dim=2)
        x1 = torch.cat([x1, x1, x1], dim=1)
        if CFG["flip"]:
            x = torch.cat([x1, torch.flip(x1, [3])], dim=0)
        else:
            x = x1
        x = self.model.forward_features(x)
        x = self.avg(x)
        x = x.view(-1, 2048)
        x = self.dropout(x)
        x = self.fc(x)
        x = torch.softmax(x, dim=-1)
        if CFG["flip"]:
            x = (x[:bs] + x[bs:]) / 2.0
        return x




## === cell 3
class Transform(nn.Module):
    def __init__(self):
        super().__init__()
        self.wave_transform = torchaudio.transforms.Spectrogram(
            n_fft=512, hop_length=25, power=1
        )
        self.am2db = torchaudio.transforms.AmplitudeToDB(stype="magnitude", top_db=80)

    def forward(self, x):
        img = self.wave_transform(x)
        img = self.am2db(img)
        n, c, h, w = img.size()
        img = img[:, :, : int(0.2 * h + 30), :]
        img = img.view(n, 4, -1, w)
        return img




## === cell 4
class NetEeg(nn.Module):
    def __init__(self, num_classes=6):
        super().__init__()
        self.preprocess = Transform()
        self.model = timm.create_model("efficientnet_b5", pretrained=False, in_chans=4)
        self.fc = nn.Linear(2048, num_classes, bias=True)
        self.dropout = nn.Dropout(0.5)
        self.avg = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        bs = x.size(0)
        x = self.preprocess(x)
        if CFG["flip"]:
            x = torch.cat([x, torch.flip(x, [3])], dim=0)
        x = self.model.forward_features(x)
        x = self.avg(x)
        x = x.view(-1, 2048)
        x = self.dropout(x)
        x = self.fc(x)
        x = torch.softmax(x, dim=-1)
        if CFG["flip"]:
            x = (x[:bs] + x[bs:]) / 2.0
        return x




## === cell 5
def inference_function(loader, model, device):
    model.eval()
    preds = []
    with torch.no_grad(), tqdm(loader, unit="batch", desc="Inference") as pbar:
        for X in pbar:
            X = X.to(device)
            out = model(X)
            preds.append(out.cpu().numpy())
    return np.concatenate(preds, axis=0)




## === cell 6
test_df = pd.read_csv(CFG["data"])
train_df = pd.read_csv(CFG["train"])
print(f"Test rows: {len(test_df)}, Train rows: {len(train_df)}")



## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_55/1058293901.py in <cell line: 0>()
----> 1 test_df = pd.read_csv(CFG["data"])
      2 train_df = pd.read_csv(CFG["train"])
      3 print(f"Test rows: {len(test_df)}, Train rows: {len(train_df)}")
      4 

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

FileNotFoundError: [Errno 2] No such file or directory: './data/hms-harmful-brain-activity-classification/test.csv'

## === cell 7
predictions = []
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if CFG["weights_spec"]:
    for weight_path in CFG["weights_spec"]:
        dataset = AlaskaDataIter(test_df, training_flag=False, shuffle=False)
        loader = DataLoader(
            dataset,
            batch_size=CFG["batch_size"],
            num_workers=CFG["num_worker"],
            shuffle=False,
        )
        model = NetSpec().to(device)
        state = torch.load(weight_path, map_location=device)
        model.load_state_dict(state, strict=False)
        pred = inference_function(loader, model, device)
        predictions.append(pred)
        torch.cuda.empty_cache()
        gc.collect()
else:
    print("No spec weights found – skipping spec inference.")

if CFG["weights_eeg"]:
    for weight_path in CFG["weights_eeg"]:
        dataset = AlaskaDataIter(
            test_df, training_flag=False, shuffle=False, use_eeg=True
        )
        loader = DataLoader(
            dataset,
            batch_size=CFG["batch_size"],
            num_workers=CFG["num_worker"],
            shuffle=False,
        )
        model = NetEeg().to(device)
        state = torch.load(weight_path, map_location=device)
        model.load_state_dict(state, strict=False)
        pred = inference_function(loader, model, device)
        predictions.append(pred)
        torch.cuda.empty_cache()
        gc.collect()
else:
    print("No EEG weights found – skipping EEG inference.")

vote_cols = [
    "seizure_vote",
    "lpd_vote",
    "gpd_vote",
    "lrda_vote",
    "grda_vote",
    "other_vote",
]

patient_group = train_df.groupby("patient_id")[vote_cols].sum()
patient_prior = patient_group.div(patient_group.sum(axis=1), axis=0)

overall_counts = train_df[vote_cols].sum()
overall_prior = overall_counts / overall_counts.sum()

priors = []
for pid in test_df["patient_id"].values:
    if pid in patient_prior.index:
        priors.append(patient_prior.loc[pid].values)
    else:
        priors.append(overall_prior.values)
priors = np.array(priors)

if not predictions:
    final_preds = priors
else:
    blend_weight = 0.0  # 0% model, 100% prior (baseline)
    model_mean = np.mean(np.stack(predictions, axis=0), axis=0)
    final_preds = blend_weight * model_mean + (1.0 - blend_weight) * priors



## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/222267450.py in <cell line: 0>()
     52 
     53 # Compute patient‑wise and overall priors from the training data
---> 54 patient_group = train_df.groupby("patient_id")[vote_cols].sum()
     55 patient_prior = patient_group.div(patient_group.sum(axis=1), axis=0)
     56 

NameError: name 'train_df' is not defined

## === cell 8
row_sums = final_preds.sum(axis=1, keepdims=True)
row_sums[row_sums == 0] = 1.0
final_preds = final_preds / row_sums

TARGETS = [
    "seizure_vote",
    "lpd_vote",
    "gpd_vote",
    "lrda_vote",
    "grda_vote",
    "other_vote",
]
sub = pd.DataFrame({"eeg_id": test_df["eeg_id"].values})
sub[TARGETS] = final_preds
sub.to_csv("submission.csv", index=False)
print(f"Submission shape: {sub.shape}")
print(sub.head())

## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1674158034.py in <cell line: 0>()
      1 # Ensure each row sums to 1
----> 2 row_sums = final_preds.sum(axis=1, keepdims=True)
      3 row_sums[row_sums == 0] = 1.0
      4 final_preds = final_preds / row_sums
      5 

NameError: name 'final_preds' is not defined
