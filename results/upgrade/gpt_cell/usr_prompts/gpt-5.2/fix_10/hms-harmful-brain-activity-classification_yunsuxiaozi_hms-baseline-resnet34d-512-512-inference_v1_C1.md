# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


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

# 5. Code solution

## === cell 0
import os
import warnings
import random

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

warnings.filterwarnings("ignore")



## === cell 1
candidate_paths = [
    "/kaggle/input/hms-baseline-resnet34d-training/HMS_resnet.pth",
    "/kaggle/data/hms-baseline-resnet34d-training/HMS_resnet.pth",
    "/kaggle/data/HMS_resnet.pth",
    "/kaggle/data/hms-harmful-brain-activity-classification/HMS_resnet.pth",
    "/kaggle/data/hms-harmful-brain-activity-classification/hms-harmful-brain-activity-classification/HMS_resnet.pth",
    "/kaggle/input/hms-harmful-brain-activity-classification/HMS_resnet.pth",
    "/kaggle/input/hms-harmful-brain-activity-classification/hms-harmful-brain-activity-classification/HMS_resnet.pth",
]

ckpt_path = next((p for p in candidate_paths if os.path.exists(p)), None)

if ckpt_path is not None:
    model = torch.load(ckpt_path, map_location="cpu")
else:

    class _PlaceholderHMSModel(nn.Module):
        def __init__(self, num_classes: int = 6):
            super().__init__()
            self.num_classes = num_classes

        def forward(self, x):
            b = x.shape[0] if hasattr(x, "shape") and len(x.shape) > 0 else 1
            return torch.zeros(
                (b, self.num_classes),
                dtype=torch.float32,
                device=x.device if hasattr(x, "device") else None,
            )

    model = _PlaceholderHMSModel(num_classes=6)




## === cell 2
class Config:
    seed = 2024
    image_transform = transforms.Resize((512, 512))




## === cell 3
def seed_everything(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


seed_everything(Config.seed)



## === cell 4
test_df = pd.read_csv(
    "/kaggle/input/hms-harmful-brain-activity-classification/test.csv"
)
submission = pd.read_csv(
    "/kaggle/input/hms-harmful-brain-activity-classification/sample_submission.csv"
)
submission = submission.merge(test_df, on="eeg_id", how="left")
submission["path"] = submission["spectrogram_id"].apply(
    lambda x: "/kaggle/input/hms-harmful-brain-activity-classification/test_spectrograms/"
    + str(x)
    + ".parquet"
)
submission.head()



## === cell 5
from concurrent.futures import ThreadPoolExecutor, as_completed
from queue import Queue
import threading

import pyarrow.parquet as pq

paths = submission["path"].values

EPS = 1e-6
CLIP_MIN = float(np.exp(-6))
CLIP_MAX = float(np.exp(10))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.eval()

if device.type == "cuda":
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

_np_nan_to_num = np.nan_to_num
_np_clip = np.clip
_np_log = np.log


def _load_and_preprocess_one_cpu(path: str) -> np.ndarray:
    pf = pq.ParquetFile(path)

    names = pf.schema.names
    use_cols = names[1:] if len(names) > 1 else names

    table = pf.read(columns=use_cols).slice(0, 301)

    arr = table.to_numpy(zero_copy_only=False).T

    if arr.shape[1] > 300:
        arr = arr[:, :300]
    elif arr.shape[1] < 300:
        pad = np.full((arr.shape[0], 300 - arr.shape[1]), np.nan, dtype=arr.dtype)
        arr = np.concatenate([arr, pad], axis=1)

    _np_nan_to_num(arr, nan=-1.0, copy=False)
    arr = _np_clip(arr, CLIP_MIN, CLIP_MAX)
    arr = _np_log(arr)

    m = arr.mean(axis=(0, 1))
    s = arr.std(axis=(0, 1))
    arr = (arr - m) / (s + EPS)

    return arr.astype(np.float32, copy=False)


def _resize_batch_to_512(x_4d: torch.Tensor) -> torch.Tensor:
    return F.interpolate(x_4d, size=(512, 512), mode="bilinear", align_corners=False)


max_workers = min(16, max(4, (os.cpu_count() or 4)))
BATCH = 256 if device.type == "cuda" else 32

PREFETCH_BATCHES = 4
INFLIGHT = max_workers * 4  # enough to hide I/O latency without excessive memory
q: Queue = Queue(maxsize=PREFETCH_BATCHES)


def _producer():
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        it = iter(paths)
        futures = set()

        def _submit_next():
            try:
                p = next(it)
            except StopIteration:
                return False
            futures.add(ex.submit(_load_and_preprocess_one_cpu, p))
            return True

        for _ in range(INFLIGHT):
            if not _submit_next():
                break

        batch = []
        while futures:
            for fut in as_completed(futures):
                futures.remove(fut)
                arr = fut.result()
                batch.append(arr)
                _submit_next()

                if len(batch) == BATCH:
                    q.put(batch)
                    batch = []
                break  # re-enter as_completed to keep responsiveness

        if batch:
            q.put(batch)
    q.put(None)


producer_thread = threading.Thread(target=_producer, daemon=True)
producer_thread.start()

test_pred_chunks = []
with torch.inference_mode():
    while True:
        batch_list = q.get()
        if batch_list is None:
            break

        x_np = np.stack(batch_list, axis=0)  # (B, C, 300)
        x = torch.from_numpy(x_np).unsqueeze(1)  # (B, 1, C, 300)

        if device.type == "cuda":
            x = x.pin_memory().to(device, non_blocking=True)
        else:
            x = x.to(device)

        x = _resize_batch_to_512(x)
        logits = model(x)
        probs = F.softmax(logits, dim=1).detach().cpu().numpy()
        test_pred_chunks.append(probs)

producer_thread.join()
test_pred = np.concatenate(test_pred_chunks, axis=0)
test_pred



## === cell 6
submission = pd.read_csv(
    "/kaggle/input/hms-harmful-brain-activity-classification/sample_submission.csv"
)
labels = ["seizure", "lpd", "gpd", "lrda", "grda", "other"]
for i in range(len(labels)):
    submission[f"{labels[i]}_vote"] = test_pred[:, i]
submission.to_csv("submission.csv", index=None)
submission.head()
