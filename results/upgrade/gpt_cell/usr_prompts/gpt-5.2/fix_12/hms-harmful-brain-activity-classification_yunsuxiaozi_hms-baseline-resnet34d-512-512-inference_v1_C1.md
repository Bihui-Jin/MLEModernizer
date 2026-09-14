# Goal

You will receive environment details and a partial notebook export.

# Requirements

- Fix the bug that causes the error in cell k.
- Do NOT adjust any other non-buggy cells.
- You may reference cell k+1 only to preserve variable/interface compatibility.
- Do not complete or extend code logic in cell k, k+1, or later cells.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (bug fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Output must follow your strict format: Diagnosis / Patch summary / Updated cells / Compatibility notes for cell k+1 / Assumptions.


# 1. Python version

3.12

# 2. Installed packages

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

# 3. Data file paths

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

# 4. Code solution

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
from concurrent.futures import ThreadPoolExecutor
import pyarrow.parquet as pq
import pyarrow.dataset as ds

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

_cpu = os.cpu_count() or 4
torch.set_num_threads(max(1, min(8, _cpu)))
torch.set_num_interop_threads(1)

_np_nan_to_num = np.nan_to_num
_np_clip = np.clip
_np_log = np.log

SPECTRO_DIR = (
    "/kaggle/input/hms-harmful-brain-activity-classification/test_spectrograms"
)
_dir_dataset = ds.dataset(SPECTRO_DIR, format="parquet")
_names = _dir_dataset.schema.names
_cols = _names[1:] if len(_names) > 1 else _names


def _load_and_preprocess_one_cpu(path: str) -> np.ndarray:
    table = pq.read_table(path, columns=_cols).slice(0, 300)
    arr = np.asarray(table.to_numpy(zero_copy_only=False)).T  # (C, 300)

    _np_nan_to_num(arr, nan=-1.0, copy=False)
    arr = _np_clip(arr, CLIP_MIN, CLIP_MAX)
    arr = _np_log(arr)

    m = arr.mean(axis=(0, 1))
    s = arr.std(axis=(0, 1))
    arr = (arr - m) / (s + EPS)

    return arr.astype(np.float32, copy=False)


def _resize_batch_to_512(x_4d: torch.Tensor) -> torch.Tensor:
    return F.interpolate(x_4d, size=(512, 512), mode="bilinear", align_corners=False)


max_workers = min(12, max(4, _cpu))
BATCH = 256 if device.type == "cuda" else 64


def _iter_batches(iterable, batch_size):
    batch = []
    for x in iterable:
        batch.append(x)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


test_pred_chunks = []
with ThreadPoolExecutor(max_workers=max_workers) as ex, torch.inference_mode():
    mapped = ex.map(_load_and_preprocess_one_cpu, paths, chunksize=16)
    for batch_list in _iter_batches(mapped, BATCH):
        x_np = np.stack(batch_list, axis=0)
        x = torch.from_numpy(x_np).unsqueeze(1)

        if device.type == "cuda":
            x = x.pin_memory().to(device, non_blocking=True)
        else:
            x = x.to(device)

        x = _resize_batch_to_512(x)
        logits = model(x)
        probs = F.softmax(logits, dim=1).detach().cpu().numpy()
        test_pred_chunks.append(probs)

test_pred = np.concatenate(test_pred_chunks, axis=0)
test_pred



## --- ERROR in cell 5, traceback:
[0;31m---------------------------------------------------------------------------[0m
[0;31mAttributeError[0m                            Traceback (most recent call last)
[0;32m/tmp/ipykernel_11/29157862.py[0m in [0;36m<cell line: 0>[0;34m()[0m
[1;32m     79[0m     [0;31m# chunksize tunes task dispatch overhead; doesn't change computation results.[0m[0;34m[0m[0;34m[0m[0m
[1;32m     80[0m     [0mmapped[0m [0;34m=[0m [0mex[0m[0;34m.[0m[0mmap[0m[0;34m([0m[0m_load_and_preprocess_one_cpu[0m[0;34m,[0m [0mpaths[0m[0;34m,[0m [0mchunksize[0m[0;34m=[0m[0;36m16[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0;32m---> 81[0;31m     [0;32mfor[0m [0mbatch_list[0m [0;32min[0m [0m_iter_batches[0m[0;34m([0m[0mmapped[0m[0;34m,[0m [0mBATCH[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     82[0m         [0mx_np[0m [0;34m=[0m [0mnp[0m[0;34m.[0m[0mstack[0m[0;34m([0m[0mbatch_list[0m[0;34m,[0m [0maxis[0m[0;34m=[0m[0;36m0[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m     83[0m         [0mx[0m [0;34m=[0m [0mtorch[0m[0;34m.[0m[0mfrom_numpy[0m[0;34m([0m[0mx_np[0m[0;34m)[0m[0;34m.[0m[0munsqueeze[0m[0;34m([0m[0;36m1[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m

[0;32m/tmp/ipykernel_11/29157862.py[0m in [0;36m_iter_batches[0;34m(iterable, batch_size)[0m
[1;32m     66[0m [0;32mdef[0m [0m_iter_batches[0m[0;34m([0m[0miterable[0m[0;34m,[0m [0mbatch_size[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m     67[0m     [0mbatch[0m [0;34m=[0m [0;34m[[0m[0;34m][0m[0;34m[0m[0;34m[0m[0m
[0;32m---> 68[0;31m     [0;32mfor[0m [0mx[0m [0;32min[0m [0miterable[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     69[0m         [0mbatch[0m[0;34m.[0m[0mappend[0m[0;34m([0m[0mx[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m     70[0m         [0;32mif[0m [0mlen[0m[0;34m([0m[0mbatch[0m[0;34m)[0m [0;34m==[0m [0mbatch_size[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/lib/python3.11/concurrent/futures/_base.py[0m in [0;36mresult_iterator[0;34m()[0m
[1;32m    617[0m                     [0;31m# Careful not to keep a reference to the popped future[0m[0;34m[0m[0;34m[0m[0m
[1;32m    618[0m                     [0;32mif[0m [0mtimeout[0m [0;32mis[0m [0;32mNone[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 619[0;31m                         [0;32myield[0m [0m_result_or_cancel[0m[0;34m([0m[0mfs[0m[0;34m.[0m[0mpop[0m[0;34m([0m[0;34m)[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    620[0m                     [0;32melse[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m    621[0m                         [0;32myield[0m [0m_result_or_cancel[0m[0;34m([0m[0mfs[0m[0;34m.[0m[0mpop[0m[0;34m([0m[0;34m)[0m[0;34m,[0m [0mend_time[0m [0;34m-[0m [0mtime[0m[0;34m.[0m[0mmonotonic[0m[0;34m([0m[0;34m)[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/lib/python3.11/concurrent/futures/_base.py[0m in [0;36m_result_or_cancel[0;34m(***failed resolving arguments***)[0m
[1;32m    315[0m     [0;32mtry[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m    316[0m         [0;32mtry[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 317[0;31m             [0;32mreturn[0m [0mfut[0m[0;34m.[0m[0mresult[0m[0;34m([0m[0mtimeout[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    318[0m         [0;32mfinally[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m    319[0m             [0mfut[0m[0;34m.[0m[0mcancel[0m[0;34m([0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/lib/python3.11/concurrent/futures/_base.py[0m in [0;36mresult[0;34m(self, timeout)[0m
[1;32m    447[0m                     [0;32mraise[0m [0mCancelledError[0m[0;34m([0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m    448[0m                 [0;32melif[0m [0mself[0m[0;34m.[0m[0m_state[0m [0;34m==[0m [0mFINISHED[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 449[0;31m                     [0;32mreturn[0m [0mself[0m[0;34m.[0m[0m__get_result[0m[0;34m([0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    450[0m [0;34m[0m[0m
[1;32m    451[0m                 [0mself[0m[0;34m.[0m[0m_condition[0m[0;34m.[0m[0mwait[0m[0;34m([0m[0mtimeout[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/lib/python3.11/concurrent/futures/_base.py[0m in [0;36m__get_result[0;34m(self)[0m
[1;32m    399[0m         [0;32mif[0m [0mself[0m[0;34m.[0m[0m_exception[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m    400[0m             [0;32mtry[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 401[0;31m                 [0;32mraise[0m [0mself[0m[0;34m.[0m[0m_exception[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    402[0m             [0;32mfinally[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m    403[0m                 [0;31m# Break a reference cycle with the exception in self._exception[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/lib/python3.11/concurrent/futures/thread.py[0m in [0;36mrun[0;34m(self)[0m
[1;32m     56[0m [0;34m[0m[0m
[1;32m     57[0m         [0;32mtry[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m---> 58[0;31m             [0mresult[0m [0;34m=[0m [0mself[0m[0;34m.[0m[0mfn[0m[0;34m([0m[0;34m*[0m[0mself[0m[0;34m.[0m[0margs[0m[0;34m,[0m [0;34m**[0m[0mself[0m[0;34m.[0m[0mkwargs[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     59[0m         [0;32mexcept[0m [0mBaseException[0m [0;32mas[0m [0mexc[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m     60[0m             [0mself[0m[0;34m.[0m[0mfuture[0m[0;34m.[0m[0mset_exception[0m[0;34m([0m[0mexc[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m

[0;32m/tmp/ipykernel_11/29157862.py[0m in [0;36m_load_and_preprocess_one_cpu[0;34m(path)[0m
[1;32m     41[0m     [0;31m# Still reads same columns and same first 300 rows.[0m[0;34m[0m[0;34m[0m[0m
[1;32m     42[0m     [0mtable[0m [0;34m=[0m [0mpq[0m[0;34m.[0m[0mread_table[0m[0;34m([0m[0mpath[0m[0;34m,[0m [0mcolumns[0m[0;34m=[0m[0m_cols[0m[0;34m)[0m[0;34m.[0m[0mslice[0m[0;34m([0m[0;36m0[0m[0;34m,[0m [0;36m300[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0;32m---> 43[0;31m     [0marr[0m [0;34m=[0m [0mnp[0m[0;34m.[0m[0masarray[0m[0;34m([0m[0mtable[0m[0;34m.[0m[0mto_numpy[0m[0;34m([0m[0mzero_copy_only[0m[0;34m=[0m[0;32mFalse[0m[0;34m)[0m[0;34m)[0m[0;34m.[0m[0mT[0m  [0;31m# (C, 300)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     44[0m [0;34m[0m[0m
[1;32m     45[0m     [0m_np_nan_to_num[0m[0;34m([0m[0marr[0m[0;34m,[0m [0mnan[0m[0;34m=[0m[0;34m-[0m[0;36m1.0[0m[0;34m,[0m [0mcopy[0m[0;34m=[0m[0;32mFalse[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m

[0;31mAttributeError[0m: 'pyarrow.lib.Table' object has no attribute 'to_numpy'

## === cell 6
submission = pd.read_csv(
    "/kaggle/input/hms-harmful-brain-activity-classification/sample_submission.csv"
)
labels = ["seizure", "lpd", "gpd", "lrda", "grda", "other"]
for i in range(len(labels)):
    submission[f"{labels[i]}_vote"] = test_pred[:, i]
submission.to_csv("submission.csv", index=None)
submission.head()
