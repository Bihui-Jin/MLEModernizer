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

0.5243078618326658

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 1.41937) has done: 'I fix the early runtime crash caused by an incompatible protobuf/h5py stack by forcing TensorFlow to use the pure-Python protobuf implementation before importing TensorFlow (a common Kaggle TF workaround). Then I make the model-loading step robust by auto-detecting the correct `/kaggle/input/...` directory and weight filenames (since `models20240130/EB2_v1_f*.h5` is not present), and fall back to writing a valid, normalized baseline submission if no weights are found. Finally, I ensure the script always produces `submission.csv` with the exact required columns and rows summing to 1, so you get a valid submission and a non-error score.'
- What this solution (achieved 1.41937) has done: 'I fix the TensorFlow/protobuf crash by forcing the pure-Python protobuf implementation *and* setting the protobuf runtime version env var before importing TensorFlow (this specific error is common when TF meets newer protobuf). Then I keep your inference logic intact but make sure the script always produces a valid `submission.csv` even if TF still fails to import (by catching the import error and writing the prior-based baseline). Finally, I keep all probabilities safely normalized/clipped to satisfy the KL-divergence submission constraints; these changes are primarily stability fixes and should not worsen your score, and if weights exist they be used as before.'
- What this solution (achieved 1.41937) has done: 'We fix the TensorFlow/protobuf import crash that stops the notebook at cell 0 by forcing safer protobuf runtime settings *and* adding a defensive fallback so the pipeline continues even if TF cannot import. Then we make the weight-file discovery actually find the pre-trained fold weights from any Kaggle dataset/input folder (instead of only a very specific filename pattern), so inference uses the intended model instead of the weak prior-baseline that yields ~1.42. Finally, we keep your model/inference logic unchanged but ensure the submission is always aligned to `sample_submission.csv` order, clipped, and row-normalized to satisfy the KL metric requirements.'
- What this solution (achieved 1.41937) has done: 'I fix the TensorFlow/protobuf crash by setting safer protobuf-related environment variables *before* any TensorFlow-related import happens, and by catching the specific import failure so the pipeline continues. Then I keep your inference core logic unchanged but make the TF availability detection reliable (so it doesn’t incorrectly proceed into TF code when the import is partially broken). Finally, I ensure the script always writes a valid `submission.csv` with the exact required columns and properly normalized probabilities, using model weights when TF loads successfully and falling back to the prior baseline otherwise (score should improve versus the current baseline-only behavior when weights are actually found/usable).'
- What this solution (achieved 1.41937) has done: 'We fix the TensorFlow/protobuf crash (`MessageFactory.GetPrototype` missing) by setting `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` early (already) and additionally pinning protobuf to the pure-Python backend via `protobuf` import order + guarding TF import so the pipeline doesn’t die before writing `submission.csv`. Next, we ensure inference actually uses pre-trained weights by broadening weight discovery to include common Kaggle dataset structures and by sorting/selecting fold files deterministically (so you don’t accidentally average unrelated `.h5` files). Finally, we keep your model and generator logic intact, but we make submission alignment strictly follow `sample_submission.csv` order and enforce safe normalization/clipping to reduce KL-divergence failures and improve score versus the baseline.'
- What this solution (achieved 1.41937) has done: 'We fix the early TensorFlow/protobuf crash that prevents the pipeline from running by forcing the pure-Python protobuf backend earlier and additionally disabling C++ protobuf in TF via environment variables before any protobuf/TF import. Then, to move the score down toward the 0.52 target (lower is better) without changing your model/inference logic, we ensure pretrained weights can actually be found by broadening the search to the specific competition input folder (not just a non-existent `models20240130` dataset) while still keeping deterministic selection. Finally, we keep the exact submission schema and enforce safe clipping + row-normalization so KL-divergence constraints are always satisfied and `submission.csv` is always produced.'
- What this solution (achieved 1.41937) has done: 'We fix the TensorFlow/protobuf crash by forcing the pure-Python protobuf backend *and* preventing an incompatible preloaded `google.protobuf` from being used (common on Kaggle with TF + protobuf>=4), so the script can actually run inference instead of always falling back to the weak prior baseline. Next, we ensure the code doesn’t crash even if TF still fails by keeping the existing baseline fallback, but only after the improved TF import attempt. Finally, we keep your model/inference logic intact and only add safe submission alignment + normalization checks (score-neutral) so the produced `submission.csv` is always valid for the KL metric and should move your score down toward the 0.52 target by enabling real model predictions.'
- What this solution (achieved 1.41937) has done: 'We stop the `MessageFactory.GetPrototype` crash by enforcing the pure-Python protobuf implementation *before any protobuf/TensorFlow import*, and by avoiding pre-importing `google.protobuf` (which can lock in an incompatible backend). Then we only proceed with the TensorFlow inference path when the TF import truly succeeds; otherwise we deterministically fall back to the prior-based baseline to always produce a valid `submission.csv`. These changes are stability-focused and should improve score versus the current baseline-only behavior by allowing the pretrained-weight inference to run when weights are present, while keeping your core model/data logic unchanged. Finally, we keep strict KL-safe clipping + row-normalization and preserve exact submission column order.'
- What this solution (achieved 1.41937) has done: 'We fix the early TensorFlow/protobuf crash by making the protobuf environment variables take effect before any TensorFlow/protobuf import and by isolating TensorFlow import in a subprocess-free, safe path (still using your exact inference logic when TF loads). Then we correct the fold-averaging logic bug: right now the code loads multiple weight files into the same model without rebuilding, so later folds overwrite earlier ones and the “mean” is effectively the last fold; we rebuild the model inside the fold loop to actually ensemble as intended (this should move your KL score down toward the 0.524 target). Finally, we keep your submission schema unchanged and continue strict clipping + row-normalization to satisfy the KL constraints.'

# 9. Code solution

## === cell 0
import os
import gc
import warnings
import glob
import sys

warnings.filterwarnings("ignore")

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION"] = "2"
os.environ["TF_DISABLE_CPP_PROTOBUF"] = "1"
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

for m in list(sys.modules.keys()):
    if m.startswith("google.protobuf") or m == "google.protobuf":
        del sys.modules[m]

PLATFORM = "kaggle"  # local 平台 或 kaggle 平台
NEEDTRAIN = False  # 是否需要训练，如果线上 infer 则不需要
READ_SPEC_FILES = False  # 是否需要预处理谱图
READ_EEG_FILES = False  # 是否需要预处理脑电数据
LOAD_MODELS_FROM = "models20240130"  # 训练好的模型保存位置，调用直接 infer
if PLATFORM == "local":  # 模型加载路径
    LOAD_MODELS_FROM = f"./input/{LOAD_MODELS_FROM}"
elif PLATFORM == "kaggle":
    LOAD_MODELS_FROM = f"/kaggle/input/{LOAD_MODELS_FROM}"

EEG_LENGTH = 20.48  # 脑电样本使用的时间长度
SFREQ = 100  # 脑电样本重采样率
HIGH = 512  # 谱图频率长度
LENGTH = 256  # 谱图时间长度

CONVERTIMAGE = False
CONVERTIMAGE_EEG = False
if CONVERTIMAGE or CONVERTIMAGE_EEG:
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_agg import FigureCanvas

import numpy as np
import pandas as pd
from scipy import signal

filter_range = [0.5, 40]  # 脑电滤波范围
BRAIN = {
    "LL": ["Fp1-F7", "F7-T3", "T3-T5", "T5-O1"],  # 重参考
    "RL": ["Fp2-F8", "F8-T4", "T4-T6", "T6-O2"],
    "LP": ["Fp1-F3", "F3-C3", "C3-P3", "P3-O1"],
    "RP": ["Fp2-F4", "F4-C4", "C4-P4", "P4-O2"],
}

VER = 1  # 版本号

os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"

TF_AVAILABLE = False
TF_IMPORT_ERROR = None
tf = None
keras = None
layers = None

try:
    import tensorflow as tf  # noqa: F401
    from tensorflow import keras  # noqa: F401
    from tensorflow.keras import layers  # noqa: F401

    TF_AVAILABLE = True
except Exception as e:
    TF_AVAILABLE = False
    TF_IMPORT_ERROR = repr(e)

if PLATFORM == "local":
    df = pd.read_csv("./input/hms-harmful-brain-activity-classification/train.csv")
elif PLATFORM == "kaggle":
    df = pd.read_csv(
        "/kaggle/input/hms-harmful-brain-activity-classification/train.csv"
    )

TARGETS = df.columns[-6:]
print("Train shape:", df.shape)
print("Targets", list(TARGETS))
print("TensorFlow available:", TF_AVAILABLE)
if not TF_AVAILABLE:
    print("TensorFlow import error:", TF_IMPORT_ERROR)



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
train = df.groupby("eeg_id")[
    ["spectrogram_id", "spectrogram_label_offset_seconds", "eeg_label_offset_seconds"]
].agg(
    {
        "spectrogram_id": "first",
        "spectrogram_label_offset_seconds": "min",
        "eeg_label_offset_seconds": "median",
    }
)
train.columns = ["spec_id", "min", "eeg_median"]

tmp = df.groupby("eeg_id")[["spectrogram_id", "spectrogram_label_offset_seconds"]].agg(
    {"spectrogram_label_offset_seconds": "max"}
)
train["max"] = tmp

tmp = df.groupby("eeg_id")[["patient_id"]].agg("first")
train["patient_id"] = tmp

tmp = df.groupby("eeg_id")[TARGETS].agg("sum")
for t in TARGETS:
    train[t] = tmp[t].values

y_data = train[TARGETS].values
y_data = y_data / y_data.sum(axis=1, keepdims=True)
train[TARGETS] = y_data

tmp = df.groupby("eeg_id")[["expert_consensus"]].agg("first")
train["target"] = tmp

train = train.reset_index()
print("Train non-overlapp eeg_id shape:", train.shape)



## === cell 2
if NEEDTRAIN:
    if PLATFORM == "local":
        PATH = "./input/hms-harmful-brain-activity-classification/train_spectrograms/"
    elif PLATFORM == "kaggle":
        PATH = "/kaggle/input/hms-harmful-brain-activity-classification/train_spectrograms/"
    files = os.listdir(PATH)
    print(f"There are {len(files)} spectrogram parquets")
    if READ_SPEC_FILES:
        spectrograms = {}
        for i, f in enumerate(files):
            if i % 100 == 0:
                print(i, ", ", end="")
            tmp = pd.read_parquet(f"{PATH}{f}")
            name = int(f.split(".")[0])
            spectrograms[name] = tmp.iloc[:, 1:].values
        if not os.path.exists("./input/brain-spectrograms"):
            os.makedirs("./input/brain-spectrograms")
        np.save("./input/brain-spectrograms/specs.npy", spectrograms, allow_pickle=True)
    else:
        if PLATFORM == "local":
            spectrograms = np.load(
                "./input/brain-spectrograms/specs.npy", allow_pickle=True
            ).item()
        elif PLATFORM == "kaggle":
            spectrograms = np.load(
                "/kaggle/input/brain-spectrograms/specs.npy", allow_pickle=True
            ).item()

    if PLATFORM == "local":
        PATH = "./input/hms-harmful-brain-activity-classification/train_eegs/"
    elif PLATFORM == "kaggle":
        PATH = "/kaggle/input/hms-harmful-brain-activity-classification/train_eegs/"
    files = os.listdir(PATH)
    print(f"There are {len(files)} eeg parquets")
    if READ_EEG_FILES:
        eegs = {}
        for i, f in enumerate(files):
            if i % 100 == 0:
                print(i, ", ", end="")
            raw_eeg = pd.read_parquet(f"{PATH}{f}")
            name = int(f.split(".")[0])
            if len(train[train.eeg_id == name]) > 0:
                eeg_default = raw_eeg.loc[:, :].reset_index(drop=True)
                list_eeg = list()
                for region in BRAIN.keys():
                    eeg = np.zeros(
                        (len(BRAIN[region]), eeg_default.shape[0]), dtype=np.float32
                    )
                    for chan_i, chan in enumerate(BRAIN[region]):
                        eeg[chan_i, :] = (
                            eeg_default.loc[:, chan.split("-")[0]]
                            - eeg_default.loc[:, chan.split("-")[1]]
                        ).values
                    eeg[np.isnan(eeg)] = 0
                    if 200 != SFREQ:
                        eeg = signal.resample_poly(eeg, SFREQ, 200, axis=1)
                    list_eeg.append(np.reshape(eeg, (eeg.shape[0], eeg.shape[1], 1)))
                list_eeg = np.concatenate(list_eeg, 0)
                eegs[name] = list_eeg
        if not os.path.exists("./input/brain-eegs"):
            os.makedirs("./input/brain-eegs")
        np.save("./input/brain-eegs/eegs.npy", eegs, allow_pickle=True)
    else:
        if PLATFORM == "local":
            eegs = np.load("./input/brain-eegs/eegs.npy", allow_pickle=True).item()
        elif PLATFORM == "kaggle":
            eegs = np.load(
                "/kaggle/input/brain-eegs/eegs.npy", allow_pickle=True
            ).item()



## === cell 3
WEIGHT_FILES = []
MODEL_DIR_CANDIDATES = []

if PLATFORM == "kaggle":
    MODEL_DIR_CANDIDATES.append(LOAD_MODELS_FROM)
    MODEL_DIR_CANDIDATES.extend(glob.glob("/kaggle/input/*"))
else:
    MODEL_DIR_CANDIDATES.append(LOAD_MODELS_FROM)
    MODEL_DIR_CANDIDATES.extend(glob.glob("./input/*"))


def discover_weight_files(dirs):
    exts = ("*.h5", "*.keras")
    found = []
    for d in dirs:
        if not d or not os.path.exists(d):
            continue
        for ext in exts:
            found.extend(glob.glob(os.path.join(d, "**", ext), recursive=True))
    found = sorted(set(found))
    return found


if TF_AVAILABLE:
    WEIGHT_FILES = discover_weight_files(MODEL_DIR_CANDIDATES)

print("Discovered weight/model files:", len(WEIGHT_FILES))
if len(WEIGHT_FILES) > 0:
    print("First few:", WEIGHT_FILES[:5])



## === cell 4
if TF_AVAILABLE:
    pass



## === cell 5
if not NEEDTRAIN:
    if PLATFORM == "local":
        test = pd.read_csv("./input/hms-harmful-brain-activity-classification/test.csv")
        sample_sub = pd.read_csv(
            "./input/hms-harmful-brain-activity-classification/sample_submission.csv"
        )
    elif PLATFORM == "kaggle":
        test = pd.read_csv(
            "/kaggle/input/hms-harmful-brain-activity-classification/test.csv"
        )
        sample_sub = pd.read_csv(
            "/kaggle/input/hms-harmful-brain-activity-classification/sample_submission.csv"
        )
    print("Test shape", test.shape)

    prior = df[TARGETS].sum().values.astype(np.float64)
    prior = prior / prior.sum()
    prior = np.clip(prior, 1e-7, 1.0)
    prior = prior / prior.sum()

    patient_mean = train.groupby("patient_id")[TARGETS].mean()
    patient_cnt = train.groupby("patient_id").size().astype(np.int64)

    def alpha_from_count(c):
        return float(c / (c + 5.0))

    eeg2pid = test.set_index("eeg_id")["patient_id"]
    sub = sample_sub[["eeg_id"]].copy()
    sub["patient_id"] = sub["eeg_id"].map(eeg2pid)

    pred_baseline = np.zeros((len(sub), len(TARGETS)), dtype=np.float64)
    for i, pid in enumerate(sub["patient_id"].values):
        if pd.notna(pid) and pid in patient_mean.index:
            pm = patient_mean.loc[pid].values.astype(np.float64)
            pm = np.clip(pm, 1e-7, 1.0)
            pm = pm / pm.sum()
            a = alpha_from_count(int(patient_cnt.loc[pid]))
            pred_baseline[i] = a * pm + (1.0 - a) * prior
        else:
            pred_baseline[i] = prior
    pred_baseline = np.clip(pred_baseline, 1e-7, 1.0)
    pred_baseline = pred_baseline / pred_baseline.sum(axis=1, keepdims=True)

    pred = pred_baseline.copy()
    if TF_AVAILABLE and len(WEIGHT_FILES) > 0:
        try:
            X = pred_baseline.astype(np.float32)

            fold_preds = []
            for wf in WEIGHT_FILES:
                model = keras.models.load_model(wf, compile=False)

                inp_shape = getattr(model, "input_shape", None)
                ok = False
                if (
                    isinstance(inp_shape, tuple)
                    and len(inp_shape) == 2
                    and inp_shape[1] == X.shape[1]
                ):
                    ok = True

                if not ok:
                    del model
                    gc.collect()
                    continue

                p = model.predict(X, batch_size=1024, verbose=0)
                p = np.asarray(p, dtype=np.float64)

                if p.ndim != 2 or p.shape[1] != len(TARGETS):
                    del model
                    gc.collect()
                    continue

                p = np.clip(p, 1e-7, 1.0)
                p = p / p.sum(axis=1, keepdims=True)
                fold_preds.append(p)

                del model
                gc.collect()

            if len(fold_preds) > 0:
                pred = np.mean(fold_preds, axis=0)
                pred = np.clip(pred, 1e-7, 1.0)
                pred = pred / pred.sum(axis=1, keepdims=True)
                print(
                    f"Used TF models for prediction. Models averaged: {len(fold_preds)}"
                )
            else:
                print(
                    "TF available but no compatible model input shapes found; using baseline."
                )
        except Exception as e:
            print("TF model inference failed; using baseline. Error:", repr(e))
            pred = pred_baseline

    pred = np.clip(pred, 1e-7, 1.0)
    pred = pred / pred.sum(axis=1, keepdims=True)

    out = sample_sub.copy()
    for k, t in enumerate(TARGETS):
        out[t] = pred[:, k]

    out[TARGETS] = np.clip(out[TARGETS].values, 1e-7, 1.0)
    out[TARGETS] = out[TARGETS].values / out[TARGETS].sum(axis=1, keepdims=True)

    out.to_csv("submission.csv", index=False)
    print("Wrote submission.csv", out.shape)
    print(
        "Row prob sums: min/mean/max =",
        float(out[TARGETS].sum(axis=1).min()),
        float(out[TARGETS].sum(axis=1).mean()),
        float(out[TARGETS].sum(axis=1).max()),
    )
    print("Submission columns:", list(out.columns))
    print("First row:", out.head(1).to_dict(orient="records")[0])

## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
InvalidIndexError                         Traceback (most recent call last)
/tmp/ipykernel_55/1124224943.py in <cell line: 0>()
     29     eeg2pid = test.set_index("eeg_id")["patient_id"]
     30     sub = sample_sub[["eeg_id"]].copy()
---> 31     sub["patient_id"] = sub["eeg_id"].map(eeg2pid)
     32 
     33     pred_baseline = np.zeros((len(sub), len(TARGETS)), dtype=np.float64)

/usr/local/lib/python3.11/dist-packages/pandas/core/series.py in map(self, arg, na_action)
   4698         dtype: object
   4699         """
-> 4700         new_values = self._map_values(arg, na_action=na_action)
   4701         return self._constructor(new_values, index=self.index, copy=False).__finalize__(
   4702             self, method="map"

/usr/local/lib/python3.11/dist-packages/pandas/core/base.py in _map_values(self, mapper, na_action, convert)
    919             return arr.map(mapper, na_action=na_action)
    920 
--> 921         return algorithms.map_array(arr, mapper, na_action=na_action, convert=convert)
    922 
    923     @final

/usr/local/lib/python3.11/dist-packages/pandas/core/algorithms.py in map_array(arr, mapper, na_action, convert)
   1730         # Since values were input this means we came from either
   1731         # a dict or a series and mapper should be an index
-> 1732         indexer = mapper.index.get_indexer(arr)
   1733         new_values = take_nd(mapper._values, indexer)
   1734 

/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in get_indexer(self, target, method, limit, tolerance)
   3883 
   3884         if not self._index_as_unique:
-> 3885             raise InvalidIndexError(self._requires_unique_msg)
   3886 
   3887         if len(target) == 0:

InvalidIndexError: Reindexing only valid with uniquely valued Index objects
